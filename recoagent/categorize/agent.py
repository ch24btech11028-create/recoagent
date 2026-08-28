"""C2: the model, on the residue the deterministic rungs could not reach.

The contract is the one `agent/citations.py` already argues for, applied to a
different decision. Stated once, since it is the only interesting thing here:

**The model does not choose a category. It quotes the words that choose one.**

A model asked "what is this row?" will answer, always, and the answer will be
fluent whether or not anything in the book supports it. So the reply is not a
category; it is a category *plus a verbatim span from that row's own text*. The
span is then checked -- literally, `in` the source string -- and a category
whose quotation does not appear is not downgraded in confidence, it is
discarded and the row goes to a human.

This is weaker than it sounds in one direction and stronger in another. Weaker:
a model can quote correctly and still classify wrongly, and that is a real
error this does not catch. Stronger: a model cannot classify a row on the
strength of a narration it imagined, which is the failure mode that produces
confident, plausible, unfalsifiable bookkeeping.

Confidence is recorded and never gates anything on its own. A floor is applied
*after* the citation check, not instead of it -- a self-reported 0.97 on a
fabricated quotation is worth less than a 0.6 on a real one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from ..budget import BudgetExhausted
from ..llm import Chat, Usage
from ..schemas import SourceBundle
from .rules import Assignment, Ledger
from .taxonomy import DEFINITIONS, PROPOSABLE, Category

#: Below this, a correctly-cited category is still filed for review. It is a
#: second gate, never the first one.
CONFIDENCE_FLOOR = 0.55

#: Token budget for one reply. Sized for a reasoning model's hidden tokens, not
#: for the four fields it eventually emits.
REPLY_BUDGET = 3000

SYSTEM = """You categorise rows from an Indian payments book for a merchant's accounts.

You will be given ONE row and the exact text that row carries. Reply with JSON:

  {"category": "<one of the categories below>",
   "quote": "<a span copied EXACTLY from the row text above>",
   "confidence": <0.0-1.0>,
   "reason": "<one sentence>"}

Rules, in order of importance:

1. `quote` must be a substring copied character-for-character out of the row
   text you were given. It is checked. If you cannot find a span in that text
   that supports your category, reply {"category": "needs_review", "quote": "",
   "confidence": 0.0, "reason": "why the row is not self-describing"}.
   Answering "needs_review" is a correct answer, not a failure.
2. Never infer a category from the amount alone. A round number is not
   evidence of anything.
3. A credit from a payment gateway into the merchant's own bank account is
   settlement_credit, NOT sales_revenue. The sale was already recorded when the
   customer paid; recording it again doubles the merchant's turnover.

Categories:
{categories}
"""


@dataclass
class Proposal:
    category: Category
    quote: str
    confidence: float
    reason: str


@dataclass
class CategoryCase:
    entity_id: str
    entity_kind: str
    outcome: str          # assigned | needs_review | uncited | refused | failed
    detail: str = ""
    proposal: Proposal | None = None
    usage: Usage = field(default_factory=Usage)


@dataclass
class CategoryReport:
    cases: list[CategoryCase] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    def _count(self, outcome: str) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def attempted(self) -> int:
        return len(self.cases)

    @property
    def assigned(self) -> int:
        return self._count("assigned")

    @property
    def uncited(self) -> int:
        """Replies discarded because the quotation was not in the source text.

        Reported on its own and never folded into "failed". A model that
        invents evidence is a different problem from a model that times out,
        and the whole design rests on knowing how often it happens.
        """
        return self._count("uncited")

    @property
    def declined(self) -> int:
        return self._count("needs_review")

    @property
    def failed(self) -> int:
        return self._count("failed")

    @property
    def not_asked(self) -> int:
        """Rows the daily request budget ran out before reaching.

        Kept strictly apart from `failed`. A row nobody asked about is not a
        row the model got wrong, and folding the two together turns "the key
        has a 20-request daily cap" into "the model could not do this" -- which
        is how a quota becomes a finding about a model.
        """
        return self._count("not_asked")


class Categoriser(Protocol):
    def propose(self, kind: str, entity_id: str, text: str) -> Proposal | str: ...


def row_text(sources: SourceBundle, entity_id: str, kind: str) -> str:
    """Every word the book carries about this row, and nothing else.

    Assembled here so the citation check has one definite string to check
    against. If the model could be shown text that is not in this string, the
    check would be a formality.
    """
    if kind == "bank_line":
        line = next((b for b in sources.bank_lines if b.bank_line_id == entity_id), None)
        if line is None:
            return ""
        return f"narration: {line.narration}\nreference: {line.bank_ref}"
    if kind == "adjustment":
        adj = next((a for a in sources.adjustments if a.adjustment_id == entity_id), None)
        if adj is None:
            return ""
        return (
            f"kind: {adj.kind}\n"
            f"linked payment: {adj.payment_id or 'none'}\n"
            f"linked settlement: {adj.settlement_id or 'none'}"
        )
    if kind == "payment":
        pay = next((p for p in sources.payments if p.payment_id == entity_id), None)
        if pay is None:
            return ""
        return (
            f"status: {pay.status}\nmethod: {pay.method}\n"
            f"order: {pay.order_id or 'none'}\n"
            f"settlement: {pay.settlement_id or 'none'}"
        )
    return ""


def _normalise(text: str) -> str:
    """Fold whitespace and case for the citation check.

    Deliberately this much and no more. Punctuation and digits are left alone,
    because a "quotation" that matches only after the digits are stripped is
    not a quotation of anything in particular.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def parse(reply: str) -> Proposal | str:
    """Read a proposal, or return the reason it could not be read."""
    text = reply.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return "reply contained no JSON object"
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return f"reply was not valid JSON: {exc}"

    raw = str(data.get("category", "")).strip().lower()
    try:
        category = Category(raw)
    except ValueError:
        return f"unknown category {raw!r}"
    if category is not Category.NEEDS_REVIEW and category not in PROPOSABLE:
        # SALES_REVENUE, GATEWAY_FEE and GST_INPUT_CREDIT come out of the
        # reconciliation with a proof attached. A model reaching for one of
        # them is reaching past the arithmetic.
        return f"{raw!r} is determined by the reconciliation, not proposable"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Proposal(
        category=category,
        quote=str(data.get("quote", "")),
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(data.get("reason", ""))[:300],
    )


class ChatCategoriser:
    """A `Chat` behind the proposal contract."""

    def __init__(self, chat: Chat) -> None:
        self.chat = chat
        self.usage = Usage()

    def propose(self, kind: str, entity_id: str, text: str) -> Proposal | str:
        categories = "\n".join(
            f"  {c.value}: {DEFINITIONS[c]}" for c in sorted(PROPOSABLE, key=lambda c: c.value)
        )
        reply = self.chat.send(
            SYSTEM.replace("{categories}", categories),
            f"Row {entity_id} ({kind}). The text this row carries:\n\n{text}\n",
            # The reply is four short fields, so 500 was the obvious budget and
            # it was wrong: a reasoning model spends the allowance on thinking
            # tokens and returns `finish_reason=length` with empty content.
            # Measured on gemini-3.6-flash -- 19 of 20 rows came back blank.
            # The cap is on the whole generation, not on the visible answer.
            max_tokens=REPLY_BUDGET,
        )
        self.usage.merge(reply.usage)
        if not reply.ok:
            return reply.error or "empty reply"
        return parse(reply.text)


def run_c2(
    sources: SourceBundle,
    ledger: Ledger,
    categoriser: Categoriser,
    *,
    floor: float = CONFIDENCE_FLOOR,
) -> CategoryReport:
    """Ask about each uncategorised row; accept only what the row's text supports."""
    from .rules import residue

    report = CategoryReport()

    exhausted = False

    for entity_id, kind, amount in residue(sources, ledger):
        text = row_text(sources, entity_id, kind)
        case = CategoryCase(entity_id=entity_id, entity_kind=kind, outcome="failed")

        if exhausted:
            case.outcome = "not_asked"
            case.detail = "daily request budget spent before this row"
            _review(ledger, entity_id, kind, amount, "not asked: request budget spent")
            report.cases.append(case)
            continue

        try:
            proposal = categoriser.propose(kind, entity_id, text)
        except BudgetExhausted as exc:
            exhausted = True
            case.outcome = "not_asked"
            case.detail = str(exc)
            _review(ledger, entity_id, kind, amount, "not asked: request budget spent")
            report.cases.append(case)
            continue
        if isinstance(proposal, str):
            case.detail = proposal
            report.cases.append(case)
            _review(ledger, entity_id, kind, amount, f"model failed: {proposal}")
            continue

        case.proposal = proposal

        if proposal.category is Category.NEEDS_REVIEW:
            case.outcome = "needs_review"
            case.detail = proposal.reason
            _review(ledger, entity_id, kind, amount, f"model declined: {proposal.reason}")
            report.cases.append(case)
            continue

        if _normalise(proposal.quote) not in _normalise(text) or not proposal.quote.strip():
            case.outcome = "uncited"
            case.detail = f"quoted {proposal.quote!r}, which is not in the row"
            _review(
                ledger, entity_id, kind, amount,
                f"cited {proposal.quote!r}, absent from the row's own text",
            )
            report.cases.append(case)
            continue

        if proposal.confidence < floor:
            case.outcome = "needs_review"
            case.detail = f"cited correctly but at {proposal.confidence:.2f}"
            _review(
                ledger, entity_id, kind, amount,
                f"{proposal.category.value} at {proposal.confidence:.2f}, below the floor",
            )
            report.cases.append(case)
            continue

        # Accepted, and *held* rather than booked. Measured on this book: of 20
        # rows the model was asked about it correctly declined 16, fabricated
        # nothing, and got 3 of the 4 it committed to wrong -- every one of
        # them quoting the row correctly. A citation proves the evidence
        # exists; it does not prove the conclusion follows from it.
        #
        # So the proposal reaches the operator queue with its quotation
        # attached and does not enter the books. This is the same treatment
        # `agent/tier.py` gives a model-chosen rate: a hypothesis a human has
        # not confirmed is a hypothesis, however well cited.
        case.outcome = "assigned"
        ledger.add(Assignment(
            entity_id=entity_id,
            entity_kind=kind,
            category=proposal.category,
            amount_paise=amount,
            rung="C2",
            rule_id="c2.cited_by_model",
            evidence=f"quoted from the row: {proposal.quote!r}",
            confidence=proposal.confidence,
            verified=False,
        ))
        report.cases.append(case)

    if isinstance(categoriser, ChatCategoriser):
        report.usage = categoriser.usage
        report.model = categoriser.chat.label
    return report


def _review(ledger: Ledger, entity_id: str, kind: str, amount: int, why: str) -> None:
    """File a row for a human, saying which way the model failed it.

    Every path out of C2 that is not an accepted assignment lands here. A row
    the model could not justify must still appear somewhere, or the exception
    list stops being honest at exactly the point it matters most.
    """
    ledger.add(Assignment(
        entity_id=entity_id,
        entity_kind=kind,
        category=Category.NEEDS_REVIEW,
        amount_paise=amount,
        rung="C2",
        rule_id="c2.unresolved",
        evidence=why,
        verified=False,
    ))
