"""The settlement Q&A agent, and the harness that scores it.

Ask it about a reconciliation run and it answers with a number or an id -- or
says it cannot. The design mirrors the reconciliation tier deliberately, because
the same failure is available here and it is the expensive one: a model that
answers confidently and wrongly is worse than a model that declines, since an
operator acts on the number either way.

So the lead metric is **wrong-answer rate**, not accuracy. Coverage is reported
second. That is the same ordering as false-match rate over match rate, for the
same reason.

Two things keep the numbers honest:

- **Facts are retrieved deterministically, not recalled.** The agent is handed a
  compact factsheet built from the run by code, and told to answer only from it.
  It has no memory of the batch and no way to invent one.
- **Answers are typed and compared exactly.** `bank.is_correct` does the
  grading, not a second model. A benchmark graded by a model is not a benchmark.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import Chat, Usage
from ..money import format_inr
from ..schemas import LabelledBatch, ReconResult
from .bank import Question, is_correct

SYSTEM = """\
You answer questions about a completed payment reconciliation run for an Indian \
merchant. You are given a factsheet built from that run and one question.

Rules:
- Answer ONLY from the factsheet. It is the complete record; nothing else exists.
- All money is in integer paise. 100 paise = 1 rupee. Never convert to rupees.
- A negative gap means the bank credited LESS than the batch rows account for.
- If the factsheet does not contain what you need, say so. Declining is a good \
answer; a confident wrong number is the worst one available to you, because the \
operator will act on it either way.

Reply with a single JSON object and nothing else -- no prose, no markdown fence:
{"answer": <number, id string, or true/false>, "confidence": 0.0, "basis": "which fact you used"}
or
{"cannot_answer": "what is missing"}"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Answer:
    qid: str
    given: Any = None
    correct: bool = False
    declined: bool = False
    failed: bool = False
    confidence: float | None = None
    basis: str = ""
    detail: str = ""
    #: False when the question had no ground truth to check against. Such an
    #: answer is shown to the operator with its factsheet attached and is
    #: excluded from every rate this module reports.
    graded: bool = True
    usage: Usage = field(default_factory=Usage)


@dataclass
class QAReport:
    answers: list[Answer] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.scored)

    @property
    def scored(self) -> list[Answer]:
        """Only answers with a ground truth behind them. Every rate uses these."""
        return [a for a in self.answers if a.graded]

    @property
    def attempted(self) -> int:
        return sum(1 for a in self.scored if not a.declined and not a.failed)

    @property
    def correct(self) -> int:
        """Includes correctly declining an unanswerable question."""
        return sum(1 for a in self.scored if a.correct)

    @property
    def correct_answers(self) -> int:
        """Correct among questions it actually answered. Excludes right declines."""
        return sum(1 for a in self.scored if a.correct and not a.declined)

    @property
    def wrong(self) -> int:
        # Must be counted against answers given, not against `correct` -- a
        # correct decline raises `correct` without raising `attempted`, which
        # drove this negative (-6.45%) on the first run with decline probes.
        return self.attempted - self.correct_answers

    @property
    def declined(self) -> int:
        return sum(1 for a in self.scored if a.declined)

    @property
    def hallucinated(self) -> int:
        """Answered a question the factsheet cannot support. The worst outcome."""
        return sum(1 for a in self.scored if a.detail.startswith("HALLUCINATED"))

    @property
    def failed(self) -> int:
        return sum(1 for a in self.scored if a.failed)

    @property
    def wrong_answer_rate(self) -> float:
        """The metric that leads. Share of answers given that were wrong."""
        return self.wrong / self.attempted if self.attempted else 0.0

    @property
    def coverage(self) -> float:
        return self.attempted / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


#: The queue is the product, so it goes in the factsheet -- but bounded, and
#: labelled with how much was left out. An agent told "here are 25" when there
#: are 40 will answer "40 exceptions?" with 25 and be confidently wrong.
EXCEPTION_LIMIT = 10

_FEE_WORDS = ("fee", "fees", "mdr", "commission", "tax", "gst", "charge", "charges")
_FX_WORDS = ("fx", "currency", "conversion", "forex", "exchange", "international")
_RATE_WORDS = ("rate", "repricing", "repriced", "notice", "schedule", "bps")
_ADJ_WORDS = ("adjustment", "adjustments", "refund", "refunds", "chargeback",
              "chargebacks", "dispute", "reversal")


def _mentions(text: str, words) -> bool:
    lowered = text.lower()
    return any(re.search(rf"\b{w}\b", lowered) for w in words)


def factsheet(batch: LabelledBatch, result: ReconResult, question: Question) -> str:
    """Everything the agent may see, built by code from the run.

    Portfolio totals and the open queue always; per-entity detail only for what
    the question names, and a topic section only when the question is about that
    topic. Handing over the whole book would make the context enormous and would
    also stop measuring anything useful -- an agent that can see every row is
    being tested on arithmetic, not on retrieval plus reasoning.

    The queue is here because it is what anyone actually asks about. It used to
    be absent, and the agent correctly declined every question a person would
    think to ask -- "why is this one open?", "which orders are duplicated?" --
    because seven aggregate counts genuinely could not answer them. That is a
    retrieval failure wearing a model's clothes.
    """
    sources = batch.sources
    matched = result.matches_for_leg(2)
    matched_ids = {m.left_ids[0] for m in matched}
    residual_excs = [
        e for e in result.exceptions
        if e.leg == 2 and e.entity_kind == "bank_line" and e.residual_paise is not None
    ]
    worst = max(residual_excs, key=lambda e: abs(e.residual_paise or 0), default=None)

    facts: dict[str, Any] = {
        "portfolio": {
            "orders": len(sources.orders),
            "payments": len(sources.payments),
            "settlement_batches": len(sources.settlements),
            "bank_credits_in_statement": len(sources.bank_lines),
            "unlinked_adjustment_rows": len(sources.unlinked_adjustments),
            "orders_matched_to_a_payment": len(result.matches_for_leg(1)),
            "credits_matched_to_a_batch": len(matched),
            "open_exception_items": len(result.exceptions),
            "open_exceptions_leg1_order_to_payment": len(result.exceptions_for_leg(1)),
            "open_exceptions_leg2_credit_to_batch": len(result.exceptions_for_leg(2)),
            "total_unexplained_paise": sum(
                abs(e.residual_paise or 0) for e in result.exceptions
            ),
            # Money on matched rows that does not agree -- a declared partial
            # capture, mostly. Not unexplained, but not nothing either.
            "documented_variance_paise": sum(m.variance_paise for m in result.matches),
        }
    }
    if worst is not None:
        facts["largest_unexplained_gap"] = {
            "bank_line_id": worst.entity_id,
            "gap_paise": worst.residual_paise,
        }

    # ── the open queue ───────────────────────────────────────────────────
    if result.exceptions:
        ranked = sorted(
            result.exceptions,
            key=lambda e: (-abs(e.residual_paise or 0), e.entity_id),
        )
        facts["open_exceptions"] = {
            "total": len(ranked),
            "shown": min(len(ranked), EXCEPTION_LIMIT),
            "ordered_by": "largest unexplained amount first",
            "note": (
                "This list is the whole queue."
                if len(ranked) <= EXCEPTION_LIMIT else
                f"Only the first {EXCEPTION_LIMIT} of {len(ranked)} are listed. "
                "Use `total` for any count question."
            ),
            "items": [
                {
                    "entity_id": e.entity_id,
                    "leg": e.leg,
                    "reason": e.reason,
                    "suspected_class": e.suspected_class.name if e.suspected_class else None,
                    "residual_paise": e.residual_paise,
                }
                for e in ranked[:EXCEPTION_LIMIT]
            ],
        }
        by_class: dict[str, int] = {}
        for e in result.exceptions:
            key = e.suspected_class.name if e.suspected_class else "unclassified"
            by_class[key] = by_class.get(key, 0) + 1
        facts["open_exceptions"]["counts_by_suspected_class"] = dict(sorted(by_class.items()))

    # ── topic sections, only when asked about ────────────────────────────
    text = question.text
    if _mentions(text, _FEE_WORDS):
        by_method: dict[str, dict[str, int]] = {}
        for pay in sources.payments:
            row = by_method.setdefault(pay.method, {"payments": 0, "fee_paise": 0, "tax_paise": 0})
            row["payments"] += 1
            row["fee_paise"] += pay.fee_paise
            row["tax_paise"] += pay.tax_paise
        facts["fees_and_tax"] = {
            "total_fee_paise": sum(p.fee_paise for p in sources.payments),
            "total_tax_paise": sum(p.tax_paise for p in sources.payments),
            "note": "tax is GST on the fee, and is stated separately from it",
            "by_method": dict(sorted(by_method.items())),
        }
    if _mentions(text, _FX_WORDS):
        facts["fx"] = {
            "international_payments": sum(1 for p in sources.payments if p.currency != "INR"),
            "advices_on_file": [
                {
                    "advice_id": a.advice_id,
                    "payment_id": a.payment_id,
                    "rate_pct_of_gross": a.rate_pct_of_gross,
                    "reference": a.reference,
                }
                for a in sources.fx_advices
            ],
        }
    if _mentions(text, _RATE_WORDS):
        facts["rate_notices_on_file"] = [
            {
                "notice_id": n.notice_id,
                "method": n.method,
                "mdr_bps": n.mdr_bps,
                "effective_from": n.effective_from.isoformat(),
                "effective_to": n.effective_to.isoformat() if n.effective_to else None,
                "reference": n.reference,
            }
            for n in sources.rate_notices
        ]
    if _mentions(text, _ADJ_WORDS):
        by_kind: dict[str, dict[str, int]] = {}
        for adj in sources.adjustments:
            row = by_kind.setdefault(adj.kind, {"rows": 0, "amount_paise": 0})
            row["rows"] += 1
            row["amount_paise"] += adj.amount_paise
        facts["adjustments"] = {
            "rows": len(sources.adjustments),
            "not_linked_to_a_settlement": len(sources.unlinked_adjustments),
            "by_kind": dict(sorted(by_kind.items())),
        }

    # ── the entities the question names ──────────────────────────────────
    # Every id shape in the book, not a hardcoded two. A live question carries
    # no `depends_on`, so typing an id is the operator's only way to ask about
    # one row -- and it used to work for bank lines and settlements alone.
    known = {o.order_id for o in sources.orders}
    known |= {p.payment_id for p in sources.payments}
    known |= {s.settlement_id for s in sources.settlements}
    known |= {b.bank_line_id for b in sources.bank_lines}
    known |= {a.adjustment_id for a in sources.adjustments}
    spoken = set(re.findall(r"[A-Za-z0-9_]+", text))
    named = set(question.depends_on) | (spoken & known)

    if named:
        detail: dict[str, Any] = {}
        for line in sources.bank_lines:
            if line.bank_line_id in named:
                exc = next((e for e in residual_excs if e.entity_id == line.bank_line_id), None)
                match = next((m for m in matched if m.left_ids[0] == line.bank_line_id), None)
                detail[line.bank_line_id] = {
                    "credited_amount_paise": line.amount_paise,
                    "value_date": line.value_date.isoformat(),
                    "matched_to_settlement": match.right_ids[0] if match else None,
                    "is_matched": line.bank_line_id in matched_ids,
                    "gap_paise": exc.residual_paise if exc else None,
                    "status": "matched" if match else "open in the exception queue",
                }
        for s_ in sources.settlements:
            if s_.settlement_id in named:
                members = sources.payments_by_settlement(s_.settlement_id)
                detail[s_.settlement_id] = {
                    "payments_in_batch": len(members),
                    "reported_net_paise": s_.net_paise,
                    "settled_at": s_.settled_at.isoformat(),
                    "status": s_.status,
                }
        for order in sources.orders:
            if order.order_id in named:
                claims = [p for p in sources.payments if p.order_id == order.order_id]
                exc = next((e for e in result.exceptions if e.entity_id == order.order_id), None)
                detail[order.order_id] = {
                    "order_amount_paise": order.amount_paise,
                    "payment_attempts": [
                        {"payment_id": p.payment_id, "status": p.status,
                         "gross_paise": p.gross_paise, "method": p.method}
                        for p in claims
                    ],
                    "status": exc.reason if exc else "matched to a payment",
                }
        for pay in sources.payments:
            if pay.payment_id in named:
                detail[pay.payment_id] = {
                    "order_id": pay.order_id,
                    "gross_paise": pay.gross_paise,
                    "fee_paise": pay.fee_paise,
                    "tax_paise": pay.tax_paise,
                    "method": pay.method,
                    "status": pay.status,
                    "settlement_id": pay.settlement_id,
                    "currency": pay.currency,
                }
        for adj in sources.adjustments:
            if adj.adjustment_id in named:
                detail[adj.adjustment_id] = {
                    "kind": adj.kind,
                    "amount_paise": adj.amount_paise,
                    "payment_id": adj.payment_id,
                    "settlement_id": adj.settlement_id,
                }
        if detail:
            facts["entities"] = detail

    return json.dumps(facts, indent=2, sort_keys=True)


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON.search(text)
        if not m:
            raise ValueError(f"no JSON object in reply: {text[:120]!r}")
        return json.loads(m.group(0))


def ask(
    chat: Chat,
    batch: LabelledBatch,
    result: ReconResult,
    question: Question,
    tolerance_paise: int = 0,
) -> Answer:
    """Ask one question. Never raises; a failed call is a recorded outcome."""
    ans = Answer(qid=question.qid)
    reply = chat.send(
        SYSTEM,
        json.dumps(
            {"factsheet": json.loads(factsheet(batch, result, question)),
             "question": question.text},
            indent=2,
        ),
        max_tokens=2000,
    )
    ans.usage.merge(reply.usage)
    if not reply.ok:
        ans.failed = True
        ans.detail = reply.error or "empty reply"
        return ans

    try:
        payload = _parse(reply.text)
    except (ValueError, json.JSONDecodeError) as exc:
        ans.failed = True
        ans.detail = f"malformed: {exc}"
        return ans

    if "cannot_answer" in payload:
        ans.declined = True
        ans.detail = str(payload["cannot_answer"])[:120]
        # Declining an unanswerable question is the right answer, not a gap in
        # coverage. Scoring it as a miss would push the agent toward guessing.
        ans.correct = question.expects_decline
        return ans

    ans.given = payload.get("answer")
    ans.basis = str(payload.get("basis", ""))[:160]
    try:
        ans.confidence = float(payload.get("confidence")) if payload.get("confidence") is not None else None
    except (TypeError, ValueError):
        ans.confidence = None
    if question.expects_decline:
        ans.correct = False
        ans.detail = (
            f"HALLUCINATED {ans.given!r} -- the factsheet does not contain this"
        )
        return ans
    if not question.graded:
        # No ground truth exists for this one. Say so rather than reporting a
        # `correct` of False, which reads as "the agent got it wrong".
        ans.graded = False
        ans.detail = "ungraded: asked live, no ground truth to check against"
        return ans
    ans.correct = is_correct(question, ans.given, tolerance_paise)
    if not ans.correct:
        ans.detail = f"said {ans.given!r}, expected {question.answer!r}"
    return ans


def render(report: QAReport, questions: list[Question]) -> str:
    by_id = {q.qid: q for q in questions}
    w = 72
    out = [
        "=" * w,
        "SETTLEMENT Q&A",
        "=" * w,
        "",
        f"  WRONG-ANSWER RATE   {report.wrong_answer_rate:>8.2%}   <- lead metric",
        f"  coverage            {report.coverage:>8.2%}   ({report.attempted} of {report.total} answered)",
        f"  accuracy            {report.accuracy:>8.2%}   (correct out of all questions)",
        "",
        f"  HALLUCINATED        {report.hallucinated:>8}   "
        "(answered a question the factsheet cannot support)",
        "",
        f"  correct {report.correct}   wrong {report.wrong}   "
        f"declined {report.declined}   call failed {report.failed}",
        f"  tokens {report.usage.input_tokens:,} in / {report.usage.output_tokens:,} out"
        f"   {report.seconds:.0f}s",
        "",
        "-" * w,
        "  WHAT IT GOT WRONG, DECLINED, OR FAILED",
        "-" * w,
    ]
    bad = [a for a in report.answers if not a.correct]
    if not bad:
        out.append("    nothing")
    for a in bad:
        q = by_id.get(a.qid)
        state = "declined" if a.declined else ("failed" if a.failed else "WRONG")
        out.append(f"    [{state}] {q.text[:60] if q else a.qid}")
        out.append(f"             {a.detail[:100]}")
    out += [
        "-" * w,
        "  A wrong answer is worse than a declined one: the operator acts on the",
        "  number either way. That is why the wrong-answer rate leads and why",
        "  declining is scored as a cost, not a failure.",
        "=" * w,
    ]
    return "\n".join(out)
