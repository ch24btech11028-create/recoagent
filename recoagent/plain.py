"""The same finding, in the language of the person whose money it is.

Everything else in this repository is written for an analyst. `agent/proposer.py`
opens by telling the model it is a reconciliation analyst; `qa/agent.py` forbids
it from converting paise to rupees so answers can be graded by exact comparison.
Both are right for their readers. Neither produces a sentence a merchant can act
on, and a merchant is who the money belongs to:

    ambiguous: 2 payments claim this order (pay_00033, pay_00033_retry)
    The residual of -947 paise equals 1.9979% of the card_domestic gross

**The rule this module exists to keep: the plain sentence is generated from the
proof, never asked of a model.** The tempting fix is a second, friendlier field
in the proposer's reply. That would put a fluent, confident sentence in front of
the reader least equipped to doubt it, with nothing checking it against the
ledger -- the one ungrounded generative surface in a system whose entire
argument is that nothing is asserted without proof.

It is also unnecessary. By the time anything reaches a screen the numbers are
already known: `agent/citations.py` has priced every cited row, named the
payments, and recorded the claimed rate against the rate on file. The facts
exist; only the phrasing is missing, and phrasing is a template. So every figure
in the text below is computed by code, which is why
`tests/test_plain.py` can assert that the rupees in the sentence equal the
rupees in the proof. No sentence a model wrote could be checked that way.

Three rules hold throughout.

**Rupees, in words, once.** Money is formatted by `money.format_inr` and nothing
else -- there is one way to write a rupee in this repository and this is not a
second one. The sign is carried by English ("short", "more than expected")
rather than by a minus sign, because "-Rs 9.47 short" says it twice and a
merchant reads the minus as a second deduction.

**No identifiers in the prose.** `pay_00033` means nothing to the person who
took the payment. Ids stay in `ReconException.reason` and in the case file's
evidence panel, where an operator needs them and expects them.

**A refusal must still read as a refusal.** The failure this module could
introduce is a warm summary of a held item that leaves the reader thinking it
was settled. Every account that was not booked says so in its own sentence, and
`test_a_held_item_never_reads_as_a_settled_one` holds the line.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from .defects import DefectClass
from .money import Paise, format_inr
from .schemas import ReconException, SourceBundle

#: How a payment method is said out loud. The keys are the gateway's own method
#: strings; the values are what a merchant would call the same thing.
METHOD_WORDS = {
    "upi": "a UPI payment",
    "rupay_debit": "a RuPay debit payment",
    "netbanking": "a netbanking payment",
    "card_domestic": "a card payment",
    "card_international": "an international card payment",
    "wallet": "a wallet payment",
    "emi": "an EMI payment",
}

PLURAL_METHOD_WORDS = {
    "upi": "UPI payments",
    "rupay_debit": "RuPay debit payments",
    "netbanking": "netbanking payments",
    "card_domestic": "card payments",
    "card_international": "international card payments",
    "wallet": "wallet payments",
    "emi": "EMI payments",
}

COUNT_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


@dataclass(frozen=True)
class PlainAccount:
    """One finding, said plainly. Four parts, because a merchant asks four things.

    `headline` is what happened and how much. `body` is why we think so.
    `status` is what the system did about it -- and for anything not booked, it
    is the sentence that has to survive being skim-read. `next_step` is the one
    action worth taking.
    """

    headline: str
    body: tuple[str, ...]
    status: str
    next_step: str

    @property
    def text(self) -> str:
        return " ".join([self.headline, *self.body, self.status, self.next_step]).strip()

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "body": list(self.body),
            "status": self.status,
            "next_step": self.next_step,
            "text": self.text,
        }


# ── saying things ────────────────────────────────────────────────────────────


def rupees(paise: Paise) -> str:
    """Money as a merchant reads it. Unsigned -- direction is a word, not a dash."""
    return format_inr(abs(paise))


def count(n: int) -> str:
    return COUNT_WORDS.get(n, str(n))


def method_words(method: str, n: int = 1) -> str:
    if n == 1:
        return METHOD_WORDS.get(method, "a payment")
    return f"{count(n)} {PLURAL_METHOD_WORDS.get(method, 'payments')}"


def when(moment: date | datetime | None) -> str:
    """`4 July`. No year: every book on one screen is one period."""
    if moment is None:
        return ""
    day = moment.date() if isinstance(moment, datetime) else moment
    return f"{day.day} {day:%B}"


def rate(bps: int) -> str:
    return f"{bps / 100:.2f}%"


def _payout_phrase(sources: SourceBundle, exc: ReconException) -> str:
    """`the payout that reached your bank on 4 July`, when we can date it."""
    line = next(
        (b for b in sources.bank_lines if b.bank_line_id == exc.entity_id), None
    )
    if line is not None:
        return f"the payout that reached your bank on {when(line.value_date)}"
    settlement = next(
        (s for s in sources.settlements
         if s.settlement_id in (exc.entity_id, exc.related_id)), None
    )
    if settlement is not None:
        return f"the payout the gateway sent on {when(settlement.settled_at)}"
    return "this payout"


def _short_or_over(residual: Paise) -> str:
    return "short" if residual < 0 else "more than expected"


# ── the deterministic queue ──────────────────────────────────────────────────
#
# These are the rows a merchant actually sees. On the shipped default book the
# model writes none of them, so this is where the readability problem mostly
# lives -- and none of it needs a model to fix.


def _duplicate_payment(sources: SourceBundle, exc: ReconException) -> PlainAccount:
    payments = [p for p in sources.payments if p.order_id == exc.entity_id]
    total = sum(p.gross_paise for p in payments)
    order = next((o for o in sources.orders if o.order_id == exc.entity_id), None)
    amount = f" for {rupees(order.amount_paise)}" if order else ""
    times = {2: "twice", 3: "three times"}.get(len(payments), "more than once")
    return PlainAccount(
        headline=f"One order{amount} was paid {times}.",
        body=(
            "This usually means the customer tried again after a payment "
            "appeared to fail, and both attempts went through.",
            f"Together they come to {rupees(total)}." if total else "",
        ),
        status="We have not picked one, because guessing which is the real "
               "sale would put the other one in your books as revenue you "
               "never earned.",
        next_step="Check whether the customer was charged twice. If they were, "
                 "refund one and this clears itself on the next run.",
    )


def _duplicate_credit(sources: SourceBundle, exc: ReconException) -> PlainAccount:
    line = next(
        (b for b in sources.bank_lines if b.bank_line_id == exc.entity_id), None
    )
    amount = f" of {rupees(line.amount_paise)}" if line else ""
    dated = f" on {when(line.value_date)}" if line else ""
    return PlainAccount(
        headline=f"The same payout{amount} appears twice on your bank statement{dated}.",
        body=(
            "Banks sometimes restate a credit, and a statement pulled across "
            "overlapping dates can carry it twice. Only one of them is money "
            "that actually arrived.",
        ),
        status="We have counted neither, because counting both would show "
               "cash you do not have.",
        next_step="Ask your bank which line is the live one. Once you tell us, "
                 "the other is written off in a click.",
    )


def _no_credit(sources: SourceBundle, exc: ReconException) -> PlainAccount:
    settlement = next(
        (s for s in sources.settlements
         if s.settlement_id in (exc.entity_id, exc.related_id)), None
    )
    amount = f" of {rupees(settlement.net_paise)}" if settlement else ""
    dated = f" on {when(settlement.settled_at)}" if settlement else ""
    held = settlement is not None and settlement.status == "on_hold"
    return PlainAccount(
        headline=f"The gateway reported a payout{amount}{dated} that never reached your bank.",
        body=(
            "The gateway is holding it in your reserve balance rather than "
            "sending it." if held else
            "The settlement report says it was paid. No matching credit "
            "appears on the statement you gave us.",
        ),
        status="We have not treated this as money received.",
        next_step="Ask the gateway when it will be released. Nothing is wrong "
                 "with your books until it is paid and still missing.",
    )


# ── a payout that does not add up ────────────────────────────────────────────


def _rate_claim_sentence(row) -> str:
    """Turn one priced fee-variance row into English, from the row's own numbers.

    Reads `ResolvedRow.detail`, never `derivation`. The prose and the audit
    string are two renderings of the same numbers, not one parsed out of the
    other.
    """
    d = row.detail or {}
    method, claimed, schedule = (
        d.get("method"), d.get("claimed_bps"), d.get("schedule_bps"),
    )
    if method is None:
        return ""
    who = method_words(method, len(row.cited_ids)).capitalize()
    if claimed is None or schedule is None:
        return f"{who} on this payout were charged at a rate your rate card does not show."
    direction = "higher" if claimed > schedule else "lower"
    return (
        f"{who} on this payout look like they were charged at {rate(claimed)} "
        f"rather than the {rate(schedule)} on your rate card — a {direction} rate."
    )


def _residual(
    sources: SourceBundle,
    exc: ReconException,
    resolution=None,
    verified: bool = False,
) -> PlainAccount:
    residual = exc.residual_paise or 0
    payout = _payout_phrase(sources, exc)
    headline = (
        f"{rupees(residual)} {_short_or_over(residual)} on {payout}."
    )

    if resolution is None or not getattr(resolution, "rows", ()):
        return PlainAccount(
            headline=headline,
            body=(
                "We checked every payment, refund and adjustment the gateway "
                "linked to this payout, and they do not account for the "
                "difference.",
            ),
            status="We have left it open rather than write the difference off.",
            next_step="Ask the gateway for a breakdown of this payout. The "
                     "amount above is exactly what is unexplained.",
        )

    sentences: list[str] = []
    for row in resolution.rows:
        if row.source == "fee_variance":
            said = _rate_claim_sentence(row)
            if said:
                sentences.append(said)
        elif row.source == "fx":
            pct = (row.detail or {}).get("pct")
            slipped = (
                f" — about {abs(pct):.2f}% of the sale"
                if isinstance(pct, (int, float)) else ""
            )
            sentences.append(
                "An international payment was converted at a rate the "
                f"settlement report does not show{slipped}."
            )
        elif row.source == "adjustment":
            sentences.append(
                "An adjustment the gateway made was taken out of this payout "
                "without appearing in its list of payments."
            )
    sentences.append(f"That accounts for the whole difference of {rupees(residual)}.")

    if verified:
        return PlainAccount(
            headline=headline,
            body=tuple(sentences),
            status="We have accepted this and your books balance, because the "
                   "rate is on the paperwork you already hold.",
            next_step="Nothing to do.",
        )
    return PlainAccount(
        headline=headline,
        body=tuple(sentences),
        status="We have not accepted this. The arithmetic works, but nothing "
               "in your paperwork confirms the rate it depends on, so it is a "
               "reasonable guess rather than a fact.",
        next_step="Ask the gateway to confirm the rate in writing. Once that "
                 "notice is on file, this closes on its own.",
    )


# ── the entry point ──────────────────────────────────────────────────────────

_BY_CLASS = {
    DefectClass.DUPLICATE_PAYMENT: _duplicate_payment,
    DefectClass.DUPLICATE_UTR: _duplicate_credit,
    DefectClass.MISSING_BANK_LINE: _no_credit,
}


def account_for(
    exc: ReconException,
    sources: SourceBundle,
    *,
    resolution=None,
    verified: bool = False,
) -> PlainAccount:
    """Say what happened to one exception, for the merchant rather than the desk.

    `resolution` is the priced citation set from `agent/citations.py` when a
    hypothesis reached this item, and it is the only route by which anything the
    model said influences this text -- and then only through rows whose money
    was computed by code. Passing nothing yields the honest version: we could
    not account for it.
    """
    handler = _BY_CLASS.get(exc.suspected_class)
    if handler is not None:
        return handler(sources, exc)
    if exc.residual_paise is not None:
        return _residual(sources, exc, resolution, verified)
    return PlainAccount(
        headline=f"Something about {_payout_phrase(sources, exc)} did not add up.",
        body=("We could not tie it to any of the usual causes.",),
        status="We have left it open rather than guess.",
        next_step="Send this one to your accountant, or ask the gateway for a "
                 "breakdown of the payout.",
    )
