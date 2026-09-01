"""Two deterministic rungs, before any model is allowed near the book.

The ladder here is the same argument the B0/B2/B3 ladder makes, applied to a
different question. It exists so that the model's contribution is a measured
number rather than an assumption.

    C0   source fields only     what a categoriser without a reconciler can do
    C1   + the reconciliation   what the matching already determined for free
    C2   + a cited model        what is genuinely left  (see agent.py)

The finding C1 exists to expose: **in a reconciled book, the reconciliation has
already categorised nearly everything, and it did so with proofs.** A payment
that closed against an order is revenue -- not "probably revenue, 0.94". A bank
credit that closed against a settlement batch is a transfer, and it is a
transfer because the arithmetic tying it to that batch balanced, not because a
narration contained the word SETTLEMENT.

That is not a smaller claim than the model's. It is a larger one, and it is the
reason the residue C2 gets handed is as small as it turns out to be.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import ReconResult, SourceBundle
from .taxonomy import Category


@dataclass(frozen=True)
class Assignment:
    """One categorised row, and what put it there.

    `evidence` is not a comment. It is the field, row id, or match id that
    determined the category, and it is required -- a category with no evidence
    is exactly the thing this design refuses, whether it came from a model or
    from a rule that nobody can now explain.
    """

    entity_id: str
    entity_kind: str  # payment | adjustment | bank_line | settlement | payout
    category: Category
    amount_paise: int
    rung: str          # C0 | C1 | C2
    rule_id: str
    evidence: str
    #: Only C2 sets this. A rule has no confidence; it either applied or it
    #: did not, and attaching 1.0 to it would put rules and guesses in one
    #: column and invite someone to average them.
    confidence: float | None = None
    #: False means *proposed, not booked*. The category is real and reaches the
    #: operator queue with its evidence attached, but it is not filed and does
    #: not count towards the wrong-category rate, because nothing has been
    #: decided yet. C2 sets this on every assignment it makes; see the note in
    #: `agent.run_c2`.
    verified: bool = True

    @property
    def booked(self) -> bool:
        """Whether this assignment is a decision or a suggestion."""
        return self.verified and self.category is not Category.NEEDS_REVIEW


@dataclass
class Ledger:
    """Every assignment made, keyed so a later rung can see what is left."""

    assignments: dict[str, Assignment] = field(default_factory=dict)

    def add(self, a: Assignment) -> None:
        # First rung wins. C1 refines what C0 could not see; it does not
        # overrule what C0 read straight off the source row, and a later rung
        # silently replacing an earlier one would make the ladder unreadable.
        self.assignments.setdefault(a.entity_id, a)

    def has(self, entity_id: str) -> bool:
        return entity_id in self.assignments

    def by_rung(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for a in self.assignments.values():
            out[a.rung] = out.get(a.rung, 0) + 1
        return out

    @property
    def needs_review(self) -> list[Assignment]:
        return [a for a in self.assignments.values() if a.category is Category.NEEDS_REVIEW]


#: The gateway's own words for what a non-payment row is. Read, not guessed:
#: these are the values `PGAdjustment.kind` is documented to take.
ADJUSTMENT_KIND = {
    "refund": Category.REFUND,
    "chargeback": Category.CHARGEBACK,
    "dispute_fee": Category.DISPUTE_FEE,
    "platform_fee": Category.GATEWAY_FEE,
    "reversal": Category.REFUND,
}

#: Payment statuses that took no money. Shares its reasoning with
#: `legs.leg1.FUNDED_STATUSES`, deliberately not its constant: this module
#: must keep working if the matcher's funding rule ever narrows, and a row that
#: is not a transaction is a statement about the row, not about Leg 1.
UNFUNDED = frozenset({"failed", "created", "authorized", "pending_authorization"})


def run_c0(sources: SourceBundle, ledger: Ledger | None = None) -> Ledger:
    """Source fields only. No reconciliation, no model, no inference.

    This rung is the honest baseline for the claim "we categorise
    transactions". Most of a payments book carries its own category in a field
    the gateway already populated, and any accuracy number that does not
    subtract this first is mostly measuring `status`.
    """
    ledger = ledger or Ledger()

    for payment in sources.payments:
        if payment.status in UNFUNDED:
            ledger.add(Assignment(
                entity_id=payment.payment_id,
                entity_kind="payment",
                category=Category.NOT_A_TRANSACTION,
                amount_paise=payment.gross_paise,
                rung="C0",
                rule_id="c0.unfunded_status",
                evidence=f"payment.status = {payment.status!r}",
            ))

    for adjustment in sources.adjustments:
        category = ADJUSTMENT_KIND.get(adjustment.kind)
        if category is not None:
            ledger.add(Assignment(
                entity_id=adjustment.adjustment_id,
                entity_kind="adjustment",
                category=category,
                amount_paise=adjustment.amount_paise,
                rung="C0",
                rule_id="c0.adjustment_kind",
                evidence=f"adjustment.kind = {adjustment.kind!r}",
            ))

    return ledger


def run_c1(sources: SourceBundle, result: ReconResult, ledger: Ledger | None = None) -> Ledger:
    """Everything the reconciliation already decided, with its proof as evidence.

    Note what is *not* here: nothing reads a narration, nothing scores a string
    similarity, nothing has a threshold. Each assignment cites a match record,
    and each of those match records carries an arithmetic proof. The category
    is as good as the reconciliation, which is a thing this repository has
    published a false-match rate for.
    """
    ledger = run_c0(sources, ledger)

    payments = {p.payment_id: p for p in sources.payments}
    bank_lines = {b.bank_line_id: b for b in sources.bank_lines}

    for match in result.matches_for_leg(1):
        for payment_id in match.right_ids:
            payment = payments.get(payment_id)
            if payment is None or ledger.has(payment_id):
                continue

            ledger.add(Assignment(
                entity_id=payment_id,
                entity_kind="payment",
                category=Category.SALES_REVENUE,
                amount_paise=payment.gross_paise,
                rung="C1",
                rule_id="c1.matched_to_order",
                evidence=f"match {match.match_id}: {match.rule_id}",
            ))

            # The fee and its tax are separate rows in the books even though
            # the gateway reports them on the payment. Splitting them here is
            # what makes GST reclaimable rather than buried in an expense.
            if payment.fee_paise:
                ledger.add(Assignment(
                    entity_id=f"{payment_id}:fee",
                    entity_kind="fee",
                    category=Category.GATEWAY_FEE,
                    amount_paise=-payment.fee_paise,
                    rung="C1",
                    rule_id="c1.fee_component",
                    evidence=f"payment[{payment_id}].fee, net of tax",
                ))
            if payment.tax_paise:
                ledger.add(Assignment(
                    entity_id=f"{payment_id}:tax",
                    entity_kind="tax",
                    category=Category.GST_INPUT_CREDIT,
                    amount_paise=-payment.tax_paise,
                    rung="C1",
                    rule_id="c1.gst_component",
                    evidence=f"payment[{payment_id}].tax on the MDR, reclaimable",
                ))

    for match in result.matches_for_leg(2):
        for bank_line_id in match.left_ids:
            settlement_id = match.right_ids[0] if match.right_ids else "?"
            line = bank_lines.get(bank_line_id)
            ledger.add(Assignment(
                entity_id=bank_line_id,
                entity_kind="bank_line",
                category=Category.SETTLEMENT_CREDIT,
                # The bank line's own amount, not the settlement's declared
                # net. The two disagree whenever a defect moved the credit --
                # a cutoff spill, for instance, shifts what the bank actually
                # paid while the gateway's row still states the original
                # figure. Booking the declared net debits the bank account
                # with money that never arrived, and the error is invisible on
                # a category-only scorecard: it is the amount that is wrong,
                # not the label. `journal.post` is what surfaced it, because a
                # clearing account either empties or it does not.
                amount_paise=line.amount_paise if line else 0,
                rung="C1",
                rule_id="c1.matched_to_settlement",
                # The point of naming the proof: this credit is a transfer
                # because its amount was shown to be the net of a batch of
                # payments already booked as revenue, not because it looked
                # like one.
                evidence=f"match {match.match_id}: batch {settlement_id} proved",
            ))

    return ledger


def residue(sources: SourceBundle, ledger: Ledger) -> list[tuple[str, str, int]]:
    """(entity_id, kind, amount) for every row still uncategorised.

    This is the input to the model tier, and its size is the headline of the
    whole exercise. If the deterministic rungs leave three rows out of two
    thousand, then "AI categorisation" is a claim about three rows.
    """
    out: list[tuple[str, str, int]] = []
    for payment in sources.payments:
        if not ledger.has(payment.payment_id):
            out.append((payment.payment_id, "payment", payment.gross_paise))
    for adjustment in sources.adjustments:
        if not ledger.has(adjustment.adjustment_id):
            out.append((adjustment.adjustment_id, "adjustment", adjustment.amount_paise))
    for line in sources.bank_lines:
        if not ledger.has(line.bank_line_id):
            out.append((line.bank_line_id, "bank_line", line.amount_paise))
    return out
