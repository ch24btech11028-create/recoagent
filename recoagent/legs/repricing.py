"""Apply the merchant's own paperwork: rate notices and FX advices.

The tier that exists because of what the agent tier could not honestly do.

A fee variance leaves a residual that any number of rates would explain. A model
asked to account for it picks one, the arithmetic closes, and the result is a
hypothesis wearing the clothes of a proof -- which is why those close as
`needs_approval` rather than `resolved`. The missing ingredient was never
reasoning. It was a document: the repricing notice the gateway actually sent.

Once that document is in the book, no model is needed. Read the notice in force
for the method on the settlement date, re-derive the fee from the payment's own
gross, and check. That is a lookup and two multiplications, so it belongs here
rather than in a tier that costs a network call and cannot be replayed. The same
judgement was already made for `TIMING_SPILL`, and it applies for the same
reason: work a deterministic tier can do, a deterministic tier should do.

**Nothing here chooses a number.** Every rate comes from a notice and every
amount is computed from it. Where two notices are in force for one method on one
day, that is a contradiction in the merchant's records and the answer is to
refuse -- picking the newer one would be exactly the guess this module exists to
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..money import GST_BPS, Paise, bps_of
from ..schemas import PGPayment, Settlement, SourceBundle


@dataclass(frozen=True)
class Correction:
    """Re-derived nets for a batch, and what justified each one."""

    #: payment_id -> the net that the paperwork implies, computed here.
    nets: dict[str, Paise] = field(default_factory=dict)
    #: The notice and advice ids relied on. These go on the match record, so an
    #: auditor can pull the same documents and repeat the arithmetic.
    cited: tuple[str, ...] = ()
    #: Set when the paperwork contradicts itself. A reason to stop, not a tie to
    #: break.
    refusal: str | None = None

    @property
    def applies(self) -> bool:
        return self.refusal is None and bool(self.nets)


def _repriced_net(payment: PGPayment, mdr_bps: int, gst_bps: int = GST_BPS) -> Paise:
    """The net this payment would have carried at the noticed rate.

    Expressed as a delta against the reported net rather than rebuilt from
    scratch, so TDS and anything else already deducted survives untouched. Fees
    round per step -- MDR first, then GST on the rounded MDR -- because that is
    how they are actually charged, and computing GST on an unrounded fee drifts
    a paise on roughly one payment in three.
    """
    fee = bps_of(payment.gross_paise, mdr_bps)
    tax = bps_of(fee, gst_bps)
    charged = fee + tax
    reported = payment.fee_paise + payment.tax_paise
    return payment.net_paise + (reported - charged)


def _fx_adjusted_net(payment: PGPayment, rate_pct_of_gross: float) -> Paise:
    """The net after the conversion slip the bank advised.

    The advice quotes the slip as a share of gross, which is the same unit the
    agent tier's `FxClaim` uses, so the two are checkable against each other.
    The multiplication happens once, here, and is floored the same way the
    conversion is.
    """
    slip = int(payment.gross_paise * rate_pct_of_gross / 100)
    return payment.net_paise - slip


def corrections(sources: SourceBundle, settlement: Settlement) -> Correction:
    """What the paperwork says this batch's payments actually netted.

    Returns an empty correction when there is no relevant paperwork, which is
    the common case and is not a failure -- it just means this tier has nothing
    to contribute and the next one should try.
    """
    members = sources.payments_by_settlement(settlement.settlement_id)
    if not members:
        return Correction()

    nets: dict[str, Paise] = {}
    cited: list[str] = []

    # One lookup per method, not per payment: every payment on a method shares
    # the notice, and a batch can easily carry a hundred of them.
    by_method: dict[str, list[PGPayment]] = {}
    for payment in members:
        by_method.setdefault(payment.method, []).append(payment)

    for method, payments in sorted(by_method.items()):
        in_force = sources.notices_covering(settlement.settled_at, method)
        if len(in_force) > 1:
            return Correction(refusal=(
                f"{len(in_force)} rate notices are in force for {method} on "
                f"{settlement.settled_at.date()}: "
                + ", ".join(sorted(n.notice_id for n in in_force))
            ))
        if not in_force:
            continue
        notice = in_force[0]
        touched = False
        for payment in payments:
            corrected = _repriced_net(payment, notice.mdr_bps)
            if corrected != payment.net_paise:
                nets[payment.payment_id] = corrected
                touched = True
        if touched:
            cited.append(notice.notice_id)

    # FX advices are per payment, so they layer on top of any repricing.
    for payment in members:
        advice = sources.fx_advice_for(payment.payment_id)
        if advice is None or advice.rate_pct_of_gross == 0:
            continue
        base = nets.get(payment.payment_id, payment.net_paise)
        slip = int(payment.gross_paise * advice.rate_pct_of_gross / 100)
        if slip:
            nets[payment.payment_id] = base - slip
            cited.append(advice.advice_id)

    return Correction(nets=nets, cited=tuple(cited))


def rate_book(sources: SourceBundle, when: datetime):
    """The paperwork, in the shape the agent tier checks its claims against.

    Built here rather than in `agent/` so there is exactly one reading of what a
    notice means. The agent tier never gets to interpret a document; it gets to
    cite one, and this decides whether the citation holds.
    """
    from ..agent.citations import RateBook

    mdr: dict[str, set[int]] = {}
    for notice in sources.rate_notices:
        if notice.covers(when, notice.method):
            mdr.setdefault(notice.method, set()).add(notice.mdr_bps)
    fx = {
        a.payment_id: a.rate_pct_of_gross
        for a in sources.fx_advices
        if sources.fx_advice_for(a.payment_id) is not None
    }
    return RateBook(mdr_bps=mdr, fx_pct=fx)
