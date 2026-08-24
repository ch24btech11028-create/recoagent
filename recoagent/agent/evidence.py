"""The evidence packet -- everything a proposer is allowed to see.

Built only from `SourceBundle`, which is the same restriction every matcher in
this system operates under. A proposer that could reach ground truth would make
the B3 numbers worthless in exactly the way `tests/test_independence.py` exists
to prevent, so this module is covered by that test too.

The packet is deliberately not a prose prompt. It is a structured record of the
batch, its rows, the fee schedule that should apply, and what the earlier tiers
already ruled out. Prose describing the data would be one more place for the
description and the data to drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

from ..money import FeeSchedule, Paise
from ..schemas import BankLine, Settlement, SourceBundle

#: Unlinked rows this far from the settlement date are shown as context. Wider
#: than the solver's own window: the model is being asked to reason about what
#: the solver could *not* close, so it should see slightly more than the solver
#: was allowed to consider.
CONTEXT_WINDOW = timedelta(days=7)


@dataclass(frozen=True)
class EvidencePacket:
    bank_credit: dict[str, Any]
    settlement: dict[str, Any]
    residual_paise: Paise
    payments: list[dict[str, Any]]
    linked_adjustments: list[dict[str, Any]]
    nearby_unlinked_rows: list[dict[str, Any]]
    fee_schedule: dict[str, Any]
    already_ruled_out: str
    repair_feedback: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build(
    sources: SourceBundle,
    line: BankLine,
    settlement: Settlement,
    residual_paise: Paise,
    fees: FeeSchedule,
    *,
    ruled_out: str = "",
    repair_feedback: str | None = None,
) -> EvidencePacket:
    members = sources.payments_by_settlement(settlement.settlement_id)
    linked = sources.adjustments_by_settlement(settlement.settlement_id)

    nearby = [
        a
        for a in sources.adjustments
        if a.settlement_id is None
        and abs(a.booked_at - settlement.settled_at) <= CONTEXT_WINDOW
    ]

    return EvidencePacket(
        bank_credit={
            "bank_line_id": line.bank_line_id,
            "value_date": line.value_date.isoformat(),
            "amount_paise": line.amount_paise,
            "narration": line.narration,
        },
        settlement={
            "settlement_id": settlement.settlement_id,
            "utr": settlement.utr,
            "settled_at": settlement.settled_at.isoformat(),
            "reported_net_paise": settlement.net_paise,
            "status": settlement.status,
        },
        residual_paise=residual_paise,
        payments=[
            {
                "payment_id": p.payment_id,
                "method": p.method,
                "currency": p.currency,
                "fx_rate": p.fx_rate,
                "gross_paise": p.gross_paise,
                "fee_paise": p.fee_paise,
                "tax_paise": p.tax_paise,
                "net_paise": p.net_paise,
                "captured_at": p.captured_at.isoformat(),
                "status": p.status,
                # The fee the published schedule would have charged, so a
                # repricing is visible as a difference rather than something the
                # proposer has to recompute and be trusted about.
                "fee_at_schedule_paise": fees.fee_and_tax(p.gross_paise, p.method)[0],
                "tax_at_schedule_paise": fees.fee_and_tax(p.gross_paise, p.method)[1],
            }
            for p in members
        ],
        linked_adjustments=[
            {
                "adjustment_id": a.adjustment_id,
                "kind": a.kind,
                "payment_id": a.payment_id,
                "amount_paise": a.amount_paise,
                "booked_at": a.booked_at.isoformat(),
            }
            for a in linked
        ],
        nearby_unlinked_rows=[
            {
                "adjustment_id": a.adjustment_id,
                "kind": a.kind,
                "payment_id": a.payment_id,
                "amount_paise": a.amount_paise,
                "booked_at": a.booked_at.isoformat(),
            }
            for a in nearby
        ],
        fee_schedule={
            "mdr_bps_by_method": dict(fees.mdr_bps),
            "gst_bps_on_fee": fees.gst_bps,
            "note": "UPI and RuPay debit carry zero MDR by regulation in India.",
        },
        already_ruled_out=ruled_out
        or (
            "An exhaustive subset-sum over the unlinked rows near this batch found "
            "no combination of up to three of them that closes this residual, and "
            "a cross-batch cutoff-spill pairing did not apply. Do not re-propose a "
            "plain combination of the rows listed under nearby_unlinked_rows."
        ),
        repair_feedback=repair_feedback,
    )
