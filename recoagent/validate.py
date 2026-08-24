"""The arithmetic gate.

Nothing in this system becomes a match because something believed it should.
A candidate pairing is accepted only when an independently re-derived sum
closes against the money that actually arrived, within a stated tolerance.

The proof for Leg 2 deliberately re-derives the batch total from the payment
rows rather than reading the settlement header's `net_paise`. The header is the
gateway's *claim* about what it paid out; checking a claim against itself is
not reconciliation. FinBalance measured exactly this failure -- a 26-41
percentage-point gap between what a model reported and what replaying its own
entries produced -- which is why the replay is the gate and not a formality.

When the LLM tier lands, it will call `prove_leg2` with a hypothesised set of
extra rows. It never gets to write a match itself; it can only propose a set
whose arithmetic this module then checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .money import Paise
from .schemas import (
    ArithmeticProof,
    BankLine,
    Order,
    PGAdjustment,
    PGPayment,
    Settlement,
)


@dataclass(frozen=True)
class Tolerance:
    """Absolute paise tolerance per leg.

    Zero at B0 on purpose. Sub-rupee drift from per-step rounding is real, and
    absorbing it requires an explicit, defended tolerance -- so the baseline
    rung refuses to absorb anything, the drift shows up in the exception list,
    and a later rung has to earn the tolerance it takes.
    """

    leg1_paise: Paise = 0
    leg2_paise: Paise = 0

    @classmethod
    def strict(cls) -> Tolerance:
        return cls(leg1_paise=0, leg2_paise=0)


def prove_leg1(order: Order, payment: PGPayment, tol: Tolerance) -> ArithmeticProof:
    """Does the captured amount agree with what the order says was owed?"""
    return ArithmeticProof(
        expression=f"payment[{payment.payment_id}].gross == order[{order.order_id}].amount",
        lhs_paise=payment.gross_paise,
        rhs_paise=order.amount_paise,
        tolerance_paise=tol.leg1_paise,
    )


def prove_leg2(
    bank_line: BankLine,
    settlement: Settlement,
    members: Iterable[PGPayment],
    adjustments: Iterable[PGAdjustment],
    tol: Tolerance,
    *,
    hypothesised: Iterable[PGAdjustment] = (),
) -> ArithmeticProof:
    """Re-derive the batch total from its rows and check it against the credit.

    `hypothesised` is the hook for the LLM tier: rows it believes explain a gap
    but which are not linked to this settlement in the source data. They are
    summed on exactly the same footing as linked rows and subjected to exactly
    the same check -- a hypothesis that does not close is simply rejected.
    """
    members = list(members)
    linked = list(adjustments)
    extra = list(hypothesised)

    payments_net = sum(p.net_paise for p in members)
    adj_net = sum(a.amount_paise for a in linked)
    hyp_net = sum(a.amount_paise for a in extra)

    expression = (
        f"bank[{bank_line.bank_line_id}].amount == "
        f"sum(net of {len(members)} payments in {settlement.settlement_id})"
        f" + {len(linked)} linked adjustments"
    )
    if extra:
        expression += f" + {len(extra)} hypothesised adjustments"

    return ArithmeticProof(
        expression=expression,
        lhs_paise=bank_line.amount_paise,
        rhs_paise=payments_net + adj_net + hyp_net,
        tolerance_paise=tol.leg2_paise,
    )


def header_agrees(
    settlement: Settlement, members: Iterable[PGPayment], adjustments: Iterable[PGAdjustment]
) -> bool:
    """Corroboration only, never proof.

    Whether the gateway's own reported net matches the rows it reported. Useful
    as a diagnostic -- a batch where the header disagrees with its own rows is a
    different kind of problem from one where the bank credit disagrees -- but it
    is never sufficient to accept a match.
    """
    derived = sum(p.net_paise for p in members) + sum(a.amount_paise for a in adjustments)
    return derived == settlement.net_paise
