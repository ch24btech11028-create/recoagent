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

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

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

    @classmethod
    def calibrated(cls) -> Tolerance:
        """Ten paise on Leg 2, and still nothing on Leg 1.

        Chosen against measurement, not intuition -- see
        `python -m recoagent.eval.tolerance_sweep`. The tempting way to set this
        is to maximise recall, and that picks the wrong number: Leg 2 recall
        keeps climbing past 10 paise all the way to Rs 10, and false-match rate
        stays flat at zero the whole way, so neither headline metric objects.

        Neither can see the damage. On Leg 2 the pairing comes from the UTR
        join, so the tolerance never governs *which* batch a credit belongs to,
        only whether its explanation is allowed to be approximate. What the
        per-class table in that sweep shows is that at 10 paise every
        ROUNDING_DRIFT closes and nothing else moves; at 50 the solver begins
        absorbing FX_CONVERSION, and by Rs 10 it is swallowing FEE_TAX_VARIANCE
        as well. Those are not recovered matches. They are real differences in
        money reconciled green -- the precise failure this system exists to
        prevent.

        Ten paise is the largest window that absorbs rounding and only rounding.

        Leg 1 stays at zero deliberately. A capture that differs from its order
        by any amount is a partial capture, not a rounding artifact, and
        absorbing it would hide the exact class the leg exists to catch.
        """
        return cls(leg1_paise=0, leg2_paise=10)


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
    repriced: Mapping[str, Paise] | None = None,
) -> ArithmeticProof:
    """Re-derive the batch total from its rows and check it against the credit.

    `hypothesised` is the hook for the LLM tier: rows it believes explain a gap
    but which are not linked to this settlement in the source data. They are
    summed on exactly the same footing as linked rows and subjected to exactly
    the same check -- a hypothesis that does not close is simply rejected.

    `repriced` replaces a payment's net where a rate notice says the gateway
    charged something other than the schedule. The corrected figure is computed
    by code from the notice and the payment's own gross; nothing may put an
    amount in here that it chose. As with `hypothesised`, the correction earns
    nothing by being applied -- it still has to close.
    """
    members = list(members)
    linked = list(adjustments)
    extra = list(hypothesised)

    repriced = dict(repriced or {})
    payments_net = sum(repriced.get(p.payment_id, p.net_paise) for p in members)
    adj_net = sum(a.amount_paise for a in linked)
    hyp_net = sum(a.amount_paise for a in extra)

    expression = (
        f"bank[{bank_line.bank_line_id}].amount == "
        f"sum(net of {len(members)} payments in {settlement.settlement_id})"
        f" + {len(linked)} linked adjustments"
    )
    if extra:
        expression += f" + {len(extra)} hypothesised adjustments"
    touched = sum(1 for p in members if p.payment_id in repriced)
    if touched:
        expression += f" ({touched} payments re-derived at a noticed rate)"

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
