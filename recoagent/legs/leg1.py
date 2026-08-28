"""Leg 1 -- merchant order to gateway payment, 1:1.

Tier 0 is an exact join on `order_id`, gated by an amount proof. Tier 1 closes
the one class where the two ledgers are *meant* to disagree: a partial capture,
where the gateway captured less than the order authorised and says so.

The rule that does the real work here is the refusal to guess. When an order
carries two payment rows -- a customer who retried after an ambiguous failure --
the join is ambiguous, and a matcher that picks one has a 50% chance of
booking revenue against the wrong transaction. BenchRec's stated principle
applies: leave it for a human rather than match it incorrectly.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from ..defects import DefectClass
from ..money import FeeSchedule
from ..schemas import (
    MatchRecord,
    Order,
    PGPayment,
    ReconException,
    ReconResult,
    SourceBundle,
    stable_hash,
)
from ..validate import Tolerance, prove_leg1, prove_leg1_capture

#: Statuses in which the money actually moved. Everything else is an attempt.
#:
#: This looked like a tautology for as long as the only books in existence were
#: generated ones, where every payment row is `captured` by construction. A
#: pull from a real gateway is not like that: a declined card leaves a `failed`
#: row carrying the full order amount, and an exact join on `order_id` gated
#: only by `gross == order.amount` matches it perfectly. The result is revenue
#: booked against a payment that never arrived -- a false match, in the leg
#: where a false match is most expensive, produced by the tier with the highest
#: confidence.
#:
#: `refunded` belongs here. The money did arrive and then left again; the
#: departure is an adjustment on Leg 2, not a reason to deny that the order was
#: ever paid.
FUNDED_STATUSES = frozenset({"captured", "partially_captured", "refunded"})

TIER = "T0"
TIER_T1 = "T1"
RULE_EXACT_ORDER_ID = "leg1.t0.exact_order_id"
RULE_PARTIAL_CAPTURE = "leg1.t1.documented_partial_capture"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _noticed_rates(
    sources: SourceBundle, payment: PGPayment, settled_at: dict[str, datetime]
) -> tuple[int, ...]:
    """MDR rates on file for this payment's method on its settlement date.

    A capture inside a repriced window was charged at the noticed rate, not the
    contracted one, so checking it against the rate card alone would refuse a
    payment whose arithmetic is perfectly sound. Every rate that a document
    supports is admissible; a rate no document supports is not, which is the
    whole point.
    """
    when = settled_at.get(payment.settlement_id or "")
    if when is None:
        return ()
    return tuple(
        sorted({n.mdr_bps for n in sources.notices_covering(when, payment.method)})
    )


def _partial_capture_match(
    sources: SourceBundle,
    order: Order,
    payment: PGPayment,
    fees: FeeSchedule,
    settled_at: dict[str, datetime],
) -> MatchRecord | None:
    """A match for a documented under-capture, or None if nothing documents it.

    Three things have to hold, and each one is doing work:

    - the report says `partially_captured` -- a difference nobody declared is
      not a partial capture, it is an unexplained difference;
    - the captured amount is *less* than what was authorised -- capturing more
      than was authorised is not a short capture in either direction, it is a
      book that needs a human;
    - the fee and tax re-derive from the captured gross at a rate the merchant
      has on file.

    Only the third is arithmetic, and only the third can be forged with any
    effort. That is why it is the one that gates the match.
    """
    if payment.status != "partially_captured":
        return None
    if payment.gross_paise >= order.amount_paise:
        return None

    for mdr in (None, *_noticed_rates(sources, payment, settled_at)):
        try:
            proof = prove_leg1_capture(order, payment, fees, mdr)
        except KeyError:
            # No rate on file for this method. Real gateway exports carry
            # methods a schedule was never configured for -- `paylater`,
            # `cardless_emi`, `bank_transfer` -- and the tier's whole claim is
            # that it only closes what a rate re-derives. Without a rate there
            # is nothing to re-derive against, so it declines and the row goes
            # to a human. Guessing the nearest rate would be exactly the
            # invented number this design exists to refuse.
            return None
        if proof.closes:
            return MatchRecord(
                match_id=f"m1_{order.order_id}",
                leg=1,
                tier=TIER_T1,
                rule_id=RULE_PARTIAL_CAPTURE,
                left_ids=(order.order_id,),
                right_ids=(payment.payment_id,),
                confidence=1.0,
                proof=proof,
                input_hash=stable_hash(order, payment),
                created_at=_now(),
                variance_paise=payment.gross_paise - order.amount_paise,
            )
    return None


def match(
    sources: SourceBundle,
    tol: Tolerance,
    result: ReconResult,
    *,
    fees: FeeSchedule | None = None,
    with_t1: bool = False,
) -> None:
    """Match orders to payments, appending to `result` in place."""
    fees = fees or FeeSchedule.default()
    settled_at = {s.settlement_id: s.settled_at for s in sources.settlements}
    by_order: dict[str, list[PGPayment]] = defaultdict(list)
    for p in sources.payments:
        if p.order_id is not None:
            by_order[p.order_id].append(p)

    for order in sources.orders:
        attempts = by_order.get(order.order_id, [])
        # Unsuccessful attempts are dropped before the ambiguity check, not
        # after. A customer who retried a declined card leaves two rows against
        # one order, and calling that ambiguous would send a perfectly ordinary
        # recovery to a human: only one of the two is money.
        candidates = [p for p in attempts if p.status in FUNDED_STATUSES]

        if not candidates:
            unfunded = ", ".join(sorted(f"{p.payment_id} ({p.status})" for p in attempts))
            result.exceptions.append(
                ReconException(
                    exception_id=f"x1_{order.order_id}",
                    leg=1,
                    entity_kind="order",
                    entity_id=order.order_id,
                    reason=(
                        f"{len(attempts)} payment attempts, none funded: {unfunded}"
                        if attempts
                        else "no payment row references this order"
                    ),
                    suspected_class=None,
                )
            )
            continue

        if len(candidates) > 1:
            ids = ", ".join(sorted(c.payment_id for c in candidates))
            result.exceptions.append(
                ReconException(
                    exception_id=f"x1_{order.order_id}",
                    leg=1,
                    entity_kind="order",
                    entity_id=order.order_id,
                    reason=f"ambiguous: {len(candidates)} payments claim this order ({ids})",
                    suspected_class=DefectClass.DUPLICATE_PAYMENT,
                )
            )
            continue

        payment = candidates[0]
        proof = prove_leg1(order, payment, tol)

        if not proof.closes:
            recovered = (
                _partial_capture_match(sources, order, payment, fees, settled_at)
                if with_t1
                else None
            )
            if recovered is not None:
                result.matches.append(recovered)
                continue
            result.exceptions.append(
                ReconException(
                    exception_id=f"x1_{order.order_id}",
                    leg=1,
                    entity_kind="order",
                    entity_id=order.order_id,
                    reason=(
                        f"captured amount differs from order amount by "
                        f"{proof.residual_paise} paise"
                    ),
                    residual_paise=proof.residual_paise,
                    suspected_class=(
                        DefectClass.PARTIAL_CAPTURE
                        if payment.status == "partially_captured"
                        else None
                    ),
                )
            )
            continue

        result.matches.append(
            MatchRecord(
                match_id=f"m1_{order.order_id}",
                leg=1,
                tier=TIER,
                rule_id=RULE_EXACT_ORDER_ID,
                left_ids=(order.order_id,),
                right_ids=(payment.payment_id,),
                confidence=1.0,
                proof=proof,
                input_hash=stable_hash(order, payment),
                created_at=_now(),
            )
        )
