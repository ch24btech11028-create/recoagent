"""Leg 1 -- merchant order to gateway payment, 1:1.

Tier 0 only: an exact join on `order_id`, gated by an amount proof.

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
from ..schemas import (
    MatchRecord,
    PGPayment,
    ReconException,
    ReconResult,
    SourceBundle,
    stable_hash,
)
from ..validate import Tolerance, prove_leg1

TIER = "T0"
RULE_EXACT_ORDER_ID = "leg1.t0.exact_order_id"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def match(sources: SourceBundle, tol: Tolerance, result: ReconResult) -> None:
    """Match orders to payments, appending to `result` in place."""
    by_order: dict[str, list[PGPayment]] = defaultdict(list)
    for p in sources.payments:
        if p.order_id is not None:
            by_order[p.order_id].append(p)

    for order in sources.orders:
        candidates = by_order.get(order.order_id, [])

        if not candidates:
            result.exceptions.append(
                ReconException(
                    exception_id=f"x1_{order.order_id}",
                    leg=1,
                    entity_kind="order",
                    entity_id=order.order_id,
                    reason="no payment row references this order",
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
