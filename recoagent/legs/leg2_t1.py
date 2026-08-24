"""Leg 2, Tier 1 -- a second pass over what exact keys could not resolve.

Two recovery rules, both gated by the same arithmetic replay as Tier 0 and
both required to be *unique* before they may act.

**T1a -- amount and date window.** A bank line whose narration was clipped has
no readable UTR, but it still has an amount and a value date. If exactly one
unclaimed settlement's re-derived total matches that amount inside the window,
the join is recoverable without the key. If two settlements match, the line is
ambiguous and stays with a human: two batches that happen to pay out the same
amount on the same day is a coincidence, not evidence.

**T1b -- SSMP residual closure.** A bank line that joined its settlement but
failed the proof is short or over by some residual. Somewhere in the book sit
unlinked rows -- an orphaned refund, a chargeback with its dispute fee, a
manual adjustment -- whose total is exactly that gap. Finding them is subset
sum (see `ssmp.py`), and the subset is only accepted if it is the *only* one
that closes.

What this tier deliberately does not attempt: FEE_TAX_VARIANCE, FX_CONVERSION
and TIMING_SPILL. Each of those needs a reason rather than an amount -- a
repricing, a rate, a payment that settled in a different cycle. No arithmetic
search over the local pool can distinguish them from a coincidence, which is
exactly the boundary where the LLM tier earns its place at B3.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from ..money import Paise
from ..schemas import (
    MatchRecord,
    PGAdjustment,
    ReconException,
    ReconResult,
    SourceBundle,
    stable_hash,
)
from ..validate import Tolerance, prove_leg2
from . import ssmp

TIER = "T1"
RULE_AMOUNT_WINDOW = "leg2.t1.amount_window"
RULE_SSMP_RESIDUAL = "leg2.t1.ssmp_residual"

#: How far from a settlement date an orphaned row may sit and still be
#: considered part of that batch. Wide enough to cover the generator's 2-60
#: hour booking lag plus a weekend; narrow enough that the candidate pool stays
#: in the tens rather than the thousands.
ORPHAN_WINDOW = timedelta(days=4)

#: Tolerance for matching a clipped line to a settlement on amount alone.
AMOUNT_WINDOW_DAYS = 1

#: Confidence priors. Hand-set at B2 and honest about being priors: exact keys
#: are certain, a unique exact-sum explanation is nearly so, and a match made
#: on amount alone is the weakest of the three. They become something to
#: calibrate rather than assert once the LLM tier reports its own confidence.
CONF_SSMP_EXACT = 0.95
CONF_SSMP_TOLERANT = 0.85
#: Several interchangeable rows explain the gap identically. The amount is
#: certain, the row-level attribution is not, and the confidence says so.
CONF_SSMP_FUNGIBLE = 0.90
CONF_AMOUNT_WINDOW = 0.80


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ssmp_confidence(search: "ssmp.SearchResult", subset: "ssmp.Subset") -> float:
    """Lower confidence when the rows were interchangeable or the sum drifted."""
    if not search.unique:
        return CONF_SSMP_FUNGIBLE
    return CONF_SSMP_EXACT if subset.residual == 0 else CONF_SSMP_TOLERANT


def _derived_total(sources: SourceBundle, settlement_id: str) -> Paise:
    members = sources.payments_by_settlement(settlement_id)
    linked = sources.adjustments_by_settlement(settlement_id)
    return sum(p.net_paise for p in members) + sum(a.amount_paise for a in linked)


def _orphan_pool(
    sources: SourceBundle, around: datetime, window: timedelta = ORPHAN_WINDOW
) -> list[PGAdjustment]:
    """Unlinked rows near a settlement date.

    Restricting by date is what keeps the search space small enough for an
    exhaustive subset-sum to be both cheap and meaningful. Over the whole book
    the pool would be large enough that some subset closes almost any gap by
    coincidence.
    """
    return [
        a
        for a in sources.adjustments
        if a.settlement_id is None and abs(a.booked_at - around) <= window
    ]


def recover(sources: SourceBundle, tol: Tolerance, result: ReconResult) -> None:
    """Attempt to resolve leg-2 exceptions left by Tier 0, in place."""
    settlement_by_id = {s.settlement_id: s for s in sources.settlements}
    line_by_id = {b.bank_line_id: b for b in sources.bank_lines}
    claimed = {m.right_ids[0] for m in result.matches_for_leg(2)}

    #: Unlinked rows already spent explaining an earlier batch. Fungible rows
    #: are interchangeable but not infinite -- one Rs 150 dispute fee explains
    #: one batch, and spending it twice would reconcile two batches with the
    #: same money.
    consumed: set[str] = set()

    survivors: list[ReconException] = []

    for exc in result.exceptions:
        if exc.leg != 2 or exc.entity_kind != "bank_line":
            survivors.append(exc)
            continue

        line = line_by_id.get(exc.entity_id)
        if line is None:
            survivors.append(exc)
            continue

        # ── T1b: the line joined, but the money did not add up ──────────
        if exc.residual_paise is not None:
            settlement_id = exc.related_id
            if settlement_id is None or settlement_id not in settlement_by_id:
                survivors.append(exc)
                continue

            settlement = settlement_by_id[settlement_id]
            pool = _orphan_pool(sources, settlement.settled_at)
            values = [a.amount_paise for a in pool]

            search = ssmp.enumerate_closing_subsets(
                values, exc.residual_paise, tol.leg2_paise
            )

            if not search.actionable:
                survivors.append(
                    replace(
                        exc,
                        reason=(
                            f"{exc.reason}; subset-sum over {len(pool)} unlinked rows "
                            + (
                                f"found {len(search.solutions)} materially different "
                                "explanations"
                                if search.ambiguous
                                else "found no explanation"
                            )
                        ),
                        escalated_from_tier=TIER,
                    )
                )
                continue

            # Among financially identical explanations, take the one that spends
            # the fewest rows already used elsewhere; a tie on that is a genuine
            # free choice, so break it deterministically on index order.
            subset = min(
                search.solutions,
                key=lambda sol: (
                    sum(1 for i in sol.indices if pool[i].adjustment_id in consumed),
                    sol.indices,
                ),
            )
            hypothesised = [pool[i] for i in subset.indices]

            if any(a.adjustment_id in consumed for a in hypothesised):
                survivors.append(
                    replace(
                        exc,
                        reason=(
                            f"{exc.reason}; the only rows that explain this gap are "
                            "already spent on another batch"
                        ),
                        escalated_from_tier=TIER,
                    )
                )
                continue
            members = sources.payments_by_settlement(settlement_id)
            linked = sources.adjustments_by_settlement(settlement_id)
            proof = prove_leg2(
                line, settlement, members, linked, tol, hypothesised=hypothesised
            )

            # The search said it closes; the gate confirms it independently.
            # A solver bug must not be able to book a match on its own say-so.
            if not proof.closes:
                survivors.append(replace(exc, escalated_from_tier=TIER))
                continue

            result.matches.append(
                MatchRecord(
                    match_id=f"m2_{line.bank_line_id}",
                    leg=2,
                    tier=TIER,
                    rule_id=RULE_SSMP_RESIDUAL,
                    left_ids=(line.bank_line_id,),
                    right_ids=(settlement_id,),
                    confidence=_ssmp_confidence(search, subset),
                    proof=proof,
                    input_hash=stable_hash(line, settlement, *members, *linked, *hypothesised),
                    created_at=_now(),
                    hypothesised_ids=tuple(a.adjustment_id for a in hypothesised),
                )
            )
            claimed.add(settlement_id)
            consumed.update(a.adjustment_id for a in hypothesised)
            continue

        # ── T1a: no readable key, recover on amount and date ────────────
        if "no readable UTR" in exc.reason:
            candidates = [
                s
                for s in sources.settlements
                if s.settlement_id not in claimed
                and abs((s.settled_at.date() - line.value_date).days) <= AMOUNT_WINDOW_DAYS
                and abs(_derived_total(sources, s.settlement_id) - line.amount_paise)
                <= tol.leg2_paise
            ]

            if len(candidates) != 1:
                survivors.append(
                    replace(
                        exc,
                        reason=(
                            f"{exc.reason}; amount and date window matched "
                            f"{len(candidates)} settlements"
                        ),
                        escalated_from_tier=TIER,
                    )
                )
                continue

            settlement = candidates[0]
            members = sources.payments_by_settlement(settlement.settlement_id)
            linked = sources.adjustments_by_settlement(settlement.settlement_id)
            proof = prove_leg2(line, settlement, members, linked, tol)
            if not proof.closes:
                survivors.append(replace(exc, escalated_from_tier=TIER))
                continue

            result.matches.append(
                MatchRecord(
                    match_id=f"m2_{line.bank_line_id}",
                    leg=2,
                    tier=TIER,
                    rule_id=RULE_AMOUNT_WINDOW,
                    left_ids=(line.bank_line_id,),
                    right_ids=(settlement.settlement_id,),
                    confidence=CONF_AMOUNT_WINDOW,
                    proof=proof,
                    input_hash=stable_hash(line, settlement, *members, *linked),
                    created_at=_now(),
                )
            )
            claimed.add(settlement.settlement_id)
            continue

        survivors.append(exc)

    result.exceptions = survivors
