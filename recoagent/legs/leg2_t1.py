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

**T1c -- cross-batch spill pairing.** A payment captured just before the T+2
cutoff is reported in one batch but credited with the next, leaving one credit
short by exactly X and another long by exactly X. That signature is mechanical,
so it is closed here rather than handed to the model.

What this tier deliberately does not attempt: FEE_TAX_VARIANCE and
FX_CONVERSION. Both need a *reason* rather than a sum -- a mid-cycle repricing,
a conversion rate the report does not carry -- and no arithmetic search can
separate either from a coincidence. That is the boundary where the LLM tier at
B3 has to earn its place, and it is deliberately narrow: every class that can
be closed mechanically is closed mechanically first, so the model's measured
contribution is its own rather than borrowed.
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


# ─────────────────────────────────────────────────────────────────────────────
# T1c -- cross-batch spill pairing
# ─────────────────────────────────────────────────────────────────────────────

RULE_SPILL_PAIR = "leg2.t1.spill_pair"

#: A payment captured this close to the settlement cutoff is a plausible
#: candidate for landing in the next cycle instead of this one.
CUTOFF_PROXIMITY = timedelta(hours=12)

#: Both halves of a spill are inferred together from the same evidence, so they
#: share a confidence. Below the fungible-subset prior: this rule reasons about
#: two batches and a capture time rather than one exact arithmetic identity.
CONF_SPILL_PAIR = 0.75


def pair_spills(sources: SourceBundle, tol: Tolerance, result: ReconResult) -> None:
    """Resolve T+2 cutoff spills by pairing a short batch with a long one.

    A payment captured just before the cutoff is reported inside one batch but
    credited with the next. The signature is distinctive and entirely
    mechanical: one credit short by exactly X, another long by exactly X, and a
    payment of net X sitting in the short batch's member list near the cutoff.

    This belongs in the deterministic tier precisely because it *is*
    deterministic. Leaving it for the LLM would have inflated that tier's
    apparent value -- the model would appear to solve something arithmetic
    could have closed on its own. What survives this rule genuinely needs a
    reason rather than a sum, which is the claim B3 has to earn.

    Uniqueness is enforced across the whole pairing: if two long batches could
    absorb the same spill, or two member payments have the same net, nothing is
    booked. A spill mis-paired reconciles two batches while attributing the
    money to the wrong cycle.
    """
    line_by_id = {b.bank_line_id: b for b in sources.bank_lines}
    settlement_by_id = {s.settlement_id: s for s in sources.settlements}

    residual_excs = [
        e
        for e in result.exceptions
        if e.leg == 2
        and e.entity_kind == "bank_line"
        and e.residual_paise is not None
        and e.related_id in settlement_by_id
    ]
    shorts = [e for e in residual_excs if (e.residual_paise or 0) < 0]
    longs = [e for e in residual_excs if (e.residual_paise or 0) > 0]
    if not shorts or not longs:
        return

    resolved: dict[str, tuple[str, str, int]] = {}  # exception_id -> (settlement, payment, delta)
    used_longs: set[str] = set()

    for short in shorts:
        gap = -(short.residual_paise or 0)
        src_id = short.related_id
        src = settlement_by_id[src_id]

        # Candidate payments: reported in this batch, net exactly the gap,
        # captured close enough to the cutoff to plausibly have slipped.
        candidates = [
            p
            for p in sources.payments_by_settlement(src_id)
            if p.net_paise == gap
            and abs(src.settled_at - p.captured_at) <= (timedelta(days=2) + CUTOFF_PROXIMITY)
        ]
        # Long batches that could absorb exactly this amount.
        absorbers = [
            l
            for l in longs
            if l.exception_id not in used_longs and (l.residual_paise or 0) == gap
        ]

        if len(candidates) != 1 or len(absorbers) != 1:
            continue

        payment, absorber = candidates[0], absorbers[0]
        used_longs.add(absorber.exception_id)
        resolved[short.exception_id] = (src_id, payment.payment_id, -gap)
        resolved[absorber.exception_id] = (absorber.related_id, payment.payment_id, gap)

    if not resolved:
        return

    survivors: list[ReconException] = []
    for exc in result.exceptions:
        pairing = resolved.get(exc.exception_id)
        if pairing is None:
            survivors.append(exc)
            continue

        settlement_id, payment_id, delta = pairing
        line = line_by_id[exc.entity_id]
        settlement = settlement_by_id[settlement_id]
        members = sources.payments_by_settlement(settlement_id)
        linked = sources.adjustments_by_settlement(settlement_id)

        # The inferred row is explicitly marked as inferred. It is a hypothesis
        # about where a payment settled, not a row anyone reported, and the
        # audit trail must never let the two look alike.
        inferred = PGAdjustment(
            adjustment_id=f"inferred:spill:{payment_id}",
            settlement_id=None,  # type: ignore[arg-type]
            kind="cutoff_spill",
            payment_id=payment_id,
            amount_paise=delta,
            booked_at=settlement.settled_at,
        )
        proof = prove_leg2(
            line, settlement, members, linked, tol, hypothesised=[inferred]
        )
        if not proof.closes:
            survivors.append(replace(exc, escalated_from_tier=TIER))
            continue

        result.matches.append(
            MatchRecord(
                match_id=f"m2_{line.bank_line_id}",
                leg=2,
                tier=TIER,
                rule_id=RULE_SPILL_PAIR,
                left_ids=(line.bank_line_id,),
                right_ids=(settlement_id,),
                confidence=CONF_SPILL_PAIR,
                proof=proof,
                input_hash=stable_hash(line, settlement, *members, *linked, inferred),
                created_at=_now(),
            )
        )

    result.exceptions = survivors
