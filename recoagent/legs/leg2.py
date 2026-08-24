"""Leg 2 -- gateway settlement batch to bank credit, N:1.

Tier 0 only: extract a UTR from the bank narration, join it to a settlement,
then gate on the arithmetic replay.

Scoring runs over bank lines rather than settlements, which is the harder and
more honest population. It means a duplicated UTR -- the same credit restated
on a second statement line -- sits in the denominator with no correct answer
available. Refusing both lines is the right call and it costs match rate. That
trade is the point: the alternative is a matcher that books the same money
twice.

Tier 1 will replace the exact-UTR join with an SSMP DP-greedy solver over
amount and date windows, which is what recovers the lines whose narration was
clipped. Tier 2 is the LLM, proposing rows that explain a residual.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone

from ..defects import DefectClass
from ..schemas import (
    BankLine,
    MatchRecord,
    ReconException,
    ReconResult,
    Settlement,
    SourceBundle,
    stable_hash,
)
from ..validate import Tolerance, prove_leg2

TIER = "T0"
RULE_EXACT_UTR = "leg2.t0.exact_utr"

#: Indian bank UTRs in these statement formats are a 12-digit run. The word
#: boundaries matter: without them a clipped narration yields a shorter run
#: that would silently match the wrong settlement.
_UTR_RE = re.compile(r"\b(\d{12})\b")


def extract_utr(narration: str) -> str | None:
    """Pull a UTR out of a raw bank narration, or None if it is not readable.

    Returns None rather than a best guess when the narration holds more than
    one 12-digit run -- an unreadable key and an ambiguous key are both reasons
    to stop, not reasons to pick.
    """
    found = _UTR_RE.findall(narration)
    if len(found) != 1:
        return None
    return found[0]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _index_settlements(settlements: tuple[Settlement, ...]) -> dict[str, list[Settlement]]:
    by_utr: dict[str, list[Settlement]] = defaultdict(list)
    for s in settlements:
        by_utr[s.utr].append(s)
    return by_utr


def _index_lines(lines: tuple[BankLine, ...]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for line in lines:
        utr = extract_utr(line.narration)
        if utr:
            counts[utr] += 1
    return counts


def match(sources: SourceBundle, tol: Tolerance, result: ReconResult) -> set[str]:
    """Match bank lines to settlement batches, appending to `result` in place.

    Returns the settlement ids this pass reached a verdict on -- matched or
    proof-failed. `unmatched_settlements` uses it to avoid filing the same
    failure twice, once against the credit and once against the batch.
    """
    settlements_by_utr = _index_settlements(sources.settlements)
    utr_line_counts = _index_lines(sources.bank_lines)
    adjudicated: set[str] = set()

    for line in sources.bank_lines:
        xid = f"x2_{line.bank_line_id}"
        utr = extract_utr(line.narration)

        if utr is None:
            result.exceptions.append(
                ReconException(
                    exception_id=xid,
                    leg=2,
                    entity_kind="bank_line",
                    entity_id=line.bank_line_id,
                    reason=f"no readable UTR in narration {line.narration!r}",
                    suspected_class=DefectClass.NARRATION_TRUNCATION,
                )
            )
            continue

        if utr_line_counts[utr] > 1:
            result.exceptions.append(
                ReconException(
                    exception_id=xid,
                    leg=2,
                    entity_kind="bank_line",
                    entity_id=line.bank_line_id,
                    reason=(
                        f"UTR {utr} appears on {utr_line_counts[utr]} statement lines; "
                        "refusing to book the same credit twice"
                    ),
                    suspected_class=DefectClass.DUPLICATE_UTR,
                )
            )
            continue

        candidates = settlements_by_utr.get(utr, [])
        if len(candidates) != 1:
            result.exceptions.append(
                ReconException(
                    exception_id=xid,
                    leg=2,
                    entity_kind="bank_line",
                    entity_id=line.bank_line_id,
                    reason=(
                        f"UTR {utr} matches {len(candidates)} settlements"
                        if candidates
                        else f"UTR {utr} matches no settlement in this batch"
                    ),
                    suspected_class=None,
                )
            )
            continue

        settlement = candidates[0]
        adjudicated.add(settlement.settlement_id)
        members = sources.payments_by_settlement(settlement.settlement_id)
        linked = sources.adjustments_by_settlement(settlement.settlement_id)
        proof = prove_leg2(line, settlement, members, linked, tol)

        if not proof.closes:
            residual = proof.residual_paise
            result.exceptions.append(
                ReconException(
                    exception_id=xid,
                    leg=2,
                    entity_kind="bank_line",
                    entity_id=line.bank_line_id,
                    related_id=settlement.settlement_id,
                    reason=(
                        f"UTR joins {settlement.settlement_id} but the credit is "
                        f"{'short' if residual < 0 else 'over'} by {abs(residual)} paise "
                        f"against {len(members)} payments and {len(linked)} linked adjustments"
                    ),
                    residual_paise=residual,
                    # T0 deliberately does not guess *why* the arithmetic failed.
                    # Naming the cause requires evidence it has not gathered --
                    # that is what the later rungs are for, and pretending
                    # otherwise here would flatter the baseline.
                    suspected_class=None,
                )
            )
            continue

        result.matches.append(
            MatchRecord(
                match_id=f"m2_{line.bank_line_id}",
                leg=2,
                tier=TIER,
                rule_id=RULE_EXACT_UTR,
                left_ids=(line.bank_line_id,),
                right_ids=(settlement.settlement_id,),
                confidence=1.0,
                proof=proof,
                input_hash=stable_hash(line, settlement, *members, *linked),
                created_at=_now(),
            )
        )

    return adjudicated


def unmatched_settlements(
    sources: SourceBundle, result: ReconResult, adjudicated: set[str]
) -> list[ReconException]:
    """Settlements no bank line ever reached.

    Two exclusions, both deliberate. Settlements already adjudicated on the
    credit side are skipped: a batch whose bank line failed its proof is one
    problem, and filing it again here would double the apparent exception
    count and hand an ops team the same item twice. And an on-hold settlement
    is *correctly* unmatched -- the money genuinely never moved -- so it is
    labelled as such rather than counted as a failure.
    """
    matched = {m.right_ids[0] for m in result.matches_for_leg(2)}
    out: list[ReconException] = []
    for s in sources.settlements:
        if s.settlement_id in matched or s.settlement_id in adjudicated:
            continue
        out.append(
            ReconException(
                exception_id=f"x2s_{s.settlement_id}",
                leg=2,
                entity_kind="settlement",
                entity_id=s.settlement_id,
                reason=(
                    "settlement held into reserve balance; no credit expected"
                    if s.status == "on_hold"
                    else "no bank line matched this settlement"
                ),
                suspected_class=(
                    DefectClass.MISSING_BANK_LINE if s.status == "on_hold" else None
                ),
            )
        )
    return out
