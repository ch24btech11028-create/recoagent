"""Leg 2, Tier 2 -- the LLM exception tier.

Runs only on residuals that survived every deterministic tier. That ordering is
the argument: `TIMING_SPILL` was moved into the solver precisely so the model
could not be credited for closing it, and what reaches here is the narrow set
that genuinely needs a reason rather than a sum -- a mid-cycle repricing, an FX
rate the report does not carry.

The loop is: propose, check, repair once or twice, escalate. The model never
writes a match. `prove_leg2` is called with its rows on exactly the same footing
as rows a human reported, and a proposal that does not close is discarded with
the residual fed back for one more attempt.

Two rules make the audit trail honest:

- **Self-reported confidence is capped, not trusted.** The model's own number is
  recorded, but match confidence is `min(model, CONF_T2_CAP)`. An LLM's stated
  confidence is not calibrated -- FinBalance measured models misreporting the
  consequences of their own entries by 26-41 points -- so it is evidence about
  the model, not evidence about the match.
- **Inferred rows are marked `inferred:llm:*`.** Nothing in the ledger should
  ever make a hypothesis look like a row someone reported.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable

from ..money import FeeSchedule
from ..schemas import (
    MatchRecord,
    PGAdjustment,
    ReconException,
    ReconResult,
    SourceBundle,
    stable_hash,
)
from ..validate import Tolerance, prove_leg2
from . import evidence
from .citations import resolve
from .contracts import AgentReport, CaseOutcome, Hypothesis, ProposerError, Refusal, Usage
from .proposer import Proposer

TIER = "T2"
RULE_LLM_HYPOTHESIS = "leg2.t2.llm_hypothesis"

#: Ceiling on match confidence for anything this tier books, regardless of what
#: the model claimed. The most confident possible LLM-derived match is still
#: less certain than an exact-key match or a unique arithmetic explanation.
CONF_T2_CAP = 0.70

#: Below this, the proposal is not even checked -- a model that is unsure is
#: telling you to send the item to a human, and spending an arithmetic check on
#: it only risks a coincidental close.
MIN_CONFIDENCE = 0.55

#: One retry after a failed check. A second failure means the model is guessing
#: against feedback, and more attempts buy noise rather than accuracy.
MAX_ATTEMPTS = 2

#: Cases are independent -- different batch, different rows, no shared state --
#: so the only reason this ran serially was that I wrote it that way. On a
#: shared endpoint the per-call latency is queue time rather than compute, which
#: is exactly the shape that parallelises well: 18 cases at 80s each is 24
#: minutes serially and about 3 with eight workers.
#:
#: Kept at 1 by default. Concurrency needs a `proposer_factory`, because a
#: proposer that investigates holds per-case state and sharing one across
#: threads would let two cases scribble over each other's context.
DEFAULT_MAX_WORKERS = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def recover_with_agent(
    sources: SourceBundle,
    tol: Tolerance,
    result: ReconResult,
    proposer: Proposer | None = None,
    *,
    fees: FeeSchedule | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    min_confidence: float = MIN_CONFIDENCE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    proposer_factory: Callable[[], Proposer] | None = None,
) -> AgentReport:
    """Attempt the residuals no deterministic tier could close. Mutates `result`.

    Pass `proposer` for the serial path, or `proposer_factory` with
    `max_workers > 1` to run cases concurrently. Results are always applied in
    the original exception order regardless of which case finishes first, so a
    parallel run and a serial run produce identical output.
    """
    if proposer is None and proposer_factory is None:
        raise ValueError("pass either proposer or proposer_factory")
    if max_workers > 1 and proposer_factory is None:
        raise ValueError(
            "concurrent runs need proposer_factory: a proposer that investigates "
            "holds per-case state and cannot be shared across threads"
        )

    fees = fees or FeeSchedule.default()
    report = AgentReport()

    settlement_by_id = {s.settlement_id: s for s in sources.settlements}
    line_by_id = {b.bank_line_id: b for b in sources.bank_lines}

    # Split into work and passengers, preserving position so the exception queue
    # comes back in the order an operator last saw it.
    plan: list[tuple[str, object]] = []
    for exc in result.exceptions:
        targetable = (
            exc.leg == 2
            and exc.entity_kind == "bank_line"
            and exc.residual_paise is not None
            and exc.related_id in settlement_by_id
            and exc.entity_id in line_by_id
        )
        if targetable:
            plan.append(("case", exc))
        else:
            plan.append(("keep", exc))

    cases = [exc for kind, exc in plan if kind == "case"]

    # One proposer per worker thread, not per case: building a client is not
    # free, and a thread only ever works one case at a time.
    local = threading.local()

    def _proposer_for_thread() -> Proposer:
        if proposer_factory is None:
            return proposer  # type: ignore[return-value]
        existing = getattr(local, "proposer", None)
        if existing is None:
            existing = proposer_factory()
            local.proposer = existing
        return existing

    def _work(exc: ReconException):
        return _run_case(
            exc,
            sources=sources,
            tol=tol,
            fees=fees,
            line=line_by_id[exc.entity_id],
            settlement=settlement_by_id[exc.related_id],  # type: ignore[index]
            proposer=_proposer_for_thread(),
            max_attempts=max_attempts,
            min_confidence=min_confidence,
        )

    if max_workers > 1 and cases:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(_work, cases))
    else:
        outcomes = [_work(exc) for exc in cases]

    by_exception = {id(exc): out for exc, out in zip(cases, outcomes)}

    survivors: list[ReconException] = []
    for kind, exc in plan:
        if kind == "keep":
            survivors.append(exc)  # type: ignore[arg-type]
            continue
        case, match, escalated = by_exception[id(exc)]
        report.cases.append(case)
        report.usage.merge(case.usage)
        if match is not None:
            result.matches.append(match)
        if escalated is not None:
            survivors.append(escalated)

    result.exceptions = survivors
    return report


def _run_case(
    exc: ReconException,
    *,
    sources: SourceBundle,
    tol: Tolerance,
    fees: FeeSchedule,
    line,
    settlement,
    proposer: Proposer,
    max_attempts: int,
    min_confidence: float,
) -> tuple[CaseOutcome, MatchRecord | None, ReconException | None]:
    """One exception, start to finish. Pure with respect to shared state.

    Returns what happened, the match if one was earned, and the escalated
    exception if it was not. Touching `result` from here would make the
    concurrent path racy; the caller applies everything in order instead.
    """
    members = sources.payments_by_settlement(settlement.settlement_id)
    linked = sources.adjustments_by_settlement(settlement.settlement_id)

    case = CaseOutcome(
        entity_id=exc.entity_id,
        settlement_id=settlement.settlement_id,
        residual_paise=exc.residual_paise or 0,
        outcome="failed",
    )

    binder = getattr(proposer, "bind", None)
    if callable(binder):
        binder(line, settlement)

    feedback: str | None = None

    for attempt in range(1, max_attempts + 1):
        case.attempts = attempt
        packet = evidence.build(
            sources, line, settlement, exc.residual_paise or 0, fees,
            repair_feedback=feedback,
        )
        proposal, usage = proposer.propose(packet)
        case.usage.merge(usage)

        if isinstance(proposal, ProposerError):
            case.outcome = "failed"
            case.detail = f"{proposal.kind}: {proposal.detail}"
            break

        if isinstance(proposal, Refusal):
            case.outcome = "refused"
            case.detail = proposal.reason
            break

        assert isinstance(proposal, Hypothesis)
        case.model_confidence = proposal.confidence

        if proposal.confidence < min_confidence:
            case.outcome = "low_confidence"
            case.detail = (
                f"model reported {proposal.confidence:.2f}, below the "
                f"{min_confidence:.2f} floor: {proposal.reason}"
            )
            break

        # Citations become money here, computed from the source rows and the fee
        # schedule. The proposer supplied no amounts, so it cannot manufacture a
        # number that makes its own total add up.
        resolution = resolve(sources, settlement, list(proposal.citations), fees)
        if not resolution.ok:
            case.outcome = "unverifiable"
            case.detail = "; ".join(resolution.errors)[:200]
            feedback = (
                f"Attempt {attempt} cited evidence that could not be verified: "
                + "; ".join(resolution.errors)[:300]
                + " Cite only adjustment_ids that appear in the unlinked rows for "
                "this batch, or name payments and a rate and let the system compute "
                "the variance. If no such evidence exists, decline."
            )
            continue

        inferred = [
            PGAdjustment(
                adjustment_id=f"cited:{'+'.join(row.cited_ids)}",
                settlement_id=None,  # type: ignore[arg-type]
                kind=row.source,
                payment_id=None,
                amount_paise=row.amount_paise,
                booked_at=settlement.settled_at,
            )
            for row in resolution.rows
        ]
        proof = prove_leg2(line, settlement, members, linked, tol, hypothesised=inferred)

        if not proof.closes:
            case.outcome = "rejected"
            case.detail = (
                f"cited evidence worth {resolution.total_paise} paise left "
                f"{proof.residual_paise} paise unexplained"
            )
            feedback = (
                f"Attempt {attempt} cited evidence totalling "
                f"{resolution.total_paise} paise. Checked against the ledger, "
                f"{proof.residual_paise} paise remain unexplained. Cite different "
                "or additional evidence, or decline -- you cannot state an amount "
                "directly, only point at rows that exist."
            )
            continue

        match = MatchRecord(
            match_id=f"m2_{line.bank_line_id}",
            leg=2,
            tier=TIER,
            rule_id=RULE_LLM_HYPOTHESIS,
            left_ids=(line.bank_line_id,),
            right_ids=(settlement.settlement_id,),
            confidence=min(proposal.confidence, CONF_T2_CAP),
            proof=proof,
            input_hash=stable_hash(line, settlement, *members, *linked, *inferred),
            hypothesised_ids=resolution.cited_ids,
            created_at=_now(),
        )
        case.cited_ids = resolution.cited_ids
        case.outcome = "resolved"
        case.detail = proposal.reason
        return case, match, None

    escalated = replace(
        exc,
        reason=f"{exc.reason}; agent tier {case.outcome}: {case.detail}",
        escalated_from_tier=TIER,
    )
    return case, None, escalated


def render_report(
    report: AgentReport,
    input_per_mtok: float = 5.00,
    output_per_mtok: float = 25.00,
) -> str:
    """Plain-text agent summary. Cost defaults are Claude Opus 5 list prices."""
    lines: list[str] = []
    w = 72
    lines.append("-" * w)
    lines.append(f"{'AGENT TIER (T2)':<34}{'CASES':>9}{'':>29}")
    lines.append("-" * w)
    lines.append(f"  {'attempted':<32}{report.attempted:>9}")
    lines.append(f"  {'resolved':<32}{report.resolved:>9}")
    lines.append(
        f"  {'rejected by the gate':<32}{report.rejected:>9}"
        "   <- confident but wrong"
    )
    lines.append(f"  {'declined by the model':<32}{report.refused:>9}")
    lines.append(f"  {'below confidence floor':<32}{report.low_confidence:>9}")
    lines.append(f"  {'proposer failed':<32}{report.failed:>9}")
    lines.append("")
    lines.append(f"  {'resolution rate':<32}{report.resolution_rate:>8.1%}")
    cost = report.usage.cost_usd(input_per_mtok, output_per_mtok)
    lines.append(
        f"  {'tokens in / out':<32}"
        f"{report.usage.input_tokens:>9,} / {report.usage.output_tokens:,}"
    )
    lines.append(f"  {'cost':<32}{'$' + format(cost, '.4f'):>9}")
    if report.resolved:
        per = report.cost_per_resolved(input_per_mtok, output_per_mtok)
        lines.append(f"  {'cost per exception resolved':<32}{'$' + format(per, '.4f'):>9}")
    lines.append("-" * w)
    return "\n".join(lines)
