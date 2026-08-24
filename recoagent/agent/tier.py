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

from dataclasses import replace
from datetime import datetime, timezone

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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def recover_with_agent(
    sources: SourceBundle,
    tol: Tolerance,
    result: ReconResult,
    proposer: Proposer,
    *,
    fees: FeeSchedule | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    min_confidence: float = MIN_CONFIDENCE,
) -> AgentReport:
    """Attempt the residuals no deterministic tier could close. Mutates `result`."""
    fees = fees or FeeSchedule.default()
    report = AgentReport()

    settlement_by_id = {s.settlement_id: s for s in sources.settlements}
    line_by_id = {b.bank_line_id: b for b in sources.bank_lines}

    survivors: list[ReconException] = []

    for exc in result.exceptions:
        targetable = (
            exc.leg == 2
            and exc.entity_kind == "bank_line"
            and exc.residual_paise is not None
            and exc.related_id in settlement_by_id
            and exc.entity_id in line_by_id
        )
        if not targetable:
            survivors.append(exc)
            continue

        line = line_by_id[exc.entity_id]
        settlement = settlement_by_id[exc.related_id]  # type: ignore[index]
        members = sources.payments_by_settlement(settlement.settlement_id)
        linked = sources.adjustments_by_settlement(settlement.settlement_id)

        case = CaseOutcome(
            entity_id=exc.entity_id,
            settlement_id=settlement.settlement_id,
            residual_paise=exc.residual_paise or 0,
            outcome="failed",
        )

        booked = False
        feedback: str | None = None

        for attempt in range(1, max_attempts + 1):
            case.attempts = attempt
            packet = evidence.build(
                sources,
                line,
                settlement,
                exc.residual_paise or 0,
                fees,
                repair_feedback=feedback,
            )
            proposal, usage = proposer.propose(packet)
            case.usage.merge(usage)
            report.usage.merge(usage)

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

            inferred = [
                PGAdjustment(
                    adjustment_id=f"inferred:llm:{line.bank_line_id}:{attempt}:{i}",
                    settlement_id=None,  # type: ignore[arg-type]
                    kind="llm_hypothesis",
                    payment_id=None,
                    amount_paise=row.amount_paise,
                    booked_at=settlement.settled_at,
                )
                for i, row in enumerate(proposal.rows)
            ]
            proof = prove_leg2(
                line, settlement, members, linked, tol, hypothesised=inferred
            )

            if not proof.closes:
                case.outcome = "rejected"
                case.detail = (
                    f"proposal of {proposal.total_paise} paise left "
                    f"{proof.residual_paise} paise unexplained"
                )
                feedback = (
                    f"Attempt {attempt} proposed rows totalling "
                    f"{proposal.total_paise} paise. Checked against the ledger, "
                    f"{proof.residual_paise} paise remain unexplained. The rows "
                    "must sum to exactly the residual. If you cannot account for "
                    "the difference, decline instead of adjusting the numbers to fit."
                )
                continue

            result.matches.append(
                MatchRecord(
                    match_id=f"m2_{line.bank_line_id}",
                    leg=2,
                    tier=TIER,
                    rule_id=RULE_LLM_HYPOTHESIS,
                    left_ids=(line.bank_line_id,),
                    right_ids=(settlement.settlement_id,),
                    confidence=min(proposal.confidence, CONF_T2_CAP),
                    proof=proof,
                    input_hash=stable_hash(line, settlement, *members, *linked, *inferred),
                    created_at=_now(),
                )
            )
            case.outcome = "resolved"
            case.detail = proposal.reason
            booked = True
            break

        report.cases.append(case)

        if not booked:
            survivors.append(
                replace(
                    exc,
                    reason=f"{exc.reason}; agent tier {case.outcome}: {case.detail}",
                    escalated_from_tier=TIER,
                )
            )

    result.exceptions = survivors
    return report


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
