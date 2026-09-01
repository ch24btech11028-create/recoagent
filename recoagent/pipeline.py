"""Rung assembly.

Each rung of the baseline ladder is a different combination of tiers over the
same sources. Keeping them behind one entry point means the ladder is a real
comparison -- same inputs, same scorer, same tolerance policy -- rather than
four scripts that happen to print similar-looking numbers.

    B0   exact join            + exact UTR                  <- implemented
    B1   + Splink linkage      + exact UTR
    B2   + documented capture  + SSMP DP-greedy
    B3   + documented capture  + SSMP + LLM exception tier
"""

from __future__ import annotations

import time

from . import trace
from .legs import leg1, leg2, leg2_t1
from .schemas import ReconResult, SourceBundle
from .validate import Tolerance

_log = trace.logger("pipeline")

RUNGS = ("B0", "B2", "B3")

#: The two legs are independent -- Splink only touches Leg 1, SSMP only Leg 2 --
#: so the ladder is really two ladders over shared inputs, and rungs can be
#: built out of order without making the comparison unfair. B1 (Splink) is not
#: built yet; B2 here means B0 plus the Leg 2 Tier 1 recovery pass.


def run_b0(sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    """Deterministic baseline: exact keys, zero tolerance, no inference.

    This rung exists to be beaten. Its job is to establish what pure
    bookkeeping recovers before any probabilistic or model-driven tier is
    allowed to claim credit for anything.
    """
    return _compose(sources, tol or Tolerance.strict(), "B0", with_recovery=False)


def _compose(
    sources: SourceBundle, tol: Tolerance, rung: str, *, with_recovery: bool
) -> ReconResult:
    """Run the tiers in order, then sweep for settlements nothing reached.

    The sweep runs last, after every recovery tier, and that ordering is not
    incidental. Running it mid-pipeline files an unmatched-settlement exception
    that a later tier then silently invalidates by matching the credit -- an
    exception queue that still lists items the system has already resolved.
    """
    started = time.perf_counter()
    result = ReconResult(rung=rung)

    leg1.match(sources, tol, result, with_t1=with_recovery)
    adjudicated = leg2.match(sources, tol, result)

    if with_recovery:
        leg2_t1.recover(sources, tol, result)
        # Spill pairing runs after local recovery: it reasons over the
        # residuals that survive, so it needs them to have settled first.
        leg2_t1.pair_spills(sources, tol, result)

    result.exceptions.extend(
        leg2.unmatched_settlements(sources, result, adjudicated)
    )
    trace.event(
        _log, "run.complete", rung=rung,
        orders=len(sources.orders), payments=len(sources.payments),
        settlements=len(sources.settlements), bank_lines=len(sources.bank_lines),
        matches=len(result.matches), exceptions=len(result.exceptions),
        leg1_exceptions=len(result.exceptions_for_leg(1)),
        leg2_exceptions=len(result.exceptions_for_leg(2)),
        tolerance_paise=tol.leg2_paise,
        seconds=f"{time.perf_counter() - started:.3f}",
    )
    return result


def run_b2(sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    """B0 plus the recovery tiers: documented captures, calibrated tolerance, SSMP.

    The tolerance change and the recovery pass arrive together on purpose. A
    tolerance without a solver just quietly absorbs small errors; a solver
    without a tolerance cannot close anything that drifted by a paise. Reported
    as one rung because they are one decision.

    Leg 1's tier is here for the same reason the repricing tier is: the book
    contains a document -- the gateway's own `partially_captured` status, with
    fees that re-derive at the contracted rate -- that settles a difference B0
    can only file. Leg 1 keeps its zero tolerance throughout; nothing is
    absorbed, the variance is carried on the match record instead.
    """
    return _compose(sources, tol or Tolerance.calibrated(), "B2", with_recovery=True)


def run_b3(
    sources: SourceBundle,
    proposer=None,
    tol: Tolerance | None = None,
    *,
    max_workers: int = 1,
    proposer_factory=None,
) -> tuple[ReconResult, "AgentReport"]:
    """B2 plus the LLM exception tier.

    Returns the result *and* the agent report, because B3's cost and its
    rejection count are part of the finding, not diagnostics. A rung that
    resolves five exceptions by spending more than a human review costs has not
    made the system better, and the scorecard alone would not show it.
    """
    from .agent.tier import recover_with_agent
    from .legs.repricing import rate_book

    tol = tol or Tolerance.calibrated()
    result = _compose(sources, tol, "B3", with_recovery=True)

    # The agent's citations are checked against the merchant's own paperwork.
    # Without this the tier can compute a claimed rate's consequences but has
    # nothing to confirm the rate itself, so every fee and FX explanation --
    # however well reasoned -- can only ever close as `needs_approval`. Built
    # from the same reading `legs/repricing.py` uses, so a rate cannot be
    # authoritative for one tier and not the other.
    #
    # Note what this does *not* do: a notice that applies cleanly has already
    # been spent by the deterministic tier, so the book is closed before the
    # model sees it. What reaches here is the harder remainder -- a batch whose
    # paperwork is missing, or partial, or contradicted -- and the rate book's
    # job there is to catch the cases where a cited rate does turn out to be on
    # file after all.
    when = max((s.settled_at for s in sources.settlements), default=None)
    book = rate_book(sources, when) if when is not None else None

    report = recover_with_agent(
        sources, tol, result, proposer,
        max_workers=max_workers, proposer_factory=proposer_factory,
        rate_book=book,
    )
    return result, report


def run(rung: str, sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    if rung == "B0":
        return run_b0(sources, tol)
    if rung == "B2":
        return run_b2(sources, tol)
    if rung == "B3":
        raise ValueError(
            "B3 needs a proposer; call run_b3(sources, proposer) directly"
        )
    raise NotImplementedError(
        f"rung {rung!r} is not built yet; implemented rungs: {', '.join(RUNGS)}"
    )
