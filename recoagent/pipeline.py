"""Rung assembly.

Each rung of the baseline ladder is a different combination of tiers over the
same sources. Keeping them behind one entry point means the ladder is a real
comparison -- same inputs, same scorer, same tolerance policy -- rather than
four scripts that happen to print similar-looking numbers.

    B0   exact join            + exact UTR                  <- implemented
    B1   + Splink linkage      + exact UTR
    B2   + Splink linkage      + SSMP DP-greedy
    B3   + Splink linkage      + SSMP + LLM exception tier
"""

from __future__ import annotations

from .legs import leg1, leg2, leg2_t1
from .schemas import ReconResult, SourceBundle
from .validate import Tolerance

RUNGS = ("B0", "B2")

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
    return _compose(sources, tol or Tolerance.strict(), "B0", with_leg2_t1=False)


def _compose(
    sources: SourceBundle, tol: Tolerance, rung: str, *, with_leg2_t1: bool
) -> ReconResult:
    """Run the tiers in order, then sweep for settlements nothing reached.

    The sweep runs last, after every recovery tier, and that ordering is not
    incidental. Running it mid-pipeline files an unmatched-settlement exception
    that a later tier then silently invalidates by matching the credit -- an
    exception queue that still lists items the system has already resolved.
    """
    result = ReconResult(rung=rung)

    leg1.match(sources, tol, result)
    adjudicated = leg2.match(sources, tol, result)

    if with_leg2_t1:
        leg2_t1.recover(sources, tol, result)
        # Spill pairing runs after local recovery: it reasons over the
        # residuals that survive, so it needs them to have settled first.
        leg2_t1.pair_spills(sources, tol, result)

    result.exceptions.extend(
        leg2.unmatched_settlements(sources, result, adjudicated)
    )
    return result


def run_b2(sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    """B0 plus Leg 2 Tier 1: a calibrated tolerance and SSMP residual closure.

    The tolerance change and the recovery pass arrive together on purpose. A
    tolerance without a solver just quietly absorbs small errors; a solver
    without a tolerance cannot close anything that drifted by a paise. Reported
    as one rung because they are one decision.
    """
    return _compose(sources, tol or Tolerance.calibrated(), "B2", with_leg2_t1=True)


def run(rung: str, sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    if rung == "B0":
        return run_b0(sources, tol)
    if rung == "B2":
        return run_b2(sources, tol)
    raise NotImplementedError(
        f"rung {rung!r} is not built yet; implemented rungs: {', '.join(RUNGS)}"
    )
