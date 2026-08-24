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

from .legs import leg1, leg2
from .schemas import ReconResult, SourceBundle
from .validate import Tolerance

RUNGS = ("B0", "B1", "B2", "B3")


def run_b0(sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    """Deterministic baseline: exact keys, zero tolerance, no inference.

    This rung exists to be beaten. Its job is to establish what pure
    bookkeeping recovers before any probabilistic or model-driven tier is
    allowed to claim credit for anything.
    """
    tol = tol or Tolerance.strict()
    result = ReconResult(rung="B0")

    leg1.match(sources, tol, result)
    adjudicated = leg2.match(sources, tol, result)
    result.exceptions.extend(
        leg2.unmatched_settlements(sources, result, adjudicated)
    )

    return result


def run(rung: str, sources: SourceBundle, tol: Tolerance | None = None) -> ReconResult:
    if rung == "B0":
        return run_b0(sources, tol)
    raise NotImplementedError(
        f"rung {rung} is not built yet; implemented rungs: B0"
    )
