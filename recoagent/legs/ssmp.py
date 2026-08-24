"""Subset Sum Matching for Leg 2.

The problem
-----------
A bank credit joins a settlement batch, but the arithmetic does not close: the
money that arrived differs from the sum of the rows the gateway linked to that
batch. Somewhere in the book there are unlinked rows -- an orphaned refund, a
chargeback and its dispute fee, a manual adjustment -- whose total is exactly
the gap. Finding them is an instance of the **Subset Sum Matching Problem**
(Wu et al., 2025): choose a disjoint subset with `|subset_sum - target| <= eps`.

Why exact bounded enumeration rather than the DP
------------------------------------------------
Wu et al. describe three solver families: MILP (optimal, intractable as eps
grows), DP-greedy (`O(M * S_max)`, near-optimal, scales), and search-based
meet-in-the-middle (exponential, small instances only). The DP is the one
usually recommended for reconciliation at scale, and it is the wrong choice
*here*, for two concrete reasons:

1. **S_max is enormous at paise granularity.** The DP is pseudo-polynomial in
   the maximum achievable sum. Settlement values run to crores -- 10^9 paise --
   so the table is not buildable. Discretising to rupees would make it
   tractable and would also throw away the paise precision that the
   ROUNDING_DRIFT class exists to test. Reconciliation that cannot see a
   two-paise discrepancy is not reconciliation.

2. **The DP returns *a* solution, not *all* of them.** Uniqueness is a
   correctness requirement in this system, not a nicety. If two different sets
   of rows both close the same gap, the honest verdict is "ambiguous, escalate"
   -- and a solver that hands back one answer cannot tell you that. Booking the
   wrong refund against a batch reconciles the total while corrupting the
   ledger underneath it.

The candidate pools here are small by construction -- orphaned rows inside a
date window around one settlement, typically fewer than thirty -- and real
netting involves one or two rows, rarely more than three. Bounded exact
enumeration over that space is both cheap and complete, so it is the primary
solver. `meet_in_middle` is provided for the larger-`max_size` case, which is
the same search family Wu et al. describe.

The tolerance is where this gets dangerous. As eps grows, spurious subsets
appear that close the arithmetic by coincidence, and every one of them is a
false match waiting to happen. `count_ambiguity` exists to measure that rate
rather than assume it away.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

from ..money import Paise

#: Real netting involves a refund, or a chargeback plus its dispute fee, or a
#: manual adjustment. Beyond three rows the search space grows fast and the
#: chance of a coincidental match grows with it.
DEFAULT_MAX_SIZE = 3

#: Above this many enumerated combinations, fall back to meet-in-the-middle.
ENUMERATION_BUDGET = 250_000


@dataclass(frozen=True)
class Subset:
    """One set of rows whose total lands within tolerance of the target."""

    indices: tuple[int, ...]
    total: Paise
    target: Paise
    #: The amounts themselves, so two subsets can be compared for financial
    #: equivalence without reaching back into the pool they came from.
    values: tuple[Paise, ...] | None = None

    @property
    def residual(self) -> Paise:
        return self.total - self.target

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class SearchResult:
    """Every closing subset found, plus what it cost to find them.

    `solutions` is deliberately a list rather than a single best answer. The
    caller needs to know whether the answer was unique before it is allowed to
    act on it.
    """

    solutions: tuple[Subset, ...]
    exhaustive: bool
    combinations_examined: int
    method: str

    @property
    def unique(self) -> bool:
        """Exactly one distinct subset closes, and the search was complete.

        A non-exhaustive search that happened to find one solution proves
        nothing about uniqueness, so it never reports unique.
        """
        return self.exhaustive and len(self.solutions) == 1

    @property
    def ambiguous(self) -> bool:
        return len(self.solutions) > 1

    @property
    def value_equivalent(self) -> bool:
        """Every competing subset is the same multiset of amounts.

        Flat-rate rows make this common and it is not a defect in the data: a
        Rs 150 dispute fee is identical to every other Rs 150 dispute fee, so
        several subsets close the same gap using different-but-interchangeable
        rows. The explanations differ in *which rows* they name, never in what
        the batch is owed.

        This distinction is the whole reason `unique` is not the only gate. A
        genuinely ambiguous residual -- two different amounts that both happen
        to fit -- is a reason to stop. Fungible rows are not, because the
        pairing being decided here comes from the join, and the subset only has
        to prove the total.
        """
        if not self.solutions:
            return False
        shapes = {
            tuple(sorted(s.values)) for s in self.solutions if s.values is not None
        }
        return len(shapes) == 1

    @property
    def actionable(self) -> bool:
        """Safe to act on: either one answer, or several that mean the same thing."""
        return self.exhaustive and bool(self.solutions) and (
            len(self.solutions) == 1 or self.value_equivalent
        )

    @property
    def best(self) -> Subset | None:
        if not self.solutions:
            return None
        return min(self.solutions, key=lambda s: (abs(s.residual), s.size, s.indices))


def _closes(total: Paise, target: Paise, tolerance: Paise) -> bool:
    return abs(total - target) <= tolerance


def enumerate_closing_subsets(
    values: list[Paise],
    target: Paise,
    tolerance: Paise = 0,
    max_size: int = DEFAULT_MAX_SIZE,
    budget: int = ENUMERATION_BUDGET,
) -> SearchResult:
    """Find every subset of size 1..max_size summing to within tolerance of target.

    Complete for the bounded size, which is what makes the uniqueness check
    meaningful. Returns `exhaustive=False` if the search space exceeded the
    budget, in which case the caller must treat any result as unconfirmed.
    """
    n = len(values)
    max_size = min(max_size, n)
    if n == 0 or max_size <= 0:
        return SearchResult((), True, 0, "enumerate")

    space = sum(comb(n, k) for k in range(1, max_size + 1))
    if space > budget:
        return meet_in_middle(values, target, tolerance, max_size, budget)

    found: list[Subset] = []
    examined = 0
    for size in range(1, max_size + 1):
        for idx in combinations(range(n), size):
            examined += 1
            total = sum(values[i] for i in idx)
            if _closes(total, target, tolerance):
                found.append(
                    Subset(
                        indices=idx,
                        total=total,
                        target=target,
                        values=tuple(values[i] for i in idx),
                    )
                )

    return SearchResult(tuple(found), True, examined, "enumerate")


def meet_in_middle(
    values: list[Paise],
    target: Paise,
    tolerance: Paise = 0,
    max_size: int = DEFAULT_MAX_SIZE,
    budget: int = ENUMERATION_BUDGET,
) -> SearchResult:
    """Search-family solver for pools too large to enumerate directly.

    Splits the pool, precomputes bounded subset sums on each half, then joins
    them. This is the third family in Wu et al.; it trades memory for reach and
    stops being exhaustive once the budget is hit, which is reported honestly
    rather than silently.
    """
    n = len(values)
    if n == 0:
        return SearchResult((), True, 0, "meet_in_middle")

    mid = n // 2
    left_idx, right_idx = list(range(mid)), list(range(mid, n))

    def sums(pool: list[int], cap: int) -> dict[Paise, list[tuple[int, ...]]]:
        out: dict[Paise, list[tuple[int, ...]]] = {0: [()]}
        for size in range(1, cap + 1):
            for idx in combinations(pool, size):
                out.setdefault(sum(values[i] for i in idx), []).append(idx)
        return out

    left = sums(left_idx, min(max_size, len(left_idx)))
    right = sums(right_idx, min(max_size, len(right_idx)))

    found: list[Subset] = []
    examined = 0
    exhaustive = True
    for ltotal, lsets in left.items():
        for rtotal, rsets in right.items():
            for lset in lsets:
                for rset in rsets:
                    if not lset and not rset:
                        continue
                    if len(lset) + len(rset) > max_size:
                        continue
                    examined += 1
                    if examined > budget:
                        exhaustive = False
                        break
                    if _closes(ltotal + rtotal, target, tolerance):
                        idx = tuple(sorted(lset + rset))
                        found.append(
                            Subset(
                                indices=idx,
                                total=ltotal + rtotal,
                                target=target,
                                values=tuple(values[i] for i in idx),
                            )
                        )
                if not exhaustive:
                    break
            if not exhaustive:
                break
        if not exhaustive:
            break

    # Deduplicate: a subset can be reachable by more than one split path.
    unique_by_idx = {s.indices: s for s in found}
    return SearchResult(
        tuple(unique_by_idx.values()), exhaustive, examined, "meet_in_middle"
    )


def count_ambiguity(
    values: list[Paise],
    target: Paise,
    tolerance: Paise,
    max_size: int = DEFAULT_MAX_SIZE,
) -> int:
    """How many distinct subsets close this gap.

    Reported so the cost of a loose tolerance is measured rather than assumed.
    Zero means no explanation was found; one means a confident answer; more than
    one means the arithmetic cannot distinguish between competing explanations
    and the item belongs with a human.
    """
    return len(enumerate_closing_subsets(values, target, tolerance, max_size).solutions)
