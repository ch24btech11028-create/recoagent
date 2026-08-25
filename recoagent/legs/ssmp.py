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

from dataclasses import dataclass, replace
from bisect import bisect_left, bisect_right
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


def prune(
    values: list[Paise], target: Paise, tolerance: Paise
) -> tuple[list[Paise], list[int]]:
    """Drop rows that cannot participate in any closing subset.

    Sound, not heuristic: a row survives if its magnitude could still be
    cancelled back to the target by every opposite-signed row in the pool
    combined. Nothing that could contribute is ever removed, so completeness --
    and therefore the uniqueness check the gate depends on -- is preserved.

    This matters more than it sounds. The candidate pool is drawn from a date
    window, so it grows with the size of the book while the search over it is
    cubic. At 20,000 orders that was 11.4 million subset sums and 80% of total
    runtime; the rows being enumerated were overwhelmingly ones no arithmetic
    could ever have used.
    """
    if not values:
        return [], []
    opposite = sum(abs(v) for v in values if (v > 0) != (target > 0))
    ceiling = abs(target) + tolerance + opposite
    kept: list[Paise] = []
    index: list[int] = []
    for i, v in enumerate(values):
        if abs(v) <= ceiling:
            kept.append(v)
            index.append(i)
    return kept, index


def _in_range(
    order: list[tuple[Paise, int]], low: Paise, high: Paise
) -> list[tuple[Paise, int]]:
    """Every (amount, original index) with amount in [low, high]. O(log n + hits)."""
    lo = bisect_left(order, (low, -1))
    hi = bisect_right(order, (high, 1 << 62))
    return order[lo:hi]


def enumerate_closing_subsets(
    values: list[Paise],
    target: Paise,
    tolerance: Paise = 0,
    max_size: int = DEFAULT_MAX_SIZE,
    budget: int = ENUMERATION_BUDGET,
) -> SearchResult:
    """Every subset of size 1..max_size summing to within tolerance of target.

    Indexed rather than enumerated. Sorting the pool once turns "is there a row
    that completes this partial sum" into a binary search, so the work drops a
    whole power for each subset size: singles become a lookup, pairs a scan, and
    triples a scan over pairs. Enumerating triples directly is cubic in the pool,
    and the pool grows with the size of the book -- at 50,000 orders that was
    eight and a half minutes, nearly all of it spent summing combinations that
    the first element had already ruled out.

    Still complete for the bounded size. Completeness is not a nicety here: the
    gate accepts a subset only when it is the *only* one that closes, so a search
    that missed a competing explanation would turn an ambiguous residual into a
    confident wrong match.
    """
    pool, back = prune(values, target, tolerance)
    n = len(pool)
    max_size = min(max_size, n)
    if n == 0 or max_size <= 0:
        return SearchResult((), True, 0, "indexed")

    order = sorted((v, i) for i, v in enumerate(pool))
    found: dict[tuple[int, ...], Subset] = {}
    examined = 0

    def record(idx_local: tuple[int, ...], total: Paise) -> None:
        key = tuple(sorted(back[i] for i in idx_local))
        if key not in found:
            found[key] = Subset(
                indices=key,
                total=total,
                target=target,
                values=tuple(pool[i] for i in sorted(idx_local)),
            )

    # size 1
    for v, i in _in_range(order, target - tolerance, target + tolerance):
        examined += 1
        record((i,), v)

    # size 2: fix one row, look up the complement
    if max_size >= 2:
        for a_val, a_idx in order:
            need = target - a_val
            for b_val, b_idx in _in_range(order, need - tolerance, need + tolerance):
                if b_idx <= a_idx:
                    continue
                examined += 1
                record((a_idx, b_idx), a_val + b_val)

    # size 3: fix two rows, look up the third
    if max_size >= 3:
        for ai in range(n):
            a_val = pool[ai]
            for bi in range(ai + 1, n):
                partial = a_val + pool[bi]
                need = target - partial
                for c_val, c_idx in _in_range(order, need - tolerance, need + tolerance):
                    if c_idx <= bi:
                        continue
                    examined += 1
                    record((ai, bi, c_idx), partial + c_val)
                if examined > budget:
                    return SearchResult(
                        tuple(found.values()), False, examined, "indexed"
                    )

    return SearchResult(tuple(found.values()), True, examined, "indexed")


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
