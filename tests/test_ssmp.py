"""Subset-sum solver: completeness, and the uniqueness distinction."""

import pytest

from recoagent.legs import ssmp


def test_finds_a_single_row():
    r = ssmp.enumerate_closing_subsets([-500, -300, -100], -300)
    assert r.unique
    assert r.solutions[0].indices == (1,)


def test_finds_a_pair():
    r = ssmp.enumerate_closing_subsets([-500, -300, -100], -400)
    assert r.unique
    assert r.solutions[0].indices == (1, 2)


def test_finds_nothing_when_nothing_fits():
    r = ssmp.enumerate_closing_subsets([-500, -300], -77)
    assert not r.solutions
    assert not r.unique and not r.actionable


def test_reports_every_solution_not_just_the_first():
    """Completeness is what makes the uniqueness check mean anything."""
    r = ssmp.enumerate_closing_subsets([-100, -100, -200], -200)
    # both {-200} and {-100, -100} close
    assert len(r.solutions) == 2
    assert r.ambiguous and not r.unique


def test_materially_different_explanations_are_not_actionable():
    """Two different amount-shapes closing one gap is a reason to stop."""
    r = ssmp.enumerate_closing_subsets([-100, -100, -200], -200)
    assert not r.value_equivalent
    assert not r.actionable


def test_fungible_rows_are_actionable():
    """Identical flat-rate rows differ in identity, never in amount.

    A Rs 150 dispute fee is interchangeable with every other Rs 150 dispute
    fee: several subsets close the gap and all of them mean the same thing.
    This is the real shape of the CHARGEBACK_NETTED case.
    """
    r = ssmp.enumerate_closing_subsets([-206625, -15000, -15000, -15000], -221625)
    assert r.ambiguous          # more than one subset closes
    assert r.value_equivalent   # but every one is the same two amounts
    assert r.actionable


def test_tolerance_widens_the_net():
    assert not ssmp.enumerate_closing_subsets([-305], -300, 0).solutions
    assert ssmp.enumerate_closing_subsets([-305], -300, 5).unique


def test_max_size_is_respected():
    values = [-100, -100, -100, -100]
    assert ssmp.enumerate_closing_subsets(values, -400, 0, max_size=3).solutions == ()
    assert ssmp.enumerate_closing_subsets(values, -300, 0, max_size=3).solutions


def test_meet_in_middle_agrees_with_enumeration():
    """The two search strategies must not disagree about what closes."""
    values = [-100, -250, -375, -500, -625, -750, -875, -1000]
    for target in (-350, -625, -1125, -1375, -99):
        a = ssmp.enumerate_closing_subsets(values, target, 0, max_size=3)
        b = ssmp.meet_in_middle(values, target, 0, max_size=3)
        assert {s.indices for s in a.solutions} == {s.indices for s in b.solutions}, target


def test_empty_pool_is_safe():
    r = ssmp.enumerate_closing_subsets([], -100)
    assert r.solutions == () and r.exhaustive and not r.actionable


def test_non_exhaustive_search_never_claims_uniqueness():
    """A truncated search that found one answer proves nothing about the rest."""
    r = ssmp.SearchResult(
        solutions=(ssmp.Subset((0,), -5, -5, (-5,)),),
        exhaustive=False,
        combinations_examined=1,
        method="meet_in_middle",
    )
    assert not r.unique and not r.actionable


@pytest.mark.parametrize("n", [10, 18])
def test_search_stays_cheap_at_realistic_pool_sizes(n):
    values = [-(i + 1) * 1000 for i in range(n)]
    r = ssmp.enumerate_closing_subsets(values, -6000, 0)
    assert r.exhaustive
    assert r.combinations_examined < 2000


def test_ambiguity_grows_with_tolerance():
    """The measured cost of a loose tolerance.

    This is why the tolerance is 10 paise and defended rather than 'a rupee or
    two, to be safe'. Widen it far enough and coincidental subsets appear,
    every one of which is a candidate false match.
    """
    values = [-100_00, -250_00, -375_00, -500_00, -625_00, -750_00]
    target = -350_00
    counts = [
        ssmp.count_ambiguity(values, target, tol) for tol in (0, 100_00, 400_00)
    ]
    assert counts[0] <= counts[1] <= counts[2]
    assert counts[0] == 1
    assert counts[-1] > 1


# ── pruning must never change the answer ─────────────────────────────────


def test_pruning_preserves_every_solution():
    """The prune is an optimisation, not a heuristic.

    Completeness is what makes `unique` meaningful, and `unique` is what the
    gate relies on -- so a prune that drops a valid row would silently turn an
    ambiguous residual into a confident wrong match.
    """
    import random

    rng = random.Random(11)
    for _ in range(300):
        pool = [rng.choice([-1, 1]) * rng.randint(50, 500_000) for _ in range(12)]
        target = sum(rng.sample(pool, rng.randint(1, 3)))
        tol = rng.choice([0, 5, 50])

        pruned = ssmp.enumerate_closing_subsets(pool, target, tol)
        kept, back = ssmp.prune(pool, target, tol)
        # Brute force over the unpruned pool, for comparison.
        from itertools import combinations
        brute = set()
        for size in (1, 2, 3):
            for idx in combinations(range(len(pool)), size):
                if abs(sum(pool[i] for i in idx) - target) <= tol:
                    brute.add(idx)
        assert {s.indices for s in pruned.solutions} == brute, (pool, target, tol)


def test_prune_actually_removes_rows():
    """If it never pruned anything the optimisation would be decorative."""
    pool = [-100, -200, -50_000_000, -80_000_000, 300]
    kept, back = ssmp.prune(pool, -250, 0)
    assert len(kept) < len(pool)
    assert -50_000_000 not in kept


def test_prune_keeps_rows_reachable_only_via_an_opposite_sign():
    """A large row can still participate if a positive row cancels it back."""
    pool = [-900, 700]          # -900 + 700 = -200
    kept, _ = ssmp.prune(pool, -200, 0)
    assert set(kept) == {-900, 700}
    assert ssmp.enumerate_closing_subsets(pool, -200, 0).unique
