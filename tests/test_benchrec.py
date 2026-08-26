"""RecoAgent on BenchRec: the matching rules, and the claim they support.

The scoring logic is tested against small hand-built pools so the suite runs
without the 116MB corpus. The two tests that need the real files skip cleanly
when it is absent, because BenchRec is CC BY 4.0 and gitignored -- a suite that
failed without it would fail on every fresh clone.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from recoagent.eval.benchrec import (
    EVAL,
    Candidate,
    Pool,
    Report,
    cents,
    match_one,
    run,
)

DATA = Path("data/benchrec")
have_corpus = pytest.mark.skipif(
    not (DATA / EVAL).exists(), reason="BenchRec corpus not present (gitignored, CC BY 4.0)"
)


def _b(amount: str, value_date: str = "2023-04-08", b_id: str = "b1") -> dict:
    return {"B_id": b_id, "B_amount": amount, "B_valueDate": value_date}


def _pool(*rows: tuple[str, str, str]) -> Pool:
    return Pool([
        Candidate(a_id=f"a{i}", allocation=alloc, amount=cents(amount),
                  value_date=date.fromisoformat(day))
        for i, (alloc, amount, day) in enumerate(rows)
    ])


# ── money ────────────────────────────────────────────────────────────────


def test_money_is_read_as_an_integer():
    assert cents("-20893751.85") == -2089375185
    assert cents("0.01") == 1
    assert cents("") is None
    assert cents("not a number") is None


def test_no_float_ever_touches_an_amount():
    """A cent lost to binary rounding is a false match waiting to happen."""
    assert isinstance(cents("133215.05"), int)
    assert cents("133215.05") == 13321505


# ── the matching rules ───────────────────────────────────────────────────


def test_a_unique_amount_and_date_matches():
    pool = _pool(("ALLOC_A", "100.00", "2023-04-08"))
    predicted, tier, refusal = match_one(pool, _b("100.00"), window_days=1, max_size=1)
    assert predicted == "ALLOC_A" and tier.startswith("T0") and refusal is None


def test_two_candidates_pointing_at_one_allocation_is_not_ambiguous():
    """Different rows, same answer. There is nothing for a human to decide."""
    pool = _pool(("ALLOC_A", "100.00", "2023-04-08"), ("ALLOC_A", "100.00", "2023-04-08"))
    predicted, tier, _ = match_one(pool, _b("100.00"), window_days=1, max_size=1)
    assert predicted == "ALLOC_A" and tier.startswith("T0")


def test_two_allocations_at_the_same_amount_and_date_is_refused():
    """The whole thesis in one test: a coin flip is not an answer."""
    pool = _pool(("ALLOC_A", "100.00", "2023-04-08"), ("ALLOC_B", "100.00", "2023-04-08"))
    predicted, tier, refusal = match_one(pool, _b("100.00"), window_days=1, max_size=1)
    assert predicted is None and tier is None and refusal == "ambiguous"


def test_a_date_that_disagrees_is_refused_by_default():
    """Amount-alone matching is measurably a bad trade; it is off unless asked for."""
    pool = _pool(("ALLOC_A", "100.00", "2023-01-01"))
    predicted, _, refusal = match_one(pool, _b("100.00", "2023-04-08"), window_days=1, max_size=1)
    assert predicted is None and refusal == "ambiguous"

    predicted, tier, _ = match_one(
        pool, _b("100.00", "2023-04-08"), window_days=1, max_size=1, amount_only=True
    )
    assert predicted == "ALLOC_A" and tier.startswith("T1")


def test_an_amount_nobody_carries_is_refused_as_such():
    pool = _pool(("ALLOC_A", "100.00", "2023-04-08"))
    predicted, _, refusal = match_one(pool, _b("999.99"), window_days=1, max_size=1)
    assert predicted is None and refusal == "no candidate"


def test_an_unreadable_amount_is_refused_not_guessed():
    pool = _pool(("ALLOC_A", "100.00", "2023-04-08"))
    predicted, _, refusal = match_one(pool, _b("wat"), window_days=1, max_size=1)
    assert predicted is None and refusal == "unparseable"


def test_a_closing_subset_spanning_two_allocations_is_refused():
    """The N:1 tier proves a total. A total is not an allocation."""
    pool = _pool(("ALLOC_A", "60.00", "2023-04-08"), ("ALLOC_B", "40.00", "2023-04-08"))
    predicted, tier, refusal = match_one(pool, _b("100.00"), window_days=0, max_size=3)
    assert predicted is None and refusal == "split allocation"


def test_a_closing_subset_within_one_allocation_is_accepted():
    pool = _pool(("ALLOC_A", "60.00", "2023-04-08"), ("ALLOC_A", "40.00", "2023-04-08"))
    predicted, tier, _ = match_one(pool, _b("100.00"), window_days=0, max_size=3)
    assert predicted == "ALLOC_A" and tier.startswith("T2")


# ── the report ───────────────────────────────────────────────────────────


def test_the_lead_metric_is_over_answers_given():
    """Refusals must not flatter the wrong-match rate, or refusing would game it."""
    from recoagent.eval.benchrec import Outcome

    r = Report(outcomes=[
        Outcome("b1", "A", "T0", None, "A"),
        Outcome("b2", "B", "T0", None, "A"),
        Outcome("b3", None, None, "ambiguous", "A"),
    ])
    assert r.attempted == 2
    assert r.wrong == 1
    assert r.wrong_match_rate == 0.5      # over answers given, not over the population
    assert r.coverage == pytest.approx(2 / 3)
    assert r.accuracy == pytest.approx(1 / 3)


def test_refusing_everything_scores_no_wrong_matches_but_no_coverage():
    from recoagent.eval.benchrec import Outcome

    r = Report(outcomes=[Outcome(f"b{i}", None, None, "ambiguous", "A") for i in range(5)])
    assert r.wrong_match_rate == 0.0 and r.coverage == 0.0


# ── against the real corpus ──────────────────────────────────────────────


@have_corpus
def test_the_published_benchrec_claim_reproduces():
    """Pins the numbers in results/benchrec_recoagent.txt and the README."""
    report = run(DATA, window_days=1, max_size=1)
    assert report.population == 32_048
    assert report.attempted == 27_037
    assert report.wrong == 77
    assert report.wrong_match_rate == pytest.approx(0.0028, abs=5e-5)
    assert report.coverage == pytest.approx(0.8436, abs=5e-5)
    # The claim only means anything next to the baseline that ships with the set.
    assert report.wrong_match_rate < 0.048 / 10


@have_corpus
def test_the_run_is_deterministic():
    a = run(DATA, window_days=1, max_size=1, limit=3000)
    b = run(DATA, window_days=1, max_size=1, limit=3000)
    assert [(o.b_id, o.predicted, o.tier) for o in a.outcomes] == \
           [(o.b_id, o.predicted, o.tier) for o in b.outcomes]
