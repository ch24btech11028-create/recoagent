"""The adversarial audit has to keep being adversarial.

Three failure modes this file exists to catch, none of which a passing pipeline
would reveal on its own:

1. **The suite stops attacking.** A mutation that silently becomes a no-op --
   because a schema field moved, or a helper started returning None -- keeps
   reporting 100% containment while testing nothing. So the mutations are
   required to actually change the book, and to actually be applicable.
2. **The declared limits stop being limits.** `KNOWN_UNCONTAINED` is a promise
   that those attacks land. If one starts being contained, the honest response
   is to remove it from the list and claim the win -- not to leave a stale
   disclaimer in the README implying a weakness that no longer exists.
3. **A contained attack quietly starts landing.** The audit's own exit code
   covers this in CI; here it is pinned per family so a regression names the
   part of the join that broke.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from recoagent.audit import mutate
from recoagent.audit.mutate import (
    CONTAINED,
    KNOWN_UNCONTAINED,
    MUTATIONS,
    WRONG_MATCH,
)
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.legs import leg2
from recoagent.pipeline import run_b2

ROOT = Path(__file__).resolve().parents[1]

#: Small enough to stay quick, large enough that every mutation finds something
#: eligible. Below roughly 400 orders the two-settlement mutations start
#: returning None and the suite silently thins out.
N = 600
TRIALS = 4


@pytest.fixture(scope="module")
def card():
    return mutate.audit(n_orders=N, trials=TRIALS)


def test_every_mutation_is_applicable_and_actually_mutates():
    """A mutation that never fires, or fires without changing anything, is a
    test that reports success for doing nothing."""
    batch = generate(GeneratorConfig(n_orders=N, seed=7, mix=DefectMix.dev()))
    baseline = run_b2(batch.sources)
    mutate._PAIR_CACHE[id(batch.sources)] = mutate._clean_pairs(
        batch.sources, batch.truth, baseline
    )
    try:
        for m in MUTATIONS:
            applied = [
                mut
                for seed in range(100_000, 100_000 + 12)
                if (mut := m.apply(batch.sources, batch.truth,
                                   __import__("random").Random(seed))) is not None
            ]
            assert applied, f"{m.name} never found anything to mutate"
            for mut in applied:
                assert mut.sources is not batch.sources
                assert (
                    mut.sources.bank_lines != batch.sources.bank_lines
                    or mut.sources.settlements != batch.sources.settlements
                    or mut.sources.payments != batch.sources.payments
                ), f"{m.name} returned a bundle identical to the original"
                assert mut.targets, f"{m.name} judged nothing"
    finally:
        mutate._PAIR_CACHE.clear()


def test_no_undeclared_attack_books_the_wrong_batch(card):
    """The lead metric. Everything not on the known-limits list is contained."""
    assert card.unexpected == [], "\n".join(
        f"{c.mutation} seed={c.seed}: {c.detail}" for c in card.unexpected
    )


def test_nothing_crashes(card):
    """A matcher that raises on malformed input stops mid-book."""
    assert card.crashes == [], "\n".join(
        f"{c.mutation} seed={c.seed}: {c.detail}" for c in card.crashes
    )


@pytest.mark.parametrize("family", ["narration", "amount", "timing"])
def test_these_families_are_fully_contained(card, family):
    """Pinned per family so a regression names what broke, not just that
    something did. Structural is excluded: it holds the declared limits."""
    cases = card.by("family").get(family, [])
    assert cases, f"no {family} cases ran"
    bad = [c for c in cases if c.verdict not in CONTAINED]
    assert not bad, "\n".join(f"{c.mutation} seed={c.seed}: {c.detail}" for c in bad)


def test_the_declared_limits_still_land():
    """If one of these is now contained, take it off the list and say so.

    Left unchecked, `KNOWN_UNCONTAINED` becomes a disclaimer for a weakness that
    was fixed years ago -- which understates the system just as dishonestly as
    an unlisted failure overstates it.

    Checked against the published scorecard rather than this file's fast
    fixture, and for a reason worth recording: `perfect_forgery` lands about
    once in twenty-four, so four trials contain it by luck roughly 85% of the
    time. Asserting it there would have produced a test that fails at random.
    The rate is a property of the attack, and the artifact is where the attack
    was run enough times to see it.
    """
    published = json.loads((ROOT / "results" / "mutation_audit.json").read_text())
    landed = {c["mutation"] for c in published["known_limits"]}
    assert set(published["declared_uncontained"]) == set(KNOWN_UNCONTAINED), (
        "the published scorecard was generated against a different "
        "KNOWN_UNCONTAINED list; regenerate it"
    )
    for name in KNOWN_UNCONTAINED:
        assert name in landed, (
            f"{name} is on KNOWN_UNCONTAINED but was contained in every case of "
            f"the published run -- remove it from the list and claim the win"
        )


def test_the_settlement_window_is_what_contains_the_undated_forgery():
    """The claim in `leg2.SETTLEMENT_WINDOW_DAYS` is that the window cuts the
    plain forgery sharply. Assert the direction and the magnitude, not the exact
    count -- the point is that removing the check makes things markedly worse."""
    original = leg2.SETTLEMENT_WINDOW_DAYS
    try:
        with_check = mutate.audit(n_orders=1200, trials=12, only="perfect_forgery")
        leg2.SETTLEMENT_WINDOW_DAYS = 10_000
        without = mutate.audit(n_orders=1200, trials=12, only="perfect_forgery")
    finally:
        leg2.SETTLEMENT_WINDOW_DAYS = original

    assert len(without.wrong_matches) > len(with_check.wrong_matches), (
        "removing the settlement window did not make the forgery easier, so "
        "the window is not what is holding it back"
    )


def test_a_credit_dated_outside_the_window_is_refused_not_matched():
    """The narrow unit behind the audit result: a well-keyed credit whose date
    does not belong to its payout is an exception, however well it adds up."""
    from dataclasses import replace
    from datetime import timedelta

    batch = generate(GeneratorConfig(n_orders=400, seed=7, mix=DefectMix.clean()))
    baseline = run_b2(batch.sources)
    pairs = mutate._clean_pairs(batch.sources, batch.truth, baseline)
    assert pairs, "clean book produced no matched credits to work from"

    line, settlement = pairs[0]
    far = replace(
        line, value_date=line.value_date + timedelta(days=leg2.SETTLEMENT_WINDOW_DAYS + 5)
    )
    sources = replace(
        batch.sources,
        bank_lines=tuple(
            far if b.bank_line_id == line.bank_line_id else b
            for b in batch.sources.bank_lines
        ),
    )
    result = run_b2(sources)

    assert not [
        m for m in result.matches_for_leg(2) if m.left_ids[0] == line.bank_line_id
    ], "a credit dated outside the settlement window was still booked"
    reasons = [
        e.reason for e in result.exceptions if e.entity_id == line.bank_line_id
    ]
    assert any("settlement window" in r for r in reasons), reasons


def test_the_published_scorecard_matches_a_fresh_run():
    """`results/mutation_audit.json` is evidence, so it has to be re-derivable.

    Regenerate with:
        python -m recoagent.audit.mutate --n 2000 --trials 25 \\
            --out results/mutation_audit.json
    """
    published = json.loads((ROOT / "results" / "mutation_audit.json").read_text())
    fresh = subprocess.run(
        [sys.executable, "-m", "recoagent.audit.mutate",
         "--n", str(published["n_orders"]),
         "--trials", str(published["trials_per_mutation"]),
         "--json"],
        capture_output=True, text=True, cwd=ROOT,
    )
    # A declared limit means the command exits non-zero only on a real failure.
    assert fresh.returncode == 0, fresh.stdout[-2000:] + fresh.stderr[-2000:]
    produced = json.loads(fresh.stdout[fresh.stdout.index("{"):])
    assert produced == published, (
        "the committed scorecard is not what the code produces; regenerate it"
    )
