"""B0 invariants. These are the claims the submission will actually make."""

import pytest

from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b0

PROFILES = [
    DefectMix.dev(),
    DefectMix.holdout(),
    DefectMix.clean(),
    # Carries three classes no tier in this repository handles. Every
    # invariant below is asserted on it too, which is the point: the
    # safety claims have to survive a defect nobody wrote code for.
    DefectMix.unknown(),
]


def _score(mix, n=1200, seed=7):
    batch = generate(GeneratorConfig(n_orders=n, seed=seed, mix=mix))
    return batch, score(batch, run_b0(batch.sources))


@pytest.mark.parametrize("mix", PROFILES, ids=lambda m: m.label)
def test_false_match_rate_is_zero(mix):
    """The lead metric. Exact keys plus an arithmetic gate cannot be wrong.

    If this ever goes non-zero, the system is booking money against the wrong
    transaction and no other number matters.
    """
    _, card = _score(mix)
    assert card.overall_false_match_rate == 0.0
    for leg in card.legs.values():
        assert leg.false_matches == 0


@pytest.mark.parametrize("mix", PROFILES, ids=lambda m: m.label)
def test_every_injected_defect_is_accounted_for(mix):
    """Injected counts must equal explained counts, per class.

    A divergence means some defect class lands where the matcher never looks,
    and every rate reported above it is flattering itself.
    """
    _, card = _score(mix)
    for a in card.accounting:
        assert a.reconciles, (
            f"{a.defect.value}: injected {a.injected}, found {a.accounted}"
        )
    assert card.fully_reconciles


@pytest.mark.parametrize("mix", PROFILES, ids=lambda m: m.label)
def test_no_clean_record_is_ever_rejected(mix):
    """Exceptions raised against undamaged entities are matcher bugs."""
    _, card = _score(mix)
    assert card.unattributed_exceptions == 0


def test_clean_book_matches_completely():
    """The control run. Anything below 100% here is a bug in the matcher."""
    _, card = _score(DefectMix.clean())
    assert card.overall_auto_match_rate == 1.0
    assert card.value.share == 1.0
    for leg in card.legs.values():
        assert leg.exceptions == 0
        assert leg.recall == 1.0


def test_defects_actually_cost_match_rate():
    """The counterpart to the control: a defective book must NOT fully match.

    If B0 resolved everything, the injected defects would be decorative and the
    later rungs would have nothing left to earn.
    """
    _, card = _score(DefectMix.dev())
    assert card.overall_auto_match_rate < 1.0
    assert card.legs[2].recall < 0.95


def test_held_settlements_are_correctly_declined_not_matched():
    batch, card = _score(DefectMix.dev())
    result = run_b0(batch.sources)
    held = {s.settlement_id for s in batch.sources.settlements if s.status == "on_hold"}
    matched = {m.right_ids[0] for m in result.matches_for_leg(2)}
    assert not (held & matched), "an on-hold settlement was matched to a credit"


def test_duplicate_utr_is_refused_on_both_lines():
    """Refusing both costs recall. Booking either would double-count real money."""
    batch = generate(GeneratorConfig(n_orders=3000, seed=11, mix=DefectMix.dev()))
    result = run_b0(batch.sources)
    matched_lines = {m.left_ids[0] for m in result.matches_for_leg(2)}

    from recoagent.defects import DefectClass

    dupes = [
        d for d in batch.truth.defects if d.defect is DefectClass.DUPLICATE_UTR
    ]
    assert dupes
    for d in dupes:
        for line_id in d.affected_ids:
            assert line_id not in matched_lines
