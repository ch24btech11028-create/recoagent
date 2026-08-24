"""Generator properties. The evaluation is only as honest as these."""

from recoagent.defects import DefectClass
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.validate import Tolerance, prove_leg2


def _batch(**kw):
    return generate(GeneratorConfig(**{"n_orders": 400, "seed": 3, **kw}))


def test_same_seed_same_world():
    a = _batch()
    b = _batch()
    assert a.sources == b.sources
    assert a.truth.defects == b.truth.defects


def test_different_seed_different_world():
    assert _batch(seed=3).sources != _batch(seed=4).sources


def test_clean_profile_is_arithmetically_perfect():
    """With no defects injected, every bank credit must be derivable exactly.

    This is the control. If it fails, the generator itself is producing
    inconsistent books and no downstream metric means anything.
    """
    batch = _batch(mix=DefectMix.clean())
    s = batch.sources
    tol = Tolerance.strict()

    for line_id, sid in batch.truth.leg2.items():
        line = next(b for b in s.bank_lines if b.bank_line_id == line_id)
        settlement = next(x for x in s.settlements if x.settlement_id == sid)
        proof = prove_leg2(
            line,
            settlement,
            s.payments_by_settlement(sid),
            s.adjustments_by_settlement(sid),
            tol,
        )
        assert proof.closes, f"{sid} off by {proof.residual_paise} paise with no defects"


def test_clean_profile_injects_nothing():
    assert _batch(mix=DefectMix.clean()).truth.defects == ()


def test_defect_counts_are_exact_not_sampled():
    """Rates produce exact counts, which is what makes the scorer's
    reconciliation check an assertion rather than an impression."""
    batch = _batch(n_orders=2000, seed=7, mix=DefectMix.dev())
    counts = batch.truth.defects_by_class()
    n_settlements = len(batch.sources.settlements)

    expected = round(DefectMix.dev().rates[DefectClass.REFUND_NETTED] * n_settlements)
    assert counts[DefectClass.REFUND_NETTED] == expected

    expected_l1 = round(DefectMix.dev().rates[DefectClass.PARTIAL_CAPTURE] * 2000)
    assert counts[DefectClass.PARTIAL_CAPTURE] == expected_l1


def test_every_defect_class_is_reachable():
    """A class that never injects is a class the taxonomy claims but cannot show."""
    batch = _batch(n_orders=3000, seed=11, mix=DefectMix.dev())
    seen = set(batch.truth.defects_by_class())
    assert seen == set(DefectClass), f"never injected: {set(DefectClass) - seen}"


def test_truncated_narration_actually_destroys_the_utr():
    """The defect must break the join key, not merely shorten a string.

    Clipping at a fixed column sometimes leaves the UTR intact, which would
    make this class silently harmless and inflate the reported match rate.
    """
    from recoagent.legs.leg2 import extract_utr

    batch = _batch(n_orders=3000, seed=11, mix=DefectMix.dev())
    truncated = [
        d.entity_id
        for d in batch.truth.defects
        if d.defect is DefectClass.NARRATION_TRUNCATION
    ]
    assert truncated
    for line_id in truncated:
        line = next(b for b in batch.sources.bank_lines if b.bank_line_id == line_id)
        assert extract_utr(line.narration) is None, (
            f"{line_id} still yields a readable UTR: {line.narration!r}"
        )


def test_held_settlements_have_no_bank_line():
    batch = _batch(n_orders=3000, seed=11, mix=DefectMix.dev())
    held = [s for s in batch.sources.settlements if s.status == "on_hold"]
    assert held
    for s in held:
        assert s.settlement_id not in batch.truth.leg2.values()
