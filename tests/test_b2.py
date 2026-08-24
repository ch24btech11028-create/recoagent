"""Rung B2 invariants, and what it must have improved over B0."""

import pytest

from recoagent.defects import DefectClass
from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b0, run_b2

PROFILES = [DefectMix.dev(), DefectMix.holdout(), DefectMix.clean()]


def _both(mix, n=1500, seed=7):
    batch = generate(GeneratorConfig(n_orders=n, seed=seed, mix=mix))
    return batch, score(batch, run_b0(batch.sources)), score(batch, run_b2(batch.sources))


@pytest.mark.parametrize("mix", PROFILES, ids=lambda m: m.label)
def test_recovery_never_costs_correctness(mix):
    """The whole point of the rung.

    A tier that raises recall by booking wrong matches has made the system
    worse, however good the headline looks.
    """
    _, _, b2 = _both(mix)
    assert b2.overall_false_match_rate == 0.0
    assert b2.mishandled_total == 0
    assert b2.unattributed_exceptions == 0


@pytest.mark.parametrize("mix", PROFILES, ids=lambda m: m.label)
def test_b2_never_loses_ground_to_b0(mix):
    _, b0, b2 = _both(mix)
    assert b2.legs[2].true_matches >= b0.legs[2].true_matches
    assert b2.value.share >= b0.value.share


def test_b2_beats_b0_where_it_claims_to():
    _, b0, b2 = _both(DefectMix.dev())
    assert b2.legs[2].recall > b0.legs[2].recall
    assert b2.legs[2].exceptions < b0.legs[2].exceptions


@pytest.mark.parametrize(
    "mix", [DefectMix.dev(), DefectMix.holdout()], ids=["dev", "holdout"]
)
def test_the_solver_closes_exactly_the_classes_it_claims(mix):
    """The tier's docstring promises four classes and disclaims the rest.

    That claim matters more than usual: the case for adding an LLM at B3 rests
    entirely on what is left over needing a *reason* rather than an amount. If
    the solver silently starts closing FX or timing spills, the B3 argument is
    gone and this test should be the thing that says so.
    """
    _, _, b2 = _both(mix)
    by_class = {a.defect: a for a in b2.accounting}

    for cls in (
        DefectClass.REFUND_NETTED,
        DefectClass.CHARGEBACK_NETTED,
        DefectClass.ADJUSTMENT_ENTRY,
        DefectClass.ROUNDING_DRIFT,
    ):
        assert by_class[cls].resolved == by_class[cls].injected, cls

    for cls in (
        DefectClass.FEE_TAX_VARIANCE,
        DefectClass.FX_CONVERSION,
        DefectClass.TIMING_SPILL,
        DefectClass.DUPLICATE_UTR,
    ):
        assert by_class[cls].flagged == by_class[cls].injected, cls


def test_clean_book_is_still_perfect_at_b2():
    _, _, b2 = _both(DefectMix.clean())
    assert b2.overall_auto_match_rate == 1.0
    assert b2.value.share == 1.0


def test_no_unlinked_row_is_spent_twice():
    """Fungible rows are interchangeable, not infinite.

    One Rs 150 dispute fee explains one batch. Spending the same row against
    two batches would reconcile both with the same money.
    """
    batch = generate(GeneratorConfig(n_orders=2500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)

    used: list[str] = []
    for m in result.matches:
        used.extend(m.hypothesised_ids)
    assert used, "no hypothesised rows were recorded at all"
    assert len(used) == len(set(used)), "an unlinked row was spent on two batches"


def test_hypothesised_rows_are_real_and_unlinked():
    """A hypothesis may only reach for rows the source data left unattached."""
    batch = generate(GeneratorConfig(n_orders=2500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    orphans = {
        a.adjustment_id for a in batch.sources.adjustments if a.settlement_id is None
    }
    for m in result.matches:
        for aid in m.hypothesised_ids:
            assert aid in orphans, f"{m.match_id} claimed a row that was already linked"


def test_matched_settlements_are_not_also_reported_unmatched():
    """An exception queue must not list items the system already resolved."""
    batch = generate(GeneratorConfig(n_orders=2500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    matched = {m.right_ids[0] for m in result.matches_for_leg(2)}
    stale = [
        e
        for e in result.exceptions
        if e.entity_kind == "settlement" and e.entity_id in matched
    ]
    assert not stale, f"{len(stale)} settlements reported unmatched after being matched"


def test_every_b2_match_still_carries_a_closing_proof():
    batch = generate(GeneratorConfig(n_orders=1500, seed=7, mix=DefectMix.dev()))
    for m in run_b2(batch.sources).matches:
        assert m.proof is not None and m.proof.closes


def test_recovered_matches_are_less_confident_than_exact_ones():
    batch = generate(GeneratorConfig(n_orders=1500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    t0 = [m.confidence for m in result.matches_for_leg(2) if m.tier == "T0"]
    t1 = [m.confidence for m in result.matches_for_leg(2) if m.tier == "T1"]
    assert t0 and t1
    assert min(t0) == 1.0
    assert max(t1) < 1.0


def test_the_explanation_is_right_not_merely_arithmetically_sufficient():
    """Closing the total is not the same as explaining it correctly.

    The pairing on Leg 2 comes from the UTR join, so naming the wrong rows in a
    hypothesis cannot produce a false match -- it produces a *false audit
    trail*, which is worse in a way no match-rate metric can see. A batch that
    reconciles against someone else's refund reconciles, and is wrong.

    Asserted on amounts rather than ids because genuinely fungible rows (every
    dispute fee is a flat Rs 150) are interchangeable by design.
    """
    batch = generate(GeneratorConfig(n_orders=2500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    by_id = {a.adjustment_id: a for a in batch.sources.adjustments}

    checked = 0
    for m in result.matches_for_leg(2):
        if not m.hypothesised_ids:
            continue
        settlement_id = m.right_ids[0]
        # What the generator actually netted out of this batch; orphan ids
        # embed the settlement they belong to.
        truth_rows = sorted(
            a.amount_paise
            for a in batch.sources.adjustments
            if a.settlement_id is None and settlement_id in a.adjustment_id
        )
        claimed = sorted(by_id[aid].amount_paise for aid in m.hypothesised_ids)
        assert claimed == truth_rows, (
            f"{settlement_id}: explained with {claimed}, actually netted {truth_rows}"
        )
        checked += 1

    assert checked >= 5, "too few recovered matches to make this assertion mean anything"


def test_a_lying_solver_cannot_book_a_match(monkeypatch):
    """The gate must hold when the thing behind it is wrong.

    This is the invariant the whole architecture rests on, and it matters far
    more at B3 than it does here: the LLM proposes rows, and a plausible-looking
    proposal that does not actually close must never become a match. Simulated
    by making the solver confidently return a subset that does not add up.
    """
    from recoagent.legs import leg2_t1, ssmp

    def liar(values, target, tolerance=0, max_size=3, budget=250_000):
        # Confident, well-formed, and wrong: claims to close, does not.
        return ssmp.SearchResult(
            solutions=(ssmp.Subset((0,), target, target, (target,)),),
            exhaustive=True,
            combinations_examined=1,
            method="liar",
        )

    monkeypatch.setattr(leg2_t1.ssmp, "enumerate_closing_subsets", liar)

    batch = generate(GeneratorConfig(n_orders=1500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)

    for m in result.matches_for_leg(2):
        assert m.proof is not None and m.proof.closes, (
            f"{m.match_id} was booked on a solver claim the arithmetic rejects"
        )
    assert score(batch, result).overall_false_match_rate == 0.0


def test_leg1_is_untouched_by_leg2_recovery():
    """The two legs are independent; B2 adds nothing to Leg 1 and must not
    accidentally change it either."""
    _, b0, b2 = _both(DefectMix.dev())
    assert b0.legs[1].true_matches == b2.legs[1].true_matches
    assert b0.legs[1].exceptions == b2.legs[1].exceptions
