"""The tier that reads the merchant's paperwork.

What is under test is mostly what it refuses. Applying a rate that a document
states is arithmetic; the judgement is in deciding which document applies, and
in stopping when the file contradicts itself.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from recoagent.defects import DefectClass
from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.legs.repricing import _repriced_net, corrections, rate_book
from recoagent.money import GST_BPS, FeeSchedule, bps_of
from recoagent.pipeline import run_b2
from recoagent.schemas import FxAdvice, RateNotice

WHEN = datetime(2026, 7, 15, 12, 0)


@pytest.fixture(scope="module")
def book():
    batch = generate(GeneratorConfig(n_orders=2000, seed=7, mix=DefectMix.dev()))
    return batch, run_b2(batch.sources)


def _notice(method="card_domestic", bps=240, frm=WHEN - timedelta(days=5), to=None, nid="n1"):
    return RateNotice(nid, method, bps, frm, to, "test")


# ── which notice applies ─────────────────────────────────────────────────


def test_a_notice_covers_its_window_and_nothing_else():
    n = _notice(frm=datetime(2026, 7, 1), to=datetime(2026, 8, 1))
    assert n.covers(datetime(2026, 7, 15), "card_domestic")
    assert n.covers(datetime(2026, 7, 1), "card_domestic")       # inclusive
    assert not n.covers(datetime(2026, 8, 1), "card_domestic")   # exclusive
    assert not n.covers(datetime(2026, 6, 30), "card_domestic")
    assert not n.covers(datetime(2026, 7, 15), "netbanking")


def test_consecutive_windows_never_both_apply():
    """`effective_to` is exclusive precisely so this cannot happen."""
    a = _notice(bps=200, frm=datetime(2026, 7, 1), to=datetime(2026, 7, 15), nid="a")
    b = _notice(bps=240, frm=datetime(2026, 7, 15), to=None, nid="b")
    boundary = datetime(2026, 7, 15)
    assert [n.notice_id for n in (a, b) if n.covers(boundary, "card_domestic")] == ["b"]


def test_two_notices_in_force_is_refused_not_broken(book):
    """A contradiction in the merchant's own file. There is no right answer to pick."""
    batch, _ = book
    settlement = batch.sources.settlements[0]
    method = batch.sources.payments_by_settlement(settlement.settlement_id)[0].method
    clashing = (
        _notice(method, 240, settlement.settled_at - timedelta(days=1), None, "n1"),
        _notice(method, 275, settlement.settled_at - timedelta(days=1), None, "n2"),
    )
    sources = replace(batch.sources, rate_notices=clashing)
    out = corrections(sources, settlement)
    assert out.refusal is not None and "in force" in out.refusal
    assert not out.applies
    assert out.nets == {}


def test_no_paperwork_is_not_a_failure(book):
    """Most batches have nothing to apply, and that is not an error."""
    batch, _ = book
    sources = replace(batch.sources, rate_notices=(), fx_advices=())
    out = corrections(sources, batch.sources.settlements[0])
    assert out.refusal is None and not out.applies


def test_a_notice_for_another_method_changes_nothing(book):
    batch, _ = book
    settlement = batch.sources.settlements[0]
    methods = {p.method for p in batch.sources.payments_by_settlement(settlement.settlement_id)}
    absent = next(m for m in ("wallet", "netbanking", "upi") if m not in methods)
    sources = replace(batch.sources, rate_notices=(
        _notice(absent, 900, settlement.settled_at - timedelta(days=1)),
    ))
    assert not corrections(sources, settlement).applies


# ── the arithmetic ───────────────────────────────────────────────────────


def test_the_repriced_net_matches_a_per_step_rounded_fee(book):
    """Fees round per step: MDR first, then GST on the rounded MDR."""
    batch, _ = book
    payment = next(
        p for p in batch.sources.payments if p.method == "card_domestic" and p.gross_paise > 10_000
    )
    fee = bps_of(payment.gross_paise, 240)
    tax = bps_of(fee, GST_BPS)
    expected = payment.net_paise + (payment.fee_paise + payment.tax_paise) - (fee + tax)
    assert _repriced_net(payment, 240) == expected
    assert isinstance(_repriced_net(payment, 240), int)


def test_a_notice_quoting_the_scheduled_rate_moves_nothing(book):
    """No change is no correction, so nothing is cited and nothing is claimed."""
    batch, _ = book
    settlement = batch.sources.settlements[0]
    fees = FeeSchedule.default()
    method = batch.sources.payments_by_settlement(settlement.settlement_id)[0].method
    sources = replace(batch.sources, rate_notices=(
        _notice(method, fees.mdr_for(method), settlement.settled_at - timedelta(days=1)),
    ), fx_advices=())
    out = corrections(sources, settlement)
    assert out.nets == {} and out.cited == ()


def test_a_zero_slip_advice_is_ignored(book):
    """The drawer holds advices for payments that converted exactly as reported."""
    batch, _ = book
    settlement = batch.sources.settlements[0]
    payment = batch.sources.payments_by_settlement(settlement.settlement_id)[0]
    sources = replace(batch.sources, rate_notices=(), fx_advices=(
        FxAdvice("adv", payment.payment_id, 0.0, settlement.settled_at, "no slip"),
    ))
    assert not corrections(sources, settlement).applies


def test_two_advices_for_one_payment_is_a_refusal_to_choose(book):
    """Contradictory advice is not resolved by taking the first one."""
    batch, _ = book
    payment = batch.sources.payments[0]
    sources = replace(batch.sources, fx_advices=(
        FxAdvice("a1", payment.payment_id, 1.0, WHEN, ""),
        FxAdvice("a2", payment.payment_id, 2.0, WHEN, ""),
    ))
    assert sources.fx_advice_for(payment.payment_id) is None


# ── end to end ───────────────────────────────────────────────────────────


def test_the_paperwork_closes_every_fee_and_fx_defect(book):
    """The point of the tier, stated as a measurement."""
    batch, result = book
    by_class = {a.defect: a for a in score(batch, result).accounting}
    for cls in (DefectClass.FEE_TAX_VARIANCE, DefectClass.FX_CONVERSION):
        acc = by_class[cls]
        assert acc.injected > 0, f"{cls} was never injected; the test proves nothing"
        assert acc.resolved == acc.injected, acc
        assert acc.mishandled == 0, acc


def test_without_the_paperwork_the_same_book_cannot_close_them(book):
    """The control. If these closed anyway, the notices would be decorative."""
    batch, _ = book
    stripped = replace(batch, sources=replace(
        batch.sources, rate_notices=(), fx_advices=()
    ))
    card = score(stripped, run_b2(stripped.sources))
    by_class = {a.defect: a for a in card.accounting}
    assert by_class[DefectClass.FEE_TAX_VARIANCE].resolved == 0
    assert by_class[DefectClass.FX_CONVERSION].resolved == 0
    # And still nothing wrong is booked -- it refuses rather than guesses.
    assert card.overall_false_match_rate == 0.0
    assert card.mishandled_total == 0


def test_every_notice_match_carries_a_proof(book):
    _, result = book
    for m in result.matches_for_leg(2):
        if m.rule_id == "leg2.t1.rate_notice":
            assert m.proof is not None and m.proof.closes
    assert any(m.rule_id == "leg2.t1.rate_notice" for m in result.matches_for_leg(2))


def test_the_decoys_outnumber_the_live_notices(book):
    """A drawer with one circular in it tests retrieval of the only row present."""
    batch, _ = book
    notices = batch.sources.rate_notices
    live = [n for n in notices if n.effective_to is None]
    assert len(notices) >= 3 * len(live), (notices, live)
    assert len({n.method for n in live}) == len(live), "two live notices on one method"


def test_the_rate_book_only_carries_what_is_in_force(book):
    batch, _ = book
    live = [n for n in batch.sources.rate_notices if n.effective_to is None]
    assert live, "no live notice in this book; the test proves nothing"
    when = max(s.settled_at for s in batch.sources.settlements)
    rb = rate_book(batch.sources, when)
    for notice in live:
        assert rb.confirms_mdr(notice.method, notice.mdr_bps)
    expired = [n for n in batch.sources.rate_notices if n.effective_to is not None]
    for notice in expired:
        if not any(
            n.method == notice.method and n.mdr_bps == notice.mdr_bps for n in live
        ):
            assert not rb.confirms_mdr(notice.method, notice.mdr_bps)
