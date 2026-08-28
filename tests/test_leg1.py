"""Leg 1, and the one difference it is allowed to close.

A partial capture is the only Leg 1 disagreement the book itself explains: the
gateway captured less than the order authorised and says so on the row. Tier 1
accepts that explanation, and the tests here are almost entirely about the
cases where it must not.

The rule under attack: a `partially_captured` label is a claim, not a proof.
What earns the match is that the fee and tax re-derive from the captured gross
at a rate the merchant has on file. Strip that check and the status field alone
buys a match, which is the same failure the agent tier's citation contract
exists to prevent -- one leg down.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from recoagent.defects import DefectClass
from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.legs import leg1
from recoagent.money import FeeSchedule
from recoagent.pipeline import run_b0, run_b2
from recoagent.schemas import ReconResult
from recoagent.validate import Tolerance


@pytest.fixture(scope="module")
def book():
    batch = generate(GeneratorConfig(n_orders=1200, seed=7, mix=DefectMix.dev()))
    return batch


def _rerun(sources):
    result = ReconResult(rung="B2")
    leg1.match(sources, Tolerance.calibrated(), result, with_t1=True)
    return result


def _a_partial_capture(sources):
    """One under-captured payment and its order, from a real generated book."""
    payment = next(p for p in sources.payments if p.status == "partially_captured")
    order = next(o for o in sources.orders if o.order_id == payment.order_id)
    return order, payment


def _replace_payment(sources, payment, **kw):
    swapped = replace(payment, **kw)
    return replace(
        sources,
        payments=tuple(swapped if p is payment else p for p in sources.payments),
    )


def _matched(result, order_id):
    return any(m.left_ids[0] == order_id for m in result.matches_for_leg(1))


# ── what it closes ───────────────────────────────────────────────────────


def test_a_declared_capture_closes_and_keeps_its_variance(book):
    order, payment = _a_partial_capture(book.sources)
    result = _rerun(book.sources)

    match = next(m for m in result.matches_for_leg(1) if m.left_ids[0] == order.order_id)
    assert match.tier == "T1"
    assert match.rule_id == leg1.RULE_PARTIAL_CAPTURE
    assert match.proof is not None and match.proof.closes
    assert match.variance_paise == payment.gross_paise - order.amount_paise < 0


def test_b0_still_refuses_it(book):
    """The ladder has to keep meaning something.

    If the baseline rung quietly gained this too, the lift would be unmeasurable
    and the comparison between rungs would stop being a comparison.
    """
    order, _ = _a_partial_capture(book.sources)
    assert not _matched(run_b0(book.sources), order.order_id)
    assert _matched(run_b2(book.sources), order.order_id)


# ── what it refuses ──────────────────────────────────────────────────────


def test_a_status_field_alone_cannot_buy_a_match(book):
    """The attack. Claim a short capture, keep fees that no rate produces.

    This is a book whose own numbers disagree with its own rate card -- a worse
    problem than a short capture, and one no label should be able to talk its
    way past.
    """
    order, payment = _a_partial_capture(book.sources)
    sources = _replace_payment(book.sources, payment, fee_paise=payment.fee_paise + 500)

    assert not _matched(_rerun(sources), order.order_id)


def test_capturing_more_than_was_authorised_is_refused(book):
    order, payment = _a_partial_capture(book.sources)
    fees = FeeSchedule.default()
    over = order.amount_paise + 100_00
    fee, tax = fees.fee_and_tax(over, payment.method)
    # Internally consistent, and still not a partial capture in any direction.
    sources = _replace_payment(
        book.sources, payment, gross_paise=over, fee_paise=fee, tax_paise=tax
    )

    assert not _matched(_rerun(sources), order.order_id)


def test_an_undeclared_shortfall_is_still_an_exception(book):
    """Same money, no declaration. That is an unexplained difference."""
    order, payment = _a_partial_capture(book.sources)
    sources = _replace_payment(book.sources, payment, status="captured")

    result = _rerun(sources)
    assert not _matched(result, order.order_id)
    exc = next(e for e in result.exceptions_for_leg(1) if e.entity_id == order.order_id)
    assert exc.residual_paise is not None


def test_ambiguity_still_outranks_the_new_tier(book):
    """Two payments claiming one order is refused whatever their statuses say.

    Tier 1 sits behind the ambiguity check, not in front of it: an under-capture
    on one of two rival rows would otherwise let the tier pick a winner.
    """
    order, payment = _a_partial_capture(book.sources)
    rival = replace(payment, payment_id=payment.payment_id + "_retry")
    sources = replace(book.sources, payments=book.sources.payments + (rival,))

    result = _rerun(sources)
    assert not _matched(result, order.order_id)
    exc = next(e for e in result.exceptions_for_leg(1) if e.entity_id == order.order_id)
    assert exc.suspected_class is DefectClass.DUPLICATE_PAYMENT


# ── what it does to the published numbers ────────────────────────────────


def test_the_whole_class_closes_and_nothing_else_moves(book):
    """Every partial capture, no false matches, and Leg 2 untouched."""
    b0, b2 = score(book, run_b0(book.sources)), score(book, run_b2(book.sources))

    partial = next(a for a in b2.accounting if a.defect is DefectClass.PARTIAL_CAPTURE)
    assert partial.flagged == 0 and partial.mishandled == 0
    assert partial.resolved == partial.injected > 0

    assert b2.legs[1].false_matches == 0
    assert b2.legs[1].true_matches - b0.legs[1].true_matches == partial.injected
    assert b2.overall_false_match_rate == 0.0
