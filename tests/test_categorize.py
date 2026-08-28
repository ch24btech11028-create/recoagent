"""The categorisation ladder, and the attacks the model tier has to survive."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from recoagent.categorize import rules, score
from recoagent.categorize.agent import (
    ChatCategoriser,
    Proposal,
    parse,
    row_text,
    run_c2,
)
from recoagent.categorize.taxonomy import PROPOSABLE, Category
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.llm import ScriptedChat
from recoagent.pipeline import run_b2
from recoagent.schemas import BankLine, Order, PGAdjustment, PGPayment, Settlement, SourceBundle

WHEN = datetime(2026, 8, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def batch():
    return generate(GeneratorConfig(n_orders=500, seed=7, mix=DefectMix.dev()))


@pytest.fixture(scope="module")
def c1(batch):
    return rules.run_c1(batch.sources, run_b2(batch.sources))


# ── the ladder, and what each rung is actually worth ────────────────────────


def test_source_fields_alone_categorise_almost_nothing(batch):
    """C0 is the honest floor for "we categorise transactions".

    Pinned low on purpose. If this number ever climbs, the rung has started
    inferring rather than reading, and the C1 and C2 lifts stop meaning what
    they say.
    """
    card = score.score(rules.run_c0(batch.sources), batch.truth.categories, "C0")
    assert card.assigned == 8
    assert card.coverage < 0.01
    assert card.wrong == 0


def test_the_reconciliation_categorises_the_book(c1, batch):
    """C1 is the finding: matching already decided nearly everything, with proofs."""
    card = score.score(c1, batch.truth.categories, "C1")
    assert card.assigned == 885
    assert card.wrong == 0
    assert card.wrong_rate == 0.0
    assert 0.94 < card.coverage < 0.96


def test_what_is_left_for_a_model_is_small_and_named(c1, batch):
    left = rules.residue(batch.sources, c1)
    assert len(left) == 20
    kinds = {kind for _, kind, _ in left}
    assert kinds == {"payment", "bank_line"}


def test_every_assignment_carries_evidence(c1):
    assert c1.assignments
    for assignment in c1.assignments.values():
        assert assignment.evidence.strip(), assignment


# ── the two mistakes the taxonomy exists to prevent ─────────────────────────


def test_a_settlement_credit_is_never_revenue(c1, batch):
    """The double-counting guard, checked on the real book.

    Every bank credit that closed against a batch must be a transfer. One of
    them booked as revenue would inflate the merchant's declared turnover by
    the whole value of the settlement.
    """
    credits = [
        a for a in c1.assignments.values() if a.entity_kind == "bank_line"
    ]
    assert credits
    assert all(a.category is Category.SETTLEMENT_CREDIT for a in credits)

    revenue = {a.entity_id for a in c1.assignments.values()
               if a.category is Category.SALES_REVENUE}
    bank_ids = {b.bank_line_id for b in batch.sources.bank_lines}
    assert not (revenue & bank_ids)


def test_gst_on_the_mdr_is_split_out_of_the_fee(c1, batch):
    """Filed as an expense it would be an unclaimed input credit, silently."""
    paid = next(
        p for p in batch.sources.payments
        if p.tax_paise and c1.has(f"{p.payment_id}:tax")
    )
    fee = c1.assignments[f"{paid.payment_id}:fee"]
    tax = c1.assignments[f"{paid.payment_id}:tax"]

    assert fee.category is Category.GATEWAY_FEE
    assert tax.category is Category.GST_INPUT_CREDIT
    assert fee.amount_paise == -paid.fee_paise
    assert tax.amount_paise == -paid.tax_paise


def test_an_earlier_rung_is_never_overruled_by_a_later_one(batch):
    ledger = rules.run_c0(batch.sources)
    before = dict(ledger.assignments)
    rules.run_c1(batch.sources, run_b2(batch.sources), ledger)
    for entity_id, assignment in before.items():
        assert ledger.assignments[entity_id] == assignment


# ── the model tier ──────────────────────────────────────────────────────────


def small_book() -> SourceBundle:
    """One unmatched bank credit and one unlinked adjustment. Nothing else."""
    return SourceBundle(
        orders=(Order("order_1", "c", "INV-1", 100000, "INR", WHEN),),
        payments=(),
        adjustments=(PGAdjustment("adj_1", None, "mystery", None, -5000, WHEN),),
        settlements=(Settlement("setl_1", "UTR1", WHEN, 100000, "processed"),),
        bank_lines=(
            BankLine("bl_1", date(2026, 8, 1), 250000,
                     "NEFT CR-ACME SUPPLIES PVT LTD-VENDOR INVOICE 8841", "REF9"),
        ),
    )


def reply(category: str, quote: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {"category": category, "quote": quote, "confidence": confidence, "reason": "because"}
    )


def categorise(book: SourceBundle, replies) -> tuple[rules.Ledger, object]:
    ledger = rules.Ledger()
    rules.run_c0(book, ledger)
    report = run_c2(book, ledger, ChatCategoriser(ScriptedChat(replies)))
    return ledger, report


def test_a_model_cannot_invent_the_evidence():
    """The attack this tier is built around.

    The model returns a perfectly plausible category, and a quotation that
    appears nowhere in the row. Accepting it would mean the book is being
    categorised on a narration the model imagined.
    """
    book = small_book()
    ledger, report = categorise(
        book,
        [reply("vendor_payment", "SALARY DISBURSEMENT PAYROLL AUGUST"),
         reply("bank_charge", "quarterly maintenance fee")],
    )

    assert report.uncited == 2
    assert report.assigned == 0
    assert ledger.assignments["bl_1"].category is Category.NEEDS_REVIEW
    assert "absent from the row" in ledger.assignments["bl_1"].evidence
    assert ledger.assignments["bl_1"].verified is False


def test_a_real_quotation_is_accepted():
    book = small_book()
    ledger, report = categorise(
        book,
        # Residue order is payments, adjustments, bank lines: adj_1 is asked
        # about first.
        [reply("bank_charge", "kind: mystery"),
         reply("vendor_payment", "ACME SUPPLIES PVT LTD")],
    )

    assert report.assigned == 2
    assert report.uncited == 0
    assert ledger.assignments["bl_1"].category is Category.VENDOR_PAYMENT
    assert "ACME SUPPLIES" in ledger.assignments["bl_1"].evidence
    assert ledger.assignments["bl_1"].confidence == 0.9


def test_the_model_may_not_reach_for_a_category_the_arithmetic_owns():
    """sales_revenue comes out of a proof. A model agreeing adds nothing;
    a model disagreeing would be overruled, so it is not on the menu."""
    assert Category.SALES_REVENUE not in PROPOSABLE
    assert Category.GST_INPUT_CREDIT not in PROPOSABLE

    outcome = parse(reply("sales_revenue", "NEFT CR-ACME"))
    assert isinstance(outcome, str)
    assert "determined by the reconciliation" in outcome


def test_declining_is_a_correct_answer_not_a_failure():
    book = small_book()
    ledger, report = categorise(
        book, [reply("needs_review", "", 0.0), reply("needs_review", "", 0.0)]
    )
    assert report.declined == 2
    assert report.failed == 0
    assert ledger.assignments["bl_1"].category is Category.NEEDS_REVIEW


def test_a_correct_citation_below_the_floor_still_goes_to_a_human():
    book = small_book()
    ledger, _ = categorise(
        book,
        [reply("bank_charge", "kind: mystery", confidence=0.2),
         reply("vendor_payment", "ACME SUPPLIES PVT LTD", confidence=0.2)],
    )
    assert ledger.assignments["bl_1"].category is Category.NEEDS_REVIEW
    assert "below the floor" in ledger.assignments["bl_1"].evidence


def test_a_broken_reply_lands_in_review_rather_than_nowhere():
    book = small_book()
    ledger, report = categorise(book, ["not json at all", "{oops"])
    assert report.failed == 2
    assert report.assigned == 0
    # The row must still appear. A row the model failed on and that then
    # vanishes is worse than one it got wrong.
    assert ledger.assignments["bl_1"].category is Category.NEEDS_REVIEW


def test_no_row_is_ever_silently_dropped():
    book = small_book()
    before = rules.residue(book, rules.run_c0(book))
    ledger, _ = categorise(book, ["garbage", "garbage"])
    for entity_id, _kind, _amount in before:
        assert ledger.has(entity_id)


def test_the_row_text_is_the_only_thing_the_citation_can_match():
    book = small_book()
    text = row_text(book, "bl_1", "bank_line")
    assert "ACME SUPPLIES" in text
    assert "250000" not in text  # the amount is not evidence of a category


def test_parse_reads_a_fenced_reply():
    got = parse('```json\n{"category":"bank_charge","quote":"x","confidence":0.7}\n```')
    assert isinstance(got, Proposal)
    assert got.category is Category.BANK_CHARGE
    assert got.confidence == 0.7


def test_confidence_outside_the_range_is_clamped_not_trusted():
    got = parse(reply("bank_charge", "x", confidence=7.5))
    assert isinstance(got, Proposal) and got.confidence == 1.0


# ── the scorer ──────────────────────────────────────────────────────────────


def test_wrong_rate_is_measured_against_what_was_assigned():
    """A row sent for review is not a wrong answer, and not a right one."""
    ledger = rules.Ledger()
    for i, (category, truth) in enumerate([
        (Category.VENDOR_PAYMENT, "vendor_payment"),
        (Category.VENDOR_PAYMENT, "bank_charge"),
        (Category.NEEDS_REVIEW, "bank_charge"),
    ]):
        ledger.add(rules.Assignment(f"e{i}", "bank_line", category, 0, "C2", "r", "e"))

    card = score.score(
        ledger, {"e0": "vendor_payment", "e1": "bank_charge", "e2": "bank_charge"}, "C2"
    )
    assert card.assigned == 2
    assert card.reviewed == 1
    assert card.wrong == 1
    assert card.wrong_rate == 0.5
    assert card.confusions[0].predicted is Category.VENDOR_PAYMENT
    assert card.confusions[0].actual is Category.BANK_CHARGE


def test_a_row_the_answer_key_does_not_cover_is_scored_neither_way():
    ledger = rules.Ledger()
    ledger.add(rules.Assignment("e0", "bank_line", Category.BANK_CHARGE, 0, "C2", "r", "e"))
    card = score.score(ledger, {"other": "refund"}, "C2")
    assert card.unlabelled == 1
    assert card.correct == 0 and card.wrong == 0


def test_render_names_the_lead_metric_first(c1, batch):
    text = score.render(score.score(c1, batch.truth.categories, "C1"))
    assert text.index("Wrong-category rate") < text.index("Coverage")
