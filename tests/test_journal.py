"""The books have to balance, and the clearing account has to be explainable.

A category scorecard grades labels. It cannot see an amount that is wrong, and
all three bugs the journal found were exactly that shape -- right label, wrong
number. So these tests check the two things a label cannot fake: that every
posting balances, and that the money left in the gateway receivable is
attributable to a named cause with nothing left over.
"""

import pytest

from recoagent.categorize.rules import run_c1
from recoagent.categorize.taxonomy import Category
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.journal.accounts import (
    ACCOUNT_TYPES,
    EXPECTED_SIGN,
    NOT_POSTED,
    POSTING_RULES,
    Account,
)
from recoagent.journal.post import (
    CAUSE_ORDER,
    ROUNDING,
    ROUNDING_CEILING_PAISE,
    explain_receivable,
    post,
    render,
)
from recoagent.pipeline import run_b2

PROFILES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout),
            "clean": (7, DefectMix.clean)}


def build(profile: str, n: int = 2000):
    seed, mix = PROFILES[profile]
    batch = generate(GeneratorConfig(n_orders=n, seed=seed, mix=mix()))
    result = run_b2(batch.sources)
    ledger = run_c1(batch.sources, result)
    return batch, result, post(ledger, batch.sources, result)


@pytest.fixture(scope="module", params=sorted(PROFILES))
def booked(request):
    return (request.param, *build(request.param))


def test_every_entry_balances(booked):
    _, _, _, journal = booked
    assert journal.entries
    assert journal.unbalanced_entries == []


def test_the_trial_balance_balances(booked):
    profile, _, _, journal = booked
    assert journal.total_debits == journal.total_credits, (
        f"{profile}: out by {journal.total_debits - journal.total_credits} paise"
    )


def test_the_receivable_is_fully_attributed(booked):
    """The headline claim. Not that the clearing account empties -- that every
    rupee left in it has a name against it."""
    profile, batch, result, journal = booked
    open_batches = explain_receivable(journal, batch.sources, result)
    attributed = sum(b.balance_paise for b in open_batches)
    assert attributed == journal.balance_of(Account.GATEWAY_RECEIVABLE), (
        f"{profile}: {journal.balance_of(Account.GATEWAY_RECEIVABLE) - attributed} "
        f"paise in the clearing account belongs to no named cause"
    )
    assert all(b.cause in CAUSE_ORDER for b in open_batches)


def test_nothing_real_hides_in_the_rounding_bucket(booked):
    """The one assertion that can actually fail.

    Every other cause is read off the matcher's rule id and is therefore true
    by construction. `ROUNDING` is the leftover -- what no other cause claimed
    -- so a misfiled cause or a genuine gap lands here, and sub-rupee drift
    between a gateway and a bank is paise.
    """
    profile, batch, result, journal = booked
    rounding = [
        b for b in explain_receivable(journal, batch.sources, result)
        if b.cause is ROUNDING
    ]
    too_big = [b for b in rounding if abs(b.balance_paise) > ROUNDING_CEILING_PAISE]
    assert not too_big, (
        f"{profile}: filed as rounding but not sub-rupee: "
        + ", ".join(f"{b.batch_id} {b.balance_paise} paise" for b in too_big)
    )


def test_a_clean_book_leaves_no_open_receivable():
    """The control. With no defects injected, every batch clears to zero.

    This is what makes the dev-profile balances meaningful: they are caused by
    the defects, not by the posting rules being wrong.
    """
    batch, result, journal = build("clean")
    open_batches = explain_receivable(journal, batch.sources, result)
    assert open_batches == [], [
        (b.batch_id, b.balance_paise, b.cause) for b in open_batches
    ]
    assert journal.balance_of(Account.GATEWAY_RECEIVABLE) == 0


def test_a_settlement_credit_is_booked_at_what_the_bank_actually_sent():
    """The first bug the clearing account found.

    C1 booked the gateway's declared `net_paise`. A defect that moves the
    credit -- a cutoff spill, say -- leaves the gateway's row stating the
    original figure while the bank sends something else, so the bank account
    was being debited with money that never arrived. Invisible on a
    category-only scorecard: the label was right the whole time.
    """
    batch, result, journal = build("dev")
    lines = {b.bank_line_id: b for b in batch.sources.bank_lines}
    credits = [
        e for e in journal.entries if e.category is Category.SETTLEMENT_CREDIT
    ]
    assert credits
    for entry in credits:
        assert entry.amount_paise == lines[entry.entity_id].amount_paise


def test_a_row_whose_sign_contradicts_its_category_posts_the_reverse_pair():
    """The third bug. A positive amount on a refund is money arriving, and
    posting it at magnitude in the refund direction moves the receivable the
    wrong way by twice its value -- while the trial balance still balances,
    because a wrong direction is wrong on both sides at once."""
    _, _, journal = build("dev")
    for anomaly in journal.anomalies:
        entry = next(e for e in journal.entries if e.entity_id == anomaly.entity_id)
        expected_debit, expected_credit = POSTING_RULES[anomaly.category]
        assert entry.postings[0].account is expected_credit
        assert entry.postings[1].account is expected_debit


def test_a_settlement_credit_never_touches_income():
    """The structural error the taxonomy exists to prevent, checked where it
    would actually cost money. Booking a payout as revenue doubles the declared
    turnover -- the revenue was recognised when the customer paid."""
    _, _, journal = build("dev")
    for entry in journal.entries:
        if entry.category is Category.SETTLEMENT_CREDIT:
            accounts = {p.account for p in entry.postings}
            assert Account.SALES_REVENUE not in accounts
            assert accounts == {Account.BANK, Account.GATEWAY_RECEIVABLE}


def test_gst_is_an_asset_and_not_an_expense():
    """Filed as a cost it overstates expenses and understates the credit
    claimable in GSTR-3B, which is a filing error rather than a presentation
    one."""
    from recoagent.journal.accounts import AccountType

    _, _, journal = build("dev")
    posted = [
        e for e in journal.entries if e.category is Category.GST_INPUT_CREDIT
    ]
    assert posted
    for entry in posted:
        debit = entry.postings[0].account
        assert debit is Account.GST_INPUT_CREDIT
        assert ACCOUNT_TYPES[debit] is AccountType.ASSET


def test_nothing_a_model_proposed_reaches_the_ledger():
    """C2 marks every assignment unverified by design. A suggestion in the
    operator queue is useful; a suggestion in the general ledger is a
    misstatement."""
    from recoagent.categorize.rules import Assignment, Ledger

    batch, result, _ = build("dev")
    ledger = Ledger()
    ledger.add(Assignment(
        entity_id="pay_fake", entity_kind="payment",
        category=Category.SALES_REVENUE, amount_paise=10_000,
        rung="C2", rule_id="c2.model", evidence="model said so",
        confidence=0.99, verified=False,
    ))
    journal = post(ledger, batch.sources, result)
    assert journal.entries == []
    assert any("not booked" in why for _, _, why in journal.unposted)


def test_every_category_either_posts_or_says_why_not():
    """A category with no posting rule and no exemption would silently vanish
    from the books, which is the one failure mode a balanced trial balance
    cannot reveal."""
    for category in Category:
        assert category in POSTING_RULES or category in NOT_POSTED, category
    for category, (debit, credit) in POSTING_RULES.items():
        assert debit is not credit, f"{category} posts to one account twice"
        assert debit in ACCOUNT_TYPES and credit in ACCOUNT_TYPES


def test_expected_signs_cover_every_posting_category_that_can_be_signed():
    signed = set(POSTING_RULES) - {Category.NEEDS_REVIEW}
    assert signed <= set(EXPECTED_SIGN), signed - set(EXPECTED_SIGN)


def test_the_report_renders_for_every_profile(booked):
    profile, batch, result, journal = booked
    text = render(journal, batch.sources, result)
    assert "TRIAL BALANCE" in text
    assert "BALANCED" in text, f"{profile} did not balance"
    # The clean profile clears every batch, so it has no attribution table to
    # print -- which is the control working, not a missing section.
    if explain_receivable(journal, batch.sources, result):
        assert "unattributed" in text
    else:
        assert "Every batch cleared to zero." in text
