"""Money arithmetic. If these are wrong, every number downstream is wrong."""

from decimal import Decimal

import pytest

from recoagent.money import (
    GST_BPS,
    FeeSchedule,
    bps_of,
    format_inr,
    rupees_to_paise,
)


def test_bps_rounds_half_up():
    # 2% of 1050 paise = 21.0 exactly
    assert bps_of(1050, 200) == 21
    # 2% of 1025 paise = 20.5 -> 21 under half-up, 20 under banker's rounding
    assert bps_of(1025, 200) == 21
    assert bps_of(0, 200) == 0
    assert bps_of(999_999, 0) == 0


def test_rupees_to_paise_is_exact():
    assert rupees_to_paise("1234.56") == 123_456
    assert rupees_to_paise(1234) == 123_400
    assert rupees_to_paise(Decimal("0.01")) == 1
    # The classic float trap: 0.1 + 0.2 in binary floats is 0.30000000000000004
    assert rupees_to_paise("0.1") + rupees_to_paise("0.2") == rupees_to_paise("0.3")


def test_indian_digit_grouping():
    assert format_inr(0) == "Rs 0.00"
    assert format_inr(99) == "Rs 0.99"
    assert format_inr(100_000) == "Rs 1,000.00"
    # 12,34,567.89 -- lakh grouping, not 1,234,567.89
    assert format_inr(123_456_789) == "Rs 12,34,567.89"
    assert format_inr(-123_456_789) == "-Rs 12,34,567.89"


def test_upi_carries_no_mdr():
    """Not a placeholder: UPI P2M MDR is zero by regulation in India.

    A fee model that charges every method would report every UPI-heavy
    settlement as short.
    """
    fees = FeeSchedule.default()
    fee, tax = fees.fee_and_tax(1_000_00, "upi")
    assert (fee, tax) == (0, 0)
    assert fees.net_of(1_000_00, "upi") == 1_000_00


def test_card_fee_and_gst_on_fee():
    fees = FeeSchedule.default()
    gross = 10_000_00  # Rs 10,000
    fee, tax = fees.fee_and_tax(gross, "card_domestic")
    assert fee == 20_000  # 2% of Rs 10,000 = Rs 200
    assert tax == 3_600  # 18% GST on Rs 200 = Rs 36
    assert fees.net_of(gross, "card_domestic") == gross - fee - tax


def test_per_step_rounding_differs_from_single_step():
    """This is the mechanism behind the ROUNDING_DRIFT defect class.

    Rounding the fee, then rounding GST on the rounded fee, does not always
    equal rounding fee*(1+gst) once. Real settlement reports round per step,
    which is why the tolerance question cannot be waved away.
    """
    fees = FeeSchedule.default()
    disagreements = 0
    for gross in range(10_000, 10_400):
        fee, tax = fees.fee_and_tax(gross, "card_domestic")
        per_step = fee + tax
        single_step = bps_of(gross, 200 + (200 * GST_BPS) // 10_000)
        if per_step != single_step:
            disagreements += 1
    assert disagreements > 0


def test_unknown_method_is_loud():
    with pytest.raises(KeyError):
        FeeSchedule.default().mdr_for("carrier_billing")


def test_only_one_module_formats_rupees():
    """Every surface must print money the same way, and the Indian way.

    `journal/post.py` and `publish.py` each grew a private `_rupees` that used
    Western digit grouping, so the ledger reported Rs 12,082,700.56 where every
    other screen in an Indian merchant's product said Rs 1,20,82,700.56. The
    whole suite passed while that was true, which is why this is a structural
    check rather than a value one: the failure is a second formatter existing
    at all, and it is invisible to any test that only reads the first.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "recoagent"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "money.py":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            # Dividing paise by 100 into a format string is the shape of a
            # money formatter, and it is also a float touching money.
            if "/ 100" in line and ":," in line:
                offenders.append(f"{path.relative_to(root.parent)}:{lineno}")
    assert not offenders, (
        "these format rupees themselves instead of calling money.format_inr: "
        + ", ".join(offenders)
    )
