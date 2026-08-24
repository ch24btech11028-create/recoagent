"""The gate. An LLM proposal that does not close must be rejected, always."""

from datetime import date, datetime

from recoagent.schemas import BankLine, Order, PGAdjustment, PGPayment, Settlement
from recoagent.validate import Tolerance, header_agrees, prove_leg1, prove_leg2

NOW = datetime(2026, 7, 1)


def _payment(pid: str, gross: int, fee: int = 0, tax: int = 0) -> PGPayment:
    return PGPayment(
        payment_id=pid,
        order_id=f"o_{pid}",
        gross_paise=gross,
        fee_paise=fee,
        tax_paise=tax,
        method="upi",
        status="captured",
        settlement_id="setl_0",
        captured_at=NOW,
    )


def _line(amount: int) -> BankLine:
    return BankLine(
        bank_line_id="bank_0",
        value_date=date(2026, 7, 3),
        amount_paise=amount,
        narration="IMPS/123456789012/RAZORPAY/SETTLEMENT",
        bank_ref="REF1",
    )


SETTLEMENT = Settlement(
    settlement_id="setl_0",
    utr="123456789012",
    settled_at=NOW,
    net_paise=300_00,
    status="processed",
)


def test_exact_batch_closes():
    payments = [_payment("p1", 100_00), _payment("p2", 200_00)]
    proof = prove_leg2(_line(300_00), SETTLEMENT, payments, [], Tolerance.strict())
    assert proof.closes
    assert proof.residual_paise == 0


def test_short_credit_does_not_close():
    payments = [_payment("p1", 100_00), _payment("p2", 200_00)]
    proof = prove_leg2(_line(250_00), SETTLEMENT, payments, [], Tolerance.strict())
    assert not proof.closes
    assert proof.residual_paise == -50_00


def test_tolerance_absorbs_drift_but_not_more():
    payments = [_payment("p1", 300_00)]
    tol = Tolerance(leg2_paise=5)
    assert prove_leg2(_line(300_00 - 5), SETTLEMENT, payments, [], tol).closes
    assert not prove_leg2(_line(300_00 - 6), SETTLEMENT, payments, [], tol).closes


def test_correct_hypothesis_closes_the_gap():
    """The shape of the LLM tier's only privilege: propose rows, never write matches."""
    payments = [_payment("p1", 100_00), _payment("p2", 200_00)]
    refund = PGAdjustment(
        adjustment_id="adj_x",
        settlement_id=None,
        kind="refund",
        payment_id="p1",
        amount_paise=-50_00,
        booked_at=NOW,
    )
    line = _line(250_00)
    assert not prove_leg2(line, SETTLEMENT, payments, [], Tolerance.strict()).closes
    proof = prove_leg2(
        line, SETTLEMENT, payments, [], Tolerance.strict(), hypothesised=[refund]
    )
    assert proof.closes


def test_wrong_hypothesis_is_rejected():
    payments = [_payment("p1", 100_00), _payment("p2", 200_00)]
    wrong = PGAdjustment(
        adjustment_id="adj_wrong",
        settlement_id=None,
        kind="refund",
        payment_id="p1",
        amount_paise=-40_00,  # plausible, but not the actual gap
        booked_at=NOW,
    )
    proof = prove_leg2(
        _line(250_00), SETTLEMENT, payments, [], Tolerance.strict(), hypothesised=[wrong]
    )
    assert not proof.closes, "a near-miss hypothesis must not be accepted"


def test_header_is_corroboration_not_proof():
    """A gateway header that agrees with its own rows says nothing about the bank."""
    payments = [_payment("p1", 300_00)]
    assert header_agrees(SETTLEMENT, payments, [])
    proof = prove_leg2(_line(250_00), SETTLEMENT, payments, [], Tolerance.strict())
    assert not proof.closes


def test_leg1_amount_proof():
    order = Order(
        order_id="o_p1",
        customer_id="c1",
        invoice_no="INV-1",
        amount_paise=100_00,
        currency="INR",
        created_at=NOW,
    )
    assert prove_leg1(order, _payment("p1", 100_00), Tolerance.strict()).closes
    assert not prove_leg1(order, _payment("p1", 60_00), Tolerance.strict()).closes
