"""The category set, and the two mistakes it exists to prevent.

A finance controller that assigns categories is a bookkeeping tool, and
bookkeeping has exactly two ways to be catastrophically wrong while looking
tidy. Both are structural, and both are visible in this enum.

**Counting the same rupee as revenue twice.** A customer pays; the gateway
later credits a batch containing that payment into the bank. Two rows, one
sale. Category `SETTLEMENT_CREDIT` exists so the bank credit is booked as a
*transfer between accounts you already own*, never as income. A tool that
calls both "Sales Revenue" doubles the merchant's declared turnover, and the
error is invisible on any dashboard that only shows a total going up.

**Booking a recoverable tax as an expense.** GST charged on the MDR is input
tax credit: the merchant claims it back. Filed as `GATEWAY_FEE` it silently
overstates costs and understates the credit claimable in GSTR-3B, which is a
filing error, not a presentation one. `GST_INPUT_CREDIT` is separate for that
reason alone.

Everything else here is ordinary. These two are why the taxonomy is not just
whatever categories a model happens to emit.
"""

from __future__ import annotations

from enum import Enum


class Category(str, Enum):
    #: A captured payment against a customer order. The sale itself.
    SALES_REVENUE = "sales_revenue"
    #: The gateway's MDR, exclusive of tax. An expense.
    GATEWAY_FEE = "gateway_fee"
    #: GST charged on the MDR. Recoverable, so not an expense.
    GST_INPUT_CREDIT = "gst_input_credit"
    #: Money returned to a customer. Contra-revenue, not a cost.
    REFUND = "refund"
    #: A disputed transaction clawed back, and the fee for handling it.
    CHARGEBACK = "chargeback"
    DISPUTE_FEE = "dispute_fee"
    #: The batch credit arriving in the bank. A transfer, never income.
    SETTLEMENT_CREDIT = "settlement_credit"
    #: Money leaving to a supplier or employee, via RazorpayX or otherwise.
    VENDOR_PAYMENT = "vendor_payment"
    #: Charges the bank levied, distinct from the gateway's.
    BANK_CHARGE = "bank_charge"
    #: An attempt that took no money. Belongs in no ledger, and saying so is
    #: a category rather than a silence, because a row with no category is
    #: indistinguishable from a row the system failed on.
    NOT_A_TRANSACTION = "not_a_transaction"
    #: Nothing in the book determines this one. Goes to a human.
    NEEDS_REVIEW = "needs_review"


#: What each category means, in the words a reviewer would use. Shipped to the
#: model verbatim: a category list without definitions is an invitation to
#: invent the boundary between two of them, and the boundary is the whole job.
DEFINITIONS: dict[Category, str] = {
    Category.SALES_REVENUE: (
        "Money a customer paid for goods or services. The gross amount of a "
        "captured payment, before the gateway took anything out of it."
    ),
    Category.GATEWAY_FEE: (
        "The payment gateway's commission (MDR), excluding tax. An expense."
    ),
    Category.GST_INPUT_CREDIT: (
        "GST charged on the gateway's commission. The merchant reclaims this, "
        "so it is an asset, not a cost. Never merge it into gateway_fee."
    ),
    Category.REFUND: (
        "Money returned to a customer for a payment they already made. "
        "Reduces revenue; it is not an expense."
    ),
    Category.CHARGEBACK: (
        "A payment reversed by the customer's bank after a dispute."
    ),
    Category.DISPUTE_FEE: (
        "The fee the gateway charges for handling a dispute, separate from the "
        "disputed amount itself."
    ),
    Category.SETTLEMENT_CREDIT: (
        "The gateway paying out a batch into the merchant's bank account. This "
        "is a transfer between accounts the merchant already owns. It is NOT "
        "revenue -- the revenue was recognised when the customer paid, and "
        "booking it again here would double the reported turnover."
    ),
    Category.VENDOR_PAYMENT: (
        "Money the merchant sent out: a supplier, a contractor, payroll."
    ),
    Category.BANK_CHARGE: (
        "A fee the bank levied on the account, as opposed to the gateway."
    ),
    Category.NOT_A_TRANSACTION: (
        "An attempt that moved no money -- failed, created, or authorised but "
        "never captured. It belongs in no ledger."
    ),
    Category.NEEDS_REVIEW: (
        "Nothing in the book determines this row. A human decides."
    ),
}

#: Categories a model is allowed to propose. It may not propose NEEDS_REVIEW --
#: that is the outcome of failing to justify something, not a thing to choose --
#: and it may not propose the three the arithmetic already determines
#: (SALES_REVENUE, GATEWAY_FEE, GST_INPUT_CREDIT), because those come out of the
#: reconciliation and a model agreeing with them adds no information while a
#: model disagreeing with them would be overruled anyway.
PROPOSABLE = frozenset({
    Category.REFUND,
    Category.CHARGEBACK,
    Category.DISPUTE_FEE,
    Category.SETTLEMENT_CREDIT,
    Category.VENDOR_PAYMENT,
    Category.BANK_CHARGE,
    Category.NOT_A_TRANSACTION,
})
