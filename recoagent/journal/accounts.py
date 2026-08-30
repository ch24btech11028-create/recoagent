"""The chart of accounts, and the one account that carries the whole argument.

A category is a label. A posting is a commitment: it says which two accounts
move and in which direction, and once you have committed, the books either
balance or they do not. That is a much harder thing to be quietly wrong about
than a category, which is why this file exists rather than stopping at
`categorize/`.

**`GATEWAY_RECEIVABLE` is the point.** When a customer pays, the merchant does
not have the money -- the gateway does. So a capture creates a receivable from
the gateway, the fee and its tax and any refund reduce it, and the settlement
credit *clears* it into the bank. Which means:

    gross - fee - tax - refunds - chargebacks - dispute fees - net credit = 0

is both the leg-2 arithmetic identity the matcher proves, and the statement
that a batch's receivable nets to zero. They are the same equation written in
two notations. So a batch that reconciled leaves no receivable balance behind,
and a batch that did not leaves exactly its unexplained residual sitting in
that account, under the batch's own id.

That is what "closes the loop" means here, and it is checkable rather than
asserted: `journal.post` reports the per-batch receivable balance and
`tests/test_journal.py` requires it to agree with the reconciliation's residual
to the paisa.

The two structural errors the taxonomy exists to prevent survive into the
postings, which is where they would actually cost money:

- **`SETTLEMENT_CREDIT` credits the receivable, never income.** Booking it to
  `SALES_REVENUE` would double the merchant's declared turnover -- the revenue
  was recognised when the customer paid.
- **`GST_INPUT_CREDIT` debits an asset, not an expense.** It is reclaimable in
  GSTR-3B. Filed as a cost it overstates expenses and understates the credit
  claimable, which is a filing error rather than a presentation one.
"""

from __future__ import annotations

from enum import Enum

from ..categorize.taxonomy import Category


class AccountType(str, Enum):
    ASSET = "asset"
    LIABILITY = "liability"
    INCOME = "income"
    EXPENSE = "expense"


class Account(str, Enum):
    BANK = "bank"
    #: The clearing account. Money the gateway owes the merchant and has not
    #: paid out yet. See the module docstring: this is where reconciliation and
    #: bookkeeping turn out to be the same statement.
    GATEWAY_RECEIVABLE = "gateway_receivable"
    #: Reclaimable GST on the MDR. An asset, deliberately not an expense.
    GST_INPUT_CREDIT = "gst_input_credit"
    SALES_REVENUE = "sales_revenue"
    GATEWAY_FEES = "gateway_fees"
    DISPUTE_FEES = "dispute_fees"
    BANK_CHARGES = "bank_charges"
    VENDOR_EXPENSE = "vendor_expense"
    #: Rows the system will not classify. Standard practice, and the honest
    #: alternative to dropping them: the books stay balanced and the problem
    #: stays visible with a number against it.
    SUSPENSE = "suspense"


ACCOUNT_TYPES: dict[Account, AccountType] = {
    Account.BANK: AccountType.ASSET,
    Account.GATEWAY_RECEIVABLE: AccountType.ASSET,
    Account.GST_INPUT_CREDIT: AccountType.ASSET,
    Account.SALES_REVENUE: AccountType.INCOME,
    Account.GATEWAY_FEES: AccountType.EXPENSE,
    Account.DISPUTE_FEES: AccountType.EXPENSE,
    Account.BANK_CHARGES: AccountType.EXPENSE,
    Account.VENDOR_EXPENSE: AccountType.EXPENSE,
    Account.SUSPENSE: AccountType.ASSET,
}


#: category -> (debit account, credit account).
#:
#: Every rule here moves exactly two accounts, so an entry cannot fail to
#: balance by construction and the trial balance is a check on the *data*
#: rather than on the arithmetic of this table.
POSTING_RULES: dict[Category, tuple[Account, Account]] = {
    # A capture: the gateway now owes the merchant, and revenue is recognised.
    Category.SALES_REVENUE: (Account.GATEWAY_RECEIVABLE, Account.SALES_REVENUE),
    # The MDR, taken out of what the gateway owes.
    Category.GATEWAY_FEE: (Account.GATEWAY_FEES, Account.GATEWAY_RECEIVABLE),
    # GST on the MDR. Debits an asset -- the merchant reclaims this.
    Category.GST_INPUT_CREDIT: (Account.GST_INPUT_CREDIT, Account.GATEWAY_RECEIVABLE),
    # Contra-revenue, not an expense: a refund reverses a sale.
    Category.REFUND: (Account.SALES_REVENUE, Account.GATEWAY_RECEIVABLE),
    Category.CHARGEBACK: (Account.SALES_REVENUE, Account.GATEWAY_RECEIVABLE),
    Category.DISPUTE_FEE: (Account.DISPUTE_FEES, Account.GATEWAY_RECEIVABLE),
    # The payout landing. A transfer between accounts the merchant owns, which
    # is the entire reason `SETTLEMENT_CREDIT` is a separate category.
    Category.SETTLEMENT_CREDIT: (Account.BANK, Account.GATEWAY_RECEIVABLE),
    Category.BANK_CHARGE: (Account.BANK_CHARGES, Account.BANK),
    Category.VENDOR_PAYMENT: (Account.VENDOR_EXPENSE, Account.BANK),
    # Unclassified money, parked where it can be seen and counted.
    Category.NEEDS_REVIEW: (Account.SUSPENSE, Account.GATEWAY_RECEIVABLE),
}

#: Categories that deliberately produce no posting at all. An attempt that took
#: no money is not a bookkeeping event, and inventing a zero-value entry for it
#: would put rows in the journal that never belonged in any ledger.
NOT_POSTED = frozenset({Category.NOT_A_TRANSACTION})


#: The cash direction each category is expected to carry, used to catch a row
#: whose amount contradicts its own label. Not enforced -- reported. A refund
#: with a positive amount is a data problem worth naming, and silently taking
#: its absolute value would hide exactly the kind of sign error that makes a
#: set of books wrong while they still balance.
EXPECTED_SIGN: dict[Category, int] = {
    Category.SALES_REVENUE: +1,
    Category.SETTLEMENT_CREDIT: +1,
    Category.GATEWAY_FEE: -1,
    Category.GST_INPUT_CREDIT: -1,
    Category.REFUND: -1,
    Category.CHARGEBACK: -1,
    Category.DISPUTE_FEE: -1,
    Category.BANK_CHARGE: -1,
    Category.VENDOR_PAYMENT: -1,
}
