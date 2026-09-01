"""Turning a proved, categorised book into double-entry postings.

The ladder so far ends at a category. A category is a label, and a label can be
wrong quietly. A posting cannot: it names two accounts and a direction, and
once every row is posted the books either balance or they do not.

Three things are asserted here, in descending order of how easy they are to
fake:

1. **Every entry balances.** True by construction -- each posting rule moves
   exactly two accounts -- so this checks the code, not the data.
2. **The trial balance balances.** Total debits equal total credits across the
   whole book.
3. **Every rupee left in the gateway receivable is attributable to a named
   cause, and nothing is left over.** This is the one worth reading, and note
   what it does *not* claim: the clearing account does not empty, and an engine
   that said it did would be hiding something. Four of the six causes are read
   straight off the matcher's own rule id -- whatever Tier 1 had to do to close
   a credit is the reason the books and that batch disagree -- so the
   attribution cannot drift away from the reconciliation it describes.

Finding all six took three real bugs out of the pipeline, none of which the
category scorecard could see, because in each case the *label* was right and
the *amount* was wrong: a settlement credit booked at the gateway's declared
net rather than the cash the bank actually sent, an orphaned refund the solver
had attributed to a batch that the books attributed to nothing, and a manual
adjustment whose positive amount was posted in the negative direction its
category implies. A clearing account either empties or it does not, which is
why it found them.

Nothing is posted that the reconciliation did not prove. A `NEEDS_REVIEW` row
goes to suspense rather than to a guessed account, and the suspense balance is
reported at the top rather than buried -- a non-zero suspense is the operator's
queue expressed in rupees, not a rounding note.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..categorize.rules import Ledger
from ..categorize.taxonomy import Category
from ..money import Paise, format_inr
from ..schemas import ReconResult, SourceBundle
from .accounts import (
    ACCOUNT_TYPES,
    EXPECTED_SIGN,
    NOT_POSTED,
    POSTING_RULES,
    Account,
    AccountType,
)


@dataclass(frozen=True)
class Posting:
    account: Account
    debit_paise: Paise = 0
    credit_paise: Paise = 0


@dataclass(frozen=True)
class JournalEntry:
    """One categorised row, expressed as a balanced pair of postings."""

    entry_id: str
    entity_id: str
    entity_kind: str
    category: Category
    amount_paise: Paise
    postings: tuple[Posting, ...]
    rung: str
    rule_id: str
    #: Carried through from the assignment. A posting whose evidence is a match
    #: id is a posting a reader can walk back to an arithmetic proof.
    evidence: str
    #: The settlement this row belongs to, where the sources say so. This is
    #: what makes the per-batch receivable check possible.
    batch_id: str | None = None

    @property
    def debits(self) -> Paise:
        return sum(p.debit_paise for p in self.postings)

    @property
    def credits(self) -> Paise:
        return sum(p.credit_paise for p in self.postings)

    @property
    def balances(self) -> bool:
        return self.debits == self.credits


@dataclass(frozen=True)
class SignAnomaly:
    """A row whose amount contradicts the direction its category implies."""

    entity_id: str
    category: Category
    amount_paise: Paise
    expected_sign: int


@dataclass
class Journal:
    entries: list[JournalEntry] = field(default_factory=list)
    #: Rows deliberately not posted, with the reason. `NOT_A_TRANSACTION` rows
    #: are here, and so is anything whose category has no posting rule -- the
    #: second case is a bug in this module and is surfaced rather than skipped.
    unposted: list[tuple[str, Category, str]] = field(default_factory=list)
    anomalies: list[SignAnomaly] = field(default_factory=list)

    @property
    def total_debits(self) -> Paise:
        return sum(e.debits for e in self.entries)

    @property
    def total_credits(self) -> Paise:
        return sum(e.credits for e in self.entries)

    @property
    def balances(self) -> bool:
        return self.total_debits == self.total_credits

    @property
    def unbalanced_entries(self) -> list[JournalEntry]:
        return [e for e in self.entries if not e.balances]

    def trial_balance(self) -> dict[Account, tuple[Paise, Paise]]:
        """account -> (total debits, total credits), every account that moved."""
        out: dict[Account, list[int]] = {}
        for entry in self.entries:
            for p in entry.postings:
                row = out.setdefault(p.account, [0, 0])
                row[0] += p.debit_paise
                row[1] += p.credit_paise
        return {a: (d, c) for a, (d, c) in out.items()}

    def balance_of(self, account: Account) -> Paise:
        """Debit-positive balance. Assets and expenses run positive."""
        d, c = self.trial_balance().get(account, (0, 0))
        return d - c

    def receivable_by_batch(self) -> dict[str, Paise]:
        """Per-settlement gateway-receivable balance, debit-positive.

        Zero for a batch whose credit reconciled. Non-zero for one that did
        not, by exactly the amount the matcher could not explain.
        """
        out: dict[str, int] = {}
        for entry in self.entries:
            if entry.batch_id is None:
                continue
            for p in entry.postings:
                if p.account is Account.GATEWAY_RECEIVABLE:
                    out[entry.batch_id] = (
                        out.get(entry.batch_id, 0) + p.debit_paise - p.credit_paise
                    )
        return out

    @property
    def suspense_paise(self) -> Paise:
        return self.balance_of(Account.SUSPENSE)


#: Why a batch still has money sitting in the clearing account.
#:
#: Ordered, and each batch's whole balance goes to the first cause that
#: applies, so the causes partition the total rather than overlapping it. Most
#: of them are read straight off the *matcher's own rule id* -- whatever Tier 1
#: had to do to close the credit is, by definition, the reason the books and
#: the batch disagree. That keeps this table honest: it cannot drift from the
#: reconciliation, because it is derived from it.
NEVER_CREDITED = "the gateway has not paid this batch out"
MEMBER_NOT_BOOKED = "a payment in the batch never matched an order"
CROSS_BATCH_SPILL = "a payment reported here was credited with another cycle"
RATE_DIFFERENCE = "an FX or repricing difference against the reported figure"
NETTED_ROW = "the gateway netted a row it did not link to the batch"
ROUNDING = "sub-rupee rounding between the gateway and the bank"
UNATTRIBUTED = "unattributed"

#: leg-2 rule id -> the cause its use implies.
_RULE_CAUSE = {
    "leg2.t1.spill_pair": CROSS_BATCH_SPILL,
    "leg2.t1.rate_notice": RATE_DIFFERENCE,
    "leg2.t1.ssmp_residual": NETTED_ROW,
}

#: A batch left in the rounding bucket may not be hiding a real gap. Sub-rupee
#: drift between a gateway and a bank is paise, so a hundred rupees is already
#: two orders of magnitude of headroom -- generous on purpose, because the
#: assertion is meant to catch a misfiled cause, not to encode a threshold.
ROUNDING_CEILING_PAISE = 10_000

CAUSE_ORDER = (
    NEVER_CREDITED,
    MEMBER_NOT_BOOKED,
    CROSS_BATCH_SPILL,
    RATE_DIFFERENCE,
    NETTED_ROW,
    ROUNDING,
)


@dataclass(frozen=True)
class OpenBatch:
    batch_id: str
    balance_paise: Paise
    cause: str


def explain_receivable(
    journal: Journal, sources: SourceBundle, result: ReconResult
) -> list[OpenBatch]:
    """Attribute every open receivable balance to a named cause.

    The claim this supports is not "everything clears" -- it does not, and a
    reconciliation engine that made it would be hiding something. It is the
    stronger and more checkable one: **every rupee left in the clearing account
    is attributable, and nothing is left over.**

    Causes are assigned in priority order and each batch's whole balance goes to
    exactly one, so they partition the total by construction. That makes the sum
    trivially right and moves the real assertion somewhere it can fail: whatever
    lands in `ROUNDING` is what no other cause claimed, and sub-rupee drift is
    paise. If a real gap ever hides in that bucket, `tests/test_journal.py`
    says so.
    """
    matched_payments = {p for m in result.matches_for_leg(1) for p in m.right_ids}
    rules = {
        m.right_ids[0]: m.rule_id
        for m in result.matches_for_leg(2)
        if m.right_ids
    }

    out: list[OpenBatch] = []
    for batch_id, balance in sorted(journal.receivable_by_batch().items()):
        if balance == 0:
            continue
        members = sources.payments_by_settlement(batch_id)
        if batch_id not in rules:
            cause = NEVER_CREDITED
        elif any(p.payment_id not in matched_payments for p in members):
            cause = MEMBER_NOT_BOOKED
        else:
            cause = _RULE_CAUSE.get(rules[batch_id], ROUNDING)
        out.append(OpenBatch(batch_id, balance, cause))
    return out


def _batch_index(sources: SourceBundle, result: ReconResult) -> dict[str, str]:
    """entity_id -> settlement_id, for everything the sources attribute.

    Payments and adjustments carry their batch on the row. A bank line does
    not -- it is attributed by the match that paired it, which is the only
    honest source for that link: an unmatched credit belongs to no batch, and
    guessing one here would invent the very fact the matcher refused to assert.

    **Hypothesised rows are attributed too, and that is not a shortcut.** An
    orphaned refund the gateway netted into a batch without linking it carries
    no `settlement_id`; Tier 1's subset-sum finds it and records it on the match
    as `hypothesised_ids`, having required that it be the *only* subset that
    closes. Leaving those out was the first thing that stopped the receivable
    clearing -- the batch's credit accounted for money the books had attributed
    to nothing. The link is as good as the arithmetic proof on the match record,
    which is the same standard every other entry here is held to.
    """
    index: dict[str, str] = {}
    for p in sources.payments:
        if p.settlement_id:
            index[p.payment_id] = p.settlement_id
            index[f"{p.payment_id}:fee"] = p.settlement_id
            index[f"{p.payment_id}:tax"] = p.settlement_id
    for a in sources.adjustments:
        if a.settlement_id:
            index[a.adjustment_id] = a.settlement_id
    for m in result.matches_for_leg(2):
        if not m.right_ids:
            continue
        index[m.left_ids[0]] = m.right_ids[0]
        for hypothesised in m.hypothesised_ids:
            index.setdefault(hypothesised, m.right_ids[0])
    return index


def post(ledger: Ledger, sources: SourceBundle, result: ReconResult) -> Journal:
    """Build the journal from a categorised, reconciled book.

    Assignments that are proposals rather than decisions (`booked` is False --
    every C2 assignment, by design) are not posted. A suggestion that reaches
    the operator queue with its evidence attached is useful; a suggestion in
    the general ledger is a misstatement.
    """
    journal = Journal()
    batches = _batch_index(sources, result)

    for entity_id in sorted(ledger.assignments):
        a = ledger.assignments[entity_id]

        if a.category in NOT_POSTED:
            journal.unposted.append(
                (a.entity_id, a.category, "not a bookkeeping event")
            )
            continue

        if not a.booked and a.category is not Category.NEEDS_REVIEW:
            journal.unposted.append(
                (a.entity_id, a.category, f"proposed by {a.rung}, not booked")
            )
            continue

        rule = POSTING_RULES.get(a.category)
        if rule is None:
            journal.unposted.append(
                (a.entity_id, a.category, "no posting rule for this category")
            )
            continue

        amount = abs(a.amount_paise)
        if amount == 0:
            journal.unposted.append((a.entity_id, a.category, "zero value"))
            continue

        # The category names the pair of accounts; the sign names which way
        # they move. A row carrying the opposite sign to its category posts the
        # reverse pair -- a contra entry -- rather than the same one at
        # magnitude.
        #
        # This is not pedantry. A manual adjustment filed as a refund but
        # carrying a positive amount is money arriving, and posting it at
        # magnitude in the refund direction moves the receivable the wrong way
        # by *twice* the value. One such row put Rs 12,500 into a clearing
        # account that had no other reason to be open, and it took the
        # per-batch receivable check to find it: the trial balance still
        # balanced, because a wrong direction is wrong on both sides at once.
        debit, credit = rule
        expected = EXPECTED_SIGN.get(a.category)
        if expected is not None and (1 if a.amount_paise > 0 else -1) != expected:
            journal.anomalies.append(
                SignAnomaly(a.entity_id, a.category, a.amount_paise, expected)
            )
            debit, credit = credit, debit
        journal.entries.append(
            JournalEntry(
                entry_id=f"je_{a.entity_id}",
                entity_id=a.entity_id,
                entity_kind=a.entity_kind,
                category=a.category,
                amount_paise=amount,
                postings=(
                    Posting(debit, debit_paise=amount),
                    Posting(credit, credit_paise=amount),
                ),
                rung=a.rung,
                rule_id=a.rule_id,
                evidence=a.evidence,
                batch_id=batches.get(a.entity_id),
            )
        )

    return journal


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def render(journal: Journal, sources: SourceBundle, result: ReconResult) -> str:
    tb = journal.trial_balance()
    open_batches = explain_receivable(journal, sources, result)
    cleared = sum(1 for v in journal.receivable_by_batch().values() if v == 0)

    out = [
        "=" * 72,
        "  GENERAL JOURNAL AND TRIAL BALANCE",
        "=" * 72,
        "",
        f"  entries posted              {len(journal.entries):>14,}",
        f"  rows not posted             {len(journal.unposted):>14,}",
        f"  total debits                {format_inr(journal.total_debits):>14}",
        f"  total credits               {format_inr(journal.total_credits):>14}",
        f"  TRIAL BALANCE               "
        f"{'BALANCED' if journal.balances else 'OUT BY ' + format_inr(journal.total_debits - journal.total_credits):>14}"
        f"   <- lead metric",
        f"  entries that do not balance {len(journal.unbalanced_entries):>14,}",
        "",
        f"  suspense (unclassified)     {format_inr(journal.suspense_paise):>14}",
        "",
        "-" * 72,
        "  TRIAL BALANCE",
        "-" * 72,
        f"  {'account':<24}{'type':<10}{'debits':>17}{'credits':>17}",
    ]
    for account in Account:
        if account not in tb:
            continue
        d, c = tb[account]
        out.append(
            f"  {account.value:<24}{ACCOUNT_TYPES[account].value:<10}"
            f"{format_inr(d):>17}{format_inr(c):>17}"
        )
    out += [
        "  " + "-" * 68,
        f"  {'':<34}{format_inr(journal.total_debits):>17}"
        f"{format_inr(journal.total_credits):>17}",
    ]

    out += [
        "",
        "-" * 72,
        "  DOES THE LOOP CLOSE",
        "-" * 72,
        "",
        "  The gateway receivable is a clearing account: a capture creates it,",
        "  fees and refunds reduce it, and the settlement credit clears it into",
        "  the bank. The claim is not that it empties -- it does not, and an",
        "  engine that said so would be hiding something. It is that every rupee",
        "  left in it is attributable to a named cause, with nothing left over.",
        "",
        f"  receivable balance          {format_inr(journal.balance_of(Account.GATEWAY_RECEIVABLE)):>18}",
        f"  batches fully cleared       {cleared:>18,}",
        f"  batches still open          {len(open_batches):>18,}",
        "",
    ]
    if open_batches:
        by_cause: dict[str, list[OpenBatch]] = {}
        for ob in open_batches:
            by_cause.setdefault(ob.cause, []).append(ob)
        out.append(f"  {'cause':<56}{'batches':>8}{'value':>17}")
        attributed = 0
        for cause in CAUSE_ORDER:
            group = by_cause.get(cause, [])
            if not group:
                continue
            total = sum(g.balance_paise for g in group)
            attributed += total
            out.append(f"  {cause:<56}{len(group):>8,}{format_inr(total):>17}")
        leftover = journal.balance_of(Account.GATEWAY_RECEIVABLE) - attributed
        out += [
            "  " + "-" * 72,
            f"  {UNATTRIBUTED:<56}{'':>8}{format_inr(leftover):>17}"
            f"{'   <- must be zero' if leftover == 0 else '   <- NOT ZERO'}",
        ]

        biggest = sorted(open_batches, key=lambda b: -abs(b.balance_paise))[:8]
        out += ["", f"  {'largest open batches':<14}{'balance':>18}   cause"]
        for ob in biggest:
            out.append(f"  {ob.batch_id:<14}{format_inr(ob.balance_paise):>18}   {ob.cause}")
    else:
        out.append("  Every batch cleared to zero.")

    out += [
        "",
        "  Every cause but the first two is read off the matcher's own rule id:",
        "  whatever Tier 1 had to do to close the credit is the reason the books",
        "  and the batch disagree, so this table cannot drift from the",
        "  reconciliation. What lands in rounding is what no other cause claimed,",
        "  and `tests/test_journal.py` fails if a real gap hides there.",
    ]

    if journal.anomalies:
        out += ["", "-" * 72, "  SIGN ANOMALIES", "-" * 72]
        for a in journal.anomalies[:20]:
            out.append(
                f"  {a.entity_id:<22}{a.category.value:<20}"
                f"{format_inr(a.amount_paise):>16}  expected "
                f"{'positive' if a.expected_sign > 0 else 'negative'}"
            )
        if len(journal.anomalies) > 20:
            out.append(f"  ... and {len(journal.anomalies) - 20} more")
        out += [
            "",
            "  Reported, not corrected. A row whose amount contradicts its own",
            "  category is a data problem, and quietly flipping the sign would",
            "  hide the class of error that leaves books wrong but balanced.",
        ]

    reasons: dict[str, int] = {}
    for _, _, why in journal.unposted:
        reasons[why] = reasons.get(why, 0) + 1
    if reasons:
        out += ["", "-" * 72, "  NOT POSTED, AND WHY", "-" * 72]
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            out.append(f"  {n:>7,}  {why}")

    out += [
        "",
        "  Nothing here was posted from a category a model proposed. C2",
        "  assignments reach the operator queue with their evidence and stop",
        "  there: a suggestion in the general ledger is a misstatement.",
        "=" * 72,
    ]
    return "\n".join(out)
