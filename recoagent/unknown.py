"""Defects the matcher has never been told about.

Every number this repository publishes on its own data has one structural
weakness, and it is worth stating plainly rather than disclaiming: the
generator and the matcher share an author. The twelve classes in
`recoagent.defects` are the ones the engine was *written for*, so a 0.00%
false-match rate over them is partly a statement about the engine and partly a
statement about the vocabulary both halves were built from. Changing the defect
*mix* between dev and held-out, which is what `DefectMix.holdout` does, varies
the proportions inside that vocabulary. It cannot vary the vocabulary itself.

This module is the vocabulary the engine does not have.

Three further defect classes are defined here, all of them real events on
Indian settlement books, and **no handling for any of them exists anywhere in
`recoagent`**. There is no tier that closes a TDS withholding, no rule that
recognises a bank transfer charge, no concept anywhere of one settlement paid
out as two credits. That absence is the experiment.

**What is being measured.** Not recall -- the engine cannot resolve what it has
no model of, and an unknown-class recall of 0% is the expected and correct
result. What is being measured is the *failure mode*. When a book contains a
discrepancy the system was never designed to explain, does it:

- file an exception and say so (**contained** -- the safety property holds
  beyond the taxonomy it was written for), or
- find some subset of rows that coincidentally closes the arithmetic and book
  a match (**a wrong match** -- the safety property was an artefact of only
  ever meeting defects the author had already thought about)?

The second outcome is the one worth hunting for, and the solver makes it a live
risk rather than a theoretical one: `legs.ssmp` searches for subsets that sum to
a residual, and a residual it was never meant to see is still a number it will
happily search against.

**The fence.** This module is under the same rule as the generator and for the
same reason: `tests/test_independence.py` asserts that no matcher may import it.
A tier that could read `UnknownDefectClass` would stop being unknown, and the
result would measure nothing at all. The injectors live here rather than in
`generator.py` so that the fence is one import away from being checked, and so
that adding handling for one of these classes is a conspicuous act rather than
a quiet edit inside a thousand-line file.

If a class here is ever given a tier, it belongs in `defects.py` and a new one
belongs here. The point is not these three events. The point is that there are
always three more.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from enum import Enum

from .money import Paise


class UnknownDefectClass(str, Enum):
    """Defect classes with no corresponding tier, rule, or tolerance."""

    TDS_194O_WITHHELD = "TDS_194O_WITHHELD"
    BANK_CHARGE_DEBIT = "BANK_CHARGE_DEBIT"
    SPLIT_PAYOUT = "SPLIT_PAYOUT"


@dataclass(frozen=True)
class UnknownSpec:
    code: str
    label: str
    description: str
    #: Why no existing tier reaches it. This is the part that has to stay true:
    #: if one of these becomes explicable by something already in the engine,
    #: the class stops being a test and starts being a duplicate.
    why_unseen: str


UNKNOWN_SPECS: dict[UnknownDefectClass, UnknownSpec] = {
    UnknownDefectClass.TDS_194O_WITHHELD: UnknownSpec(
        code="TDS_194O_WITHHELD",
        label="Section 194-O TDS withheld at source",
        description=(
            "The gateway withholds 0.1% TDS on the gross value of sales made "
            "through it, as an e-commerce operator is required to, and remits "
            "it against the merchant's PAN. The settlement credit is short by "
            "that amount and the settlement report says nothing about it."
        ),
        why_unseen=(
            "Every deduction the engine models is derived from the fee "
            "schedule and is a function of fee or of net. This one is a "
            "percentage of gross, remitted to a third party, and no rate on "
            "file describes it."
        ),
    ),
    UnknownDefectClass.BANK_CHARGE_DEBIT: UnknownSpec(
        code="BANK_CHARGE_DEBIT",
        label="Bank transfer charge debited from the credit",
        description=(
            "The receiving bank takes its own RTGS/NEFT handling charge plus "
            "GST out of the credit before it lands. The gateway paid the full "
            "amount; the merchant received less, and only the bank statement "
            "knows why."
        ),
        why_unseen=(
            "It is a bank-side deduction. Every explanation the engine can "
            "construct is built from gateway rows -- payments, refunds, "
            "adjustments, notices -- and there is no gateway row behind this "
            "one at all."
        ),
    ),
    UnknownDefectClass.SPLIT_PAYOUT: UnknownSpec(
        code="SPLIT_PAYOUT",
        label="One settlement paid out as two credits",
        description=(
            "A large payout is split across two transfers -- a per-transaction "
            "ceiling on the rail, or a gateway spreading a payout across "
            "windows. Two bank lines, two UTRs, one settlement, and neither "
            "line reconciles against the batch on its own."
        ),
        why_unseen=(
            "The engine's leg 2 is N:1 by construction: many payments "
            "consolidate into one credit. Nothing in it can express one "
            "settlement arriving as two credits, so the correct answer is not "
            "merely unreached -- it is unrepresentable."
        ),
    ),
}


#: Statutory TDS under section 194-O, in basis points of gross.
TDS_194O_BPS = 10

#: A typical RTGS handling charge, in paise, before tax.
BANK_CHARGE_PAISE = 2_500

#: GST on a bank charge.
BANK_CHARGE_GST_BPS = 1_800


def _inject_tds_194o(w, sid: str) -> bool:
    members = w.members.get(sid, [])
    if not members or not w.claim(sid):
        return False
    gross = sum(w.payment(pid).gross_paise for pid in members)
    tds: Paise = (gross * TDS_194O_BPS) // 10_000
    if tds <= 0 or not w.shift_bank_line(sid, -tds):
        return False
    w.record(
        UnknownDefectClass.TDS_194O_WITHHELD,
        "settlement",
        sid,
        f"194-O TDS of {tds} paise withheld on gross of {gross} paise",
    )
    return True


def _inject_bank_charge(w, sid: str) -> bool:
    if not w.claim(sid):
        return False
    charge = BANK_CHARGE_PAISE + (BANK_CHARGE_PAISE * BANK_CHARGE_GST_BPS) // 10_000
    if not w.shift_bank_line(sid, -charge):
        return False
    w.record(
        UnknownDefectClass.BANK_CHARGE_DEBIT,
        "settlement",
        sid,
        f"bank took {charge} paise in transfer charge and GST out of the credit",
    )
    return True


def _inject_split_payout(w, sid: str) -> bool:
    line = w.bank_line_for(sid)
    if line is None or not w.claim(sid):
        return False
    # Split off between a fifth and a half, and never at a round number: an
    # even split would let a solver close the batch by halving it, which would
    # be a coincidence dressed as an explanation.
    share = w.rng.randint(20, 50)
    second = (line.amount_paise * share) // 100
    first = line.amount_paise - second
    if first <= 0 or second <= 0:
        return False

    tail_id = f"{line.bank_line_id}_b"
    for i, b in enumerate(w.bank_lines):
        if b.bank_line_id == line.bank_line_id:
            w.bank_lines[i] = replace(b, amount_paise=first)
            break
    tail_utr = f"{w.rng.randint(10**11, 10**12 - 1)}"
    w.bank_lines.append(
        replace(
            line,
            bank_line_id=tail_id,
            amount_paise=second,
            narration=f"NEFT CR-HDFC0000123-RAZORPAY SOFTWARE PVT LTD-{tail_utr}",
            bank_ref=f"{line.bank_ref}B",
            value_date=line.value_date + timedelta(days=1),
        )
    )
    # Both lines are this settlement's money, so both are in the truth map. The
    # second one carries a UTR the settlement report has never heard of, which
    # is exactly what makes the pair unrepresentable rather than merely hard.
    w.leg2[tail_id] = sid
    w.record(
        UnknownDefectClass.SPLIT_PAYOUT,
        "settlement",
        sid,
        f"payout split into {first} and {second} paise across two transfers",
        affected_ids=(line.bank_line_id, tail_id),
    )
    return True


UNKNOWN_INJECTORS = {
    UnknownDefectClass.TDS_194O_WITHHELD: _inject_tds_194o,
    UnknownDefectClass.BANK_CHARGE_DEBIT: _inject_bank_charge,
    UnknownDefectClass.SPLIT_PAYOUT: _inject_split_payout,
}
