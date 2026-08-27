"""The exception taxonomy.

These twelve classes are the vocabulary of the whole system. The generator
injects them, the matcher fails on them, and the scorer reports exceptions
broken down by them. Getting the names and the leg assignment right is the
domain-depth signal: each one is a real thing that happens to real merchant
settlements, not a synthetic perturbation invented to make the data look hard.

`deterministically_resolvable` records the design claim being tested: whether a
tier that only does arithmetic can, in principle, resolve this class. Classes
marked False are the ones the LLM tier exists to attempt -- and the baseline
ladder is what proves whether it actually earns that place.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Leg(int, Enum):
    """Which matching leg a defect disturbs."""

    LEG1 = 1  # order <-> PSP payment, 1:1
    LEG2 = 2  # PSP settlement batch <-> bank credit, N:1
    BOTH = 3


@dataclass(frozen=True)
class DefectSpec:
    code: str
    label: str
    leg: Leg
    description: str
    deterministically_resolvable: bool


class DefectClass(str, Enum):
    FEE_TAX_VARIANCE = "FEE_TAX_VARIANCE"
    REFUND_NETTED = "REFUND_NETTED"
    CHARGEBACK_NETTED = "CHARGEBACK_NETTED"
    TIMING_SPILL = "TIMING_SPILL"
    PARTIAL_CAPTURE = "PARTIAL_CAPTURE"
    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"
    DUPLICATE_UTR = "DUPLICATE_UTR"
    ROUNDING_DRIFT = "ROUNDING_DRIFT"
    NARRATION_TRUNCATION = "NARRATION_TRUNCATION"
    MISSING_BANK_LINE = "MISSING_BANK_LINE"
    FX_CONVERSION = "FX_CONVERSION"
    ADJUSTMENT_ENTRY = "ADJUSTMENT_ENTRY"

    @property
    def spec(self) -> DefectSpec:
        return DEFECT_SPECS[self]

    @property
    def leg(self) -> Leg:
        return DEFECT_SPECS[self].leg


DEFECT_SPECS: dict[DefectClass, DefectSpec] = {
    DefectClass.FEE_TAX_VARIANCE: DefectSpec(
        code="FEE_TAX_VARIANCE",
        label="Fee or tax variance",
        leg=Leg.LEG2,
        description=(
            "The fee actually charged differs from the published schedule -- a "
            "negotiated rate, a promotional period, or a mid-cycle repricing. "
            "The batch is short by an amount that is not explained by the fee "
            "model the matcher is carrying."
        ),
        deterministically_resolvable=False,
    ),
    DefectClass.REFUND_NETTED: DefectSpec(
        code="REFUND_NETTED",
        label="Refund netted into the batch",
        leg=Leg.LEG2,
        description=(
            "A refund issued after capture is deducted from the settlement "
            "credit rather than debited separately, so the bank line is short "
            "by the refund amount and no bank-side entry explains it."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.CHARGEBACK_NETTED: DefectSpec(
        code="CHARGEBACK_NETTED",
        label="Chargeback or dispute debit netted",
        leg=Leg.LEG2,
        description=(
            "A chargeback, plus its dispute-handling fee, is withheld from the "
            "batch. Two deductions of different kinds against one credit."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.TIMING_SPILL: DefectSpec(
        code="TIMING_SPILL",
        label="T+2 cutoff spill",
        leg=Leg.LEG2,
        description=(
            "A payment captured just before the settlement cutoff lands in the "
            "next cycle. The batch it 'should' belong to is short; a later "
            "batch is long. Neither reconciles in isolation."
        ),
        deterministically_resolvable=False,
    ),
    DefectClass.PARTIAL_CAPTURE: DefectSpec(
        code="PARTIAL_CAPTURE",
        label="Partial capture",
        leg=Leg.BOTH,
        description=(
            "The captured amount is less than the authorised order amount, so "
            "the order ledger and the PSP report disagree on Leg 1 while the "
            "settlement itself is internally consistent."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.DUPLICATE_PAYMENT: DefectSpec(
        code="DUPLICATE_PAYMENT",
        label="Duplicate payment against one order",
        leg=Leg.LEG1,
        description=(
            "A customer double-pays -- retry after an ambiguous failure. Two "
            "payment rows claim the same order, breaking the 1:1 assumption "
            "Leg 1 rests on."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.DUPLICATE_UTR: DefectSpec(
        code="DUPLICATE_UTR",
        label="Duplicate UTR across bank lines",
        leg=Leg.LEG2,
        description=(
            "The same UTR appears on two bank lines -- a bank-side reposting or "
            "a statement pulled twice across an overlapping window. Matching on "
            "UTR alone silently double-counts."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.ROUNDING_DRIFT: DefectSpec(
        code="ROUNDING_DRIFT",
        label="Sub-rupee rounding drift",
        leg=Leg.LEG2,
        description=(
            "Per-step versus per-batch rounding of fee and GST leaves the credit "
            "off by a few paise. Individually trivial, but it defeats exact "
            "equality and so must be absorbed by an explicit tolerance rather "
            "than ignored."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.NARRATION_TRUNCATION: DefectSpec(
        code="NARRATION_TRUNCATION",
        label="Narration truncated, UTR unreadable",
        leg=Leg.LEG2,
        description=(
            "The bank statement narration is clipped to a fixed width, cutting "
            "the UTR short or mangling it. The amount is correct but the join "
            "key is gone, so the line must be matched on amount and date alone."
        ),
        deterministically_resolvable=False,
    ),
    DefectClass.MISSING_BANK_LINE: DefectSpec(
        code="MISSING_BANK_LINE",
        label="Settlement on hold, no bank line",
        leg=Leg.LEG2,
        description=(
            "The settlement is marked processed on the PSP side but held back "
            "into reserve balance, so no credit ever lands. The correct verdict "
            "is 'unmatched, and legitimately so' -- not an error."
        ),
        deterministically_resolvable=True,
    ),
    DefectClass.FX_CONVERSION: DefectSpec(
        code="FX_CONVERSION",
        label="International payment, FX converted",
        leg=Leg.LEG2,
        description=(
            "An international payment settles in INR at a rate applied by the "
            "bank, at a higher MDR, so the credited amount cannot be derived "
            "from the order amount without the rate actually used."
        ),
        deterministically_resolvable=False,
    ),
    DefectClass.ADJUSTMENT_ENTRY: DefectSpec(
        code="ADJUSTMENT_ENTRY",
        label="Manual adjustment or reversal",
        leg=Leg.LEG2,
        description=(
            "A support-issued credit, a reversal of an earlier deduction, or a "
            "platform-fee adjustment appears against the batch with no "
            "corresponding payment row at all."
        ),
        deterministically_resolvable=False,
    ),
}


#: Classes an arithmetic-only tier should be able to close, given the right
#: evidence. Used by the scorer to separate "the solver missed something it
#: should have caught" from "this is genuinely what the LLM tier is for".
DETERMINISTIC_CLASSES: frozenset[DefectClass] = frozenset(
    d for d, spec in DEFECT_SPECS.items() if spec.deterministically_resolvable
)
