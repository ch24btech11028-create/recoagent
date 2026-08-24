"""Domain entities, match records, and the source/truth split.

The single most important structure here is the split between `SourceBundle`
and `LabelledBatch`.

`SourceBundle` is everything a real merchant would actually possess: three
files that disagree. It is the *only* thing ever handed to a matcher.

`LabelledBatch` is what the generator emits -- a SourceBundle plus the ground
truth. It never crosses into `recoagent.legs` or `recoagent.validate`; only the
generator produces it and only the scorer consumes it.

This is enforced structurally rather than by convention, and there is a test
(`tests/test_independence.py`) asserting that the matching modules import
nothing from the generator. A matcher that cannot reach the answer key cannot
accidentally be tuned against it, which is what makes the reported numbers
worth anything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from .defects import DefectClass
from .money import Paise

# ─────────────────────────────────────────────────────────────────────────────
# Source entities -- what the merchant actually has
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Order:
    """A row from the merchant's own order ledger / ERP."""

    order_id: str
    customer_id: str
    invoice_no: str
    amount_paise: Paise
    currency: str
    created_at: datetime


@dataclass(frozen=True)
class PGPayment:
    """A transaction row from the payment gateway's settlement report."""

    payment_id: str
    order_id: str | None
    gross_paise: Paise
    fee_paise: Paise
    tax_paise: Paise
    method: str
    status: str  # captured | partially_captured | refunded
    settlement_id: str | None
    captured_at: datetime
    currency: str = "INR"
    fx_rate: float | None = None  # populated only for international payments

    @property
    def net_paise(self) -> Paise:
        """What this payment contributes to its settlement batch."""
        return self.gross_paise - self.fee_paise - self.tax_paise


@dataclass(frozen=True)
class PGAdjustment:
    """A non-payment line netted into a settlement batch.

    `amount_paise` is signed from the merchant's perspective: negative values
    reduce the credit. Refunds and chargebacks are negative; a reversal of an
    earlier deduction is positive.
    """

    adjustment_id: str
    settlement_id: str
    kind: str  # refund | chargeback | dispute_fee | reversal | platform_fee
    payment_id: str | None
    amount_paise: Paise
    booked_at: datetime


@dataclass(frozen=True)
class Settlement:
    """The batch header the gateway reports: N payments consolidated into one payout."""

    settlement_id: str
    utr: str
    settled_at: datetime
    net_paise: Paise
    status: str  # processed | on_hold | reversed


@dataclass(frozen=True)
class BankLine:
    """A credit on the merchant's bank statement.

    `narration` is the raw string as the bank printed it -- possibly truncated,
    possibly with the UTR mangled. Extracting a usable key from it is part of
    the matching problem, not a preprocessing step we get for free.
    """

    bank_line_id: str
    value_date: date
    amount_paise: Paise
    narration: str
    bank_ref: str


@dataclass(frozen=True)
class SourceBundle:
    """Everything the matcher is allowed to see. No labels, ever."""

    orders: tuple[Order, ...]
    payments: tuple[PGPayment, ...]
    adjustments: tuple[PGAdjustment, ...]
    settlements: tuple[Settlement, ...]
    bank_lines: tuple[BankLine, ...]

    def payments_by_settlement(self, settlement_id: str) -> tuple[PGPayment, ...]:
        return tuple(p for p in self.payments if p.settlement_id == settlement_id)

    def adjustments_by_settlement(self, settlement_id: str) -> tuple[PGAdjustment, ...]:
        return tuple(a for a in self.adjustments if a.settlement_id == settlement_id)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "orders": len(self.orders),
            "payments": len(self.payments),
            "adjustments": len(self.adjustments),
            "settlements": len(self.settlements),
            "bank_lines": len(self.bank_lines),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth -- generator and scorer only
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InjectedDefect:
    """One injected defect and every entity it damages.

    `affected_ids` exists because some defects break two things at once. A T+2
    timing spill leaves one batch short and the next one long; a duplicated UTR
    makes both the original line and its restatement unmatchable. Recording
    only the injection site would leave the collateral exception looking
    unexplained, which would make the scorer's reconciliation check noisy
    exactly where the data is most interesting.
    """

    defect: DefectClass
    entity_kind: str  # order | payment | settlement | bank_line
    entity_id: str
    note: str
    affected_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.affected_ids:
            object.__setattr__(self, "affected_ids", (self.entity_id,))


@dataclass(frozen=True)
class GroundTruth:
    #: order_id -> payment_id, for orders that have exactly one payment
    leg1: dict[str, str]
    #: bank_line_id -> settlement_id
    leg2: dict[str, str]
    #: settlement_id -> the payment_ids genuinely inside that batch
    members: dict[str, tuple[str, ...]]
    defects: tuple[InjectedDefect, ...]

    def defects_by_class(self) -> dict[DefectClass, int]:
        out: dict[DefectClass, int] = {}
        for d in self.defects:
            out[d.defect] = out.get(d.defect, 0) + 1
        return out

    def defect_on(self, entity_id: str) -> InjectedDefect | None:
        for d in self.defects:
            if d.entity_id == entity_id:
                return d
        return None


@dataclass(frozen=True)
class LabelledBatch:
    """Generator output. Held by the scorer; never handed to a matcher."""

    sources: SourceBundle
    truth: GroundTruth
    seed: int
    profile: str


# ─────────────────────────────────────────────────────────────────────────────
# Matcher output
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArithmeticProof:
    """Why a match was accepted, in a form that can be re-checked from the record alone."""

    expression: str
    lhs_paise: Paise
    rhs_paise: Paise
    tolerance_paise: Paise

    @property
    def residual_paise(self) -> Paise:
        return self.lhs_paise - self.rhs_paise

    @property
    def closes(self) -> bool:
        return abs(self.residual_paise) <= self.tolerance_paise


@dataclass(frozen=True)
class MatchRecord:
    """One accepted match, with everything needed to replay the decision."""

    match_id: str
    leg: int
    tier: str  # T0 | T1 | T2
    rule_id: str
    left_ids: tuple[str, ...]
    right_ids: tuple[str, ...]
    confidence: float
    proof: ArithmeticProof | None
    input_hash: str
    created_at: datetime
    #: Rows that were not linked to this batch in the source data but were
    #: needed to make the arithmetic close. Recorded by id because "the credit
    #: reconciles once you count these three rows" is the actual finding, and
    #: an audit trail that proves a total without naming what went into it
    #: cannot be checked by the person who has to sign it off.
    hypothesised_ids: tuple[str, ...] = ()

    @property
    def pair(self) -> tuple[str, str]:
        """The canonical (left, right) identity of this match, for scoring."""
        return (self.left_ids[0], self.right_ids[0])


@dataclass(frozen=True)
class ReconException:
    """Something the system declined to match, and why.

    `suspected_class` is the matcher's own read of what went wrong. It is a
    guess made without access to ground truth, and the scorer reports how often
    that guess is right -- separately from whether the item was matched.
    """

    exception_id: str
    leg: int
    entity_kind: str
    entity_id: str
    reason: str
    #: The counterparty this item was adjudicated against, when there was
    #: one -- the settlement a credit joined before its arithmetic failed.
    #: Later tiers need it to resume where the previous one stopped, and
    #: carrying it explicitly beats re-parsing it out of `reason`.
    related_id: str | None = None
    residual_paise: Paise | None = None
    suspected_class: DefectClass | None = None
    escalated_from_tier: str = "T0"


@dataclass
class ReconResult:
    """The output of one pipeline run at one rung of the baseline ladder."""

    rung: str
    matches: list[MatchRecord] = field(default_factory=list)
    exceptions: list[ReconException] = field(default_factory=list)

    def matches_for_leg(self, leg: int) -> list[MatchRecord]:
        return [m for m in self.matches if m.leg == leg]

    def exceptions_for_leg(self, leg: int) -> list[ReconException]:
        return [e for e in self.exceptions if e.leg == leg]


# ─────────────────────────────────────────────────────────────────────────────
# Hashing -- audit records must be replayable
# ─────────────────────────────────────────────────────────────────────────────


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, DefectClass):
        return obj.value
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def stable_hash(*objs: Any) -> str:
    """A deterministic hash of the inputs a decision was made from.

    Recorded on every match so the audit log can prove which bytes produced
    which verdict. Sorted keys and ISO timestamps keep it stable across runs
    and across machines.
    """
    payload = json.dumps(
        [asdict(o) if hasattr(o, "__dataclass_fields__") else o for o in objs],
        sort_keys=True,
        default=_json_default,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
