"""Synthetic batch generator with labelled defect injection.

Builds a clean, internally consistent three-source world -- orders, a gateway
settlement report, and a bank statement -- and then injects a controlled number
of defects drawn from the taxonomy in `recoagent.defects`, recording a label
for every one.

Two properties matter more than realism here:

1. **Nothing downstream can see this module.** The generator emits a
   `LabelledBatch`; matchers receive only `batch.sources`. See the note in
   `recoagent.schemas` and `tests/test_independence.py`.

2. **Defect counts are exact, not sampled.** A rate of 0.10 over 40
   settlements injects exactly 4, not "about 4". This is what makes the
   scorer's reconciliation check -- injected counts versus explained
   exceptions -- a real assertion rather than a statistical impression.

The one modelling decision worth stating out loud: the settlement header's
reported net is *not* used as proof. The proof is re-derived from the payment
rows themselves. The header is the gateway's claim about what it paid; the
whole job of reconciliation is to check that claim against the money that
actually arrived. This is the same "replay it through the ledger rather than
trusting the reported total" discipline that FinBalance found LLMs failing at
by 26-41 percentage points.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from .defects import DefectClass
from .money import FeeSchedule, Paise, bps_of
from .schemas import (
    BankLine,
    GroundTruth,
    InjectedDefect,
    LabelledBatch,
    Order,
    PGAdjustment,
    PGPayment,
    Settlement,
    SourceBundle,
)

BASE_DATE = datetime(2026, 7, 1, 9, 0, 0)

#: How many days of trading a batch covers.
SPAN_DAYS = 20

BANK_NARRATION_TEMPLATES = (
    "NEFT CR-{ifsc}-RAZORPAY SOFTWARE PVT LTD-{utr}",
    "IMPS/{utr}/RAZORPAY/SETTLEMENT",
    "UPI/CR/{utr}/RAZORPAYSOFT/PAYOUT",
    "RTGS CR {utr} RAZORPAY SOFTWARE PRIVATE LIMITED",
)

IFSC_CODES = ("HDFC0000123", "ICIC0000456", "UTIB0000789", "KKBK0000321")

#: Rough share of Indian online payment volume by method. Skews heavily to UPI,
#: which matters: UPI carries zero MDR, so a fee model that assumes every
#: payment is charged will be wrong on most of the book.
METHOD_WEIGHTS = {
    "upi": 0.58,
    "card_domestic": 0.16,
    "netbanking": 0.11,
    "wallet": 0.07,
    "rupay_debit": 0.05,
    "emi": 0.02,
    "card_international": 0.01,
}


@dataclass(frozen=True)
class DefectMix:
    """Defect rates, expressed as a fraction of the eligible population.

    Leg-2 rates are fractions of settlements; leg-1 rates are fractions of orders.
    """

    rates: dict[DefectClass, float]
    label: str

    @classmethod
    def clean(cls) -> DefectMix:
        return cls(rates={}, label="clean")

    @classmethod
    def dev(cls) -> DefectMix:
        """The mix used while building and tuning.

        Rates are calibrated to a plausible mid-size merchant book: roughly 23%
        of settlement batches carry some defect, and about 5% of orders. A
        healthy production reconciliation runs at 85-95% straight-through, so a
        synthetic set where most batches are broken would not be a hard problem
        -- it would be a different and less useful one, and it would make the
        LLM tier look far more valuable than it is.
        """
        return cls(
            label="dev",
            rates={
                # Leg 2 -- fractions of settlements, summing to ~0.23
                DefectClass.REFUND_NETTED: 0.045,
                DefectClass.ROUNDING_DRIFT: 0.030,
                DefectClass.FEE_TAX_VARIANCE: 0.025,
                DefectClass.ADJUSTMENT_ENTRY: 0.022,
                DefectClass.TIMING_SPILL: 0.022,
                DefectClass.CHARGEBACK_NETTED: 0.020,
                DefectClass.NARRATION_TRUNCATION: 0.020,
                DefectClass.FX_CONVERSION: 0.020,
                DefectClass.MISSING_BANK_LINE: 0.015,
                DefectClass.DUPLICATE_UTR: 0.012,
                # Leg 1 -- fractions of orders, summing to ~0.046
                DefectClass.PARTIAL_CAPTURE: 0.028,
                DefectClass.DUPLICATE_PAYMENT: 0.018,
            },
        )

    @classmethod
    def holdout(cls) -> DefectMix:
        """A deliberately different mix, never tuned against.

        Held-out evaluation changes both the seed and the distribution. The
        totals per leg are held roughly constant so the comparison stays fair,
        but the composition is inverted: classes the dev mix made common are
        made rare here and vice versa. A system that has quietly overfitted the
        dev proportions -- a threshold tuned to how often rounding drift shows
        up, say -- will show it as a drop here rather than passing unnoticed.
        """
        return cls(
            label="holdout",
            rates={
                # Leg 2 -- same ~0.23 total, redistributed
                DefectClass.REFUND_NETTED: 0.018,
                DefectClass.ROUNDING_DRIFT: 0.014,
                DefectClass.FEE_TAX_VARIANCE: 0.040,
                DefectClass.ADJUSTMENT_ENTRY: 0.038,
                DefectClass.TIMING_SPILL: 0.036,
                DefectClass.CHARGEBACK_NETTED: 0.032,
                DefectClass.NARRATION_TRUNCATION: 0.010,
                DefectClass.FX_CONVERSION: 0.012,
                DefectClass.MISSING_BANK_LINE: 0.022,
                DefectClass.DUPLICATE_UTR: 0.008,
                # Leg 1 -- same ~0.046 total, inverted
                DefectClass.PARTIAL_CAPTURE: 0.016,
                DefectClass.DUPLICATE_PAYMENT: 0.030,
            },
        )


@dataclass(frozen=True)
class GeneratorConfig:
    n_orders: int = 500
    seed: int = 7
    batch_size_min: int = 6
    batch_size_max: int = 18
    #: Settlements per day. Leave None for the fixed small-batch default, which
    #: is a stress case rather than a realistic one: it holds batch size constant
    #: as volume grows, so a 20,000-order book produces 78 payouts a day. Real
    #: gateways settle on a T+2 cycle -- roughly one payout a day -- and a
    #: merchant doing ten times the volume gets batches ten times bigger, not ten
    #: times as many. Set this (2.0 is typical) to model that instead. It matters
    #: for more than realism: the solver's candidate pool is drawn from a date
    #: window, so settlement density drives its search space directly.
    settlements_per_day: float | None = None
    linked_adjustment_rate: float = 0.10  # clean, correctly-linked adjustments
    mix: DefectMix = None  # type: ignore[assignment]
    fees: FeeSchedule = None  # type: ignore[assignment]

    def resolved(self) -> GeneratorConfig:
        return replace(
            self,
            mix=self.mix or DefectMix.dev(),
            fees=self.fees or FeeSchedule.default(),
        )


class _World:
    """Mutable scratch space while the batch is under construction."""

    def __init__(self, rng: random.Random, fees: FeeSchedule) -> None:
        self.rng = rng
        self.fees = fees
        self.orders: list[Order] = []
        self.payments: list[PGPayment] = []
        self.adjustments: list[PGAdjustment] = []
        self.settlements: list[Settlement] = []
        self.bank_lines: list[BankLine] = []
        self.members: dict[str, list[str]] = {}
        self.leg1: dict[str, str] = {}
        self.leg2: dict[str, str] = {}
        self.defects: list[InjectedDefect] = []
        self.claimed: set[str] = set()  # entities already carrying a defect

    # -- helpers ---------------------------------------------------------

    def payment(self, payment_id: str) -> PGPayment:
        return next(p for p in self.payments if p.payment_id == payment_id)

    def settlement(self, settlement_id: str) -> Settlement:
        return next(s for s in self.settlements if s.settlement_id == settlement_id)

    def orphan_booked_at(self, settlement_id: str) -> datetime:
        """A plausible booking time for a row netted into this batch.

        Orphans must carry realistic timestamps or a date-window filter over
        the candidate pool is meaningless -- every orphan would sit in every
        window, and the solver's search space would be the whole book rather
        than the handful of rows genuinely near the batch.
        """
        settled = self.settlement(settlement_id).settled_at
        return settled - timedelta(hours=self.rng.randint(2, 60))

    def replace_payment(self, payment_id: str, **changes) -> None:
        for i, p in enumerate(self.payments):
            if p.payment_id == payment_id:
                self.payments[i] = replace(p, **changes)
                return
        raise KeyError(payment_id)

    def bank_line_for(self, settlement_id: str) -> BankLine | None:
        line_id = next(
            (lid for lid, sid in self.leg2.items() if sid == settlement_id), None
        )
        if line_id is None:
            return None
        return next(b for b in self.bank_lines if b.bank_line_id == line_id)

    def shift_bank_line(self, settlement_id: str, delta: Paise) -> bool:
        """Adjust the credit that actually landed. Returns False if no line exists."""
        line = self.bank_line_for(settlement_id)
        if line is None:
            return False
        for i, b in enumerate(self.bank_lines):
            if b.bank_line_id == line.bank_line_id:
                self.bank_lines[i] = replace(b, amount_paise=b.amount_paise + delta)
                return True
        return False

    def claim(self, *entity_ids: str) -> bool:
        """Reserve entities for a defect. Returns False if any is already taken."""
        if any(e in self.claimed for e in entity_ids):
            return False
        self.claimed.update(entity_ids)
        return True

    def record(
        self,
        defect: DefectClass,
        kind: str,
        entity_id: str,
        note: str,
        affected_ids: tuple[str, ...] = (),
    ) -> None:
        self.defects.append(
            InjectedDefect(
                defect=defect,
                entity_kind=kind,
                entity_id=entity_id,
                note=note,
                affected_ids=affected_ids,
            )
        )


def _weighted_method(rng: random.Random) -> str:
    r = rng.random()
    cumulative = 0.0
    for method, weight in METHOD_WEIGHTS.items():
        cumulative += weight
        if r <= cumulative:
            return method
    return "upi"


def _order_amount(rng: random.Random) -> Paise:
    """Log-ish distribution: many small orders, a long tail of large ones."""
    bucket = rng.random()
    if bucket < 0.55:
        rupees = rng.randint(99, 1_500)
    elif bucket < 0.88:
        rupees = rng.randint(1_500, 12_000)
    else:
        rupees = rng.randint(12_000, 150_000)
    return rupees * 100 + rng.choice([0, 0, 0, 50, 99, 25])


def _utr(rng: random.Random) -> str:
    return f"{rng.randint(10**11, 10**12 - 1)}"


def _build_clean(w: _World, cfg: GeneratorConfig) -> None:
    rng = w.rng

    for i in range(cfg.n_orders):
        oid = f"order_{i:05d}"
        created = BASE_DATE + timedelta(
            days=rng.randint(0, SPAN_DAYS), minutes=rng.randint(0, 1439)
        )
        amount = _order_amount(rng)
        w.orders.append(
            Order(
                order_id=oid,
                customer_id=f"cust_{rng.randint(1, max(2, cfg.n_orders // 3)):05d}",
                invoice_no=f"INV-2026-{i:05d}",
                amount_paise=amount,
                currency="INR",
                created_at=created,
            )
        )

        method = _weighted_method(rng)
        fee, tax = cfg.fees.fee_and_tax(amount, method)
        pid = f"pay_{i:05d}"
        w.payments.append(
            PGPayment(
                payment_id=pid,
                order_id=oid,
                gross_paise=amount,
                fee_paise=fee,
                tax_paise=tax,
                method=method,
                status="captured",
                settlement_id=None,
                captured_at=created + timedelta(minutes=rng.randint(1, 30)),
                currency="INR",
            )
        )
        w.leg1[oid] = pid

    # Batch payments into settlements in capture order, T+2.
    ordered = sorted(w.payments, key=lambda p: p.captured_at)
    idx = 0
    batch_no = 0

    if cfg.settlements_per_day:
        target = max(1, round(SPAN_DAYS * cfg.settlements_per_day))
        per_batch = max(1, len(ordered) // target)
        size_lo, size_hi = max(1, int(per_batch * 0.7)), max(2, int(per_batch * 1.3))
    else:
        size_lo, size_hi = cfg.batch_size_min, cfg.batch_size_max

    while idx < len(ordered):
        size = rng.randint(size_lo, size_hi)
        chunk = ordered[idx : idx + size]
        idx += size
        if not chunk:
            break

        sid = f"setl_{batch_no:04d}"
        batch_no += 1
        settled_at = chunk[-1].captured_at + timedelta(days=2)

        for p in chunk:
            w.replace_payment(p.payment_id, settlement_id=sid)
        w.members[sid] = [p.payment_id for p in chunk]

        # A minority of batches carry a correctly-linked adjustment. These are
        # part of the clean world: the matcher can see them and must include
        # them in the sum, so a matcher that ignores adjustments entirely will
        # fail on these even with no defects injected at all.
        adj_total = 0
        if rng.random() < cfg.linked_adjustment_rate:
            victim = rng.choice(chunk)
            refund = -min(victim.gross_paise, _order_amount(rng) // 3)
            w.adjustments.append(
                PGAdjustment(
                    adjustment_id=f"adj_{sid}_0",
                    settlement_id=sid,
                    kind="refund",
                    payment_id=victim.payment_id,
                    amount_paise=refund,
                    booked_at=settled_at - timedelta(hours=rng.randint(1, 40)),
                )
            )
            adj_total += refund

        net = sum(p.net_paise for p in chunk) + adj_total
        utr = _utr(rng)
        w.settlements.append(
            Settlement(
                settlement_id=sid,
                utr=utr,
                settled_at=settled_at,
                net_paise=net,
                status="processed",
            )
        )

        line_id = f"bank_{batch_no - 1:04d}"
        narration = rng.choice(BANK_NARRATION_TEMPLATES).format(
            utr=utr, ifsc=rng.choice(IFSC_CODES)
        )
        w.bank_lines.append(
            BankLine(
                bank_line_id=line_id,
                value_date=settled_at.date(),
                amount_paise=net,
                narration=narration,
                bank_ref=f"REF{rng.randint(10**7, 10**8 - 1)}",
            )
        )
        w.leg2[line_id] = sid


# ─────────────────────────────────────────────────────────────────────────────
# Defect injectors. Each takes one target and mutates the world, or returns
# False if the target is unsuitable so the caller can try another.
# ─────────────────────────────────────────────────────────────────────────────


def _inject_refund_netted(w: _World, sid: str) -> bool:
    members = w.members.get(sid, [])
    if not members or not w.claim(sid):
        return False
    victim = w.rng.choice(members)
    amount = -max(1000, w.payment(victim).gross_paise // w.rng.randint(2, 4))
    # Orphaned: the refund reduced the payout but carries no settlement link,
    # so summing the batch's linked rows overstates what should have arrived.
    w.adjustments.append(
        PGAdjustment(
            adjustment_id=f"adj_orphan_{sid}",
            settlement_id=None,  # type: ignore[arg-type]
            kind="refund",
            payment_id=victim,
            amount_paise=amount,
            booked_at=w.orphan_booked_at(sid),
        )
    )
    if not w.shift_bank_line(sid, amount):
        return False
    w.record(
        DefectClass.REFUND_NETTED,
        "settlement",
        sid,
        f"orphan refund of {amount} paise against {victim}",
    )
    return True


def _inject_chargeback_netted(w: _World, sid: str) -> bool:
    members = w.members.get(sid, [])
    if not members or not w.claim(sid):
        return False
    victim = w.rng.choice(members)
    cb = -w.payment(victim).gross_paise
    fee = -150_00  # Rs 150 dispute handling fee
    for i, (kind, amt) in enumerate((("chargeback", cb), ("dispute_fee", fee))):
        w.adjustments.append(
            PGAdjustment(
                adjustment_id=f"adj_cb_{sid}_{i}",
                settlement_id=None,  # type: ignore[arg-type]
                kind=kind,
                payment_id=victim if kind == "chargeback" else None,
                amount_paise=amt,
                booked_at=w.orphan_booked_at(sid),
            )
        )
    if not w.shift_bank_line(sid, cb + fee):
        return False
    w.record(
        DefectClass.CHARGEBACK_NETTED,
        "settlement",
        sid,
        f"chargeback {cb} + dispute fee {fee} paise, both unlinked",
    )
    return True


def _inject_adjustment_entry(w: _World, sid: str) -> bool:
    if not w.claim(sid):
        return False
    amount = w.rng.choice([-1, 1]) * w.rng.randint(500, 9_000) * 100
    w.adjustments.append(
        PGAdjustment(
            adjustment_id=f"adj_manual_{sid}",
            settlement_id=None,  # type: ignore[arg-type]
            kind=w.rng.choice(["reversal", "platform_fee"]),
            payment_id=None,  # no payment to tie it to at all
            amount_paise=amount,
            booked_at=w.orphan_booked_at(sid),
        )
    )
    if not w.shift_bank_line(sid, amount):
        return False
    w.record(
        DefectClass.ADJUSTMENT_ENTRY,
        "settlement",
        sid,
        f"manual adjustment of {amount} paise with no payment reference",
    )
    return True


def _inject_fee_tax_variance(w: _World, sid: str) -> bool:
    """A mid-cycle repricing: one method's MDR changed, and the report is stale.

    Applied as a uniform rate change across every payment of one fee-bearing
    method in the batch, because that is what a repricing is. An earlier version
    added an arbitrary extra to half the charged payments, which no rate could
    reproduce -- so the defect had no coherent explanation and the LLM tier was
    being asked to find one that did not exist. It kept declining with "fees
    match the schedule", which was the correct read of a badly modelled defect.
    """
    members = w.members.get(sid, [])
    if not members or not w.claim(sid):
        return False

    by_method: dict[str, list] = {}
    for pid in members:
        p = w.payment(pid)
        if w.fees.mdr_for(p.method) > 0:
            by_method.setdefault(p.method, []).append(p)
    if not by_method:
        return False

    method = max(by_method, key=lambda m: len(by_method[m]))
    targets = by_method[method]
    scheduled_bps = w.fees.mdr_for(method)
    actual_bps = scheduled_bps + w.rng.choice([25, 40, 50, 75])

    delta = 0
    for p in targets:
        fee = bps_of(p.gross_paise, actual_bps)
        charged = fee + bps_of(fee, w.fees.gst_bps)
        delta += (p.fee_paise + p.tax_paise) - charged
    if delta == 0 or not w.shift_bank_line(sid, delta):
        return False

    w.record(
        DefectClass.FEE_TAX_VARIANCE,
        "settlement",
        sid,
        f"{method} repriced {scheduled_bps}->{actual_bps} bps across "
        f"{len(targets)} payments; report still shows the old rate",
    )
    return True


def _inject_rounding_drift(w: _World, sid: str) -> bool:
    if not w.claim(sid):
        return False
    drift = w.rng.choice([-9, -5, -3, -2, -1, 1, 2, 3, 5, 7])
    if not w.shift_bank_line(sid, drift):
        return False
    w.record(
        DefectClass.ROUNDING_DRIFT,
        "settlement",
        sid,
        f"sub-rupee drift of {drift} paise from per-step rounding",
    )
    return True


def _inject_timing_spill(w: _World, sid: str) -> bool:
    members = w.members.get(sid, [])
    if len(members) < 3 or not w.claim(sid):
        return False
    # A payment reported inside this batch actually settled in a later cycle:
    # this batch's credit is short by its net, and the later one is long.
    stray = w.rng.choice(members)
    net = w.payment(stray).net_paise
    later = [s for s in w.settlements if s.settlement_id > sid and s.settlement_id not in w.claimed]
    if not later:
        return False
    target = later[0].settlement_id
    if not w.shift_bank_line(sid, -net):
        return False
    if not w.shift_bank_line(target, net):
        w.shift_bank_line(sid, net)  # roll back
        return False
    w.claim(target)
    w.record(
        DefectClass.TIMING_SPILL,
        "settlement",
        sid,
        f"{stray} ({net} paise) reported in {sid} but credited with {target}",
        affected_ids=(sid, target),
    )
    return True


def _inject_narration_truncation(w: _World, sid: str) -> bool:
    line = w.bank_line_for(sid)
    if line is None or not w.claim(sid):
        return False
    # Clip *inside* the UTR rather than at an arbitrary column, so the defect
    # actually destroys the join key. Truncating at a fixed width sometimes
    # leaves the UTR intact, which would make this class silently harmless and
    # quietly inflate the match rate.
    settlement = next(s for s in w.settlements if s.settlement_id == sid)
    pos = line.narration.find(settlement.utr)
    if pos == -1:
        return False
    keep = pos + w.rng.randint(2, max(2, len(settlement.utr) - 3))
    cut = line.narration[:keep]
    for i, b in enumerate(w.bank_lines):
        if b.bank_line_id == line.bank_line_id:
            w.bank_lines[i] = replace(b, narration=cut)
            break
    w.record(
        DefectClass.NARRATION_TRUNCATION,
        "bank_line",
        line.bank_line_id,
        "narration clipped by the bank; UTR no longer extractable",
    )
    return True


def _inject_missing_bank_line(w: _World, sid: str) -> bool:
    line = w.bank_line_for(sid)
    if line is None or not w.claim(sid):
        return False
    w.bank_lines = [b for b in w.bank_lines if b.bank_line_id != line.bank_line_id]
    del w.leg2[line.bank_line_id]
    for i, s in enumerate(w.settlements):
        if s.settlement_id == sid:
            w.settlements[i] = replace(s, status="on_hold")
            break
    w.record(
        DefectClass.MISSING_BANK_LINE,
        "settlement",
        sid,
        "settlement held into reserve balance; no credit ever landed",
    )
    return True


def _inject_duplicate_utr(w: _World, sid: str) -> bool:
    line = w.bank_line_for(sid)
    if line is None or not w.claim(sid):
        return False
    ghost_id = f"{line.bank_line_id}_dup"
    w.bank_lines.append(
        replace(line, bank_line_id=ghost_id, bank_ref=f"{line.bank_ref}D")
    )
    # Deliberately NOT added to leg2 truth: the duplicate matches nothing.
    w.record(
        DefectClass.DUPLICATE_UTR,
        "bank_line",
        ghost_id,
        f"same UTR restated on a second line, cloned from {line.bank_line_id}",
        affected_ids=(ghost_id, line.bank_line_id),
    )
    return True


def _inject_fx_conversion(w: _World, sid: str) -> bool:
    members = w.members.get(sid, [])
    if not members or not w.claim(sid):
        return False
    pid = w.rng.choice(members)
    p = w.payment(pid)
    rate = round(w.rng.uniform(82.5, 89.5), 4)
    # The bank converted at a rate slightly off the one implied by the report.
    slip = int(p.gross_paise * w.rng.uniform(0.004, 0.02))
    w.replace_payment(
        pid, method="card_international", currency="USD", fx_rate=rate
    )
    fee, tax = w.fees.fee_and_tax(p.gross_paise, "card_international")
    w.replace_payment(pid, fee_paise=fee, tax_paise=tax)
    if not w.shift_bank_line(sid, -slip):
        return False
    w.record(
        DefectClass.FX_CONVERSION,
        "settlement",
        sid,
        f"{pid} settled at FX {rate}; credit differs by {slip} paise from the reported figure",
    )
    return True


def _inject_partial_capture(w: _World, oid: str) -> bool:
    pid = w.leg1.get(oid)
    if pid is None or not w.claim(oid, pid):
        return False
    p = w.payment(pid)
    captured = int(p.gross_paise * w.rng.uniform(0.35, 0.85))
    fee, tax = w.fees.fee_and_tax(captured, p.method)
    delta = (captured - fee - tax) - p.net_paise
    w.replace_payment(
        pid,
        gross_paise=captured,
        fee_paise=fee,
        tax_paise=tax,
        status="partially_captured",
    )
    if p.settlement_id:
        w.shift_bank_line(p.settlement_id, delta)
    w.record(
        DefectClass.PARTIAL_CAPTURE,
        "payment",
        pid,
        f"captured {captured} of an authorised {p.gross_paise} paise",
    )
    return True


def _inject_duplicate_payment(w: _World, oid: str) -> bool:
    pid = w.leg1.get(oid)
    if pid is None or not w.claim(oid, pid):
        return False
    p = w.payment(pid)
    ghost_id = f"{pid}_retry"
    w.payments.append(
        replace(
            p,
            payment_id=ghost_id,
            settlement_id=None,  # the retry was never settled
            status="captured",
            captured_at=p.captured_at + timedelta(minutes=3),
        )
    )
    w.record(
        DefectClass.DUPLICATE_PAYMENT,
        "order",
        oid,
        f"customer retried; {ghost_id} duplicates {pid} against the same order",
    )
    return True


LEG2_INJECTORS = {
    DefectClass.REFUND_NETTED: _inject_refund_netted,
    DefectClass.CHARGEBACK_NETTED: _inject_chargeback_netted,
    DefectClass.ADJUSTMENT_ENTRY: _inject_adjustment_entry,
    DefectClass.FEE_TAX_VARIANCE: _inject_fee_tax_variance,
    DefectClass.ROUNDING_DRIFT: _inject_rounding_drift,
    DefectClass.TIMING_SPILL: _inject_timing_spill,
    DefectClass.NARRATION_TRUNCATION: _inject_narration_truncation,
    DefectClass.MISSING_BANK_LINE: _inject_missing_bank_line,
    DefectClass.DUPLICATE_UTR: _inject_duplicate_utr,
    DefectClass.FX_CONVERSION: _inject_fx_conversion,
}

LEG1_INJECTORS = {
    DefectClass.PARTIAL_CAPTURE: _inject_partial_capture,
    DefectClass.DUPLICATE_PAYMENT: _inject_duplicate_payment,
}


def _apply_defects(w: _World, mix: DefectMix) -> None:
    """Inject an exact count per class, in a fixed class order for determinism."""
    settlement_ids = [s.settlement_id for s in w.settlements]
    order_ids = [o.order_id for o in w.orders]

    for defect in DefectClass:
        rate = mix.rates.get(defect, 0.0)
        if rate <= 0:
            continue

        if defect in LEG2_INJECTORS:
            pool, injector = settlement_ids, LEG2_INJECTORS[defect]
        elif defect in LEG1_INJECTORS:
            pool, injector = order_ids, LEG1_INJECTORS[defect]
        else:
            continue

        target_count = int(round(rate * len(pool)))
        if target_count == 0:
            continue

        candidates = [c for c in pool if c not in w.claimed]
        w.rng.shuffle(candidates)

        injected = 0
        for candidate in candidates:
            if injected >= target_count:
                break
            if injector(w, candidate):
                injected += 1


def generate(cfg: GeneratorConfig | None = None) -> LabelledBatch:
    """Build one labelled batch. Deterministic in `cfg.seed`."""
    cfg = (cfg or GeneratorConfig()).resolved()
    rng = random.Random(cfg.seed)
    w = _World(rng, cfg.fees)

    _build_clean(w, cfg)
    _apply_defects(w, cfg.mix)

    sources = SourceBundle(
        orders=tuple(w.orders),
        payments=tuple(sorted(w.payments, key=lambda p: p.payment_id)),
        adjustments=tuple(sorted(w.adjustments, key=lambda a: a.adjustment_id)),
        settlements=tuple(w.settlements),
        bank_lines=tuple(sorted(w.bank_lines, key=lambda b: b.bank_line_id)),
    )
    truth = GroundTruth(
        leg1=dict(w.leg1),
        leg2=dict(w.leg2),
        members={k: tuple(v) for k, v in w.members.items()},
        defects=tuple(w.defects),
    )
    return LabelledBatch(
        sources=sources, truth=truth, seed=cfg.seed, profile=cfg.mix.label
    )
