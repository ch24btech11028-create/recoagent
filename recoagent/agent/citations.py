"""What a proposer may cite, and how citations become money.

This module exists because the previous design was unsound and a reviewer caught
it. The model used to return `(label, amount_paise)` pairs, and the tier turned
them into ledger rows. Since the model chose the amount, it could always choose
the residual -- and a proposal of "there was an adjustment of exactly this
amount" closed the arithmetic every time. Reproduced: 7 of 7 cases resolved on a
fabricated number, false-match rate still reporting 0.00%.

The gate was checking that the model's own number made the model's own total add
up. That is not verification.

So the model no longer supplies amounts. It supplies **citations**: pointers at
evidence that already exists, or a rule to apply. Every rupee is then computed by
the code in this module from the source data:

- `CitedAdjustment` names an unlinked row by id. The amount comes from that row.
  A row that does not exist, or is already linked to a batch, resolves to an
  error rather than to money.
- `FeeVarianceClaim` names payments and a rate. The variance is recomputed from
  the fee schedule; the model cannot state the delta.
- `FxClaim` names one international payment and a rate. Same.

If a citation cannot be resolved against the sources, the proposal is not
downgraded or partially accepted -- it is rejected. A proposal that cites
nothing resolvable is a hypothesis for a human, not a match.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..money import FeeSchedule, Paise, bps_of
from ..schemas import PGAdjustment, Settlement, SourceBundle

MAX_BPS = 2000


@dataclass(frozen=True)
class CitedAdjustment:
    """An unlinked row the model believes belongs to this batch."""

    adjustment_id: str
    rationale: str = ""


@dataclass(frozen=True)
class FeeVarianceClaim:
    """A repricing: these payments were charged at a different MDR than reported."""

    payment_ids: tuple[str, ...]
    actual_mdr_bps: int
    rationale: str = ""


@dataclass(frozen=True)
class FxClaim:
    """An international payment converted at a rate the report does not carry."""

    payment_id: str
    actual_rate_pct_of_gross: float
    rationale: str = ""


Citation = CitedAdjustment | FeeVarianceClaim | FxClaim


@dataclass(frozen=True)
class RateBook:
    """Authoritative rates, when the merchant actually has them.

    A repricing notice from the gateway, or the bank's FX advice. Without one,
    a claimed rate is the model's guess -- reasonable, checkable against the
    arithmetic, but not a fact anyone can point at.
    """

    #: method -> the set of MDR rates known to have applied in this period.
    mdr_bps: dict[str, set[int]] = field(default_factory=dict)
    #: payment_id -> the conversion slip the bank advised, as a % of gross.
    fx_pct: dict[str, float] = field(default_factory=dict)
    fx_tolerance_pct: float = 0.01

    def confirms_mdr(self, method: str, bps: int) -> bool:
        return bps in self.mdr_bps.get(method, set())

    def confirms_fx(self, payment_id: str, pct: float) -> bool:
        known = self.fx_pct.get(payment_id)
        return known is not None and abs(known - pct) <= self.fx_tolerance_pct


@dataclass(frozen=True)
class ResolvedRow:
    """One citation, turned into money by code rather than by the model."""

    source: str  # adjustment | fee_variance | fx
    cited_ids: tuple[str, ...]
    amount_paise: Paise
    derivation: str
    #: The numbers behind `derivation`, kept structured as well as prose.
    #: `derivation` is an audit string and reads like one; anything that needs
    #: to *say* what happened in another register -- `recoagent.plain`, for the
    #: merchant -- would otherwise have to parse it back out, and a sentence
    #: assembled by re-reading a sentence is one edit away from being wrong.
    #: Keys are per `source`: fee_variance carries `method`, `claimed_bps` and
    #: `schedule_bps`; fx carries `pct`.
    detail: dict = field(default_factory=dict)
    #: True when the row rests entirely on data that already existed. False when
    #: a parameter came from the model -- a claimed MDR or FX rate. The
    #: arithmetic is computed by code either way, but an unverified rate is a
    #: hypothesis: plausible, self-consistent, and nobody has confirmed it. The
    #: tier books those as needing approval rather than as reconciled.
    verified: bool = True


@dataclass(frozen=True)
class Resolution:
    rows: tuple[ResolvedRow, ...]
    errors: tuple[str, ...]

    @property
    def fully_verified(self) -> bool:
        """Every row rests on data that existed before the model was asked."""
        return bool(self.rows) and all(r.verified for r in self.rows)

    @property
    def unverified_reasons(self) -> tuple[str, ...]:
        return tuple(r.derivation for r in self.rows if not r.verified)

    @property
    def ok(self) -> bool:
        return not self.errors and bool(self.rows)

    @property
    def total_paise(self) -> Paise:
        return sum(r.amount_paise for r in self.rows)

    @property
    def cited_ids(self) -> tuple[str, ...]:
        out: list[str] = []
        for r in self.rows:
            out.extend(r.cited_ids)
        return tuple(out)


def resolve(
    sources: SourceBundle,
    settlement: Settlement,
    citations: list[Citation],
    fees: FeeSchedule | None = None,
    rate_book: RateBook | None = None,
) -> Resolution:
    """Turn citations into rows, or into the reasons they could not be trusted."""
    fees = fees or FeeSchedule.default()
    members = {p.payment_id: p for p in sources.payments_by_settlement(settlement.settlement_id)}
    unlinked: dict[str, PGAdjustment] = {
        a.adjustment_id: a for a in sources.unlinked_adjustments
    }

    rows: list[ResolvedRow] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for c in citations:
        if isinstance(c, CitedAdjustment):
            row = unlinked.get(c.adjustment_id)
            if row is None:
                errors.append(
                    f"cited adjustment {c.adjustment_id!r} is not an unlinked row "
                    "in this book"
                )
                continue
            if c.adjustment_id in seen_ids:
                errors.append(f"adjustment {c.adjustment_id!r} cited twice")
                continue
            seen_ids.add(c.adjustment_id)
            rows.append(ResolvedRow(
                source="adjustment",
                cited_ids=(c.adjustment_id,),
                amount_paise=row.amount_paise,  # from the source row, never the model
                derivation=f"{row.kind} row {c.adjustment_id} booked {row.booked_at:%Y-%m-%d}",
            ))

        elif isinstance(c, FeeVarianceClaim):
            if not 0 <= c.actual_mdr_bps <= MAX_BPS:
                errors.append(f"claimed MDR {c.actual_mdr_bps} bps is out of range")
                continue
            targets = []
            for pid in c.payment_ids:
                p = members.get(pid)
                if p is None:
                    errors.append(f"payment {pid!r} is not in {settlement.settlement_id}")
                    break
                targets.append(p)
            else:
                if not targets:
                    errors.append("fee variance cited no payments")
                    continue
                delta = 0
                for p in targets:
                    fee = bps_of(p.gross_paise, c.actual_mdr_bps)
                    charged = fee + bps_of(fee, fees.gst_bps)
                    delta += (p.fee_paise + p.tax_paise) - charged
                method = targets[0].method
                confirmed = bool(
                    rate_book and rate_book.confirms_mdr(method, c.actual_mdr_bps)
                )
                rows.append(ResolvedRow(
                    source="fee_variance",
                    cited_ids=tuple(p.payment_id for p in targets),
                    amount_paise=delta,
                    verified=confirmed,
                    detail={
                        "method": method,
                        "claimed_bps": c.actual_mdr_bps,
                        "schedule_bps": fees.mdr_for(method),
                    },
                    derivation=(
                        f"{len(targets)} {method} payments repriced at "
                        f"{c.actual_mdr_bps} bps + {fees.gst_bps} bps GST, "
                        + ("confirmed by the rate book"
                           if confirmed else
                           "rate claimed by the model and not independently confirmed")
                    ),
                ))

        elif isinstance(c, FxClaim):
            p = members.get(c.payment_id)
            if p is None:
                errors.append(
                    f"payment {c.payment_id!r} is not in {settlement.settlement_id}"
                )
                continue
            if p.currency == "INR" and p.fx_rate is None:
                errors.append(
                    f"payment {c.payment_id!r} is domestic INR; no conversion applies"
                )
                continue
            if not -5.0 <= c.actual_rate_pct_of_gross <= 5.0:
                errors.append(
                    f"claimed FX slip of {c.actual_rate_pct_of_gross}% is implausible"
                )
                continue
            # One convention, signed the same way as every other row: negative
            # means less money arrived. A slip that cost the merchant 1.6% of
            # gross is -1.6. The earlier version took abs() and then flipped,
            # which made a positive rate produce a positive amount against a
            # negative residual -- the two could never meet.
            amount = round(p.gross_paise * c.actual_rate_pct_of_gross / 100)
            confirmed = bool(
                rate_book
                and rate_book.confirms_fx(p.payment_id, c.actual_rate_pct_of_gross)
            )
            rows.append(ResolvedRow(
                source="fx",
                cited_ids=(p.payment_id,),
                amount_paise=amount,
                verified=confirmed,
                detail={"pct": c.actual_rate_pct_of_gross},
                derivation=(
                    f"{c.actual_rate_pct_of_gross:+.4f}% of {p.payment_id} gross "
                    f"({p.gross_paise} paise), "
                    + ("confirmed by the rate book"
                       if confirmed else
                       "rate claimed by the model and not independently confirmed")
                ),
            ))

        else:  # pragma: no cover - guarded by the parser
            errors.append(f"unknown citation type {type(c).__name__}")

    if not rows and not errors:
        errors.append("no citations offered")

    return Resolution(rows=tuple(rows), errors=tuple(errors))
