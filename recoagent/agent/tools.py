"""Tools the agent calls to investigate a residual itself.

The earlier design pre-computed an evidence packet and handed it over. That put
*me* in charge of deciding what mattered, and it went wrong exactly as you would
expect: the packet exposed a per-payment fee-vs-schedule comparison that can
never differ under this generator, and the model read it as positive evidence
that no repricing had occurred. It declined five cases on the strength of a
field I chose badly.

An agent that queries for what it needs cannot be trapped by my framing in the
same way. So the packet becomes a starting summary and these tools do the rest.

Three properties hold for every tool here:

- **Source-only.** Each is built from `SourceBundle`, the same restriction every
  matcher operates under. None can reach ground truth, and
  `tests/test_independence.py` covers this module.
- **Read-only, except one calculator.** Nothing mutates state. `check_hypothesis`
  runs the same arithmetic the gate runs, but its answer is advice: it does not
  book anything, and passing it is not acceptance.
- **A tool that fails returns an error dict, never raises.** A model that sends
  a bad argument should get a usable correction back, not kill the case.

`check_hypothesis` is the interesting one. It lets the model test a guess for
free before committing, which is what a human analyst does with a calculator.
It does not weaken the gate: the gate re-runs the same check on the final
proposal from the tier, so a model that lies about having checked gains nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from ..money import FeeSchedule, Paise, bps_of
from ..schemas import BankLine, PGAdjustment, Settlement, SourceBundle
from ..validate import Tolerance, prove_leg2

MAX_ROWS_RETURNED = 60


@dataclass
class ToolContext:
    """Everything the tools may see, scoped to one exception under investigation."""

    sources: SourceBundle
    line: BankLine
    settlement: Settlement
    residual_paise: Paise
    tol: Tolerance
    fees: FeeSchedule

    @property
    def members(self):
        return self.sources.payments_by_settlement(self.settlement.settlement_id)

    @property
    def linked(self):
        return self.sources.adjustments_by_settlement(self.settlement.settlement_id)


TOOL_GUIDE = """\
Available tools. Call exactly one per turn:

{"tool": "list_payments", "args": {"method": "card_domestic"}}
    Payment rows in this batch. Omit "method" for all of them. Optional
    "min_gross_paise" to see only larger payments.

{"tool": "list_unlinked_rows", "args": {"window_days": 7}}
    Refunds, chargebacks and adjustments near this settlement date that the
    gateway did not link to any batch.

{"tool": "compute_fee_scenario", "args": {"mdr_bps": 240, "methods": ["card_domestic"]}}
    What fee + GST would have been charged at a different MDR, and how far that
    is from the fee actually reported. Use it to test a repricing theory: if the
    residual matches the delta at some plausible rate, that is your explanation.
    Rates are basis points -- 200 bps is 2.00%.

{"tool": "check_hypothesis", "args": {"rows": [{"amount_paise": -12345}]}}
    Test whether a set of rows closes the residual, WITHOUT committing to it.
    Returns what would remain unexplained. Use this before proposing.

Then finish with exactly one of:

{"action": "propose", "rows": [{"label": "...", "amount_paise": -123, "rationale": "..."}],
 "reason": "...", "confidence": 0.0}

{"action": "decline", "reason": "..."}
"""


def _payment_row(p) -> dict[str, Any]:
    return {
        "payment_id": p.payment_id,
        "method": p.method,
        "currency": p.currency,
        "fx_rate": p.fx_rate,
        "gross_paise": p.gross_paise,
        "fee_paise": p.fee_paise,
        "tax_paise": p.tax_paise,
        "net_paise": p.net_paise,
        "captured_at": p.captured_at.isoformat(),
        "status": p.status,
    }


def _adjustment_row(a: PGAdjustment) -> dict[str, Any]:
    return {
        "adjustment_id": a.adjustment_id,
        "kind": a.kind,
        "payment_id": a.payment_id,
        "amount_paise": a.amount_paise,
        "booked_at": a.booked_at.isoformat(),
    }


def list_payments(ctx: ToolContext, args: dict) -> dict[str, Any]:
    rows = list(ctx.members)
    method = args.get("method")
    if method:
        rows = [p for p in rows if p.method == method]
    floor = args.get("min_gross_paise")
    if isinstance(floor, int):
        rows = [p for p in rows if p.gross_paise >= floor]
    return {
        "count": len(rows),
        "truncated": len(rows) > MAX_ROWS_RETURNED,
        "payments": [_payment_row(p) for p in rows[:MAX_ROWS_RETURNED]],
    }


def list_unlinked_rows(ctx: ToolContext, args: dict) -> dict[str, Any]:
    days = args.get("window_days", 7)
    if not isinstance(days, int) or not 1 <= days <= 30:
        return {"error": "window_days must be an integer between 1 and 30"}
    window = timedelta(days=days)
    rows = [
        a
        for a in ctx.sources.adjustments
        if a.settlement_id is None
        and abs(a.booked_at - ctx.settlement.settled_at) <= window
    ]
    return {
        "count": len(rows),
        "note": (
            "An exhaustive subset-sum over these rows already failed to close "
            "this residual, so a plain combination of them is not the answer."
        ),
        "rows": [_adjustment_row(a) for a in rows[:MAX_ROWS_RETURNED]],
    }


def compute_fee_scenario(ctx: ToolContext, args: dict) -> dict[str, Any]:
    """What the fee would have been at a different MDR, versus what was reported."""
    mdr_bps = args.get("mdr_bps")
    if not isinstance(mdr_bps, int) or not 0 <= mdr_bps <= 2000:
        return {"error": "mdr_bps must be an integer between 0 and 2000"}

    methods = args.get("methods")
    rows = list(ctx.members)
    if methods:
        if not isinstance(methods, list):
            return {"error": "methods must be a list of payment-method strings"}
        rows = [p for p in rows if p.method in methods]
    if not rows:
        return {"error": "no payments in this batch match that filter", "count": 0}

    reported = sum(p.fee_paise + p.tax_paise for p in rows)
    scenario = 0
    for p in rows:
        fee = bps_of(p.gross_paise, mdr_bps)
        scenario += fee + bps_of(fee, ctx.fees.gst_bps)

    delta = reported - scenario  # negative => the scenario costs more than reported
    return {
        "payments_considered": len(rows),
        "gross_paise": sum(p.gross_paise for p in rows),
        "reported_fee_plus_tax_paise": reported,
        "scenario_fee_plus_tax_paise": scenario,
        "delta_paise": delta,
        "residual_paise": ctx.residual_paise,
        "delta_matches_residual": abs(delta - ctx.residual_paise) <= ctx.tol.leg2_paise,
        "note": (
            "delta_paise is reported minus scenario. If the actual deduction used "
            "this rate, the batch would be short by -delta_paise."
        ),
    }


def check_hypothesis(ctx: ToolContext, args: dict) -> dict[str, Any]:
    """Run the gate's arithmetic without committing to the result."""
    rows = args.get("rows")
    if not isinstance(rows, list) or not rows:
        return {"error": "rows must be a non-empty list of {amount_paise: int}"}

    amounts: list[int] = []
    for r in rows:
        amount = r.get("amount_paise") if isinstance(r, dict) else None
        if isinstance(amount, bool) or not isinstance(amount, int):
            return {"error": f"amount_paise must be a whole integer, got {amount!r}"}
        amounts.append(amount)

    trial = [
        PGAdjustment(
            adjustment_id=f"trial:{i}",
            settlement_id=None,  # type: ignore[arg-type]
            kind="trial",
            payment_id=None,
            amount_paise=amount,
            booked_at=ctx.settlement.settled_at,
        )
        for i, amount in enumerate(amounts)
    ]
    proof = prove_leg2(
        ctx.line, ctx.settlement, ctx.members, ctx.linked, ctx.tol, hypothesised=trial
    )
    return {
        "rows_total_paise": sum(amounts),
        "residual_paise": ctx.residual_paise,
        "would_close": proof.closes,
        "still_unexplained_paise": proof.residual_paise,
        "tolerance_paise": ctx.tol.leg2_paise,
        "note": (
            "This is advice, not acceptance. The gate re-runs the same check on "
            "whatever you finally propose."
        ),
    }


REGISTRY = {
    "list_payments": list_payments,
    "list_unlinked_rows": list_unlinked_rows,
    "compute_fee_scenario": compute_fee_scenario,
    "check_hypothesis": check_hypothesis,
}


def execute(ctx: ToolContext, name: str, args: dict | None) -> dict[str, Any]:
    """Run one tool. Returns an error dict rather than raising, always."""
    fn = REGISTRY.get(name)
    if fn is None:
        return {
            "error": f"unknown tool {name!r}",
            "available": sorted(REGISTRY),
        }
    try:
        return fn(ctx, args or {})
    except Exception as exc:  # a bad argument must not kill the case
        return {"error": f"{type(exc).__name__}: {exc}"}
