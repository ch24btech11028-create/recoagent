"""Razorpay JSON into the same `SourceBundle` the generator produces.

The interesting content of this module is what it refuses to do.

**It does not invent a bank side.** Razorpay knows what it paid out; it does
not know what your bank credited, and the gap between those two is the entire
subject of Leg 2. A settlement header carries `amount` and `utr`, so it is
trivially possible to write `BankLine(amount=settlement.amount,
bank_ref=settlement.utr)` and report 100% Leg 2 recall for ever. That number
would be a measurement of this function, not of the reconciler. The bank side
comes from a statement CSV or Leg 2 does not run.

**It does not invent a partial capture.** Razorpay has no `partially_captured`
payment status, and Leg 1's recovery tier is gated on the gateway having
*declared* the shortfall. The only declaration Razorpay actually makes is on
the order: `amount_paid` below `amount`. That, and nothing looser, is what
gets translated.

**It does not guess a status it has not seen.** An unrecognised Razorpay
status passes through under its own name and lands in the exception queue,
rather than being folded into the nearest familiar one. A book that reconciles
because an unknown state was quietly read as `captured` is worse than a book
with an unexplained exception in it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from ..schemas import Order, PGAdjustment, PGPayment, Settlement, SourceBundle

#: Razorpay payment status -> the status the engine reasons about. Deliberately
#: not a `.get(status, "captured")` fallback: see the module docstring.
STATUS = {
    "captured": "captured",
    "refunded": "refunded",
    "authorized": "authorized",
    "failed": "failed",
    "created": "created",
}

#: Razorpay payment method -> the key `FeeSchedule` prices. Razorpay reports
#: one `card` method and carries the domestic/international distinction in a
#: separate boolean; the schedule prices them differently because the MDR is
#: different, so the two fields have to be recombined here rather than in a
#: matcher that should never have heard of Razorpay.
METHOD = {
    ("card", False): "card_domestic",
    ("card", True): "card_international",
    ("netbanking", False): "netbanking",
    ("upi", False): "upi",
    ("wallet", False): "wallet",
    ("emi", False): "emi",
}


def _ts(value: Any) -> datetime:
    """Razorpay timestamps are unix seconds. Anything else is a bug, loudly."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, (int, float)):
        raise ValueError(f"expected a unix timestamp, got {value!r}")
    return datetime.fromtimestamp(int(value), timezone.utc)


def _paise(value: Any) -> int:
    """Razorpay amounts are already integer paise. Keep them that way.

    A float here would be a silent precision bug in the one place this project
    least tolerates one, so a non-integer is an error rather than a rounding.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected integer paise, got {value!r} ({type(value).__name__})")
    return value


def split_fee(row: dict[str, Any]) -> tuple[int, int]:
    """Razorpay's `fee` includes GST. The engine's `fee_paise` does not.

    This is the single most consequential line in the module, and it is one
    subtraction. `PGPayment.net_paise` is `gross - fee - tax`, while Razorpay
    documents `fee` as the charge *including* tax and `tax` as the tax
    component inside it. Copying both fields across unchanged books the GST
    twice, understating the net on every priced row -- which does not fail
    loudly, it just makes every settlement look short by the total GST in it,
    and produces a beautifully consistent wall of fee-variance exceptions that
    a model would then be asked to explain.

    Verified against the arithmetic Razorpay itself publishes in the recon
    report: `credit == amount - fee`, with `fee` inclusive.
    """
    fee_inclusive = _paise(row.get("fee") or 0)
    tax = _paise(row.get("tax") or 0)
    if tax > fee_inclusive:
        raise ValueError(
            f"{row.get('id')}: tax ({tax}) exceeds the inclusive fee "
            f"({fee_inclusive}), which cannot be true if the fee contains it"
        )
    return fee_inclusive - tax, tax


def method_of(row: dict[str, Any]) -> str:
    """The rate-card key for this payment, or its raw name if there is none."""
    raw = row.get("method", "unknown")
    return METHOD.get((raw, bool(row.get("international"))), raw)


def order_from(row: dict[str, Any]) -> Order:
    notes = row.get("notes") or {}
    return Order(
        order_id=row["id"],
        # Razorpay orders carry no customer field. `notes` is the merchant's
        # own free-form dictionary and is where a real integration puts one;
        # empty is honest when it is absent.
        customer_id=str(notes.get("customer_id", "")),
        invoice_no=str(row.get("receipt") or ""),
        amount_paise=_paise(row["amount"]),
        currency=row.get("currency", "INR"),
        created_at=_ts(row["created_at"]),
    )


def payment_from(
    row: dict[str, Any],
    *,
    settlement_id: str | None = None,
    under_captured: bool = False,
) -> PGPayment:
    status = STATUS.get(row.get("status", ""), row.get("status", "unknown"))
    if under_captured and status == "captured":
        status = "partially_captured"
    # `fee` and `tax` are null until the payment is captured and priced.
    fee, tax = split_fee(row)
    return PGPayment(
        payment_id=row["id"],
        order_id=row.get("order_id"),
        gross_paise=_paise(row["amount"]),
        fee_paise=fee,
        tax_paise=tax,
        method=method_of(row),
        status=status,
        settlement_id=settlement_id,
        captured_at=_ts(row["created_at"]),
        currency=row.get("currency", "INR"),
        # `international` is a bool; the rate Razorpay converted at is not in
        # the payment object, so an FX advice has to come from the bank.
        fx_rate=None,
    )


def settlement_from(row: dict[str, Any]) -> Settlement:
    return Settlement(
        settlement_id=row["id"],
        utr=str(row.get("utr") or ""),
        settled_at=_ts(row["created_at"]),
        net_paise=_paise(row["amount"]),
        status=row.get("status", "unknown"),
    )


def refund_as_adjustment(
    row: dict[str, Any], *, settlement_id: str | None = None
) -> PGAdjustment:
    """A refund is a negative line inside somebody's settlement batch.

    Signed from the merchant's perspective, matching `PGAdjustment`'s contract:
    Razorpay reports a refund as a positive amount because it is describing the
    refund, and the engine stores it negative because it is describing the
    effect on the credit.
    """
    return PGAdjustment(
        adjustment_id=row["id"],
        settlement_id=settlement_id,
        kind="refund",
        payment_id=row.get("payment_id"),
        amount_paise=-_paise(row["amount"]),
        booked_at=_ts(row["created_at"]),
    )


def settlement_index(recon: Iterable[dict[str, Any]]) -> dict[str, str]:
    """entity id -> the settlement it was netted into, from the recon report.

    This linkage exists nowhere else. `/payments` will not tell you which batch
    a payment landed in, and without that Leg 2 has no left-hand side to build
    a subset out of -- so on a book with no recon rows, Leg 2 is not merely
    inaccurate, it is undefined.
    """
    out: dict[str, str] = {}
    for row in recon:
        entity = row.get("entity_id")
        settlement = row.get("settlement_id")
        if entity and settlement:
            out[entity] = settlement
    return out


def bundle_from_payload(
    payload: dict[str, Any],
    *,
    bank_lines: Iterable[Any] = (),
    rate_notices: Iterable[Any] = (),
    fx_advices: Iterable[Any] = (),
) -> SourceBundle:
    """Build a `SourceBundle` from one recorded pull.

    `bank_lines` is a parameter rather than something derived here, and the
    module docstring says why. Pass what your bank actually sent.
    """
    recon = payload.get("recon") or []
    settled_by_entity = settlement_index(recon)

    orders = [order_from(r) for r in payload.get("orders", [])]

    # An order Razorpay itself reports as underpaid is the only under-capture
    # declaration available. `amount_paid` is the order's, so the flag is
    # carried across to the payment by order id.
    under_captured = {
        r["id"]
        for r in payload.get("orders", [])
        if r.get("amount_paid") is not None
        and _paise(r["amount_paid"]) < _paise(r["amount"])
        and _paise(r["amount_paid"]) > 0
    }

    payments = [
        payment_from(
            r,
            settlement_id=settled_by_entity.get(r["id"]),
            under_captured=r.get("order_id") in under_captured,
        )
        for r in payload.get("payments", [])
    ]

    adjustments = [
        refund_as_adjustment(r, settlement_id=settled_by_entity.get(r["id"]))
        for r in payload.get("refunds", [])
    ]

    settlements = [settlement_from(r) for r in payload.get("settlements", [])]

    # Tuples, not lists: `SourceBundle` is frozen and hashes its contents,
    # and a mutable member would make an "immutable" book quietly editable.
    return SourceBundle(
        orders=tuple(orders),
        payments=tuple(payments),
        adjustments=tuple(adjustments),
        settlements=tuple(settlements),
        bank_lines=tuple(bank_lines),
        rate_notices=tuple(rate_notices),
        fx_advices=tuple(fx_advices),
    )


def readiness(payload: dict[str, Any], bundle: SourceBundle) -> list[str]:
    """What this book cannot be asked, and why. Printed before any number is.

    Test mode does not run settlement cycles, so a pull from a fresh test
    account usually has payments and no settlements at all. A reconciler handed
    that book will report a Leg 2 recall of zero and be perfectly correct, and
    a reader who was not told why will read it as a failure of the reconciler.
    """
    notes: list[str] = []
    if not bundle.settlements:
        notes.append(
            "No settlements. Razorpay test mode does not run settlement cycles, "
            "so Leg 2 has no batches to reconcile -- only Leg 1 is measurable "
            "on this book."
        )
    if not payload.get("recon"):
        notes.append(
            "No recon rows. Without /settlements/recon/combined nothing links a "
            "payment to the batch it settled in, so Leg 2 is undefined rather "
            "than merely unmatched."
        )
    if not bundle.bank_lines:
        notes.append(
            "No bank statement. Leg 2 matches a gateway batch to a bank credit; "
            "with no credits there is no right-hand side. Pass --bank with a "
            "statement CSV. Deriving one from the settlement headers would make "
            "Leg 2 reconcile against itself and report a meaningless 100%."
        )
    unpriced = [p for p in bundle.payments if p.status == "captured" and not p.fee_paise]
    if unpriced:
        notes.append(
            f"{len(unpriced)} captured payments carry no fee. Test mode prices "
            "some methods at zero, so a fee-variance explanation has nothing to "
            "vary from on those rows."
        )
    return notes
