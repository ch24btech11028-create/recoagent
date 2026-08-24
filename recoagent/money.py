"""Money arithmetic in integer paise.

Every amount in this system is an integer number of paise. Floats are never
used for money -- `0.1 + 0.2 != 0.3` is a cosmetic annoyance in a report and a
false match in a reconciliation engine.

Rates are expressed in basis points (bps): 200 bps == 2.00%.

Rounding is ROUND_HALF_UP at each step, applied independently to the fee and
then to the tax on that fee. This is deliberate and it is where genuine
sub-rupee drift comes from: rounding `gross * mdr` and then `fee * gst` gives a
different total than rounding `gross * mdr * (1 + gst)` once. Real settlement
reports round per-step, so we do too, and the ROUNDING_DRIFT defect class
exercises the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

Paise = int

#: GST charged on the payment-gateway fee (MDR). 18% in India.
GST_BPS = 1800

#: TDS under s.194-O, withheld on gross for e-commerce operators. 0.1%.
TDS_BPS = 10

BPS_DENOMINATOR = 10_000


def bps_of(amount: Paise, bps: int) -> Paise:
    """Return `bps` basis points of `amount`, rounded half-up to whole paise."""
    if bps == 0:
        return 0
    exact = Decimal(amount) * Decimal(bps) / Decimal(BPS_DENOMINATOR)
    return int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def rupees_to_paise(rupees: str | int | Decimal) -> Paise:
    """Parse a rupee amount into integer paise. Accepts '1234.56', 1234, Decimal."""
    exact = Decimal(str(rupees)) * 100
    return int(exact.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def format_inr(paise: Paise) -> str:
    """Format paise as rupees with Indian digit grouping: -12,34,567.89 -> '-Rs 12,34,567.89'."""
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(paise), 100)
    digits = str(whole)

    # Indian grouping: final group of 3, then groups of 2.
    if len(digits) <= 3:
        grouped = digits
    else:
        head, tail = digits[:-3], digits[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        grouped = ",".join(parts) + "," + tail

    return f"{sign}Rs {grouped}.{frac:02d}"


@dataclass(frozen=True)
class FeeSchedule:
    """MDR by payment method, in basis points.

    The UPI row is not a placeholder: MDR on UPI P2M is zero by regulation in
    India. A recon engine that assumes every payment carries a fee will flag
    every UPI settlement as short, which is the single most common way a naive
    matcher produces a wall of false exceptions.
    """

    mdr_bps: dict[str, int]
    gst_bps: int = GST_BPS
    tds_bps: int = 0  # off by default; enabled per-merchant

    @classmethod
    def default(cls) -> FeeSchedule:
        return cls(
            mdr_bps={
                "upi": 0,
                "rupay_debit": 0,
                "netbanking": 175,
                "card_domestic": 200,
                "card_international": 300,
                "wallet": 200,
                "emi": 250,
            }
        )

    def mdr_for(self, method: str) -> int:
        if method not in self.mdr_bps:
            raise KeyError(f"no MDR configured for payment method {method!r}")
        return self.mdr_bps[method]

    def fee_and_tax(self, gross: Paise, method: str) -> tuple[Paise, Paise]:
        """Return (fee, tax) for a gross amount, rounded half-up at each step."""
        fee = bps_of(gross, self.mdr_for(method))
        tax = bps_of(fee, self.gst_bps)
        return fee, tax

    def tds_on(self, gross: Paise) -> Paise:
        return bps_of(gross, self.tds_bps)

    def net_of(self, gross: Paise, method: str) -> Paise:
        fee, tax = self.fee_and_tax(gross, method)
        return gross - fee - tax - self.tds_on(gross)

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(self.mdr_bps)
