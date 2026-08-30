"""Write the same settlement book out as five real bank statement layouts.

The point of the fixtures this produces is that they are *the same book*. Every
file describes the identical credits; only the column names, the date format,
the money format and the way the amount is split differ. So a test can load
each one and require the reconciliation to come out identical, which is the
only way to show the ingest layer is reading the format rather than getting
lucky on a file it was written against.

Header layouts are the export shapes these banks actually produce -- split
debit/credit columns, `Chq./Ref.No.` for a reference, `Value Dt` for a date,
and three of the five carrying no line identifier at all.

Regenerate:
    python tests/fixtures/banks/make_fixtures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from recoagent.generator import DefectMix, GeneratorConfig, generate  # noqa: E402

HERE = Path(__file__).resolve().parent

#: Small on purpose. These fixtures are read by eye when a mapping argument is
#: being settled, and a 2,000-order book is not read by eye.
N_ORDERS = 120
SEED = 7


def rupees(paise: int, *, grouped: bool = False) -> str:
    """Paise back to a rupee string. `grouped` uses Indian lakh grouping."""
    sign = "-" if paise < 0 else ""
    whole, frac = divmod(abs(paise), 100)
    text = str(whole)
    if grouped and len(text) > 3:
        head, tail = text[:-3], text[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        text = ",".join(parts + [tail])
    return f"{sign}{text}.{frac:02d}"


# Each layout: the header row, and a function turning one BankLine into a row.
# `n` is the 1-based line number, for the formats that carry a serial.
LAYOUTS = {
    # HDFC: no line id, split amounts, `Value Dt`, dd/mm/yy.
    "hdfc": (
        ["Date", "Narration", "Chq./Ref.No.", "Value Dt",
         "Withdrawal Amt.", "Deposit Amt.", "Closing Balance"],
        lambda b, n: [
            b.value_date.strftime("%d/%m/%Y"), b.narration, b.bank_ref,
            b.value_date.strftime("%d/%m/%Y"),
            "" if b.amount_paise >= 0 else rupees(-b.amount_paise),
            rupees(b.amount_paise) if b.amount_paise >= 0 else "",
            "",
        ],
    ),
    # ICICI: serial number, trailing space inside the money header, lakh grouping.
    "icici": (
        ["S No.", "Value Date", "Transaction Date", "Cheque Number",
         "Transaction Remarks", "Withdrawal Amount (INR )",
         "Deposit Amount (INR )", "Balance (INR )"],
        lambda b, n: [
            str(n), b.value_date.isoformat(), b.value_date.isoformat(), b.bank_ref,
            b.narration,
            "" if b.amount_paise >= 0 else rupees(-b.amount_paise, grouped=True),
            rupees(b.amount_paise, grouped=True) if b.amount_paise >= 0 else "",
            "",
        ],
    ),
    # SBI: no line id, `Ref No./Cheque No.`, bare Debit/Credit.
    "sbi": (
        ["Txn Date", "Value Date", "Description", "Ref No./Cheque No.",
         "Debit", "Credit", "Balance"],
        lambda b, n: [
            b.value_date.strftime("%d-%m-%Y"), b.value_date.strftime("%d-%m-%Y"),
            b.narration, b.bank_ref,
            "" if b.amount_paise >= 0 else rupees(-b.amount_paise),
            rupees(b.amount_paise) if b.amount_paise >= 0 else "",
            "",
        ],
    ),
    # Axis: the terse one -- no line id, uppercase PARTICULARS, DR/CR.
    "axis": (
        ["Tran Date", "CHQNO", "PARTICULARS", "DR", "CR", "BAL", "SOL"],
        lambda b, n: [
            b.value_date.strftime("%d/%m/%Y"), b.bank_ref, b.narration,
            "" if b.amount_paise >= 0 else rupees(-b.amount_paise),
            rupees(b.amount_paise) if b.amount_paise >= 0 else "",
            "", "0001",
        ],
    ),
    # Kotak: serial with a dot, `Chq / Ref No.` spaced around the slash.
    "kotak": (
        ["Sl. No.", "Transaction Date", "Value Date", "Description",
         "Chq / Ref No.", "Debit", "Credit", "Balance"],
        lambda b, n: [
            f"{n}.", b.value_date.isoformat(), b.value_date.isoformat(),
            b.narration, b.bank_ref,
            "" if b.amount_paise >= 0 else rupees(-b.amount_paise),
            rupees(b.amount_paise) if b.amount_paise >= 0 else "",
            "",
        ],
    ),
}


def write(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    batch = generate(GeneratorConfig(n_orders=N_ORDERS, seed=SEED, mix=DefectMix.dev()))
    src = batch.sources

    for name, (header, row_of) in LAYOUTS.items():
        write(
            HERE / f"{name}.csv",
            header,
            [row_of(b, n) for n, b in enumerate(src.bank_lines, start=1)],
        )

    # The other three sources, in the field names the schema already uses, so
    # the only thing varying across the test is the bank statement layout.
    write(
        HERE / "orders.csv",
        ["order_id", "customer_id", "invoice_no", "amount_paise", "currency", "created_at"],
        [[o.order_id, o.customer_id, o.invoice_no, rupees(o.amount_paise),
          o.currency, o.created_at.isoformat()] for o in src.orders],
    )
    write(
        HERE / "payments.csv",
        ["payment_id", "order_id", "gross_paise", "fee_paise", "tax_paise",
         "method", "status", "settlement_id", "captured_at", "currency"],
        [[p.payment_id, p.order_id or "", rupees(p.gross_paise), rupees(p.fee_paise),
          rupees(p.tax_paise), p.method, p.status, p.settlement_id or "",
          p.captured_at.isoformat(), p.currency] for p in src.payments],
    )
    write(
        HERE / "settlements.csv",
        ["settlement_id", "utr", "settled_at", "net_paise", "status"],
        [[s.settlement_id, s.utr, s.settled_at.isoformat(), rupees(s.net_paise),
          s.status] for s in src.settlements],
    )

    print(f"wrote {len(LAYOUTS)} bank layouts + 3 source files for "
          f"{len(src.bank_lines)} credits, {N_ORDERS} orders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
