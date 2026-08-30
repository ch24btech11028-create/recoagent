"""Point it at your own CSVs.

    python -m recoagent.ingest --orders orders.csv --payments payments.csv \\
        --settlements settlements.csv --bank bank.csv

Every number this repository publishes is measured on a book it generated
itself or on BenchRec, and both of those are answers to "is it right?". This
module answers a different question -- "does it run on mine?" -- which is the
one a reader with a real settlement file actually has.

**There is no scorecard here, and that is not an omission.** A false-match rate
needs an answer key. Your CSVs do not come with one, so what this prints is
coverage and the exception list: how much credit tied out, which tier tied it,
and every item that did not. Any tool that offers you an accuracy figure on
unlabelled data is computing it against its own opinion.

Column names are matched case-insensitively with punctuation folded, so
`Order ID`, `order-id` and `ORDER_ID` all land on `order_id`. A handful of
industry synonyms are recognised too (`utr` for a bank reference, `amount` for
`amount_paise`). Anything else, name explicitly:

    --map map.json      {"orders": {"amount_paise": "Gross Value"}}

Money is read as rupees with an optional decimal part and parsed as a string --
`12,345.67` becomes `1234567` paise exactly. Floats never touch it. Pass
`--money paise` if your files are already in the smallest unit.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import MISSING, fields as dataclass_fields
from datetime import date, datetime
from pathlib import Path

from .money import Paise, format_inr
from .pipeline import run_b0, run_b2
from .schemas import (
    BankLine,
    FxAdvice,
    Order,
    PGAdjustment,
    PGPayment,
    RateNotice,
    Settlement,
    SourceBundle,
)

#: Field name -> the other names a real export might use for it. Deliberately
#: short: a guess that silently maps the wrong column is worse than an error
#: message naming the column it could not find.
ALIASES: dict[str, tuple[str, ...]] = {
    "amount_paise": ("amount", "value", "gross", "gross_amount", "order_amount"),
    "gross_paise": ("gross", "amount", "gross_amount", "captured_amount"),
    "fee_paise": ("fee", "mdr", "commission"),
    "tax_paise": ("tax", "gst", "service_tax"),
    "net_paise": ("net", "net_amount", "settlement_amount", "payout"),
    "bank_ref": (
        "utr", "reference", "ref", "transaction_ref",
        # Real statement headers, in the shape `_norm` leaves them.
        "chq_ref_no", "chqno", "chq_no", "cheque_number", "cheque_no",
        "ref_no_cheque_no", "chq_ref_number",
    ),
    "narration": (
        "description", "particulars", "remarks", "detail",
        "transaction_remarks", "transaction_description", "narration_particulars",
    ),
    "value_date": (
        "date", "txn_date", "posting_date", "credit_date",
        "value_dt", "tran_date", "transaction_date", "txn_posted_date",
    ),
    "settled_at": ("settlement_date", "payout_date"),
    "captured_at": ("payment_date", "txn_date", "created_at"),
    "booked_at": ("adjustment_date", "date"),
    "bank_line_id": ("id", "line_id", "statement_line_id",
                     "s_no", "sl_no", "sr_no", "srl_no", "serial_no"),
    "settlement_id": ("payout_id", "batch_id"),
    "order_id": ("order_ref", "merchant_order_id", "receipt"),
    "payment_id": ("txn_id", "transaction_id"),
    "method": ("payment_method", "instrument", "mode"),
    "currency": ("ccy", "curr"),
}

#: A bank statement almost never has one signed amount column. It has two --
#: money in and money out -- with the other cell blank, and every major Indian
#: bank names the pair differently. Fixtures for five of them are in
#: `tests/fixtures/banks/`.
#:
#: Folding them is not cosmetic. Reading only the credit column silently drops
#: every debit, and a reconciliation that cannot see money leaving the account
#: is not a reconciliation -- it is a report on deposits.
CREDIT_COLUMNS = (
    "deposit", "deposit_amt", "deposit_amount", "deposit_amount_inr",
    "credit", "credit_amount", "credit_amt", "cr", "cr_amount",
)
DEBIT_COLUMNS = (
    "withdrawal", "withdrawal_amt", "withdrawal_amount", "withdrawal_amount_inr",
    "debit", "debit_amount", "debit_amt", "dr", "dr_amount",
)

#: Fields a given kind of file may legitimately not carry, and that can be
#: reconstructed rather than demanded. Only one so far, and it is worth being
#: precise about why: HDFC, SBI and Axis statements have no line identifier at
#: all, so requiring one would reject three of the five real formats over a
#: column the bank never had. A statement line's identity is its position in the
#: file, and that is what gets synthesised -- visibly, as `<file>:<row>`, so
#: nobody mistakes it for something the bank issued.
SYNTHESISABLE = {"bank": ("bank_line_id",)}

#: The bundle field each file feeds, and the row type it builds.
ENTITIES = {
    "orders": ("orders", Order),
    "payments": ("payments", PGPayment),
    "adjustments": ("adjustments", PGAdjustment),
    "settlements": ("settlements", Settlement),
    "bank": ("bank_lines", BankLine),
    "notices": ("rate_notices", RateNotice),
    "fx": ("fx_advices", FxAdvice),
}

REQUIRED = ("orders", "payments", "settlements", "bank")


class IngestError(Exception):
    """A problem in the input, phrased for the person holding the file."""


def _norm(name: str) -> str:
    """Fold a header down to a comparable key: `Order ID ` -> `order_id`."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _paise(raw: str, *, unit: str, where: str) -> Paise:
    """Rupees-with-decimals to integer paise, without ever building a float.

    `1,234.5` is 1,23,450 paise, not 1,23,45. The half-written decimal is the
    one that costs money, so a single-digit fraction is padded rather than
    guessed at, and more than two is refused instead of rounded -- a file with
    three decimal places is not a rupee file and someone should look at it.
    """
    text = raw.strip().replace(",", "").replace("₹", "").replace("Rs.", "").replace("Rs", "")
    text = text.strip()
    if text in ("", "-"):
        raise IngestError(f"{where}: empty amount")
    sign = -1 if text.startswith("-") else 1
    if text[0] in "+-":
        text = text[1:]
    if text.startswith("(") and text.endswith(")"):  # accounting negative
        sign, text = -1, text[1:-1]

    if unit == "paise":
        if not text.isdigit():
            raise IngestError(f"{where}: {raw!r} is not a whole number of paise")
        return sign * int(text)

    whole, dot, frac = text.partition(".")
    if not whole.isdigit() or (dot and not frac.isdigit()):
        raise IngestError(f"{where}: {raw!r} is not an amount")
    if len(frac) > 2:
        raise IngestError(
            f"{where}: {raw!r} has {len(frac)} decimal places. Rupees have two; "
            "if this file is already in paise, pass --money paise"
        )
    return sign * (int(whole) * 100 + int(frac.ljust(2, "0") or 0))


def _datetime(raw: str, *, where: str) -> datetime:
    text = raw.strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise IngestError(f"{where}: {raw!r} is not a date/time this reads. ISO 8601 always works")


def _date(raw: str, *, where: str) -> date:
    return _datetime(raw, where=where).date()


def _convert(field, raw: str, *, unit: str, where: str):
    """One CSV cell to one dataclass field, by the field's declared type."""
    ann = str(field.type)
    optional = "None" in ann
    text = raw.strip() if raw is not None else ""

    if text == "":
        if optional:
            return None
        raise IngestError(f"{where}: {field.name} is required and this row leaves it blank")

    if field.name.endswith("_paise"):
        return _paise(text, unit=unit, where=f"{where}.{field.name}")
    if "datetime" in ann:
        return _datetime(text, where=f"{where}.{field.name}")
    if "date" in ann:
        return _date(text, where=f"{where}.{field.name}")
    if "int" in ann:
        return int(text)
    if "float" in ann:
        return float(text)
    return text


def _optional(field) -> bool:
    """A field the file may leave out entirely: it has a default, or admits None."""
    return (
        field.default is not MISSING
        or field.default_factory is not MISSING  # type: ignore[misc]
        or "None" in str(field.type)
    )


def _split_amount(by_norm: dict[str, str]) -> tuple[str | None, str | None] | None:
    """Find a credit/debit column pair standing in for one signed amount.

    Either side may be absent -- a statement of receipts only is a real thing --
    but not both, or there is nothing here to fold.
    """
    credit = next((by_norm[c] for c in CREDIT_COLUMNS if c in by_norm), None)
    debit = next((by_norm[c] for c in DEBIT_COLUMNS if c in by_norm), None)
    return (credit, debit) if (credit or debit) else None


def _resolve_columns(
    row_type, headers, overrides: dict[str, str], kind: str = ""
) -> tuple[dict[str, str], tuple[str | None, str | None] | None, tuple[str, ...]]:
    """Decide which CSV column feeds which field, or say exactly what is missing.

    Returns the column map, the credit/debit pair to fold into `amount_paise`
    if the file splits it, and the fields that have to be synthesised because
    this format genuinely does not carry them.
    """
    by_norm = {_norm(h): h for h in headers}
    chosen: dict[str, str] = {}
    missing: list[str] = []
    fold: tuple[str | None, str | None] | None = None
    synthesised: list[str] = []

    for field in dataclass_fields(row_type):
        if field.name in overrides:
            column = overrides[field.name]
            if column not in headers:
                raise IngestError(
                    f"the map points {field.name} at {column!r}, which this file "
                    f"does not have. Columns present: {', '.join(headers)}"
                )
            chosen[field.name] = column
            continue

        candidates = (field.name, *ALIASES.get(field.name, ()))
        hit = next((by_norm[c] for c in candidates if c in by_norm), None)
        if hit is not None:
            chosen[field.name] = hit
        elif field.name == "amount_paise" and (pair := _split_amount(by_norm)):
            fold = pair
        elif field.name in SYNTHESISABLE.get(kind, ()):
            synthesised.append(field.name)
        elif _optional(field):
            continue  # a column nobody has to supply
        else:
            missing.append(field.name)

    if missing:
        raise IngestError(
            f"no column found for: {', '.join(missing)}.\n"
            f"  columns in the file: {', '.join(headers)}\n"
            f"  name them with --map, e.g. "
            + json.dumps({missing[0]: "<your column>"})
        )
    return chosen, fold, tuple(synthesised)



def _folded_amount(raw_row, fold, *, unit: str, where: str) -> Paise:
    """One signed amount from a credit column and a debit column.

    A row carrying both is refused rather than netted. On a statement that is
    not a smaller deposit -- it is a mis-mapped file, and netting it would turn
    a mapping error into a plausible-looking number.
    """
    credit_col, debit_col = fold

    def read(column: str | None) -> Paise:
        if column is None:
            return 0
        text = (raw_row.get(column) or "").strip()
        # A blank cell is the bank saying "not this side", not a bad amount.
        if text in ("", "-"):
            return 0
        return _paise(text, unit=unit, where=where)

    credit, debit = read(credit_col), read(debit_col)
    if credit and debit:
        raise IngestError(
            f"{where}: this row is filled in on both sides -- credit {credit} "
            f"and debit {debit}. A statement line is one or the other; check "
            f"that {credit_col!r} and {debit_col!r} are the columns you meant"
        )
    return credit - debit


def read_rows(path: Path, kind: str, *, unit: str, overrides: dict[str, str]) -> tuple:
    """Read one CSV into its dataclass, reporting the first handful of problems.

    Errors are collected rather than raised one at a time: someone whose export
    uses a different date format has that problem on every row, and finding out
    once per run is a bad way to spend an afternoon.
    """
    _, row_type = ENTITIES[kind]
    try:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            headers = reader.fieldnames or []
            if not headers:
                raise IngestError(f"{path}: no header row")
            columns, fold, synthesised = _resolve_columns(
                row_type, headers, overrides, kind
            )

            built, problems = [], []
            for n, raw_row in enumerate(reader, start=2):  # 1 is the header
                where = f"{path.name}:{n}"
                try:
                    values = {}
                    for field in dataclass_fields(row_type):
                        if fold is not None and field.name == "amount_paise":
                            values[field.name] = _folded_amount(
                                raw_row, fold, unit=unit, where=where
                            )
                            continue
                        if field.name in synthesised:
                            # Visibly derived, so it cannot be mistaken for a
                            # reference the bank issued.
                            values[field.name] = f"{path.stem}:{n}"
                            continue
                        column = columns.get(field.name)
                        if column is None:
                            continue
                        values[field.name] = _convert(
                            field, raw_row.get(column) or "", unit=unit, where=where
                        )
                    built.append(row_type(**values))
                except (IngestError, ValueError, TypeError) as exc:
                    problems.append(f"  {exc}")
                    if len(problems) >= 10:
                        problems.append("  ... stopping after 10")
                        break
    except FileNotFoundError:
        raise IngestError(f"{path}: no such file") from None

    if problems:
        raise IngestError(f"{path}: {len(problems)} rows could not be read:\n" + "\n".join(problems))
    if not built:
        raise IngestError(f"{path}: header row only, no data")
    return tuple(built)


def load(paths: dict[str, Path], *, unit: str = "rupees", mapping: dict | None = None) -> SourceBundle:
    """Build a `SourceBundle` from CSVs. The matcher cannot tell the difference."""
    mapping = mapping or {}
    # Adjustments are optional to *supply* and not optional to *pass*: a book
    # with no netted rows is an ordinary book, and the bundle wants an empty
    # tuple rather than nothing at all.
    kwargs = {"adjustments": ()}
    for kind, path in paths.items():
        field_name, _ = ENTITIES[kind]
        kwargs[field_name] = read_rows(path, kind, unit=unit, overrides=mapping.get(kind, {}))
    return SourceBundle(**kwargs)


def report(sources: SourceBundle, result, *, top: int = 10) -> str:
    """Coverage and the exception list. No accuracy, because none is knowable."""
    from .views import shape

    sh = shape(sources, result)
    credit = sh["credit"]
    out = [
        "",
        "  sources: " + "  ".join(f"{k}={v}" for k, v in sources.counts.items()),
        "",
        "=" * 72,
        f"  YOUR BOOK   rung={result.rung}",
        "=" * 72,
        "",
        f"  Bank credits tied to a batch   {credit['lines_matched']:>6} of {credit['lines_total']}"
        f"   ({credit['matched_share'] * 100:.2f}%)",
        f"  Credit value cleared           {credit['matched']:>18}",
        f"  Outstanding                    {credit['outstanding']:>18}",
        f"  Open exceptions                {len(result.exceptions):>7}",
        "",
        "  There is no false-match rate on this page. Your files carry no answer",
        "  key, so accuracy cannot be computed -- only coverage, and the list below.",
        "",
        "-" * 72,
        "  WHAT TIED IT OUT",
        "-" * 72,
    ]
    for rule in sh["rules"]:
        out.append(f"  {rule['tier']:<4} {rule['label']:<46}{rule['count']:>8}")

    variance = [m for m in result.matches if m.variance_paise]
    if variance:
        total = sum(m.variance_paise for m in variance)
        out += ["", f"  Documented variance   {format_inr(total):>18}"
                    f"   ({len(variance)} matched rows carry a declared gap)"]

    ranked = sorted(
        result.exceptions, key=lambda e: abs(e.residual_paise or 0), reverse=True
    )
    out += ["", "-" * 72, f"  EXCEPTIONS  ({len(ranked)} total, showing {min(top, len(ranked))})", "-" * 72]
    for e in ranked[:top]:
        money = format_inr(e.residual_paise) if e.residual_paise is not None else "--"
        out.append(f"  {e.entity_id:<22}{money:>18}  {e.reason[:60]}")
    out += ["", "=" * 72, ""]
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m recoagent.ingest",
        description="Reconcile your own CSV exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    for kind in ENTITIES:
        ap.add_argument(f"--{kind}", type=Path, help=f"{kind} CSV"
                        + ("" if kind in REQUIRED else "  (optional)"))
    ap.add_argument("--money", choices=["rupees", "paise"], default="rupees",
                    help="unit the amount columns are in (default: rupees)")
    ap.add_argument("--map", type=Path, help="JSON of {entity: {field: column}} overrides")
    ap.add_argument("--rung", default="B2", choices=["B0", "B2"])
    ap.add_argument("--out", type=Path, help="write matches and exceptions here as JSON")
    ap.add_argument("--exceptions", type=int, default=10, metavar="K",
                    help="how many exceptions to print (default: 10)")
    args = ap.parse_args(argv)

    given = {k: getattr(args, k) for k in ENTITIES if getattr(args, k) is not None}
    absent = [k for k in REQUIRED if k not in given]
    if absent:
        ap.error("these are required: " + ", ".join("--" + a for a in absent))

    mapping = {}
    if args.map:
        try:
            mapping = json.loads(args.map.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"could not read {args.map}: {exc}", file=sys.stderr)
            return 2

    try:
        sources = load(given, unit=args.money, mapping=mapping)
    except IngestError as exc:
        # An input problem is the user's to fix, so it prints as a message
        # rather than a traceback. Exit 2 keeps it distinguishable from a book
        # that read fine and did not reconcile.
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    result = run_b0(sources) if args.rung == "B0" else run_b2(sources)
    print(report(sources, result, top=args.exceptions))

    if args.out:
        args.out.write_text(json.dumps({
            "rung": result.rung,
            "matches": [
                {"match_id": m.match_id, "leg": m.leg, "tier": m.tier, "rule_id": m.rule_id,
                 "left": list(m.left_ids), "right": list(m.right_ids),
                 "hypothesised": list(m.hypothesised_ids),
                 "variance_paise": m.variance_paise,
                 "proof": None if m.proof is None else m.proof.expression,
                 "residual_paise": m.proof.residual_paise if m.proof else None}
                for m in sorted(result.matches, key=lambda m: m.match_id)
            ],
            "exceptions": [
                {"exception_id": e.exception_id, "leg": e.leg, "entity_kind": e.entity_kind,
                 "entity_id": e.entity_id, "reason": e.reason,
                 "residual_paise": e.residual_paise,
                 "suspected_class": e.suspected_class.value if e.suspected_class else None}
                for e in result.exceptions
            ],
        }, indent=2, sort_keys=True))
        print(f"  wrote {args.out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
