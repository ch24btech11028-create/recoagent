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
    "bank_ref": ("utr", "reference", "ref", "transaction_ref"),
    "narration": ("description", "particulars", "remarks", "detail"),
    "value_date": ("date", "txn_date", "posting_date", "credit_date"),
    "settled_at": ("settlement_date", "payout_date"),
    "captured_at": ("payment_date", "txn_date", "created_at"),
    "booked_at": ("adjustment_date", "date"),
    "bank_line_id": ("id", "line_id", "statement_line_id"),
    "settlement_id": ("payout_id", "batch_id"),
    "order_id": ("order_ref", "merchant_order_id", "receipt"),
    "payment_id": ("txn_id", "transaction_id"),
    "method": ("payment_method", "instrument", "mode"),
    "currency": ("ccy", "curr"),
}

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


def _resolve_columns(row_type, headers, overrides: dict[str, str]) -> dict[str, str]:
    """Decide which CSV column feeds which field, or say exactly what is missing."""
    by_norm = {_norm(h): h for h in headers}
    chosen: dict[str, str] = {}
    missing: list[str] = []

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
    return chosen



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
            columns = _resolve_columns(row_type, headers, overrides)

            built, problems = [], []
            for n, raw_row in enumerate(reader, start=2):  # 1 is the header
                where = f"{path.name}:{n}"
                try:
                    values = {}
                    for field in dataclass_fields(row_type):
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
