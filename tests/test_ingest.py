"""Reading someone else's CSVs.

The round trip is the test that matters: write a generated book out as the
files a merchant would actually hand over, read them back through the public
entry point, and require the reconciliation to be identical. Anything the
reader silently mangles -- a paise lost to a float, a date read as a string, an
optional column dropped -- changes a match or an exception, and the comparison
catches it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import fields as dataclass_fields
from datetime import date, datetime

import pytest

from recoagent import ingest
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b2

FILES = {
    "orders": "orders",
    "payments": "payments",
    "adjustments": "adjustments",
    "settlements": "settlements",
    "bank": "bank_lines",
    "notices": "rate_notices",
    "fx": "fx_advices",
}


def _cell(value) -> str:
    """The way a real export writes it: rupees with two decimals, ISO dates."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _write(path, rows, *, rename=None):
    rename = rename or {}
    names = [f.name for f in dataclass_fields(type(rows[0]))]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([rename.get(n, n) for n in names])
        for row in rows:
            out = []
            for name in names:
                value = getattr(row, name)
                if name.endswith("_paise"):
                    # Rupees with two decimals -- the format that costs a naive
                    # reader a paise per row.
                    out.append(f"{value // 100}.{abs(value) % 100:02d}"
                               if value >= 0 else f"-{abs(value) // 100}.{abs(value) % 100:02d}")
                else:
                    out.append(_cell(value))
            w.writerow(out)


def _export(sources, tmp_path, *, rename=None):
    paths = {}
    for kind, attr in FILES.items():
        rows = getattr(sources, attr)
        if not rows:
            continue
        p = tmp_path / f"{kind}.csv"
        _write(p, list(rows), rename=(rename or {}).get(kind))
        paths[kind] = p
    return paths


@pytest.fixture(scope="module")
def book():
    return generate(GeneratorConfig(n_orders=400, seed=7, mix=DefectMix.dev()))


def test_a_book_survives_the_round_trip(book, tmp_path):
    """Same sources, same matches, same exceptions -- through CSV and back."""
    paths = _export(book.sources, tmp_path)
    loaded = ingest.load(paths)

    before, after = run_b2(book.sources), run_b2(loaded)
    assert [(m.match_id, m.rule_id, m.left_ids, m.right_ids, m.variance_paise)
            for m in sorted(before.matches, key=lambda m: m.match_id)] == \
           [(m.match_id, m.rule_id, m.left_ids, m.right_ids, m.variance_paise)
            for m in sorted(after.matches, key=lambda m: m.match_id)]
    assert [(e.exception_id, e.residual_paise) for e in before.exceptions] == \
           [(e.exception_id, e.residual_paise) for e in after.exceptions]


def test_headers_do_not_have_to_be_ours(book, tmp_path):
    """`Order ID`, `UTR`, `Amount` -- what an export actually calls things."""
    paths = _export(book.sources, tmp_path, rename={
        "orders": {"order_id": "Order ID", "amount_paise": "Amount"},
        "bank": {"bank_ref": "UTR", "value_date": "Value Date", "narration": "Particulars"},
        "settlements": {"settlement_id": "Payout ID", "net_paise": "Net"},
    })
    assert run_b2(ingest.load(paths)).matches


def test_a_column_it_cannot_place_is_named_not_guessed(book, tmp_path):
    paths = _export(book.sources, tmp_path)
    text = paths["bank"].read_text().replace("amount_paise", "mystery_column", 1)
    paths["bank"].write_text(text)

    with pytest.raises(ingest.IngestError) as exc:
        ingest.load(paths)
    assert "amount_paise" in str(exc.value)
    assert "mystery_column" in str(exc.value)   # it lists what the file does have
    assert "--map" in str(exc.value)            # ...and how to fix it


def test_a_map_points_a_field_at_a_column(book, tmp_path):
    paths = _export(book.sources, tmp_path)
    paths["bank"].write_text(paths["bank"].read_text().replace("amount_paise", "Weird Name", 1))

    loaded = ingest.load(paths, mapping={"bank": {"amount_paise": "Weird Name"}})
    assert len(loaded.bank_lines) == len(book.sources.bank_lines)


# ── money ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw, expected", [
    ("1234.56", 123456),
    ("1,234.56", 123456),
    ("1,234.5", 123450),      # one decimal is tenths of a rupee, not paise
    ("1234", 123400),
    ("-12.30", -1230),
    ("(12.30)", -1230),       # accounting negative
    ("Rs 1,234.56", 123456),
    ("₹1,234.56", 123456),
])
def test_rupees_become_paise_exactly(raw, expected):
    assert ingest._paise(raw, unit="rupees", where="t") == expected


def test_three_decimal_places_is_refused_not_rounded():
    with pytest.raises(ingest.IngestError) as exc:
        ingest._paise("12.345", unit="rupees", where="t")
    assert "--money paise" in str(exc.value)


def test_a_file_already_in_paise_is_read_as_paise():
    assert ingest._paise("123456", unit="paise", where="t") == 123456
    with pytest.raises(ingest.IngestError):
        ingest._paise("1234.56", unit="paise", where="t")


def test_a_bad_date_says_what_would_work():
    with pytest.raises(ingest.IngestError) as exc:
        ingest._datetime("last tuesday", where="t")
    assert "ISO 8601" in str(exc.value)


# ── what it reports ──────────────────────────────────────────────────────


def test_the_report_offers_coverage_and_refuses_to_offer_accuracy(book, tmp_path):
    """No answer key, no accuracy. The absence has to be stated, not implied."""
    sources = ingest.load(_export(book.sources, tmp_path))
    text = ingest.report(sources, run_b2(sources))

    assert "no false-match rate" in text.lower()
    assert "answer" in text.lower()
    assert "Credit value cleared" in text
    assert "EXCEPTIONS" in text
    # The words a scorecard would use are absent because the number is unknowable.
    assert "recall" not in text.lower()


def test_the_cli_runs_end_to_end(book, tmp_path, capsys):
    paths = _export(book.sources, tmp_path)
    out = tmp_path / "run.json"
    code = ingest.main([
        "--orders", str(paths["orders"]), "--payments", str(paths["payments"]),
        "--settlements", str(paths["settlements"]), "--bank", str(paths["bank"]),
        "--adjustments", str(paths["adjustments"]),
        "--notices", str(paths["notices"]), "--fx", str(paths["fx"]),
        "--out", str(out),
    ])
    assert code == 0
    written = json.loads(out.read_text())
    assert written["rung"] == "B2"
    assert written["matches"] and "exceptions" in written


def test_a_missing_file_is_a_message_not_a_traceback(book, tmp_path, capsys):
    paths = _export(book.sources, tmp_path)
    code = ingest.main([
        "--orders", str(tmp_path / "nope.csv"), "--payments", str(paths["payments"]),
        "--settlements", str(paths["settlements"]), "--bank", str(paths["bank"]),
    ])
    assert code == 2
    assert "no such file" in capsys.readouterr().err


def test_the_four_required_files_are_enough(book, tmp_path):
    """Adjustments, notices and FX advices are optional and must stay optional."""
    paths = _export(book.sources, tmp_path)
    minimal = {k: paths[k] for k in ("orders", "payments", "settlements", "bank")}

    sources = ingest.load(minimal)
    assert sources.adjustments == ()
    assert sources.rate_notices == () and sources.fx_advices == ()
    assert run_b2(sources).matches
