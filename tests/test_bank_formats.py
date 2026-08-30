"""Five real bank statement layouts, one book, one answer.

`tests/fixtures/banks/` holds the same nine settlement credits written out as
HDFC, ICICI, SBI, Axis and Kotak actually export them. Only the presentation
differs: column names, date format, lakh grouping, and -- the one that matters
-- the amount split across a debit and a credit column instead of carried as
one signed number.

The assertion is not "each file loads". It is that all five reconcile to the
*same* result as the native-format load. A loader can succeed on a file and
still have read it wrong; identical output across five presentations is what
rules that out.

Regenerate the fixtures with `python tests/fixtures/banks/make_fixtures.py`.
"""

import csv
from pathlib import Path

import pytest

from recoagent.ingest import IngestError, load, read_rows
from recoagent.pipeline import run_b2

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "banks"
BANKS = ("hdfc", "icici", "sbi", "axis", "kotak")


def _load(bank: str):
    return load(
        {
            "orders": FIXTURES / "orders.csv",
            "payments": FIXTURES / "payments.csv",
            "settlements": FIXTURES / "settlements.csv",
            "bank": FIXTURES / f"{bank}.csv",
        }
    )


def _fingerprint(sources):
    """What the matcher concluded, independent of how the file was written.

    Bank line *ids* are deliberately excluded: three of these formats carry no
    identifier, so the ids are synthesised from the filename and can never
    agree across layouts. Everything downstream of the money can.
    """
    result = run_b2(sources)
    return {
        "matches": sorted(
            (m.leg, m.right_ids[0], m.rule_id) for m in result.matches
        ),
        "exception_reasons": sorted(
            e.reason for e in result.exceptions if e.entity_kind != "bank_line"
        ),
        "credits": sorted(b.amount_paise for b in sources.bank_lines),
        "narrations": sorted(b.narration for b in sources.bank_lines),
        "dates": sorted(b.value_date for b in sources.bank_lines),
    }


@pytest.mark.parametrize("bank", BANKS)
def test_every_layout_loads(bank):
    sources = _load(bank)
    assert sources.bank_lines, f"{bank}: no credits read"
    assert all(b.amount_paise > 0 for b in sources.bank_lines), (
        f"{bank}: a settlement credit came out as zero or negative, which means "
        f"the deposit column was not the one that got read"
    )


def test_all_five_layouts_reconcile_identically():
    """The real assertion. Five presentations, one conclusion."""
    reference = _fingerprint(_load(BANKS[0]))
    for bank in BANKS[1:]:
        assert _fingerprint(_load(bank)) == reference, (
            f"{bank} reconciled differently from {BANKS[0]} -- the ingest layer "
            f"is reading the presentation, not the book"
        )


def test_the_layouts_are_actually_different():
    """A fixture set that quietly became five copies of one file would make the
    test above pass while proving nothing."""
    headers = {}
    for bank in BANKS:
        with (FIXTURES / f"{bank}.csv").open(newline="", encoding="utf-8") as fh:
            headers[bank] = tuple(next(csv.reader(fh)))
    assert len(set(headers.values())) == len(BANKS), (
        f"two layouts share a header row: {headers}"
    )
    # And at least one format must be missing a line identifier, or the
    # synthesis path is never exercised by this suite.
    idless = [b for b, h in headers.items()
              if not any(k in " ".join(h).lower() for k in ("s no", "sl. no"))]
    assert idless, "no layout exercises the synthesised bank_line_id path"


def test_a_row_filled_in_on_both_sides_is_refused(tmp_path):
    """Netting a both-sides row would turn a mapping mistake into a number that
    looks plausible. It is refused instead, naming the two columns."""
    path = tmp_path / "confused.csv"
    path.write_text(
        "Date,Narration,Chq./Ref.No.,Withdrawal Amt.,Deposit Amt.\n"
        "05/07/2026,NEFT CR-HDFC0000123-RAZORPAY-123456789012,REF1,100.00,250.00\n",
        encoding="utf-8",
    )
    with pytest.raises(IngestError) as exc:
        read_rows(path, "bank", unit="rupees", overrides={})
    assert "both sides" in str(exc.value)
    assert "Withdrawal Amt." in str(exc.value)
    assert "Deposit Amt." in str(exc.value)


def test_a_debit_only_row_comes_back_negative(tmp_path):
    """Money out has to survive the fold. A loader that reads only the credit
    column silently drops every debit and still looks like it worked."""
    path = tmp_path / "debit.csv"
    path.write_text(
        "Date,Narration,Chq./Ref.No.,Withdrawal Amt.,Deposit Amt.\n"
        "05/07/2026,CHARGEBACK DEBIT RAZORPAY 123456789012,REF1,1500.50,\n",
        encoding="utf-8",
    )
    (line,) = read_rows(path, "bank", unit="rupees", overrides={})
    assert line.amount_paise == -150050


def test_a_synthesised_line_id_says_where_it_came_from(tmp_path):
    """It must not be mistakable for a reference the bank issued."""
    path = tmp_path / "hdfc_july.csv"
    path.write_text(
        "Date,Narration,Chq./Ref.No.,Deposit Amt.\n"
        "05/07/2026,NEFT CR RAZORPAY 123456789012,REF1,10.00\n"
        "06/07/2026,NEFT CR RAZORPAY 123456789013,REF2,20.00\n",
        encoding="utf-8",
    )
    rows = read_rows(path, "bank", unit="rupees", overrides={})
    assert [r.bank_line_id for r in rows] == ["hdfc_july:2", "hdfc_july:3"]


def test_an_explicit_map_still_wins_over_the_folding(tmp_path):
    """`--map` is the escape hatch for a file nothing recognises, so it has to
    beat the automatic pair detection rather than race it."""
    path = tmp_path / "odd.csv"
    path.write_text(
        "Date,Narration,Chq./Ref.No.,Deposit Amt.,Settled Value\n"
        "05/07/2026,NEFT CR RAZORPAY 123456789012,REF1,10.00,77.00\n",
        encoding="utf-8",
    )
    (line,) = read_rows(
        path, "bank", unit="rupees", overrides={"amount_paise": "Settled Value"}
    )
    assert line.amount_paise == 7700
