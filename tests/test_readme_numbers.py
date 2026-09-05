"""Every headline figure in the README must still be in an artifact.

This has already gone wrong once. Three of the four rows the README quoted from
`results/C2_dev.txt` were not in that file: `--out` wrote the scorecard and
dropped the model block that had gone to stdout, so the document described a run
the repository could not produce. It was found by hand, months later.

Nobody mistypes these on purpose. That is exactly why it needs a machine to
notice -- a figure is copied into prose once and then outlives the run it came
from, and a reader has no way to tell which numbers are generated and which were
typed. `test_the_readme_states_the_real_test_count` does this for the test
count; this does it for the results.

The check is deliberately narrow. It does not parse the README's tables or try
to understand them. It takes a handful of claims that carry the argument, says
where each one must appear verbatim, and fails if the artifact stops saying it.
A number the README stops quoting is removed from this list; a number that moves
is regenerated in both places by the command that produced it.
"""

from __future__ import annotations

import pathlib

import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


#: (label, the string the README states, the artifact that has to contain it).
#: Every entry is a number a judge would read off the front page.
CLAIMS = [
    # The external benchmark -- the only figures here not measured on our data.
    ("BenchRec coverage", "84.36%", "results/benchrec_recoagent.txt"),
    ("BenchRec wrong-match rate", "0.28%", "results/benchrec_recoagent.txt"),
    ("BenchRec baseline coverage", "64.90%", "results/benchrec_baseline.txt"),
    ("BenchRec baseline wrong-match", "4.80%", "results/benchrec_baseline.txt"),
    # The B2 ladder.
    ("B2 dev auto-match", "98.15%", "results/B2_dev.txt"),
    ("B2 dev leg-2 recall", "97.56%", "results/B2_dev.txt"),
    ("B2 held-out leg-2 recall", "98.78%", "results/B2_holdout.txt"),
    ("B2 dev documented variance", "Rs 4,17,173.79", "results/B2_dev.txt"),
    # The unknown-class holdout.
    ("unknown leg-2 recall", "88.10%", "results/B2_unknown.txt"),
    ("unknown containment", "14 of 14", "results/B2_unknown.txt"),
    # Categorisation.
    ("C0 coverage", "0.86%", "results/C0_dev.txt"),
    ("C1 coverage", "94.86%", "results/C1_dev.txt"),
]


@pytest.mark.parametrize(
    "label, figure, artifact", CLAIMS, ids=[c[0] for c in CLAIMS]
)
def test_a_headline_figure_is_backed_by_its_artifact(label, figure, artifact):
    assert figure in README, (
        f"the README no longer states {label} as {figure!r}. If the number "
        "changed, regenerate the artifact and update both; if the claim was "
        "dropped, remove it from CLAIMS."
    )
    path = ROOT / artifact
    assert path.is_file(), f"{artifact} is missing"
    assert figure in path.read_text(), (
        f"the README claims {label} = {figure!r}, and {artifact} does not say "
        "it. Regenerate the artifact with the command the README publishes, or "
        "correct the README."
    )


def test_the_unknown_class_verdict_is_not_quietly_downgraded():
    """The README's strongest claim, tied to the run that proves it.

    Every other figure here would merely be stale if it drifted. This one would
    be false: the README says nothing unexplainable was guessed at, and the only
    thing standing behind that sentence is a line in the artifact.
    """
    artifact = (ROOT / "results/B2_unknown.txt").read_text()
    assert "Safety property beyond the written taxonomy: HOLDS" in artifact
    assert "WRONG OR UNNOTICED           0" in artifact
    assert "nothing guessed" in README or "wrong-matched or missed" in README


# ─────────────────────────────────────────────────────────────────────────────
# The Q&A counts
#
# These were caught by a reader, not by this file. The README stated "17 derived
# questions", then a table reading Correct 17 / Declined 2 / Coverage 88.24% --
# three numbers that look like three different denominators and are in fact one
# arithmetic: 15 answered right + 2 correctly declined = 17 correct of 17 asked,
# with coverage counting only the 15 it answered. Nothing pinned that, so
# nothing noticed the ambiguity.
#
# The artifact is parsed rather than string-matched, because the failure mode
# here is arithmetic between three figures rather than one stale figure.
# ─────────────────────────────────────────────────────────────────────────────

QA_ARTIFACTS = ["results/qa_dev.txt", "results/qa_holdout.txt"]


def _qa_counts(artifact: str) -> dict[str, int | float]:
    text = (ROOT / artifact).read_text()
    coverage = re.search(r"coverage\s+([\d.]+)%\s+\((\d+) of (\d+) answered\)", text)
    tallies = re.search(
        r"correct (\d+)(?: of \d+)?.*?wrong (\d+)\s+declined (\d+)\s+call failed (\d+)",
        text, re.S,
    )
    assert coverage, f"{artifact} no longer prints a coverage line"
    assert tallies, f"{artifact} no longer prints the correct/wrong/declined tallies"
    return {
        "coverage_pct": float(coverage.group(1)),
        "answered": int(coverage.group(2)),
        "total": int(coverage.group(3)),
        "correct": int(tallies.group(1)),
        "wrong": int(tallies.group(2)),
        "declined": int(tallies.group(3)),
        "failed": int(tallies.group(4)),
    }


@pytest.mark.parametrize("artifact", QA_ARTIFACTS)
def test_the_qa_counts_add_up_within_the_artifact(artifact):
    """Answered + declined + failed must be every question asked, and `correct`
    must be the answers it got right plus the declines it was right to make.

    Either identity failing means the report is quoting two populations under
    one heading, which is exactly what made the README unreadable.
    """
    c = _qa_counts(artifact)
    assert c["answered"] + c["declined"] + c["failed"] == c["total"], c
    assert c["correct"] == (c["answered"] - c["wrong"]) + c["declined"], c
    assert round(c["answered"] / c["total"] * 100, 2) == c["coverage_pct"], c


@pytest.mark.parametrize("artifact", QA_ARTIFACTS)
def test_the_readme_states_the_qa_numbers_the_artifacts_hold(artifact):
    c = _qa_counts(artifact)
    assert f"{c['total']} derived questions" in README
    assert f"{c['answered']} of {c['total']}" in README
    assert f"{c['correct']} of {c['total']}" in README
    assert f"{c['coverage_pct']:.2f}%" in README


def test_the_readme_names_the_seed_each_published_run_used():
    """A judge reproducing `--profile holdout` without `--seed 21` gets a
    different book and a table that looks fabricated. The profile does not fix
    the seed, so the README has to."""
    assert "--seed 7" in README and "--seed 21" in README
    assert "97.98%" in README, (
        "the README no longer shows what the wrong seed produces, which is the "
        "part that makes the warning concrete"
    )


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE.md's B3 table
#
# Also caught by a reader. EVIDENCE carried needs-approval 4/5, declined 2/3 and
# malformed 1/1 while the artifacts said 4/7, 1/2 and 2/0 -- the whole held-out
# column wrong and dev's declined and malformed transposed. The README was
# right, which is worse rather than better: two documents disagreeing is what a
# judge cross-checking actually finds.
#
# Only this file pinned any prose against artifacts, and it only read README.md.
# ─────────────────────────────────────────────────────────────────────────────

EVIDENCE = (ROOT / "EVIDENCE.md").read_text()

#: (row label in EVIDENCE.md, the line label in results/B3_*_nopaper.txt)
B3_ROWS = [
    ("needs approval", "needs approval"),
    ("declined by the model", "declined by the model"),
    ("rejected by the gate", "rejected by the gate"),
    ("malformed reply", "malformed reply"),
]


def _b3_count(profile: str, label: str) -> int:
    text = (ROOT / f"results/B3_{profile}_nopaper.txt").read_text()
    m = re.search(rf"^\s*{re.escape(label)}\s+(\d+)", text, re.M)
    assert m, f"B3_{profile}_nopaper.txt no longer reports {label!r}"
    return int(m.group(1))


@pytest.mark.parametrize("row, label", B3_ROWS, ids=[r[0] for r in B3_ROWS])
def test_evidence_b3_table_matches_the_artifacts(row, label):
    dev, holdout = _b3_count("dev", label), _b3_count("holdout", label)
    pattern = rf"^\|\s*{re.escape(row)}\s*\|\s*\**(\d+)\**\s*\|\s*\**(\d+)\**\s*\|"
    m = re.search(pattern, EVIDENCE, re.M)
    assert m, f"EVIDENCE.md no longer has a B3 row for {row!r}"
    assert (int(m.group(1)), int(m.group(2))) == (dev, holdout), (
        f"EVIDENCE.md says {row} = {m.group(1)}/{m.group(2)}, the artifacts say "
        f"{dev}/{holdout}. Regenerate the run or correct EVIDENCE.md."
    )


def test_evidence_and_readme_agree_on_what_the_agent_tier_resolved():
    """The one B3 number that would be a lie rather than merely stale."""
    for profile in ("dev", "holdout"):
        assert _b3_count(profile, r"RESOLVED \(source-backed\)".replace("\\", "")) == 0 \
            or _b3_count(profile, "RESOLVED (source-backed)") == 0
    assert "**0** | **0**" in EVIDENCE or "| **0** | **0** |" in EVIDENCE


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE.md's clearing-account table
#
# Same class as the B3 table above: prose quoting an artifact, with nothing
# checking it. Pinned here because the journal artifacts have already drifted
# once -- `format_inr` changed the rupee notation repository-wide and the
# committed reports were not regenerated, so CI's own diff would have failed.
# ─────────────────────────────────────────────────────────────────────────────

CAUSE_ROWS = [
    "The gateway has not paid this batch out",
    "A payment in the batch never matched an order",
    "A payment reported here was credited with another cycle",
    "An FX or repricing difference against the reported figure",
    "Sub-rupee rounding between the gateway and the bank",
]


def _journal_cause(profile: str, cause: str) -> str:
    text = (ROOT / f"results/journal_{profile}.txt").read_text()
    # Case-insensitive rather than lowercased: the artifact writes "an FX or
    # repricing difference", and .lower() turned FX into fx and silently
    # stopped matching the one row whose wording carries an acronym.
    m = re.search(rf"^\s*{re.escape(cause)}\s+\d+\s+(-?Rs [\d,]+\.\d\d)\s*$",
                  text, re.M | re.I)
    assert m, f"journal_{profile}.txt no longer reports {cause!r}"
    return m.group(1)


@pytest.mark.parametrize("cause", CAUSE_ROWS, ids=[c[:28] for c in CAUSE_ROWS])
def test_evidence_clearing_account_table_matches_the_artifacts(cause):
    dev, holdout = _journal_cause("dev", cause), _journal_cause("holdout", cause)
    pattern = rf"^\|\s*{re.escape(cause)}\s*\|\s*\**(-?Rs [\d,]+\.\d\d)\**\s*\|\s*\**(-?Rs [\d,]+\.\d\d)\**\s*\|"
    m = re.search(pattern, EVIDENCE, re.M)
    assert m, f"EVIDENCE.md no longer has a clearing-account row for {cause!r}"
    assert (m.group(1), m.group(2)) == (dev, holdout), (
        f"EVIDENCE.md says {cause} = {m.group(1)}/{m.group(2)}, the artifacts "
        f"say {dev}/{holdout}."
    )


@pytest.mark.parametrize("profile", ["dev", "holdout"])
def test_the_published_books_balance_and_leave_nothing_unattributed(profile):
    """The two claims the journal exists to make, read off the artifact rather
    than recomputed -- so a stale report fails here as loudly as a broken one."""
    text = (ROOT / f"results/journal_{profile}.txt").read_text()
    assert re.search(r"TRIAL BALANCE\s+BALANCED", text), f"{profile} does not balance"
    assert re.search(r"entries that do not balance\s+0\s*$", text, re.M)
    assert re.search(r"unattributed\s+Rs 0\.00", text), f"{profile} has unattributed money"


# ─────────────────────────────────────────────────────────────────────────────
# The caveat on the headline zero
#
# A 0.00% false-match rate is the first thing a reader sees and the easiest
# thing in the repository to over-read. `recoagent.audit.gate` measures what
# actually produces it -- forcing every failing proof open leaves it unchanged,
# because the pairing comes from an identifier join and the gate only checks
# the money. That finding is worth nothing if it can quietly leave the README.
# ─────────────────────────────────────────────────────────────────────────────


def test_the_readme_says_what_the_zero_does_not_mean():
    assert "What the 0.00% does not mean" in README
    for probe in ("recoagent.audit.gate", "0.28%", "17 wrong",
                  "DUPLICATE_PAYMENT"):
        assert probe in README, probe


@pytest.mark.parametrize("profile", ["dev", "holdout"])
def test_the_gate_probe_artifact_still_shows_the_gate_is_not_what_holds_it(profile):
    """If this ever stops being true, the README paragraph is wrong and must be
    rewritten rather than left standing."""
    text = (ROOT / f"results/gate_{profile}.txt").read_text()
    rows = re.findall(r"^  (as shipped|every proof forced open)\s+([\d.]+)%",
                      text, re.M)
    assert len(rows) == 2, text[:400]
    assert rows[0][1] == rows[1][1] == "0.0000", rows
    forced = re.search(r"accepted anyway\s+([\d,]+)", text)
    assert forced and int(forced.group(1).replace(",", "")) > 0, (
        "no proofs were forced open, so the probe measured nothing"
    )
