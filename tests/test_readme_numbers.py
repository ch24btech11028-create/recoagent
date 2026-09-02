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
