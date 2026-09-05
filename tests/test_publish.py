"""The summary page may not say more than the artifacts do.

A one-page scorecard is the easiest place in a project to start overstating
things, so the page is generated rather than written: every figure is read from
a file in `results/`. These tests check that property directly -- that the
numbers on the page are the numbers in the artifacts, and that the unflattering
ones are on it too.
"""

import json
import re
from pathlib import Path

import pytest

from recoagent import publish

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def page():
    return publish.build("https://example.invalid/repo")


def test_it_is_self_contained(page):
    """Nothing is *fetched*, same as everything else here. A scorecard that
    needs a CDN renders blank on a locked-down machine.

    Anchor hrefs are deliberately not checked: a link a reader may choose to
    follow is not a resource the page loads, and the header's link back to the
    repository is the whole point of having one.
    """
    for pattern in (r"\ssrc\s*=", r"<link\b", r"@import", r"url\(\s*['\"]?http",
                    r"fonts\.googleapis", r"<script\b"):
        assert not re.search(pattern, page, re.I), pattern


def test_the_headline_numbers_come_from_the_artifact(page):
    card = json.loads((ROOT / "results" / "B2_dev.json").read_text())["scorecard"]
    assert f"{card['false_match_rate']:.2%}" in page
    assert f"{card['auto_match_rate']:.2%}" in page
    for leg in ("1", "2"):
        assert f"{card['legs'][leg]['population']:,}" in page


def test_the_unflattering_numbers_are_on_the_page_too(page):
    """The declared limits, the open clearing-account items and the agent
    tier's zero. A summary that shows only the wins is an advertisement."""
    audit = json.loads((ROOT / "results" / "mutation_audit.json").read_text())
    assert "declared limits" in page
    for name in audit["declared_uncontained"]:
        assert name in page
    assert "the gateway has not paid this batch out" in page
    assert "Booked by the model" in page


def test_a_missing_artifact_leaves_a_visible_gap(monkeypatch, tmp_path):
    """A panel that vanished with its evidence would make the page look
    complete when it is not."""
    monkeypatch.setattr(publish, "RESULTS", tmp_path)
    bare = publish.build("https://example.invalid/repo")
    assert bare.count("has nothing to report") >= 5


def test_every_panel_renders(page):
    for heading in ("Reconciliation", "Attacking it on purpose", "The books",
                    "The agent tier, measured", "Throughput"):
        assert f"<h2>{heading}</h2>" in page


def test_the_page_calls_the_project_what_the_readme_calls_it(page):
    """A judge following the link from a repo called RecoAgent must not land on
    a page titled something else. The project was renamed once and the page
    kept the old name, which is exactly the kind of drift a reader reads as
    carelessness."""
    import re as _re
    from pathlib import Path as _Path

    name = _Path(__file__).resolve().parents[1] / "README.md"
    heading = _re.match(r"#\s+(\S+)", name.read_text()).group(1)
    assert f"<title>{heading}" in page, f"page title does not start with {heading!r}"
    assert f"<h1>{heading}</h1>" in page
