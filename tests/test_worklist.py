"""The queue has to survive being run again, which is the only hard part.

An exception list is easy. A queue is a list that remembers, and everything
that makes it useful is a property across *two* runs rather than one:
re-running must not duplicate, must not trample an analyst's work, and must
close what a later batch genuinely explained -- without closing what it merely
failed to mention.
"""

from dataclasses import replace

import pytest

from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b2
from recoagent.schemas import ReconException
from recoagent.worklist import Worklist, WorklistError
from recoagent.worklist.store import (
    INVESTIGATING,
    OPEN,
    RESOLVED,
    WRITTEN_OFF,
    fingerprint,
)

N = 800
SEED = 7


@pytest.fixture
def book():
    return generate(GeneratorConfig(n_orders=N, seed=SEED, mix=DefectMix.dev()))


@pytest.fixture
def wl():
    with Worklist(":memory:") as w:
        yield w


def _halves(book):
    """One book, two statements: month-to-date, then the whole month."""
    lines = sorted(book.sources.bank_lines, key=lambda b: b.value_date)
    cut = len(lines) * 2 // 3
    return replace(book.sources, bank_lines=tuple(lines[:cut])), book.sources


def test_the_same_run_twice_updates_and_never_duplicates(wl, book):
    src = book.sources
    first = wl.record(src, run_b2(src))
    count_after_one = len(wl.items())
    second = wl.record(src, run_b2(src))

    assert first["opened"] > 0, "the dev book produced no exceptions to queue"
    assert second["opened"] == 0, "a repeat of the same batch opened new items"
    assert second["still_open"] == count_after_one
    assert len(wl.items()) == count_after_one
    assert second["carried_forward"] == 0, (
        "nothing changed between the runs, so nothing should have closed"
    )


def test_an_analysts_work_survives_the_pipeline(wl, book):
    src = book.sources
    wl.record(src, run_b2(src))
    item = wl.items(status=OPEN)[0]

    wl.transition(item.fingerprint, INVESTIGATING, actor="asha")
    wl.annotate(item.fingerprint, assignee="asha", notes="chased the bank")

    wl.record(src, run_b2(src))  # the pipeline runs again over the same book

    after = wl.get(item.fingerprint)
    assert after.status == INVESTIGATING, "a re-run reset a human's status"
    assert after.assignee == "asha", "a re-run wiped the assignee"
    assert after.notes == "chased the bank", "a re-run wiped the notes"


def test_a_later_batch_closes_what_it_explains(wl, book):
    early, full = _halves(book)
    wl.record(early, run_b2(early), label="month-to-date")
    opened = {i.fingerprint for i in wl.items(status=OPEN)}
    assert opened, "the truncated statement produced no exceptions"

    changed = wl.record(full, run_b2(full), label="full month")

    assert changed["carried_forward"] > 0, (
        "the full statement explained nothing the partial one had open"
    )
    closed = [i for i in wl.items(status=RESOLVED)]
    assert closed
    for item in closed:
        assert item.closed_reason.startswith("matched in run "), item.closed_reason
        assert item.closed_run == changed["run_id"]


def test_an_item_the_run_never_saw_is_left_alone(wl, book):
    """Silence is not evidence. Resolving on absence would close every July
    item the moment somebody ran August."""
    early, _ = _halves(book)
    wl.record(early, run_b2(early))
    before = {i.fingerprint: i.status for i in wl.items()}

    # A batch with no bank lines at all: every bank-line item is now out of
    # scope, and none of them may move.
    empty = replace(early, bank_lines=())
    wl.record(empty, run_b2(empty))

    for fp, status in before.items():
        if fp.startswith("2:bank_line:"):
            assert wl.get(fp).status == status, (
                f"{fp} changed status on a run that could not see it"
            )


def test_an_item_still_unmatched_stays_open(wl, book):
    """The other half of carry-forward: a run that looked and still could not
    match must not close anything."""
    early, full = _halves(book)
    wl.record(early, run_b2(early))
    result = run_b2(full)
    still_failing = {
        fingerprint(e) for e in result.exceptions
    }
    wl.record(full, result)

    for fp in still_failing:
        item = wl.get(fp)
        assert item.status in (OPEN, INVESTIGATING), (
            f"{fp} is still an exception in the later run but was closed"
        )


def test_a_written_off_item_is_never_reopened_by_a_later_run(wl, book):
    """Somebody decided that money was not worth chasing. A machine quietly
    reversing that is worse than leaving it."""
    early, full = _halves(book)
    wl.record(early, run_b2(early))
    # Pick one that the later run *would* have closed, to make the test bite.
    result = run_b2(full)
    matched = {i for m in result.matches for i in (*m.left_ids, *m.right_ids)}
    victim = next(i for i in wl.items(status=OPEN) if i.entity_id in matched)

    wl.transition(victim.fingerprint, WRITTEN_OFF, actor="controller",
                  detail="below the chase threshold")
    wl.record(full, result)

    after = wl.get(victim.fingerprint)
    assert after.status == WRITTEN_OFF
    assert after.closed_reason == "below the chase threshold"


def test_illegal_transitions_are_refused_with_the_legal_ones_named(wl, book):
    src = book.sources
    wl.record(src, run_b2(src))
    item = wl.items(status=OPEN)[0]
    wl.transition(item.fingerprint, RESOLVED, actor="asha")

    with pytest.raises(WorklistError) as exc:
        wl.transition(item.fingerprint, OPEN)
    assert "illegal transition" in str(exc.value)
    assert "closed" in str(exc.value)

    with pytest.raises(WorklistError) as exc:
        wl.transition(item.fingerprint, "banana")
    assert "not a status" in str(exc.value)


def test_the_fingerprint_ignores_the_reason(wl):
    """A tier that learns to describe a failure better has not found a new
    problem, and must not open a second item for it."""
    a = ReconException(
        exception_id="x2_bank_0007", leg=2, entity_kind="bank_line",
        entity_id="bank_0007", reason="no readable UTR in narration",
    )
    b = replace(
        a,
        exception_id="different_id_scheme",
        reason="no readable UTR; amount and date window matched 3 settlements",
    )
    assert fingerprint(a) == fingerprint(b)


def test_history_records_who_did_what(wl, book):
    early, full = _halves(book)
    wl.record(early, run_b2(early))
    result = run_b2(full)
    matched = {i for m in result.matches for i in (*m.left_ids, *m.right_ids)}
    item = next(i for i in wl.items(status=OPEN) if i.entity_id in matched)

    wl.transition(item.fingerprint, INVESTIGATING, actor="asha")
    wl.record(full, result)

    trail = [(h["from_status"], h["to_status"], h["actor"])
             for h in wl.history(item.fingerprint)]
    assert trail == [
        ("", OPEN, "pipeline"),
        (OPEN, INVESTIGATING, "asha"),
        (INVESTIGATING, RESOLVED, "pipeline"),
    ], trail


def test_the_queue_is_ordered_by_money_then_age(wl, book):
    src = book.sources
    wl.record(src, run_b2(src))
    values = [abs(i.residual_paise or 0) for i in wl.items()]
    assert values == sorted(values, reverse=True), (
        "the queue is not biggest-money-first, which is the order it is worked in"
    )


def test_it_persists_across_open_and_close(tmp_path, book):
    src = book.sources
    db = tmp_path / "work.db"
    with Worklist(db) as w:
        w.record(src, run_b2(src))
        expected = {i.fingerprint for i in w.items()}
        w.transition(sorted(expected)[0], INVESTIGATING, actor="asha")

    with Worklist(db) as w:
        assert {i.fingerprint for i in w.items()} == expected
        assert w.get(sorted(expected)[0]).status == INVESTIGATING
