"""A persistent exception queue, and the one thing it exists to do.

Reconciliation is a pure function: same sources, same result, byte for byte --
that is asserted in CI and it is deliberate. **Reconciliation *work* is not.**
A credit that is missing on Tuesday arrives on Thursday. An analyst opens an
item, writes a note, and comes back to it after the weekend. A month-end run
happens forty times before anyone closes the book.

So the engine stays stateless and the *outcomes* are persisted here, keyed by
something stable enough to recognise the same item across runs. That split is
the whole design:

    the run says what is wrong today
    the worklist remembers what was wrong yesterday, and who is on it

Three properties this has to have, none of which are free:

**Idempotent.** Running the same batch twice must update the queue, not
duplicate it. The key is `(leg, entity_kind, entity_id)` -- the business
identity of the thing that did not reconcile -- and *not* the exception's
reason or id. A reason changes when a tier improves; it is not a different
problem when it does.

**Human work survives a re-run.** Status, assignee and notes are written by
people and are never overwritten by the pipeline. The pipeline owns the
reason, the residual and the timestamps; a person owns everything else. A
queue that loses an analyst's note on the next run is a queue nobody uses
twice.

**Carry-forward.** This is the point. An item that was unresolved in an earlier
run and *is matched* in a later one closes itself, citing the run that explains
it -- which is what "closing the loop" actually means for a finance team, as
opposed to printing a fresh exception list every morning and letting a person
diff it by eye.

Carry-forward is deliberately narrow. An item auto-resolves only when its
entity is **in scope** for the run -- present in the sources -- and **matched**
in the result. An entity that simply is not in this batch is not evidence of
anything and is left alone. Resolving on absence would close every July item
the moment someone ran August.

sqlite, because it is in the standard library, needs no service, and a
merchant's reconciliation history is measured in megabytes. The zero-dependency
property survives this file.

Usage:
    python -m recoagent.worklist --db work.db --n 2000 --seed 7
    python -m recoagent.worklist --db work.db --show
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import ReconException, ReconResult, SourceBundle

OPEN = "open"
INVESTIGATING = "investigating"
RESOLVED = "resolved"
WRITTEN_OFF = "written_off"

STATUSES = (OPEN, INVESTIGATING, RESOLVED, WRITTEN_OFF)

#: What a person may do. Anything not in here is refused rather than performed,
#: because a queue that lets an item go from `resolved` back to `open` without
#: saying so is a queue whose history cannot be trusted.
ALLOWED: dict[str, tuple[str, ...]] = {
    OPEN: (INVESTIGATING, RESOLVED, WRITTEN_OFF),
    INVESTIGATING: (OPEN, RESOLVED, WRITTEN_OFF),
    RESOLVED: (),
    WRITTEN_OFF: (),
}

#: Statuses the pipeline is allowed to close on its own. A written-off item
#: stays written off even if a later batch would have explained it: somebody
#: decided that money was not worth chasing, and a machine silently reopening
#: that decision is worse than leaving it.
AUTO_CLOSEABLE = (OPEN, INVESTIGATING)

ILLEGAL = "illegal transition"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    at          TEXT NOT NULL,
    rung        TEXT NOT NULL,
    label       TEXT NOT NULL DEFAULT '',
    n_sources   INTEGER NOT NULL,
    n_matches   INTEGER NOT NULL,
    n_exceptions INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    fingerprint     TEXT PRIMARY KEY,
    leg             INTEGER NOT NULL,
    entity_kind     TEXT NOT NULL,
    entity_id       TEXT NOT NULL,
    status          TEXT NOT NULL,
    reason          TEXT NOT NULL,
    residual_paise  INTEGER,
    suspected_class TEXT,
    assignee        TEXT NOT NULL DEFAULT '',
    notes           TEXT NOT NULL DEFAULT '',
    first_seen_run  INTEGER NOT NULL,
    last_seen_run   INTEGER NOT NULL,
    closed_run      INTEGER,
    closed_reason   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL,
    run_id      INTEGER,
    at          TEXT NOT NULL,
    from_status TEXT NOT NULL,
    to_status   TEXT NOT NULL,
    actor       TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS items_status ON items(status);
CREATE INDEX IF NOT EXISTS history_fp ON history(fingerprint);
"""


class WorklistError(Exception):
    """A refused operation, phrased for the person who attempted it."""


@dataclass(frozen=True)
class Item:
    fingerprint: str
    leg: int
    entity_kind: str
    entity_id: str
    status: str
    reason: str
    residual_paise: int | None
    suspected_class: str | None
    assignee: str
    notes: str
    first_seen_run: int
    last_seen_run: int
    closed_run: int | None
    closed_reason: str

    @property
    def is_open(self) -> bool:
        return self.status in AUTO_CLOSEABLE

    @property
    def age_in_runs(self) -> int:
        """How many runs this has survived. The number an ops lead cares about."""
        return max(0, self.last_seen_run - self.first_seen_run)


def fingerprint(exc: ReconException) -> str:
    """The stable identity of a problem, across runs and across tiers.

    Deliberately *not* the exception id or the reason. `x2_bank_0031` is stable
    only while the id scheme is; the reason changes the moment a tier learns to
    describe the same failure better. What does not change is which leg, and
    which thing on that leg, failed to reconcile.
    """
    return f"{exc.leg}:{exc.entity_kind}:{exc.entity_id}"


def _entities_in_scope(sources: SourceBundle) -> dict[str, set[str]]:
    """Every entity the run could have had an opinion about, by kind.

    Carry-forward needs this: an item is only closed by a run that actually
    looked at it. Silence is not evidence.
    """
    return {
        "bank_line": {b.bank_line_id for b in sources.bank_lines},
        "settlement": {s.settlement_id for s in sources.settlements},
        "order": {o.order_id for o in sources.orders},
        "payment": {p.payment_id for p in sources.payments},
        "adjustment": {a.adjustment_id for a in sources.adjustments},
    }


def _matched_ids(result: ReconResult) -> set[str]:
    ids: set[str] = set()
    for m in result.matches:
        ids.update(m.left_ids)
        ids.update(m.right_ids)
    return ids


class Worklist:
    """The queue. Open it on a path, or on `:memory:` for a test."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Worklist:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── ingestion ────────────────────────────────────────────────────────

    def record(
        self, sources: SourceBundle, result: ReconResult, *, label: str = ""
    ) -> dict[str, int]:
        """Fold one run into the queue. Returns what changed.

        The counts are the point of the return value: `carried_forward` is the
        number an ops team feels, because it is the work that closed without
        anyone touching it.
        """
        now = datetime.now(timezone.utc).isoformat()
        cur = self.db.execute(
            "INSERT INTO runs (at, rung, label, n_sources, n_matches, n_exceptions)"
            " VALUES (?,?,?,?,?,?)",
            (now, result.rung, label, sum(sources.counts.values()),
             len(result.matches), len(result.exceptions)),
        )
        run_id = int(cur.lastrowid)

        seen: dict[str, ReconException] = {}
        for exc in result.exceptions:
            # First occurrence wins. Two exceptions on one entity in one run is
            # the same problem described twice, not two problems.
            seen.setdefault(fingerprint(exc), exc)

        opened = updated = 0
        for fp, exc in seen.items():
            row = self.db.execute(
                "SELECT status FROM items WHERE fingerprint = ?", (fp,)
            ).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO items (fingerprint, leg, entity_kind, entity_id,"
                    " status, reason, residual_paise, suspected_class,"
                    " first_seen_run, last_seen_run)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (fp, exc.leg, exc.entity_kind, exc.entity_id, OPEN, exc.reason,
                     exc.residual_paise,
                     exc.suspected_class.value if exc.suspected_class else None,
                     run_id, run_id),
                )
                self._log(fp, run_id, now, "", OPEN, "pipeline", exc.reason)
                opened += 1
            else:
                # The pipeline owns the description; the person owns the state.
                # Note what is *not* in this UPDATE: status, assignee, notes.
                self.db.execute(
                    "UPDATE items SET reason = ?, residual_paise = ?,"
                    " suspected_class = ?, last_seen_run = ?"
                    " WHERE fingerprint = ?",
                    (exc.reason, exc.residual_paise,
                     exc.suspected_class.value if exc.suspected_class else None,
                     run_id, fp),
                )
                updated += 1

        carried = self._carry_forward(sources, result, seen.keys(), run_id, now)
        self.db.commit()
        return {
            "run_id": run_id,
            "opened": opened,
            "still_open": updated,
            "carried_forward": carried,
        }

    def _carry_forward(
        self, sources, result, present, run_id: int, now: str
    ) -> int:
        """Close items this run actually explained.

        Two conditions, both required. The entity must be **in scope** -- this
        run looked at it -- and it must be **matched** in the result. An item
        whose entity is absent from the batch is left exactly as it was: a run
        that never saw something has said nothing about it.
        """
        in_scope = _entities_in_scope(sources)
        matched = _matched_ids(result)
        closed = 0

        rows = self.db.execute(
            "SELECT fingerprint, entity_kind, entity_id, status FROM items"
            f" WHERE status IN ({','.join('?' * len(AUTO_CLOSEABLE))})",
            AUTO_CLOSEABLE,
        ).fetchall()

        for row in rows:
            fp = row["fingerprint"]
            if fp in present:
                continue  # still an exception in this run
            if row["entity_id"] not in in_scope.get(row["entity_kind"], set()):
                continue  # not in this batch; this run has no opinion
            if row["entity_id"] not in matched:
                continue  # in scope, still unmatched, still open
            detail = f"matched in run {run_id}"
            self.db.execute(
                "UPDATE items SET status = ?, last_seen_run = ?, closed_run = ?,"
                " closed_reason = ? WHERE fingerprint = ?",
                (RESOLVED, run_id, run_id, detail, fp),
            )
            self._log(fp, run_id, now, row["status"], RESOLVED, "pipeline", detail)
            closed += 1
        return closed

    # ── human actions ────────────────────────────────────────────────────

    def transition(
        self, fp: str, to: str, *, actor: str = "analyst", detail: str = ""
    ) -> Item:
        """Move an item, or refuse and say why."""
        if to not in STATUSES:
            raise WorklistError(
                f"{to!r} is not a status. Known: {', '.join(STATUSES)}"
            )
        item = self.get(fp)
        if to not in ALLOWED[item.status]:
            raise WorklistError(
                f"{ILLEGAL}: {item.status} -> {to} for {fp}. "
                f"From {item.status} you may go to: "
                f"{', '.join(ALLOWED[item.status]) or 'nowhere -- it is closed'}"
            )
        now = datetime.now(timezone.utc).isoformat()
        closed = to in (RESOLVED, WRITTEN_OFF)
        self.db.execute(
            "UPDATE items SET status = ?, closed_reason = ? WHERE fingerprint = ?",
            (to, detail if closed else "", fp),
        )
        self._log(fp, None, now, item.status, to, actor, detail)
        self.db.commit()
        return self.get(fp)

    def annotate(self, fp: str, *, assignee: str | None = None,
                 notes: str | None = None) -> Item:
        """Attach human work. Never touched by a re-run."""
        item = self.get(fp)
        self.db.execute(
            "UPDATE items SET assignee = ?, notes = ? WHERE fingerprint = ?",
            (item.assignee if assignee is None else assignee,
             item.notes if notes is None else notes, fp),
        )
        self.db.commit()
        return self.get(fp)

    # ── reading ──────────────────────────────────────────────────────────

    def get(self, fp: str) -> Item:
        row = self.db.execute(
            "SELECT * FROM items WHERE fingerprint = ?", (fp,)
        ).fetchone()
        if row is None:
            raise WorklistError(f"no item {fp!r} in this worklist")
        return Item(**dict(row))

    def items(self, *, status: str | None = None) -> list[Item]:
        sql = "SELECT * FROM items"
        args: tuple = ()
        if status is not None:
            sql += " WHERE status = ?"
            args = (status,)
        # Biggest money first, then oldest: the order an analyst works.
        sql += " ORDER BY COALESCE(ABS(residual_paise), 0) DESC, first_seen_run ASC"
        return [Item(**dict(r)) for r in self.db.execute(sql, args)]

    def history(self, fp: str) -> list[dict]:
        return [
            dict(r) for r in self.db.execute(
                "SELECT * FROM history WHERE fingerprint = ? ORDER BY id", (fp,)
            )
        ]

    def counts(self) -> dict[str, int]:
        out = {s: 0 for s in STATUSES}
        for r in self.db.execute(
            "SELECT status, COUNT(*) AS n FROM items GROUP BY status"
        ):
            out[r["status"]] = r["n"]
        return out

    def _log(self, fp, run_id, at, frm, to, actor, detail) -> None:
        self.db.execute(
            "INSERT INTO history (fingerprint, run_id, at, from_status,"
            " to_status, actor, detail) VALUES (?,?,?,?,?,?,?)",
            (fp, run_id, at, frm, to, actor, detail),
        )
