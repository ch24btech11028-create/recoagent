"""Run a batch into a worklist, and show what the queue looks like afterwards.

The demo worth watching is `--carry-forward`. It reconciles a book twice: once
with the bank statement truncated to the first part of the month, and once with
the whole thing. Settlements whose credit had not arrived yet are exceptions on
the first pass and **close themselves** on the second, citing the run that
explained them -- with any note an analyst wrote in between still attached.

    python -m recoagent.worklist --carry-forward
    python -m recoagent.worklist --db work.db --n 2000 --seed 7
    python -m recoagent.worklist --db work.db --show
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from ..generator import DefectMix, GeneratorConfig, generate
from ..money import format_inr
from ..pipeline import run_b2
from .store import OPEN, RESOLVED, Worklist

MIXES = {"dev": DefectMix.dev, "holdout": DefectMix.holdout, "clean": DefectMix.clean}


def show(wl: Worklist, *, limit: int = 15) -> str:
    counts = wl.counts()
    out = [
        "=" * 72,
        "  EXCEPTION WORKLIST",
        "=" * 72,
        "",
        f"  open           {counts['open']:>5}",
        f"  investigating  {counts['investigating']:>5}",
        f"  resolved       {counts['resolved']:>5}",
        f"  written off    {counts['written_off']:>5}",
        "",
        "-" * 72,
        f"  {'entity':<14}{'status':<15}{'runs':>5}{'value':>15}  reason",
        "-" * 72,
    ]
    for item in wl.items()[:limit]:
        value = format_inr(item.residual_paise) if item.residual_paise else "--"
        out.append(
            f"  {item.entity_id:<14}{item.status:<15}{item.age_in_runs:>5}"
            f"{value:>15}  {item.reason[:60]}"
        )
    remaining = len(wl.items()) - limit
    if remaining > 0:
        out.append(f"  ... and {remaining} more")
    out += ["=" * 72]
    return "\n".join(out)


def carry_forward_demo(n_orders: int, seed: int, profile: str) -> str:
    """Two runs over one book, the second one carrying a longer statement."""
    batch = generate(
        GeneratorConfig(n_orders=n_orders, seed=seed, mix=MIXES[profile]())
    )
    full = batch.sources
    lines = sorted(full.bank_lines, key=lambda b: b.value_date)
    if len(lines) < 4:
        raise SystemExit("book is too small to split a statement in two")

    cut = len(lines) * 2 // 3
    early = replace(full, bank_lines=tuple(lines[:cut]))

    out = [
        "=" * 72,
        "  CARRY-FORWARD: the same book, reconciled twice",
        "=" * 72,
        "",
        f"  Run 1 sees the first {cut} of {len(lines)} bank credits -- the month is",
        "  not over. Run 2 sees the whole statement.",
        "",
    ]

    with Worklist(":memory:") as wl:
        first = wl.record(early, run_b2(early), label="month-to-date")
        after_first = wl.counts()

        # An analyst picks up an item between the runs, so the demo also shows
        # human work surviving the pipeline. The one chosen is a settlement
        # whose credit does arrive in the back half of the statement -- that is
        # staging, and it is stated rather than hidden, because the interesting
        # half of this demo is the item that does *not* close.
        arriving = {
            b.bank_line_id for b in lines[cut:]
        }
        later_settlements = {
            batch.truth.leg2[b] for b in arriving if b in batch.truth.leg2
        }
        open_items = [i for i in wl.items(status=OPEN) if i.entity_kind == "settlement"]
        touched = next((i for i in open_items if i.entity_id in later_settlements), None)
        stuck = next((i for i in open_items if i.entity_id not in later_settlements), None)
        if touched:
            wl.transition(touched.fingerprint, "investigating", actor="asha")
            wl.annotate(touched.fingerprint, assignee="asha",
                        notes="chased the bank, credit expected Thursday")

        second = wl.record(full, run_b2(full), label="full month")
        after_second = wl.counts()

        out += [
            f"  Run 1  opened {first['opened']:>3} items"
            f"    (open {after_first['open']}, resolved {after_first['resolved']})",
            "",
            f"         asha takes {touched.entity_id if touched else '--'} "
            f"-> investigating, and leaves a note",
            "",
            f"  Run 2  opened {second['opened']:>3} items,"
            f" carried forward {second['carried_forward']:>3}"
            f"    (open {after_second['open']}, resolved {after_second['resolved']})",
            "",
        ]

        if touched:
            item = wl.get(touched.fingerprint)
            out += [
                "-" * 72,
                f"  {item.entity_id} -- the credit arrived",
                "-" * 72,
                f"    status        {item.status}"
                + ("   <- closed itself, nobody re-filed it"
                   if item.status == RESOLVED else ""),
                f"    closed by     {item.closed_reason or '--'}",
                f"    assignee      {item.assignee or '--'}"
                "        <- survived the re-run",
                f"    notes         {item.notes or '--'}",
                "",
                "    history:",
            ]
            for h in wl.history(item.fingerprint):
                frm = h["from_status"] or "(new)"
                out.append(
                    f"      {frm:>13} -> {h['to_status']:<14} "
                    f"{h['actor']:<9} {h['detail'][:34]}"
                )

        if stuck:
            item = wl.get(stuck.fingerprint)
            out += [
                "",
                "-" * 72,
                f"  {item.entity_id} -- the credit did not",
                "-" * 72,
                f"    status        {item.status}   <- still open, correctly",
                f"    reason        {item.reason[:56]}",
                "",
                "    Run 2 looked at this settlement and still could not match it,",
                "    so it stays. Carry-forward closes an item only when the run",
                "    both saw the entity and matched it -- an item whose entity is",
                "    merely absent from a batch is left alone. Resolving on",
                "    absence would close every July item the moment somebody ran",
                "    August.",
            ]

        out += [
            "",
            "  The engine is stateless and gave the same answer both times. What",
            "  changed is that the second run *closed* the first run's work",
            "  instead of reprinting it, and the note written in between is",
            "  still on the item. That is the difference between an exception",
            "  list and a queue.",
            "=" * 72,
        ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.worklist")
    ap.add_argument("--db", default=":memory:", help="sqlite file for the queue")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--show", action="store_true", help="print the queue and stop")
    ap.add_argument(
        "--carry-forward", action="store_true",
        help="reconcile one book twice and show the queue closing itself",
    )
    args = ap.parse_args(argv)

    if args.carry_forward:
        print(carry_forward_demo(args.n, args.seed, args.profile))
        return 0

    with Worklist(args.db) as wl:
        if args.show:
            print(show(wl))
            return 0
        batch = generate(
            GeneratorConfig(n_orders=args.n, seed=args.seed, mix=MIXES[args.profile]())
        )
        changed = wl.record(
            batch.sources, run_b2(batch.sources),
            label=f"{args.profile} seed={args.seed}",
        )
        print(
            f"run {changed['run_id']}: opened {changed['opened']}, "
            f"still open {changed['still_open']}, "
            f"carried forward {changed['carried_forward']}"
        )
        print()
        print(show(wl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
