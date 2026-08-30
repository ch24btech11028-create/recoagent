"""Post a reconciled book to the general ledger and print the trial balance.

Usage:
    python -m recoagent.journal --n 2000 --seed 7 --profile dev
    python -m recoagent.journal --n 2000 --profile holdout --out results/journal_holdout.txt
"""

from __future__ import annotations

import argparse
import sys

from ..categorize.rules import run_c1
from ..generator import DefectMix, GeneratorConfig, generate
from ..pipeline import run_b2
from .post import post, render

MIXES = {"dev": DefectMix.dev, "holdout": DefectMix.holdout, "clean": DefectMix.clean}
SEEDS = {"dev": 7, "holdout": 21, "clean": 7}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.journal")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, help="defaults to the profile's own seed")
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--out", help="write the report here")
    args = ap.parse_args(argv)

    seed = args.seed if args.seed is not None else SEEDS[args.profile]
    batch = generate(
        GeneratorConfig(n_orders=args.n, seed=seed, mix=MIXES[args.profile]())
    )
    result = run_b2(batch.sources)
    ledger = run_c1(batch.sources, result)
    journal = post(ledger, batch.sources, result)

    text = render(journal, batch.sources, result)
    print(text)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    # A book that does not balance is not a report, it is a bug.
    return 0 if journal.balances and not journal.unbalanced_entries else 1


if __name__ == "__main__":
    sys.exit(main())
