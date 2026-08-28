"""Run the categorisation ladder and print what each rung actually added.

    python -m recoagent.categorize.run --n 500 --seed 7
    python -m recoagent.categorize.run --n 500 --profile holdout --rung C2 \
        --model gemini/gemini-3.6-flash

C0 and C1 need no key and no network. C2 does, and the point of running the
ladder rather than C2 alone is that the model's contribution is then the
difference between two measured numbers instead of a claim.
"""

from __future__ import annotations

import argparse
import sys

from ..generator import DefectMix, GeneratorConfig, generate
from ..pipeline import run_b2
from . import rules
# `score` the function, not the module: the package re-exports the name and
# `from . import score` would bind the function here.
from .score import render, score

MIXES = {"dev": DefectMix.dev, "holdout": DefectMix.holdout, "clean": DefectMix.clean}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.categorize.run", description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--rung", choices=("C0", "C1", "C2"), default="C1")
    ap.add_argument("--model", default="gemini/gemini-3.6-flash")
    ap.add_argument("--rpm", type=int, help="override the requests-per-minute limit")
    ap.add_argument("--rpd", type=int, help="override the requests-per-day budget")
    ap.add_argument(
        "--no-cache", action="store_true",
        help="ask the model again even for requests already answered on disk",
    )
    ap.add_argument("--out", help="write the rendered scorecard here")
    args = ap.parse_args(argv)

    batch = generate(GeneratorConfig(n_orders=args.n, seed=args.seed, mix=MIXES[args.profile]()))

    if args.rung == "C0":
        ledger = rules.run_c0(batch.sources)
    else:
        ledger = rules.run_c1(batch.sources, run_b2(batch.sources))

    report = None
    if args.rung == "C2":
        from ..budget import budgeted, limits_for
        from ..llm import client_for
        from .agent import ChatCategoriser, run_c2

        left = rules.residue(batch.sources, ledger)
        print(f"\n  C1 left {len(left)} rows for the model.")
        if not left:
            print("\n  Nothing to ask. That is the finding, not an error.\n")
        else:
            chat = client_for(args.model)
            rpm, rpd = limits_for(chat.label)
            rpm, rpd = args.rpm or rpm, rpd if args.rpd is None else args.rpd
            print(
                f"  Budget: {rpm} requests/minute, {rpd}/day"
                + ("" if args.no_cache else ", cached on disk")
                + (f" -- {len(left) - rpd} rows will go unasked.\n"
                   if len(left) > rpd else ".\n")
            )
            report = run_c2(
                batch.sources, ledger,
                ChatCategoriser(budgeted(
                    chat, rpm=rpm, rpd=rpd,
                    cache=None if args.no_cache else "data/llm-cache",
                )),
            )

    card = score(ledger, batch.truth.categories, args.rung)
    text = render(card)
    print(text)

    if report is not None:
        print(f"  model                    {report.model}")
        print(f"  rows asked about         {report.attempted:>8}")
        print(f"  assigned on a quotation  {report.assigned:>8}")
        print(f"  discarded, quote not in the row  {report.uncited:>8}")
        print(f"  model declined           {report.declined:>8}")
        print(f"  call failed              {report.failed:>8}")
        if report.not_asked:
            print(f"  never asked (budget)     {report.not_asked:>8}"
                  "     unmeasured, not unresolved")
        print()

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"  wrote {args.out}\n")

    # Non-zero if the system filed anything under a wrong category, so CI
    # catches a regression the way it does for a false match.
    return 0 if card.wrong == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
