"""CLI entry point.

    python -m recoagent.run --n 500 --seed 7
    python -m recoagent.run --n 500 --seed 21 --profile holdout --exceptions 10
    python -m recoagent.run --n 500 --seed 7 --out run.json

The deterministic core has no third-party dependencies -- this runs on a clean
Python 3.11+ with nothing installed. Splink and the Anthropic SDK arrive with
rungs B1 and B3 respectively.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .eval.scorer import render, score
from .generator import DefectMix, GeneratorConfig, generate
from .money import format_inr
from .pipeline import run
from .schemas import ReconResult

MIXES = {
    "dev": DefectMix.dev,
    "holdout": DefectMix.holdout,
    "clean": DefectMix.clean,
}


def _canonical(result: ReconResult, card) -> dict:
    """A reproducibility artifact: everything the run decided, nothing about when.

    Wall-clock timestamps live on the audit records, where they belong, but
    they are excluded here so two runs of the same seed produce byte-identical
    output and drift becomes visible with `diff`.
    """
    return {
        "rung": result.rung,
        # Money that was matched but not reconciled away. Published at the top
        # of the artifact rather than left to be summed out of the match list,
        # because a reader who takes the headline rates and skips the detail
        # should still see it.
        "documented_variance_paise": sum(m.variance_paise for m in result.matches),
        "scorecard": {
            "profile": card.profile,
            "seed": card.seed,
            "false_match_rate": round(card.overall_false_match_rate, 6),
            "auto_match_rate": round(card.overall_auto_match_rate, 6),
            "value_share": round(card.value.share, 6) if card.value else None,
            "legs": {
                str(leg): {
                    "population": s.population,
                    "attempted": s.attempted,
                    "true_matches": s.true_matches,
                    "false_matches": s.false_matches,
                    "exceptions": s.exceptions,
                }
                for leg, s in sorted(card.legs.items())
            },
            "accounting": [asdict(a) | {"defect": a.defect.value} for a in card.accounting],
            "unattributed_exceptions": card.unattributed_exceptions,
            "fully_reconciles": card.fully_reconciles,
        },
        "matches": [
            {
                "match_id": m.match_id,
                "leg": m.leg,
                "tier": m.tier,
                "rule_id": m.rule_id,
                "left": list(m.left_ids),
                "right": list(m.right_ids),
                "confidence": m.confidence,
                "input_hash": m.input_hash,
                "residual_paise": m.proof.residual_paise if m.proof else None,
                "variance_paise": m.variance_paise,
            }
            for m in sorted(result.matches, key=lambda m: m.match_id)
        ],
        "exceptions": [
            {
                "exception_id": e.exception_id,
                "leg": e.leg,
                "entity_kind": e.entity_kind,
                "entity_id": e.entity_id,
                "reason": e.reason,
                "residual_paise": e.residual_paise,
                "suspected_class": e.suspected_class.value if e.suspected_class else None,
            }
            for e in sorted(result.exceptions, key=lambda e: e.exception_id)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.run", description=__doc__)
    ap.add_argument("--n", type=int, default=500, help="number of orders to generate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--rung", default="B0", choices=["B0", "B2"], help="baseline ladder rung to run")
    ap.add_argument("--out", help="write the canonical run artifact to this path")
    ap.add_argument(
        "--exceptions",
        type=int,
        default=0,
        metavar="K",
        help="print the K largest-residual exceptions",
    )
    args = ap.parse_args(argv)

    batch = generate(
        GeneratorConfig(n_orders=args.n, seed=args.seed, mix=MIXES[args.profile]())
    )
    result = run(args.rung, batch.sources)
    card = score(batch, result)

    counts = batch.sources.counts
    print()
    print(
        "  sources: "
        + "  ".join(f"{k}={v}" for k, v in counts.items())
        + f"  defects={len(batch.truth.defects)}"
    )
    print()
    print(render(card))

    variance = [m for m in result.matches if m.variance_paise]
    if variance:
        total = sum(m.variance_paise for m in variance)
        print(
            f"  Documented variance   {format_inr(total):>18}"
            f"     ({len(variance)} matches carry a declared gap; matched, not reconciled away)"
        )
        print()

    if args.exceptions:
        # Split by leg. Leg-1 partial captures are individually large and would
        # otherwise crowd out every leg-2 residual, which are the ones the SSMP
        # solver and the LLM tier exist to close.
        for leg, title in ((2, "Leg 2 -- unexplained credit residuals"),
                           (1, "Leg 1 -- order/capture disagreements")):
            pool = [
                e for e in result.exceptions
                if e.leg == leg and e.residual_paise is not None
            ]
            pool.sort(key=lambda e: abs(e.residual_paise or 0), reverse=True)
            print()
            print(f"  {title}  ({len(pool)} total, showing {min(args.exceptions, len(pool))})")
            for e in pool[: args.exceptions]:
                print(f"    {e.entity_id:<16} {format_inr(e.residual_paise or 0):>18}  {e.reason[:64]}")

        structural = [e for e in result.exceptions if e.residual_paise is None]
        by_reason: dict[str, int] = {}
        for e in structural:
            key = e.suspected_class.value if e.suspected_class else "unclassified"
            by_reason[key] = by_reason.get(key, 0) + 1
        print()
        print(f"  Structural exceptions: {len(structural)} "
              f"(no residual to explain -- broken keys, ambiguity, held funds)")
        for key, count in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"    {key:<28} {count:>5}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(_canonical(result, card), fh, indent=2, sort_keys=True)
        print(f"\n  wrote {args.out}")

    # Non-zero exit if the run failed its own honesty checks, so CI catches it.
    ok = card.fully_reconciles and card.overall_false_match_rate == 0.0
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
