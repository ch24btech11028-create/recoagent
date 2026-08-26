"""Score the LLM baseline that ships inside BenchRec.

BenchRec (ICAIF 2023, CC BY 4.0) includes `MatcherByChatGPT_submission.csv` --
a language-model matcher's predictions over the same 32,048 labelled rows the
solution file covers. Scoring it gives an external, third-party data point for
the argument this whole system is built around, on data nobody here generated.

**This does not evaluate RecoAgent, and must never be presented as if it does.**
BenchRec is a different problem shape: prediction of an allocation key from a
bank statement line, resolved by text similarity, against one account in one
currency. RecoAgent solves N:1 subset matching over a payment batch with a fee
model. There is no fee to model here, so `prove_leg2` has nothing to say about
it.

It is *mostly* 1:1 but not entirely -- 6,000 of BenchRec's 56,074 training
groups carry more than one row a side, including 532 that are 2:1 and 424 that
are 3:1. `recoagent.eval.benchrec` runs our own matcher over the same rows and
reports what that structure is and is not worth.

What transfers is the *metric*, and that is the point of running this at all.

Usage:
    python -m recoagent.eval.benchrec_baseline --data data/benchrec
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(10**7)

SOLUTION = "BenchRec_cash_v1.0_solution.csv"
SUBMISSION = "MatcherByChatGPT_submission.csv"

#: The submission's own `targetAllocation` column holds a JSON-encoded list, not
#: a prediction. `A_allocation` is the predicted key. Scoring the wrong column
#: reports 0.00% accuracy and looks like a catastrophic baseline rather than a
#: reading error -- which is exactly what happened the first time.
PREDICTION_COLUMN = "A_allocation"


def score(data_dir: Path) -> dict:
    solution: dict[str, str] = {}
    with open(data_dir / SOLUTION, newline="") as fh:
        for row in csv.DictReader(fh):
            solution[row["B_id"]] = row["targetAllocation"].strip()

    submission: dict[str, dict] = {}
    with open(data_dir / SUBMISSION, newline="") as fh:
        for row in csv.DictReader(fh):
            submission[row["B_id"]] = row

    attempted = correct = wrong = 0
    bins: dict[float, list[int]] = {}

    for b_id, truth in solution.items():
        row = submission.get(b_id)
        predicted = (row.get(PREDICTION_COLUMN) or "").strip() if row else ""
        if not predicted:
            continue
        attempted += 1
        hit = predicted == truth
        correct += hit
        wrong += not hit
        try:
            confidence = float(row.get("confidence") or 0)
        except (TypeError, ValueError):
            confidence = 0.0
        bucket = min(int(confidence * 10) / 10, 0.9)
        entry = bins.setdefault(bucket, [0, 0])
        entry[0] += hit
        entry[1] += 1

    population = len(solution)
    return {
        "population": population,
        "attempted": attempted,
        "correct": correct,
        "wrong": wrong,
        "coverage": attempted / population if population else 0.0,
        "precision": correct / attempted if attempted else 0.0,
        "wrong_match_rate": wrong / attempted if attempted else 0.0,
        "overall_accuracy": correct / population if population else 0.0,
        "calibration": {b: tuple(v) for b, v in sorted(bins.items())},
    }


def render(result: dict) -> str:
    lines = [
        "=" * 72,
        "BenchRec -- the LLM baseline shipped inside the dataset",
        "ICAIF 2023, CC BY 4.0. Scored against BenchRec_cash_v1.0_solution.csv.",
        "=" * 72,
        "",
        f"  population              {result['population']:>10,}",
        f"  attempted (coverage)    {result['attempted']:>10,}   {result['coverage']:.2%}",
        f"  correct                 {result['correct']:>10,}",
        f"  WRONG MATCHES           {result['wrong']:>10,}",
        "",
        f"  match precision         {result['precision']:>10.2%}",
        f"  WRONG-MATCH RATE        {result['wrong_match_rate']:>10.2%}",
        f"  overall accuracy        {result['overall_accuracy']:>10.2%}",
        "",
        "  its reported confidence vs what actually happened:",
    ]
    for bucket, (hits, n) in result["calibration"].items():
        if n >= 100:
            lines.append(f"    conf {bucket:.1f}-{bucket + 0.1:.1f}   actual {hits / n:6.1%}   n={n:,}")
    lines += [
        "",
        "  Read the wrong-match rate, not the accuracy. Every one of those is",
        "  money filed against the wrong entry, and the matcher reported high",
        "  confidence on all of them -- the failure mode RecoAgent's arithmetic",
        "  gate exists to make structurally impossible.",
        "",
        "  NOT a RecoAgent result -- see recoagent.eval.benchrec for that.",
        "  Different problem shape: allocation-key",
        "  prediction by text similarity, not N:1 subset matching over a batch.",
        "=" * 72,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.eval.benchrec_baseline")
    ap.add_argument("--data", default="data/benchrec", type=Path)
    args = ap.parse_args(argv)

    if not (args.data / SOLUTION).exists():
        print(
            f"BenchRec not found in {args.data}. It is not committed (122MB).\n"
            "  kaggle datasets download -d benchmarkteam/"
            "benchrec-real-world-cash-reconciliation-dataset -p data/benchrec --unzip",
            file=sys.stderr,
        )
        return 2

    print(render(score(args.data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
