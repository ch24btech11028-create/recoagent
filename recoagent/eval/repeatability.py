"""Does the agent tier give the same answer twice?

The deterministic rungs are byte-reproducible -- same seed, same output, and a
test asserts it. The LLM tier is not, and running the same profile twice made
that obvious: at temperature 0, individual cases flipped between `resolved` and
`refused` across runs while the headline resolution rate happened to land in the
same place. A single run reported as a point estimate would hide that entirely.

So this measures two different things, and only one of them is allowed to move:

- **Verdict stability** is expected to wobble. It is reported per case, as the
  share of runs that reached the modal outcome, rather than smoothed away.
- **False-match rate is not allowed to wobble.** The gate is arithmetic; it does
  not care how confident or how inconsistent the model was. If FMR is ever
  non-zero on any run, the safety property is not a property.

That second line is the real reason this file exists. Demonstrating the gate
holds once is an anecdote. Demonstrating it holds across every run while the
model underneath is visibly unstable is the claim worth making.

Usage:
    python -m recoagent.eval.repeatability --runs 3 --model nvidia/nemotron-3-ultra-550b-a55b
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

from ..agent import OpenAICompatibleProposer
from ..generator import DefectMix, GeneratorConfig, generate
from ..pipeline import run_b2, run_b3
from .scorer import score

PROFILES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout)}


def measure(profile: str, runs: int, model: str, workers: int, n_orders: int) -> dict:
    seed, mix_factory = PROFILES[profile]
    batch = generate(GeneratorConfig(n_orders=n_orders, seed=seed, mix=mix_factory()))
    baseline = score(batch, run_b2(batch.sources))

    per_run: list[dict] = []
    verdicts: dict[str, list[str]] = collections.defaultdict(list)
    failures: dict[str, list[str]] = collections.defaultdict(list)

    for i in range(1, runs + 1):
        started = time.time()
        result, report = run_b3(
            batch.sources,
            max_workers=workers,
            proposer_factory=lambda: OpenAICompatibleProposer(model=model, timeout=300),
        )
        card = score(batch, result)
        for case in report.cases:
            # A `failed` outcome is a transport or parsing failure, not the
            # model changing its mind. Folding the two together overstates
            # how inconsistent the model is and understates how flaky the
            # endpoint is -- two problems with two different fixes.
            verdicts[case.entity_id].append(case.outcome)
            if case.outcome == "failed":
                failures[case.entity_id].append(case.detail[:80])
        per_run.append(
            {
                "run": i,
                "resolved": report.resolved,
                "needs_approval": report.needs_approval,
                "attempted": report.attempted,
                "rejected": report.rejected,
                "refused": report.refused,
                "failed": report.failed,
                "leg2_recall": card.legs[2].recall,
                "false_match_rate": card.overall_false_match_rate,
                "mishandled": card.mishandled_total,
                "seconds": time.time() - started,
                "input_tokens": report.usage.input_tokens,
                "output_tokens": report.usage.output_tokens,
            }
        )
        print(
            f"    run {i}/{runs}: resolved {report.resolved}/{report.attempted}  "
            f"FMR {card.overall_false_match_rate:.2%}  "
            f"mishandled {card.mishandled_total}  {time.time() - started:.0f}s",
            flush=True,
        )

    return {
        "profile": profile,
        "model": model,
        "runs": per_run,
        "verdicts": dict(verdicts),
        "failures": dict(failures),
        "b2_recall": baseline.legs[2].recall,
    }


def render(data: dict) -> str:
    runs = data["runs"]
    resolved = [r["resolved"] for r in runs]
    recalls = [r["leg2_recall"] for r in runs]
    fmrs = [r["false_match_rate"] for r in runs]
    attempted = runs[0]["attempted"] if runs else 0

    out = [
        "=" * 72,
        f"REPEATABILITY  profile={data['profile']}  runs={len(runs)}",
        f"model={data['model']}",
        "=" * 72,
        "",
        f"  cases per run              {attempted}",
        f"  resolved       min/max     {min(resolved)} / {max(resolved)}"
        f"   mean {sum(resolved) / len(resolved):.1f}",
        f"  Leg 2 recall   min/max     {min(recalls):.2%} / {max(recalls):.2%}"
        f"   (B2 baseline {data['b2_recall']:.2%})",
        "",
        f"  needs approval (claimed rate) {sum(r.get('needs_approval', 0) for r in runs)}",
        "",
        f"  FALSE-MATCH RATE, every run  {max(fmrs):.2%}"
        f"   {'INVARIANT' if max(fmrs) == 0 else 'BROKEN'}",
        f"  defects mishandled, worst    {max(r['mishandled'] for r in runs)}",
        "",
        "-" * 72,
        "  PER-CASE VERDICT STABILITY",
        "-" * 72,
    ]

    model_unstable = infra_only = 0
    for entity, seen in sorted(data["verdicts"].items()):
        counts = collections.Counter(seen)
        answered = [v for v in seen if v != "failed"]
        distinct = set(answered)
        if len(distinct) > 1:
            model_unstable += 1
            flag = "   <- model changed its mind"
        elif "failed" in counts:
            infra_only += 1
            flag = "   <- call failed; the verdict itself never disagreed"
        else:
            flag = ""
        modal = counts.most_common(1)[0][0]
        share = (
            max(collections.Counter(answered).values()) / len(answered)
            if answered
            else 0.0
        )
        out.append(
            f"    {entity:<14} {modal:<13} {share:>5.0%} of answers"
            f"   {dict(counts)}{flag}"
        )

    total = len(data["verdicts"])
    failed_calls = sum(len(v) for v in data.get("failures", {}).values())
    out += [
        "-" * 72,
        f"  model reached different conclusions on : {model_unstable}/{total} cases",
        f"  affected only by a failed call         : {infra_only}/{total} cases",
        f"  failed calls across all runs           : {failed_calls}",
        "",
        "  The verdict column is expected to move; the false-match line is not.",
        "  A model that reaches different conclusions on identical input is a",
        "  reason to publish a range rather than a point estimate -- and a reason",
        "  the gate is arithmetic rather than a confidence threshold.",
        "=" * 72,
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.eval.repeatability")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="dev")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args(argv)

    print(f">>> {args.profile}: {args.runs} runs of {args.model}", flush=True)
    data = measure(args.profile, args.runs, args.model, args.workers, args.n)
    print()
    print(render(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
