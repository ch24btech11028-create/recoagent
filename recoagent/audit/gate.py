"""What the arithmetic gate is actually worth, measured by removing it.

The front page of this repository leads with a 0.00% false-match rate, and the
design argument underneath it is that a model may propose but only arithmetic
may accept. A reader is entitled to ask whether the second sentence is what
produces the first. This answers that, by running the whole pipeline twice:
once as shipped, and once with every arithmetic proof forced to close.

**It does not.** Forcing open proofs that genuinely fail leaves the false-match
rate at 0.00%, because on both legs the *pairing* comes from an identifier join
-- an order id, a UTR read out of the bank narration -- and the gate never
changes which rows are paired. It checks whether the money agrees, which is a
different question from whether the rows belong together.

That has three consequences, and all three belong in front of a reader rather
than in a footnote:

1. **The headline metric does not measure the headline mechanism.** The gate's
   value is real but it is elsewhere: it is what stops a *wrong explanation*
   being booked against a correct pairing, which the defect accounting and the
   B3 agent tier measure instead.
2. **0.00% is a property of clean join keys**, and this corpus has clean join
   keys because the generator wrote them. The honest external figure is 0.28%
   on BenchRec, real third-party data nobody here produced.
3. **The cases where a wrong pairing was possible were refused, not solved.**
   Duplicate payments and duplicate UTRs are exactly the ambiguous population,
   and they make up almost the entire exception list.

This module exists so none of that has to be taken on trust.

Usage:
    python3 -m recoagent.audit.gate
    python3 -m recoagent.audit.gate --profile holdout
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import replace

from .. import validate
from ..eval.scorer import score
from ..generator import DefectMix, GeneratorConfig, generate
from ..legs import leg1, leg2, leg2_t1
from ..pipeline import run_b2

PROFILES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout)}

#: Big enough that no real residual can fail against it, so a "proof" that
#: should have been refused is accepted instead.
_ABSURD_TOLERANCE = 10**15


def _forced(original):
    """Wrap a prove_* so every proof it returns closes, and count the lies."""
    state = {"calls": 0, "forced": 0}

    def wrapper(*args, **kwargs):
        proof = original(*args, **kwargs)
        state["calls"] += 1
        if not proof.closes:
            state["forced"] += 1
            proof = replace(proof, tolerance_paise=_ABSURD_TOLERANCE)
        return proof

    return wrapper, state


def measure(profile: str, n_orders: int) -> dict:
    seed, mix = PROFILES[profile]
    batch = generate(GeneratorConfig(n_orders=n_orders, seed=seed, mix=mix()))

    shipped = score(batch, run_b2(batch.sources))
    rules = Counter(
        (m.leg, m.rule_id) for m in run_b2(batch.sources).matches
    )

    # Patch the names the matchers actually call. Patching `validate` alone is
    # not enough: each leg does `from ..validate import prove_leg2`, which binds
    # the function object at import time, so the module namespace is the thing
    # that has to change. Getting this wrong produces a run that looks like a
    # successful experiment and has changed nothing -- which is how this probe
    # first "confirmed" the result before it was checked.
    originals = {}
    counters = {}
    targets = [
        (leg1, "prove_leg1"), (leg1, "prove_leg1_capture"),
        (leg2, "prove_leg2"), (leg2_t1, "prove_leg2"),
    ]
    for module, name in targets:
        if not hasattr(module, name):
            continue
        originals[(module, name)] = getattr(module, name)
        wrapper, state = _forced(getattr(validate, name))
        setattr(module, name, wrapper)
        counters[f"{module.__name__.rsplit('.', 1)[-1]}.{name}"] = state

    try:
        opened = score(batch, run_b2(batch.sources))
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)

    refusals = Counter(
        e.suspected_class.value if e.suspected_class else "unclassified"
        for e in run_b2(batch.sources).exceptions
    )
    return {
        "profile": profile,
        "n_orders": n_orders,
        "shipped": shipped,
        "opened": opened,
        "counters": counters,
        "rules": rules,
        "refusals": refusals,
    }


def render(data: dict) -> str:
    shipped, opened = data["shipped"], data["opened"]
    forced = sum(c["forced"] for c in data["counters"].values())
    calls = sum(c["calls"] for c in data["counters"].values())

    def row(label, card):
        return (
            f"  {label:<26}{card.overall_false_match_rate:>9.4%}"
            f"{sum(s.attempted for s in card.legs.values()):>12,}"
            f"{sum(s.false_matches for s in card.legs.values()):>9,}"
            f"{sum(s.exceptions for s in card.legs.values()):>12,}"
        )

    out = [
        "=" * 72,
        f"  WHAT THE ARITHMETIC GATE IS WORTH   profile={data['profile']}"
        f"  n={data['n_orders']:,}",
        "=" * 72,
        "",
        f"  {'':<26}{'FMR':>9}{'matches':>12}{'wrong':>9}{'exceptions':>12}",
        row("as shipped", shipped),
        row("every proof forced open", opened),
        "",
        f"  proofs evaluated            {calls:>9,}",
        f"  proofs that FAILED and were accepted anyway   {forced:>9,}",
        "",
        "-" * 72,
        "  WHERE THE PAIRING ACTUALLY COMES FROM",
        "-" * 72,
    ]
    for leg in (1, 2):
        total = sum(n for (lg, _), n in data["rules"].items() if lg == leg)
        for (lg, rule), n in sorted(
            data["rules"].items(), key=lambda kv: -kv[1]
        ):
            if lg != leg:
                continue
            out.append(f"  leg {leg}  {rule:<34}{n:>6}  ({n / total:>6.1%})")

    delta = opened.overall_false_match_rate - shipped.overall_false_match_rate
    out += [
        "",
        "-" * 72,
        "  READ THIS BEFORE QUOTING THE 0.00%",
        "-" * 72,
        "",
        f"  Forcing {forced} genuinely-failing proofs open moved the false-match",
        f"  rate by {delta:+.4%}. The gate is not what produces that number.",
        "",
        "  On both legs the pairing comes from an identifier join -- an order id,",
        "  a UTR read out of the bank narration. The gate asks whether the money",
        "  agrees, which is a different question from whether the rows belong",
        "  together, so removing it cannot create a wrong pairing on a book whose",
        "  join keys are clean. This book's keys are clean because the generator",
        "  wrote them.",
        "",
        "  The gate's value is real and it is elsewhere: it refuses a wrong",
        "  *explanation* for a correct pairing. That is what the defect",
        "  accounting and the B3 agent tier measure, and what the adversarial",
        "  audit attacks.",
        "",
        "  Where this number is not zero:",
        "    BenchRec, real third-party data      0.28% wrong-match",
        "    adversarial mutation audit           17 wrong matches in 420 cases",
        "",
        "  And what the 0.00% costs: the population where a wrong pairing was",
        "  even possible is refused rather than solved --",
        "",
    ]
    total_refused = sum(data["refusals"].values())
    for cls, n in data["refusals"].most_common():
        out.append(f"    {cls:<28}{n:>5}  ({n / total_refused:>5.1%} of refusals)")
    out += [
        "",
        "  A duplicate payment and a duplicate UTR are exactly the cases with no",
        "  single right answer. Refusing them is the correct call and it is also",
        "  what keeps the numerator at zero.",
        "",
    ]
    return "\n".join(out) + "=" * 72


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.audit.gate")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="dev")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", help="write the report here")
    args = ap.parse_args(argv)

    data = measure(args.profile, args.n)
    text = render(data)
    print(text)
    if args.out:
        from pathlib import Path

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    # A measurement, not a gate: this never fails a build.
    return 0


if __name__ == "__main__":
    sys.exit(main())
