"""Measure the agent tier: what it closes, what it refuses, and what it cost.

There was no way to run B3 from the command line. The result this repository
published for it came out of a script that was never committed, so the headline
number for the most contested tier in the system was the one number a reader
could not reproduce. This is that command.

**What B3 is now measured on.** Since `legs/repricing.py` landed, a book with
its paperwork intact has almost nothing left for a model: the gateway's
repricing notice and the bank's FX advice close the fee and FX variances
deterministically, before the tier is ever asked. That was the point of building
it. So the honest question is no longer "how much can the model close" but "what
is left once every cheaper tier has had its turn", and the answer depends on
whether the merchant actually has the document:

- `--paperwork` (default): the real book. Whatever survives here is what the
  agent tier genuinely owns in production, and it is deliberately a small,
  awkward population -- ambiguity, missing lines, compound defects.
- `--no-paperwork`: the same book with the notices withheld, which is the case
  the tier was designed for -- a gap that is real and that no document explains.
  A hypothesis is the best thing available, and `RateBook` has nothing to
  confirm it with, so those close as `needs_approval` by construction.

Both are worth publishing and neither is the whole story on its own.

**The lead number is `resolved`, and it is allowed to be zero.** A tier that
explains twelve residuals and books none of them has still done the work a
human would otherwise do by hand; it just has not reconciled anything, and
saying so is the difference between this and a demo.

Usage:
    python -m recoagent.eval.b3 --profile dev
    python -m recoagent.eval.b3 --profile dev --no-paperwork --out results/B3_dev.txt
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..agent.contracts import AgentReport
from ..defects import DefectClass
from ..eval.scorer import Scorecard, score
from ..generator import DefectMix, GeneratorConfig, generate
from ..money import format_inr
from ..pipeline import run_b2, run_b3
from ..schemas import LabelledBatch

MIXES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout)}

#: Cost per million tokens (input, output), for the line that says whether this
#: was worth it. Keyed by model prefix, and deliberately sparse: a rate here is
#: one that was looked up for that host, not a house average. A model with no
#: entry prints its token counts and says the rate is unknown -- quoting a
#: dollar figure computed from some other vendor's price list is the kind of
#: number that gets repeated in a slide and is simply false.
PRICES: dict[str, tuple[float, float]] = {
    # Nemotron on NVIDIA NIM, at the time of writing.
    "nvidia/": (0.60, 0.60),
    "deepseek-ai/": (0.60, 0.60),
}


def prices_for(model: str) -> tuple[float, float] | None:
    for prefix, rate in PRICES.items():
        if model.startswith(prefix):
            return rate
    return None


@dataclass
class B3Run:
    profile: str
    seed: int
    n_orders: int
    model: str
    paperwork: bool
    before: Scorecard
    after: Scorecard
    report: AgentReport
    seconds: float = 0.0
    provenance: tuple[int, int] = (0, 0)
    #: (correct, checked) over every case that produced citations, booked or
    #: not. `provenance` covers only what was booked, which on a book with
    #: nothing left to book is a metric over an empty set.
    hypothesis: tuple[int, int] = (0, 0)
    open_before: int = 0

    @property
    def recall_moved(self) -> bool:
        return self.after.legs[2].recall != self.before.legs[2].recall


def truth_ids_for(batch: LabelledBatch) -> dict[str, set[str]]:
    """bank_line_id -> the source ids an honest explanation of it would name.

    Used only by the provenance metric, which is the one check that can catch an
    explanation naming the wrong rows while still closing the arithmetic. The
    scorer's false-match rate cannot see that: a B3 match is graded on its
    bank-line to settlement pairing, and that pairing came from the UTR join
    rather than from the model.
    """
    by_settlement: dict[str, set[str]] = {}
    for defect in batch.truth.defects:
        if defect.defect not in (
            DefectClass.REFUND_NETTED,
            DefectClass.CHARGEBACK_NETTED,
            DefectClass.ADJUSTMENT_ENTRY,
            DefectClass.FEE_TAX_VARIANCE,
            DefectClass.FX_CONVERSION,
        ):
            continue
        for sid in defect.affected_ids:
            by_settlement.setdefault(sid, set())

    # An explanation of a batch should name rows belonging to that batch: the
    # unlinked adjustments the solver would have found, or the payments a rate
    # claim covers.
    for sid in list(by_settlement):
        for adjustment in batch.sources.adjustments:
            if adjustment.settlement_id == sid:
                by_settlement[sid].add(adjustment.adjustment_id)
        for payment in batch.sources.payments_by_settlement(sid):
            by_settlement[sid].add(payment.payment_id)

    return {
        line_id: by_settlement[sid]
        for line_id, sid in batch.truth.leg2.items()
        if sid in by_settlement and by_settlement[sid]
    }


def build(profile: str, n_orders: int, *, paperwork: bool) -> LabelledBatch:
    seed, mix = MIXES[profile]
    batch = generate(GeneratorConfig(n_orders=n_orders, seed=seed, mix=mix()))
    if paperwork:
        return batch
    return replace(batch, sources=replace(
        batch.sources, rate_notices=(), fx_advices=()
    ))


def run(
    profile: str,
    *,
    n_orders: int = 2000,
    model: str = "",
    workers: int = 6,
    paperwork: bool = True,
    proposer_factory=None,
) -> B3Run:
    """One B3 pass, with the B2 baseline measured on the same book.

    `proposer_factory` is the seam the tests use: pass one and no client is ever
    built, so the whole path is exercised without an API call.
    """
    seed, _ = MIXES[profile]
    batch = build(profile, n_orders, paperwork=paperwork)

    baseline = score(batch, run_b2(batch.sources))
    open_before = sum(
        1 for e in run_b2(batch.sources).exceptions
        if e.leg == 2 and e.entity_kind == "bank_line" and e.residual_paise is not None
    )

    if proposer_factory is None:
        from ..agent.openai_proposer import OpenAICompatibleProposer

        def proposer_factory():  # noqa: F811
            return OpenAICompatibleProposer(model=model, timeout=300)

    started = time.perf_counter()
    result, report = run_b3(
        batch.sources, None, max_workers=workers, proposer_factory=proposer_factory
    )
    seconds = time.perf_counter() - started

    return B3Run(
        profile=profile,
        seed=seed,
        n_orders=n_orders,
        model=model or "scripted",
        paperwork=paperwork,
        before=baseline,
        after=score(batch, result),
        report=report,
        seconds=seconds,
        provenance=report.provenance(truth_ids_for(batch)),
        hypothesis=report.hypothesis_precision(truth_ids_for(batch)),
        open_before=open_before,
    )


def render(run: B3Run) -> str:
    w = 72
    r, before, after = run.report, run.before, run.after
    correct, checked = run.provenance
    h_correct, h_checked = run.hypothesis
    book = "with the merchant's paperwork" if run.paperwork else "paperwork withheld"

    out = [
        "=" * w,
        f"B3 -- AGENT TIER   profile={run.profile}  seed={run.seed}  "
        f"n={run.n_orders:,}",
        f"{book}   model={run.model}",
        "=" * w,
        "",
        f"  RESOLVED (source-backed)  {r.resolved:>8}   <- lead metric",
        f"  attempted                 {r.attempted:>8}   "
        f"(of {run.open_before} residual-bearing leg-2 items)",
        "",
        f"  needs approval            {r.needs_approval:>8}   "
        "arithmetic closed on a rate nothing confirms",
        f"  rejected by the gate      {r.rejected:>8}   "
        "proposed confidently, did not add up",
        f"  cited unverifiable        {r.unverifiable:>8}   "
        "named evidence that does not exist here",
        f"  declined by the model     {r.refused:>8}",
        f"  below confidence floor    {r.low_confidence:>8}",
        f"  malformed reply           {r.failed_malformed:>8}   "
        "answered, and the answer could not be used",
        f"  endpoint failed           {r.failed_transport:>8}   "
        "rate limit, timeout or 5xx -- never reached the model",
        "",
        "-" * w,
        "  DID IT MOVE THE BOOK",
        "-" * w,
        f"  Leg 2 recall        {before.legs[2].recall:>8.2%} -> {after.legs[2].recall:>8.2%}",
        f"  Leg 2 exceptions    {before.legs[2].exceptions:>8} -> {after.legs[2].exceptions:>8}",
        f"  FALSE-MATCH RATE    {after.overall_false_match_rate:>8.2%}   "
        "<- must be 0.00% whatever the model did",
        f"  defects mishandled  {after.mishandled_total:>8}",
        "",
        "-" * w,
        "  DID IT CITE THE RIGHT EVIDENCE",
        "-" * w,
    ]
    if checked:
        out.append(
            f"  provenance          {correct}/{checked} accepted explanations "
            f"named applicable evidence ({correct / checked:.0%})"
        )
    else:
        out.append("  provenance          nothing was resolved, so there is "
                   "nothing to check")
    out.append(
        "  The false-match rate cannot see this. A B3 match is graded on a"
    )
    out.append(
        "  pairing that came from the UTR join, so an explanation can name the"
    )
    out.append("  wrong rows, close the arithmetic, and still score perfectly.")

    out += ["", "-" * w, "  WAS THE MODEL RIGHT ABOUT WHY", "-" * w]
    if h_checked:
        out.append(
            f"  hypothesis          {h_correct}/{h_checked} explanations named "
            f"applicable evidence ({h_correct / h_checked:.0%})"
        )
        out.append("")
        out.append(
            "  Graded over every case the model worked, held-for-approval"
        )
        out.append(
            "  included, because `resolved` measures the gate and this measures"
        )
        out.append(
            "  the model. A tier that reasons correctly and is declined on"
        )
        out.append(
            "  policy and one that reasons badly and is caught both report zero"
        )
        out.append("  resolutions; only this number tells them apart.")
    else:
        out.append(
            "  hypothesis          the model produced no citable explanation"
        )

    out += [
        "",
        "-" * w,
        "  WHAT IT COST",
        "-" * w,
        f"  tokens              {r.usage.input_tokens:,} in / "
        f"{r.usage.output_tokens:,} out",
        f"  wall clock          {run.seconds:.0f}s over {r.attempted} cases",
    ]
    rate = prices_for(run.model)
    if rate is None:
        out.append(
            "  spend               unpriced -- no published rate on file for "
            f"{run.model}"
        )
    else:
        out.append(f"  spend               ${r.usage.cost_usd(*rate):.4f}")
    if r.resolved and rate is not None:
        out.append(
            f"  per resolved        "
            f"${r.cost_per_resolved(*rate):.4f}"
        )
    elif not r.resolved:
        out.append("  per resolved        n/a -- nothing was resolved")
    else:
        out.append(
            f"  per resolved        {r.usage.input_tokens / r.resolved:,.0f} in / "
            f"{r.usage.output_tokens / r.resolved:,.0f} out tokens per case"
        )

    out += [
        "",
        "-" * w,
        "  CASE BY CASE",
        "-" * w,
    ]
    if not r.cases:
        out.append("    no residual-bearing leg-2 item survived the cheaper tiers")
    for c in sorted(r.cases, key=lambda c: c.entity_id):
        out.append(
            f"    {c.entity_id:<14}{c.outcome:<16}"
            f"{format_inr(c.residual_paise):>14}  {c.detail[:44]}"
        )

    out += [
        "",
        "  Read `resolved` first, and note that zero is a legitimate answer. An",
        "  explanation that closes the arithmetic on a rate nobody issued is a",
        "  hypothesis, and it is held for approval rather than booked. The tier",
        "  earns its place by turning a bare residual into a worked account of",
        "  it -- not by reconciling anything it cannot prove.",
        "=" * w,
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    from ..llm import default_model

    ap = argparse.ArgumentParser(prog="recoagent.eval.b3", description=__doc__)
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", default=default_model(),
                    help="provider/model; defaults to RECOAGENT_MODEL")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--no-paperwork", dest="paperwork", action="store_false",
        help="withhold the rate notices, putting the tier back in the territory "
             "it was designed for",
    )
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args(argv)

    out = run(
        args.profile, n_orders=args.n, model=args.model,
        workers=args.workers, paperwork=args.paperwork,
    )
    text = render(out)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\n  wrote {args.out}")

    # The invariant, not the resolution rate: whatever the model proposed, the
    # gate must not have let a wrong match through.
    ok = (
        out.after.overall_false_match_rate == 0.0
        and out.after.mishandled_total == 0
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
