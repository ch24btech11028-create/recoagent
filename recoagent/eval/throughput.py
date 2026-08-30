"""How the whole pipeline scales, measured rather than extrapolated.

"2,000 orders in 0.15s" is one point, and one point cannot tell you whether the
next order costs the same as the last. A reconciliation engine has two things
in it that go quadratic if nobody watches -- a per-settlement scan over
payments, and a candidate pool drawn from a date window -- and both of them
look fine at 2,000 rows.

So this sweeps the book size and reports the *shape*, including the number that
matters more than the peak: how much throughput degrades across the range. A
straight line means the cost per record is constant; a curve means somebody has
to look at the solver before a real merchant arrives.

**Two settlement densities, and the difference is the point.** The generator's
default holds batch size constant as volume grows, so a 20,000-order book turns
into hundreds of payouts a day -- a stress case, not a merchant. Real gateways
settle on a T+2 cycle, roughly one payout a day, and a merchant doing ten times
the volume gets batches ten times bigger rather than ten times as many. The
solver's candidate pool is drawn from a date window, so settlement density
drives its search space directly. Reporting only the flattering one would be a
choice; both are here.

Timing excludes generation. What is being measured is the matcher, not the
fixture.

Usage:
    python -m recoagent.eval.throughput
    python -m recoagent.eval.throughput --sizes 500,2000,10000 --repeats 5
"""

from __future__ import annotations

import argparse
import sys
import time

from ..generator import DefectMix, GeneratorConfig, generate
from ..pipeline import run_b2

#: Spans a 50x range, which is enough to separate linear from quadratic without
#: making the command something nobody runs.
DEFAULT_SIZES = (500, 2_000, 5_000, 10_000, 25_000)

DENSITIES = (
    (None, "fixed batch size (stress: payouts grow with volume)"),
    (2.0, "T+2 cycle (realistic: batches grow with volume)"),
)


def measure(sizes, repeats: int, settlements_per_day, progress=None) -> list[dict]:
    rows = []
    for n in sizes:
        batch = generate(
            GeneratorConfig(
                n_orders=n,
                seed=7,
                mix=DefectMix.dev(),
                settlements_per_day=settlements_per_day,
            )
        )
        counts = batch.sources.counts
        records = sum(counts.values())

        # Best of N. The interest is the cost of the work, and a scheduler
        # interruption is not a property of the matcher -- a mean would fold
        # the machine's noise into the number being claimed.
        best = min(
            (lambda t0=time.perf_counter(): (run_b2(batch.sources), time.perf_counter() - t0))()[1]
            for _ in range(repeats)
        )
        row = {
            "orders": n,
            "records": records,
            "settlements": counts["settlements"],
            "seconds": best,
            "records_per_sec": records / best,
        }
        rows.append(row)
        if progress:
            progress(row)
    return rows


def render(by_density: list[tuple[str, list[dict]]]) -> str:
    out = ["=" * 72, "  THROUGHPUT", "=" * 72]
    for label, rows in by_density:
        peak = max(r["records_per_sec"] for r in rows)
        floor = min(r["records_per_sec"] for r in rows)
        span = rows[-1]["records"] / rows[0]["records"]
        out += [
            "",
            f"  {label}",
            "  " + "-" * 68,
            f"  {'orders':>8}{'records':>10}{'batches':>9}"
            f"{'seconds':>10}{'records/sec':>14}",
        ]
        for r in rows:
            out.append(
                f"  {r['orders']:>8,}{r['records']:>10,}{r['settlements']:>9,}"
                f"{r['seconds']:>10.3f}{r['records_per_sec']:>14,.0f}"
            )
        out += [
            "  " + "-" * 68,
            f"  across a {span:.0f}x range, throughput moves {peak / floor:.1f}x"
            + ("   <- effectively linear" if peak / floor < 2 else
               "   <- superlinear cost; look at the solver"),
        ]

    out += [
        "",
        "  Single process, single thread, standard library only -- no pandas, no",
        "  database, nothing installed. Timing excludes generating the book: what",
        "  is measured is the matcher, not the fixture.",
        "",
        "  The two blocks are the same code on different settlement densities.",
        "  The stress case holds batch size constant so payouts grow with volume;",
        "  the realistic one settles on a T+2 cycle so batches grow instead. The",
        "  solver draws candidates from a date window, so density drives its",
        "  search space -- which is why both are reported rather than the better.",
        "=" * 72,
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.eval.throughput")
    ap.add_argument(
        "--sizes",
        default=",".join(str(s) for s in DEFAULT_SIZES),
        help="comma-separated order counts",
    )
    ap.add_argument("--repeats", type=int, default=3, help="runs per size; best is kept")
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args(argv)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]

    by_density = []
    for density, label in DENSITIES:
        print(f">>> {label}", flush=True)
        rows = measure(
            sizes, args.repeats, density,
            progress=lambda r: print(
                f"    {r['orders']:>7,} orders  {r['records']:>8,} records  "
                f"{r['seconds']:>7.3f}s  {r['records_per_sec']:>10,.0f} rec/s",
                flush=True,
            ),
        )
        by_density.append((label, rows))

    text = render(by_density)
    print()
    print(text)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
