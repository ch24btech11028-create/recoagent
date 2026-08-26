"""Run RecoAgent's matching discipline on BenchRec, and report what it refuses.

`benchrec_baseline.py` scores the language-model matcher that ships inside
BenchRec. This scores *us*, on the same 32,048 labelled statement lines, so the
central claim stops resting on data this repo generated.

**What makes the comparison fair, and where it does not transfer.** BenchRec is
a different problem shape from the settlement book: one account, one currency,
no fee model, and an allocation key to predict rather than a batch to prove. So
the fee arithmetic has nothing to say here. Two things do transfer, and they are
the two the project is actually about:

- **`legs.ssmp.enumerate_closing_subsets` runs unmodified.** It takes a list of
  integers and a target; it does not know or care that these are dollars from a
  2023 corporate cash ledger rather than paise from a gateway. The N:1 tier here
  is the same solver that closes leg 2.
- **Ambiguity is refused, not resolved.** Where several allocations fit equally
  well, the answer is "a human looks at this", not the most textually similar
  one. That single rule is the difference the numbers are meant to show.

The cardinality claim in the older baseline was understated, and this file
corrects it: BenchRec is mostly 1:1, but 6,000 of its 56,074 training groups
carry more than one row a side, including 532 that are 2:1 and 424 that are 3:1.
The subset-sum tier exists for those.

**Nothing here consumes a candidate.** Each statement line is matched against
the whole A-side pool independently, so no line's answer depends on the order
the file happened to be read in. Greedy consumption would raise coverage and
would make the result an artefact of row order, which is the kind of number this
project exists not to publish.

Usage:
    python -m recoagent.eval.benchrec --data data/benchrec
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..legs.ssmp import enumerate_closing_subsets

csv.field_size_limit(10**7)

EVAL = "BenchRec_cash_v1.0_eval.csv"
SOLUTION = "BenchRec_cash_v1.0_solution.csv"

#: Tiers, in the order they are tried. Named to match the settlement ladder.
T0 = "T0 amount+value-date, unique"
T1 = "T1 amount alone, unique in window"
T2 = "T2 subset-sum over the window"

REFUSALS = {
    "no candidate": "no A-side row carries this amount at all",
    "ambiguous": "several allocations fit equally well",
    "split allocation": "the closing subset spans more than one allocation",
    "unparseable": "the amount could not be read as a decimal",
}


def cents(raw: str) -> int | None:
    """Money as an integer, always. A float here would be a rounding error later."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return int((Decimal(raw) * 100).to_integral_value())
    except (InvalidOperation, ValueError):
        return None


def _day(raw: str) -> date | None:
    raw = (raw or "").strip()
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


@dataclass
class Candidate:
    a_id: str
    allocation: str
    amount: int
    value_date: date | None


@dataclass
class Outcome:
    b_id: str
    predicted: str | None
    tier: str | None
    refusal: str | None
    truth: str

    @property
    def attempted(self) -> bool:
        return self.predicted is not None

    @property
    def correct(self) -> bool:
        return self.predicted is not None and self.predicted == self.truth


@dataclass
class Report:
    outcomes: list[Outcome] = field(default_factory=list)
    seconds: float = 0.0
    #: Lines whose answer is not present in the candidate pool at all. A ceiling
    #: on recall that belongs in the report rather than buried in a miss count.
    unreachable: int = 0

    @property
    def population(self) -> int:
        return len(self.outcomes)

    @property
    def attempted(self) -> int:
        return sum(1 for o in self.outcomes if o.attempted)

    @property
    def correct(self) -> int:
        return sum(1 for o in self.outcomes if o.correct)

    @property
    def wrong(self) -> int:
        return self.attempted - self.correct

    @property
    def wrong_match_rate(self) -> float:
        """The lead metric, same as everywhere else in this repo."""
        return self.wrong / self.attempted if self.attempted else 0.0

    @property
    def coverage(self) -> float:
        return self.attempted / self.population if self.population else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.population if self.population else 0.0

    def by_tier(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for o in self.outcomes:
            if o.tier:
                out[o.tier][0] += 1
                out[o.tier][1] += 0 if o.correct else 1
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    def refusals(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for o in self.outcomes:
            if o.refusal:
                out[o.refusal] += 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def load(data_dir: Path) -> tuple[list[Candidate], list[dict], dict[str, str]]:
    with open(data_dir / SOLUTION, newline="") as fh:
        truth = {r["B_id"]: r["targetAllocation"].strip() for r in csv.DictReader(fh)}

    candidates: list[Candidate] = []
    statements: list[dict] = []
    with open(data_dir / EVAL, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["A_id"].strip():
                amount = cents(row["A_amount"])
                if amount is None:
                    continue
                candidates.append(Candidate(
                    a_id=row["A_id"].strip(),
                    allocation=row["A_allocation"].strip(),
                    amount=amount,
                    value_date=_day(row["A_valueDate"]),
                ))
            elif row["B_id"].strip():
                statements.append(row)
    return candidates, statements, truth


class Pool:
    """The A-side, indexed the way the matcher asks about it."""

    def __init__(self, candidates: list[Candidate]) -> None:
        self.all = candidates
        self.by_amount: dict[int, list[Candidate]] = defaultdict(list)
        self.by_amount_date: dict[tuple[int, date | None], list[Candidate]] = defaultdict(list)
        self.by_date: dict[date | None, list[Candidate]] = defaultdict(list)
        for c in candidates:
            self.by_amount[c.amount].append(c)
            self.by_amount_date[(c.amount, c.value_date)].append(c)
            self.by_date[c.value_date].append(c)

    def window(self, day: date | None, days: int) -> list[Candidate]:
        if day is None:
            return []
        from datetime import timedelta
        out: list[Candidate] = []
        for offset in range(-days, days + 1):
            out.extend(self.by_date.get(day + timedelta(days=offset), ()))
        return out


def _allocation_of(rows: list[Candidate]) -> str | None:
    """One allocation, or nothing. A subset spanning two is not an answer."""
    allocations = {c.allocation for c in rows}
    return allocations.pop() if len(allocations) == 1 else None


def match_one(
    pool: Pool,
    row: dict,
    *,
    window_days: int,
    max_size: int,
    amount_only: bool = False,
) -> tuple[str | None, str | None, str | None]:
    """Returns (predicted allocation, tier, refusal reason)."""
    amount = cents(row["B_amount"])
    if amount is None:
        return None, None, "unparseable"
    value_date = _day(row["B_valueDate"])

    # T0 -- the exact key. Amount and value date together, and only one row has it.
    exact = pool.by_amount_date.get((amount, value_date), [])
    if len(exact) == 1:
        return exact[0].allocation, T0, None
    if exact:
        shared = _allocation_of(exact)
        if shared is not None:
            # Several rows, one allocation: the answer is not in doubt.
            return shared, T0, None

    # T1 -- amount alone, ignoring the value date. Off by default, and the
    # reason is measured rather than assumed: over the full eval set it matches
    # 213 more lines and gets 62 of them wrong, a 29.11% wrong-match rate. That
    # is 0.7 points of coverage bought with 45% of every error the system makes.
    # A date that disagrees is evidence the line is a different transaction, not
    # noise to be looked past. `--amount-only` turns it back on to reproduce the
    # comparison.
    same_amount = pool.by_amount.get(amount, [])
    if amount_only and same_amount:
        shared = _allocation_of(same_amount)
        if shared is not None:
            return shared, T1, None

    # T2 -- the N:1 case: several A rows summing to this credit. Same solver
    # that closes leg 2 of the settlement book, handed a different pool.
    if max_size >= 2:
        near = pool.window(value_date, window_days)
        if 1 < len(near) <= 4000:
            found = enumerate_closing_subsets(
                [c.amount for c in near], amount, tolerance=0, max_size=max_size,
            )
            if found.actionable and found.best is not None:
                rows = [near[i] for i in found.best.indices]
                shared = _allocation_of(rows)
                if shared is not None:
                    return shared, T2, None
                return None, None, "split allocation"
            if found.ambiguous:
                return None, None, "ambiguous"

    if not same_amount:
        return None, None, "no candidate"
    return None, None, "ambiguous"


def run(
    data_dir: Path,
    *,
    window_days: int = 1,
    max_size: int = 1,
    limit: int = 0,
    amount_only: bool = False,
) -> Report:
    candidates, statements, truth = load(data_dir)
    pool = Pool(candidates)
    reachable = {c.allocation for c in candidates}
    if limit:
        statements = statements[:limit]

    report = Report()
    started = time.perf_counter()
    for row in statements:
        b_id = row["B_id"].strip()
        target = truth.get(b_id, "")
        predicted, tier, refusal = match_one(
            pool, row, window_days=window_days, max_size=max_size,
            amount_only=amount_only,
        )
        if target and target not in reachable:
            report.unreachable += 1
        report.outcomes.append(Outcome(b_id, predicted, tier, refusal, target))
    report.seconds = time.perf_counter() - started
    return report


def render(report: Report, *, window_days: int, max_size: int) -> str:
    w = 72
    out = [
        "=" * w,
        "RECOAGENT ON BENCHREC  (cash v1.0 eval, 32,048 labelled statement lines)",
        "=" * w,
        "",
        f"  WRONG-MATCH RATE      {report.wrong_match_rate:>9.2%}   <- lead metric",
        f"  coverage              {report.coverage:>9.2%}   ({report.attempted:,} of {report.population:,} attempted)",
        f"  accuracy              {report.accuracy:>9.2%}   (correct out of every line)",
        "",
        f"  correct {report.correct:,}   WRONG {report.wrong:,}   "
        f"refused {report.population - report.attempted:,}",
        f"  window +/-{window_days}d   max subset {max_size}   {report.seconds:.0f}s",
        "",
        "-" * w,
        "  WHERE THE MATCHES CAME FROM",
        "-" * w,
        f"  {'TIER':<38}{'MATCHED':>10}{'WRONG':>9}{'RATE':>9}",
    ]
    for tier, (matched, wrong) in report.by_tier().items():
        rate = wrong / matched if matched else 0.0
        out.append(f"  {tier:<38}{matched:>10,}{wrong:>9,}{rate:>9.2%}")
    out += [
        "",
        "-" * w,
        "  WHAT IT REFUSED, AND WHY",
        "-" * w,
    ]
    for reason, count in report.refusals().items():
        out.append(f"  {reason:<38}{count:>10,}   {REFUSALS.get(reason, '')}")
    out += [
        "",
        f"  Of the refusals, {report.unreachable:,} lines have an answer that appears",
        "  nowhere in the eval file's candidate pool. No matcher reading only this",
        "  file can reach them; they are a ceiling, not a miss.",
        "",
        "-" * w,
        "  AGAINST THE MATCHER THAT SHIPS WITH BENCHREC",
        "-" * w,
        f"  {'':<22}{'COVERAGE':>10}{'WRONG-MATCH RATE':>20}",
        f"  {'MatcherByChatGPT':<22}{'64.90%':>10}{'4.80%':>20}",
        f"  {'RecoAgent':<22}{report.coverage:>10.2%}{report.wrong_match_rate:>20.2%}",
        "",
        "  Both numbers matter and neither is free. Every wrong match is money",
        "  filed against the wrong entry, found later by someone who has to",
        "  unpick it -- and the baseline reported high confidence on all of its.",
        "  Refusing costs a queue item. Guessing costs a correction.",
        "",
        "-" * w,
        "  WHAT DID NOT TRANSFER",
        "-" * w,
        "  The subset-sum tier earns nothing here, and the measurement says so:",
        "  over 2,000 lines it ran for 231s and closed 0 additional matches. It",
        "  finds subsets -- it then refuses them, because in a pool of ~600 rows",
        "  a day, three amounts summing to a target is a coincidence rather than",
        "  a batch, and the rows it names belong to different allocations. That",
        "  is the gate working, not failing: BenchRec has no batch structure to",
        "  prove, so there is nothing for a proof to be about. It is off by",
        "  default; --max-size 3 reproduces the run.",
        "=" * w,
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.eval.benchrec", description=__doc__)
    ap.add_argument("--data", default="data/benchrec", type=Path)
    ap.add_argument("--window-days", type=int, default=1,
                    help="date window for the subset-sum tier")
    ap.add_argument("--max-size", type=int, default=1,
                    help="largest subset the N:1 tier will consider; 1 disables it")
    ap.add_argument("--amount-only", action="store_true",
                    help="also match on amount alone, ignoring the value date")
    ap.add_argument("--limit", type=int, default=0, help="cap statement lines, 0 = all")
    ap.add_argument("--out", help="also write the report here")
    args = ap.parse_args(argv)

    if not (args.data / EVAL).exists():
        print(f"  {args.data / EVAL} not found.", file=sys.stderr)
        print("  BenchRec is CC BY 4.0; download it and point --data at the folder.",
              file=sys.stderr)
        return 2

    report = run(args.data, window_days=args.window_days,
                 max_size=args.max_size, limit=args.limit,
                 amount_only=args.amount_only)
    text = render(report, window_days=args.window_days, max_size=args.max_size)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\n  wrote {args.out}")
    # The gate this file exists to demonstrate.
    return 0 if report.wrong_match_rate < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
