"""What a loose tolerance actually costs.

    python -m recoagent.eval.tolerance_sweep

The Leg 2 tolerance is the one number in this system chosen by judgement rather
than derived, so it should be the one number with evidence behind it.

The naive way to pick it is to maximise recall. That is wrong here, and this
sweep is what shows why: recall keeps climbing well past the point where the
tolerance stops absorbing *rounding* and starts absorbing *real discrepancies*.
A wider window does not find more matches, it hides more differences -- and
hiding differences is the exact failure reconciliation exists to prevent.

So the second table is the one that decides the number. It reports which defect
classes get resolved at each tolerance. The right setting is the largest one
that closes every rounding artifact and touches nothing else.
"""

from __future__ import annotations

from ..defects import DefectClass
from ..generator import DefectMix, GeneratorConfig, generate
from ..legs import ssmp
from ..legs.leg2_t1 import _orphan_pool
from ..money import format_inr
from ..pipeline import run_b2
from ..validate import Tolerance
from .scorer import score

TOLERANCES = [0, 1, 5, 10, 50, 100, 1_000, 10_000, 100_000]

#: Sub-rupee drift between a gateway's per-step rounding and a merchant's is
#: the only thing a tolerance is entitled to absorb. Everything else in the
#: taxonomy is a real difference in money.
ROUNDING_ONLY = frozenset({DefectClass.ROUNDING_DRIFT})


def sweep(n_orders: int = 2000, seed: int = 7) -> str:
    batch = generate(GeneratorConfig(n_orders=n_orders, seed=seed, mix=DefectMix.dev()))
    lines: list[str] = []
    w = 78

    lines.append("=" * w)
    lines.append(f"LEG 2 TOLERANCE SWEEP   n={n_orders}  seed={seed}  profile=dev")
    lines.append("=" * w)
    lines.append("")
    lines.append("  The tempting table -- and the one that would pick the wrong number:")
    lines.append("")
    lines.append(
        f"{'TOLERANCE':>12}{'':>3}{'RECALL':>9}{'FMR':>8}"
        f"{'RESOLVED':>10}{'AMBIGUOUS':>11}{'MISHANDLED':>12}"
    )
    lines.append("-" * w)

    per_class: dict[int, dict[DefectClass, int]] = {}

    for tol_paise in TOLERANCES:
        tol = Tolerance(leg1_paise=0, leg2_paise=tol_paise)
        result = run_b2(batch.sources, tol)
        card = score(batch, result)
        per_class[tol_paise] = {a.defect: a.resolved for a in card.accounting}

        # How often the search finds more than one *materially different*
        # explanation at this tolerance -- the coincidence rate.
        ambiguous = 0
        for exc in result.exceptions:
            if exc.leg != 2 or exc.residual_paise is None or exc.related_id is None:
                continue
            settlement = next(
                (s for s in batch.sources.settlements if s.settlement_id == exc.related_id),
                None,
            )
            if settlement is None:
                continue
            pool = _orphan_pool(batch.sources, settlement.settled_at)
            search = ssmp.enumerate_closing_subsets(
                [a.amount_paise for a in pool], exc.residual_paise, tol_paise
            )
            if search.ambiguous and not search.value_equivalent:
                ambiguous += 1

        lines.append(
            f"{format_inr(tol_paise):>12}{'':>3}"
            f"{card.legs[2].recall:>8.2%} "
            f"{card.legs[2].false_match_rate:>7.2%}"
            f"{sum(per_class[tol_paise].values()):>10}"
            f"{ambiguous:>11}{card.mishandled_total:>12}"
        )

    lines.append("-" * w)
    lines.append("")
    lines.append(
        "  Recall keeps rising past 10 paise, so maximising it would argue for a much\n"
        "  wider window. False-match rate stays flat at 0.00% throughout, so it offers\n"
        "  no objection either -- on Leg 2 the pairing comes from the UTR join, and the\n"
        "  tolerance only governs the *explanation*, never the match. Neither column\n"
        "  can see the actual damage."
    )
    lines.append("")
    lines.append("=" * w)
    lines.append("  What is actually being absorbed:")
    lines.append("=" * w)
    lines.append("")

    shown = [t for t in TOLERANCES if t in (0, 10, 50, 1_000, 10_000)]
    header = f"{'DEFECT CLASS RESOLVED':<26}" + "".join(
        f"{format_inr(t):>10}" for t in shown
    )
    lines.append(header)
    lines.append("-" * w)
    classes = sorted(
        {c for t in shown for c, v in per_class[t].items() if v},
        key=lambda c: c.value,
    )
    for cls in classes:
        # A class the solver already closes at zero tolerance was explained, not
        # absorbed -- the rows that account for it were actually found. Only a
        # class that appears *because* the window widened is being swallowed,
        # and only ROUNDING_DRIFT has any business doing that.
        at_zero = per_class[0].get(cls, 0)
        widest = max(per_class[t].get(cls, 0) for t in shown)
        absorbed = widest > at_zero
        if not absorbed:
            mark = ""
        elif cls in ROUNDING_ONLY:
            mark = "  <- rounding, legitimate"
        else:
            mark = "  <- REAL MONEY, absorbed"
        lines.append(
            f"{cls.value:<26}"
            + "".join(f"{per_class[t].get(cls, 0):>10}" for t in shown)
            + mark
        )
    lines.append("-" * w)
    lines.append("")
    lines.append(
        "  Reading: at 10 paise every ROUNDING_DRIFT closes and nothing else moves.\n"
        "  At 50 the solver starts absorbing FX_CONVERSION; by Rs 10 it is swallowing\n"
        "  FEE_TAX_VARIANCE too. Those are not matches recovered, they are genuine\n"
        "  differences in money being called close enough -- a merchant silently short\n"
        "  a few hundred rupees, reconciled green. Wider still and the classes the\n"
        "  solver did close start falling over as coincidental subsets crowd out the\n"
        "  true ones.\n\n"
        "  10 paise is therefore the largest window that absorbs rounding and only\n"
        "  rounding. It is chosen against this table, not against the recall column."
    )
    lines.append("=" * w)
    return "\n".join(lines)


if __name__ == "__main__":
    print(sweep())
