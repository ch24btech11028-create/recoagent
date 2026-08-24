"""Scoring, with false-match rate as the headline.

The metric order here is deliberate and is not the usual one. Match rate is
reported second, not first, because a reconciliation engine that matches
everything and is sometimes wrong is worse than useless -- it books money
against the wrong transaction and hides the error behind a green number.
BenchRec, the ICAIF 2023 industry benchmark for exactly this task, states the
principle directly: it is better to leave a transaction unmatched, and leave it
for manual review, than to match it incorrectly.

So: false-match rate first, then throughput, then an honest exception list.

The reconciliation check at the end is what makes the rest trustworthy. Every
defect the generator injected must show up as an exception on an entity it
actually damaged. If injected counts and accounted counts diverge, some defect
class is landing somewhere the matcher never looks, and every rate above it is
flattering itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..defects import DefectClass
from ..money import Paise, format_inr
from ..schemas import GroundTruth, LabelledBatch, ReconResult


@dataclass
class LegScore:
    leg: int
    population: int
    attempted: int
    true_matches: int
    false_matches: int
    exceptions: int

    @property
    def false_match_rate(self) -> float:
        """Share of accepted matches that are wrong. The money metric."""
        return self.false_matches / self.attempted if self.attempted else 0.0

    @property
    def auto_match_rate(self) -> float:
        """Share of the population the system resolved without a human."""
        return self.attempted / self.population if self.population else 0.0

    @property
    def precision(self) -> float:
        return self.true_matches / self.attempted if self.attempted else 0.0

    @property
    def recall(self) -> float:
        return self.true_matches / self.population if self.population else 0.0


@dataclass
class ClassAccounting:
    defect: DefectClass
    injected: int
    accounted: int

    @property
    def reconciles(self) -> bool:
        return self.injected == self.accounted


@dataclass
class ValueCoverage:
    """How much money, not how many rows. A finance track cares about both."""

    total_credit: Paise
    matched_credit: Paise

    @property
    def share(self) -> float:
        return self.matched_credit / self.total_credit if self.total_credit else 0.0


@dataclass
class Scorecard:
    rung: str
    profile: str
    seed: int
    legs: dict[int, LegScore] = field(default_factory=dict)
    accounting: list[ClassAccounting] = field(default_factory=list)
    unattributed_exceptions: int = 0
    value: ValueCoverage | None = None

    @property
    def overall_false_match_rate(self) -> float:
        attempted = sum(s.attempted for s in self.legs.values())
        false = sum(s.false_matches for s in self.legs.values())
        return false / attempted if attempted else 0.0

    @property
    def overall_auto_match_rate(self) -> float:
        pop = sum(s.population for s in self.legs.values())
        attempted = sum(s.attempted for s in self.legs.values())
        return attempted / pop if pop else 0.0

    @property
    def fully_reconciles(self) -> bool:
        """Every injected defect class is accounted for in the exception list."""
        return all(a.reconciles for a in self.accounting)


def _reverse(mapping: dict[str, str]) -> dict[str, str]:
    return {v: k for k, v in mapping.items()}


def _expand(entity_id: str, truth: GroundTruth) -> set[str]:
    """Map an entity id onto every id an exception about it might be filed under.

    A defect injected on a settlement surfaces as an exception on the bank line
    that settlement should have produced; a defect on a payment surfaces on its
    order. Without this normalisation the reconciliation check would report
    false divergence purely because the generator and the matcher name things
    from opposite ends of the same relationship.
    """
    ids = {entity_id}
    rev_leg2 = _reverse(truth.leg2)  # settlement_id -> bank_line_id
    rev_leg1 = _reverse(truth.leg1)  # payment_id    -> order_id

    if entity_id in rev_leg2:
        ids.add(rev_leg2[entity_id])
    if entity_id in truth.leg2:
        ids.add(truth.leg2[entity_id])
    if entity_id in rev_leg1:
        ids.add(rev_leg1[entity_id])
    if entity_id in truth.leg1:
        ids.add(truth.leg1[entity_id])
    return ids


def score(batch: LabelledBatch, result: ReconResult) -> Scorecard:
    truth = batch.truth
    sources = batch.sources
    card = Scorecard(rung=result.rung, profile=batch.profile, seed=batch.seed)

    # ── Leg 1: orders -> payments ────────────────────────────────────────
    l1_matches = result.matches_for_leg(1)
    l1_true = sum(1 for m in l1_matches if truth.leg1.get(m.left_ids[0]) == m.right_ids[0])
    card.legs[1] = LegScore(
        leg=1,
        population=len(sources.orders),
        attempted=len(l1_matches),
        true_matches=l1_true,
        false_matches=len(l1_matches) - l1_true,
        exceptions=len(result.exceptions_for_leg(1)),
    )

    # ── Leg 2: bank lines -> settlements ─────────────────────────────────
    # Scored over bank lines: a duplicated credit sits in the denominator with
    # no correct answer, so refusing it costs match rate. That cost is the
    # honest price of not booking the same money twice.
    l2_matches = result.matches_for_leg(2)
    l2_true = sum(1 for m in l2_matches if truth.leg2.get(m.left_ids[0]) == m.right_ids[0])
    card.legs[2] = LegScore(
        leg=2,
        population=len(sources.bank_lines),
        attempted=len(l2_matches),
        true_matches=l2_true,
        false_matches=len(l2_matches) - l2_true,
        exceptions=len(
            [e for e in result.exceptions_for_leg(2) if e.entity_kind == "bank_line"]
        ),
    )

    # ── Value coverage ───────────────────────────────────────────────────
    line_amounts = {b.bank_line_id: b.amount_paise for b in sources.bank_lines}
    matched_ids = {m.left_ids[0] for m in l2_matches}
    card.value = ValueCoverage(
        total_credit=sum(line_amounts.values()),
        matched_credit=sum(v for k, v in line_amounts.items() if k in matched_ids),
    )

    # ── Reconciliation: injected defects vs the exception list ───────────
    flagged = {e.entity_id for e in result.exceptions}

    injected_by_class: dict[DefectClass, int] = {}
    accounted_by_class: dict[DefectClass, int] = {}
    explained_entities: set[str] = set()

    for d in truth.defects:
        injected_by_class[d.defect] = injected_by_class.get(d.defect, 0) + 1
        touched: set[str] = set()
        for aid in d.affected_ids:
            touched |= _expand(aid, truth)
        hit = touched & flagged
        if hit:
            accounted_by_class[d.defect] = accounted_by_class.get(d.defect, 0) + 1
            explained_entities |= hit

    card.accounting = [
        ClassAccounting(
            defect=cls,
            injected=injected_by_class[cls],
            accounted=accounted_by_class.get(cls, 0),
        )
        for cls in sorted(injected_by_class, key=lambda c: c.value)
    ]

    # Exceptions on entities no injected defect can explain. At B0 this should
    # be zero: the baseline should only ever complain about genuine damage. A
    # non-zero value means the matcher is rejecting clean records, which is a
    # bug in the matcher, not a property of the data.
    card.unattributed_exceptions = len(
        {e.entity_id for e in result.exceptions if e.entity_id not in explained_entities}
    )

    return card


def render(card: Scorecard) -> str:
    """Plain-text scorecard. Deterministic, so it can be diffed across runs."""
    lines: list[str] = []
    w = 72

    lines.append("=" * w)
    lines.append(f"RUNG {card.rung}   profile={card.profile}   seed={card.seed}")
    lines.append("=" * w)
    lines.append("")
    lines.append(
        f"  FALSE-MATCH RATE      {card.overall_false_match_rate:>8.2%}"
        "     <- lead metric"
    )
    lines.append(f"  Auto-match rate       {card.overall_auto_match_rate:>8.2%}")
    if card.value:
        lines.append(
            f"  Credit value matched  {card.value.share:>8.2%}"
            f"     ({format_inr(card.value.matched_credit)}"
            f" of {format_inr(card.value.total_credit)})"
        )
    lines.append("")

    lines.append("-" * w)
    lines.append(
        f"{'LEG':<5}{'POP':>7}{'MATCHED':>9}{'TRUE':>7}{'FALSE':>7}"
        f"{'FMR':>8}{'RECALL':>9}{'EXC':>7}"
    )
    lines.append("-" * w)
    for leg in sorted(card.legs):
        s = card.legs[leg]
        lines.append(
            f"{s.leg:<5}{s.population:>7}{s.attempted:>9}{s.true_matches:>7}"
            f"{s.false_matches:>7}{s.false_match_rate:>8.2%}"
            f"{s.recall:>9.2%}{s.exceptions:>7}"
        )
    lines.append("")

    lines.append("-" * w)
    lines.append(f"{'EXCEPTION ACCOUNTING':<44}{'INJECTED':>10}{'FOUND':>9}{'':>9}")
    lines.append("-" * w)
    for a in card.accounting:
        flag = "ok" if a.reconciles else "MISMATCH"
        lines.append(f"{a.defect.value:<44}{a.injected:>10}{a.accounted:>9}{flag:>9}")
    lines.append("-" * w)
    lines.append(
        f"{'exceptions with no injected cause':<44}"
        f"{'':>10}{card.unattributed_exceptions:>9}"
        f"{'ok' if card.unattributed_exceptions == 0 else 'INVESTIGATE':>9}"
    )
    lines.append("")
    verdict = "RECONCILES" if card.fully_reconciles else "DOES NOT RECONCILE"
    lines.append(f"  Ground-truth accounting: {verdict}")
    lines.append("=" * w)
    return "\n".join(lines)
