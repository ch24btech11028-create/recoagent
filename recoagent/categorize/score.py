"""Scoring, with the same metric ordering the reconciliation scorer uses.

**Wrong-category rate on auto-assigned rows comes first.** Not accuracy over
everything, which mixes two different failures into one number and hides the
expensive one. A row sent to a human costs a minute. A row confidently filed as
revenue when it was a transfer costs a wrong GST return, and the merchant finds
out from a notice rather than from a dashboard. BenchRec's stated principle --
better unmatched than wrongly matched -- transfers to categorisation unchanged.

Coverage comes second, because a system that reviews everything has a wrong
rate of zero and is worth nothing.

Per-category precision and recall come third, and the confusion pairs after
them, because an overall figure conceals which boundary the system cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .rules import Ledger
from .taxonomy import Category


@dataclass
class Confusion:
    predicted: Category
    actual: Category
    count: int
    entity_ids: tuple[str, ...]


@dataclass
class CategoryScorecard:
    rung: str
    population: int          # rows with a ground-truth label
    assigned: int            # rows the system committed to
    correct: int
    wrong: int
    reviewed: int            # rows sent to a human, correctly or otherwise
    unlabelled: int          # assigned rows the answer key has no entry for
    per_category: dict[Category, tuple[int, int, int]] = field(default_factory=dict)
    confusions: tuple[Confusion, ...] = ()
    by_rung: dict[str, int] = field(default_factory=dict)

    @property
    def wrong_rate(self) -> float:
        """Of the rows the system committed to, how many were wrong.

        The lead metric. Denominator is what it auto-assigned, not the whole
        population: a system is only accountable for the decisions it made.
        """
        return self.wrong / self.assigned if self.assigned else 0.0

    @property
    def coverage(self) -> float:
        return self.assigned / self.population if self.population else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.assigned if self.assigned else 0.0


def score(ledger: Ledger, truth: dict[str, str], rung: str) -> CategoryScorecard:
    correct = wrong = reviewed = unlabelled = 0
    per: dict[Category, list[int]] = {}
    confusions: dict[tuple[Category, Category], list[str]] = {}

    for entity_id, assignment in ledger.assignments.items():
        if assignment.category is Category.NEEDS_REVIEW:
            reviewed += 1
            continue

        expected_raw = truth.get(entity_id)
        if expected_raw is None:
            # The system categorised a row the answer key says nothing about.
            # Counted separately rather than as a win or a loss: scoring it
            # either way would be inventing a label, which is the offence this
            # whole file exists to detect.
            unlabelled += 1
            continue

        expected = Category(expected_raw)
        tp_fp_fn = per.setdefault(assignment.category, [0, 0, 0])
        if assignment.category is expected:
            correct += 1
            tp_fp_fn[0] += 1
        else:
            wrong += 1
            tp_fp_fn[1] += 1
            per.setdefault(expected, [0, 0, 0])[2] += 1
            confusions.setdefault((assignment.category, expected), []).append(entity_id)

    ranked = sorted(confusions.items(), key=lambda kv: -len(kv[1]))
    return CategoryScorecard(
        rung=rung,
        population=len(truth),
        assigned=correct + wrong,
        correct=correct,
        wrong=wrong,
        reviewed=reviewed,
        unlabelled=unlabelled,
        per_category={k: tuple(v) for k, v in per.items()},  # type: ignore[misc]
        confusions=tuple(
            Confusion(predicted=p, actual=a, count=len(ids), entity_ids=tuple(sorted(ids)[:5]))
            for (p, a), ids in ranked
        ),
        by_rung=ledger.by_rung(),
    )


def render(card: CategoryScorecard) -> str:
    out = [
        "",
        "=" * 72,
        f"  CATEGORISATION   rung={card.rung}",
        "=" * 72,
        "",
        f"  Wrong-category rate      {card.wrong_rate:>8.2%}"
        f"     ({card.wrong} of {card.assigned} auto-assigned)",
        f"  Coverage                 {card.coverage:>8.2%}"
        f"     ({card.assigned} of {card.population} labelled rows)",
        f"  Sent for review          {card.reviewed:>8}",
    ]
    if card.unlabelled:
        out.append(
            f"  Assigned, unlabelled     {card.unlabelled:>8}"
            "     (the answer key has no entry; scored neither way)"
        )
    out += ["", "  Assignments by rung"]
    for rung in sorted(card.by_rung):
        label = {
            "C0": "source fields alone",
            "C1": "determined by the reconciliation",
            "C2": "cited by the model",
        }.get(rung, "")
        out.append(f"    {rung:<6}{card.by_rung[rung]:>7}    {label}")

    out += ["", "-" * 72, "  PER CATEGORY", "-" * 72,
            f"  {'category':<24}{'correct':>9}{'wrong':>8}{'missed':>9}"
            f"{'precision':>12}{'recall':>9}"]
    for category in sorted(card.per_category, key=lambda c: c.value):
        tp, fp, fn = card.per_category[category]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        out.append(
            f"  {category.value:<24}{tp:>9}{fp:>8}{fn:>9}{precision:>11.1%}{recall:>9.1%}"
        )

    if card.confusions:
        out += ["", "-" * 72, "  WHERE IT GOES WRONG", "-" * 72]
        for c in card.confusions:
            out.append(
                f"  called {c.predicted.value} when it was {c.actual.value}"
                f"   ({c.count})"
            )
            out.append(f"      e.g. {', '.join(c.entity_ids)}")
    else:
        out += ["", "  No misclassifications on this book."]

    out += ["", "=" * 72, ""]
    return "\n".join(out)
