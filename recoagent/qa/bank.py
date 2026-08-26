"""Questions about a reconciliation run, with answers computed from it.

The bar for this track is "throughput plus measured accuracy plus an honest
exception list", and a Q&A agent is the easiest thing in the world to demo
without measuring. So the questions come with programmatic ground truth, and
they are derived from the run itself rather than written by hand -- a fixed list
of hand-written questions would drift out of agreement with the data the moment
the generator changed, and would quietly become a set of answers I had approved
rather than a set the system can be checked against.

Every question here has an answer that is either a number or an identifier, so
scoring is exact rather than a judgement call. Questions whose honest answer is
prose -- "why was this flagged" -- are deliberately excluded: grading those
needs a second model, and a benchmark that grades itself with a model is not a
benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..money import Paise
from ..schemas import LabelledBatch, ReconResult


@dataclass(frozen=True)
class Question:
    """One question, its exact answer, and what an agent needs to see to answer it."""

    qid: str
    kind: str
    text: str
    answer: Any
    #: `paise` answers are compared with a tolerance; `count`, `id` and `bool`
    #: must match exactly.
    answer_type: str
    #: Entity ids the answer depends on, so retrieval can be scored separately
    #: from reasoning when one of them goes wrong.
    depends_on: tuple[str, ...] = ()
    #: True when the factsheet deliberately cannot answer this. The only correct
    #: response is to decline. Without these the bank cannot detect hallucination
    #: at all -- an agent that always answers scores perfectly on a bank where
    #: everything is answerable, which is exactly the blind spot that makes a
    #: confident wrong number reach an operator.
    expects_decline: bool = False
    #: False for a question typed by an operator at runtime. There is no ground
    #: truth for those, so they are answered but never scored -- folding them
    #: into the wrong-answer rate would let an unmeasurable question move a
    #: measured number.
    graded: bool = True


def build(batch: LabelledBatch, result: ReconResult, limit_per_kind: int = 4) -> list[Question]:
    """Derive a question bank from one run. Deterministic given the same run."""
    qs: list[Question] = []
    sources = batch.sources

    matched_l2 = result.matches_for_leg(2)
    line_exceptions = [
        e for e in result.exceptions if e.leg == 2 and e.entity_kind == "bank_line"
    ]
    with_residual = sorted(
        (e for e in line_exceptions if e.residual_paise is not None),
        key=lambda e: -abs(e.residual_paise or 0),
    )
    amounts = {b.bank_line_id: b.amount_paise for b in sources.bank_lines}

    # ── portfolio-level counts and totals ────────────────────────────────
    qs.append(Question(
        "q_open_count", "count",
        "How many items are still open in the exception queue?",
        len(result.exceptions), "count",
    ))
    qs.append(Question(
        "q_matched_credits", "count",
        "How many bank credits were matched to a settlement batch?",
        len(matched_l2), "count",
    ))
    qs.append(Question(
        "q_unexplained_total", "paise",
        "What is the total unexplained amount across all open items, in paise? "
        "Use the absolute value of each gap.",
        sum(abs(e.residual_paise or 0) for e in result.exceptions), "paise",
    ))
    qs.append(Question(
        "q_credit_count", "count",
        "How many bank credits are in this statement?",
        len(sources.bank_lines), "count",
    ))
    qs.append(Question(
        "q_unlinked_rows", "count",
        "How many adjustment rows did the gateway leave unlinked to any settlement?",
        len(sources.unlinked_adjustments), "count",
    ))

    # ── the biggest problem, which is what an operator asks first ────────
    if with_residual:
        worst = with_residual[0]
        qs.append(Question(
            "q_worst_gap_id", "id",
            "Which bank credit has the largest unexplained gap? Answer with its id.",
            worst.entity_id, "id", (worst.entity_id,),
        ))
        qs.append(Question(
            "q_worst_gap_amount", "paise",
            "How many paise is the largest unexplained gap? Signed: negative "
            "means the credit was short.",
            worst.residual_paise, "paise", (worst.entity_id,),
        ))

    # ── per-credit questions, the bread and butter ───────────────────────
    for exc in with_residual[:limit_per_kind]:
        qs.append(Question(
            f"q_gap_{exc.entity_id}", "paise",
            f"By how many paise does bank credit {exc.entity_id} differ from the "
            "rows its settlement batch accounts for? Signed.",
            exc.residual_paise, "paise", (exc.entity_id,),
        ))
        qs.append(Question(
            f"q_amount_{exc.entity_id}", "paise",
            f"What is the credited amount of bank credit {exc.entity_id}, in paise?",
            amounts.get(exc.entity_id), "paise", (exc.entity_id,),
        ))

    for m in sorted(matched_l2, key=lambda m: m.match_id)[:limit_per_kind]:
        line_id, settlement_id = m.left_ids[0], m.right_ids[0]
        qs.append(Question(
            f"q_match_{line_id}", "id",
            f"Which settlement batch does bank credit {line_id} belong to? "
            "Answer with the settlement id.",
            settlement_id, "id", (line_id, settlement_id),
        ))
        qs.append(Question(
            f"q_members_{settlement_id}", "count",
            f"How many payments are in settlement batch {settlement_id}?",
            len(sources.payments_by_settlement(settlement_id)), "count",
            (settlement_id,),
        ))

    # ── resolved-or-not, where a wrong answer is operationally expensive ──
    matched_ids = {m.left_ids[0] for m in matched_l2}
    for exc in with_residual[:2]:
        qs.append(Question(
            f"q_isopen_{exc.entity_id}", "bool",
            f"Has bank credit {exc.entity_id} been matched to a settlement batch? "
            "Answer true or false.",
            exc.entity_id in matched_ids, "bool", (exc.entity_id,),
        ))
    for m in sorted(matched_l2, key=lambda m: m.match_id)[:2]:
        qs.append(Question(
            f"q_isopen_{m.left_ids[0]}", "bool",
            f"Has bank credit {m.left_ids[0]} been matched to a settlement batch? "
            "Answer true or false.",
            True, "bool", (m.left_ids[0],),
        ))

    # ── derived: arithmetic over the facts, not a field to read off ──────
    if with_residual:
        qs.append(Question(
            "q_unmatched_credits", "count",
            "How many bank credits have NOT been matched to a settlement batch?",
            len(sources.bank_lines) - len(matched_l2), "count",
        ))
        worst = with_residual[0]
        qs.append(Question(
            "q_remaining_if_worst_fixed", "count",
            f"If bank credit {worst.entity_id} were resolved, how many items would "
            "remain in the exception queue?",
            len(result.exceptions) - 1, "count", (worst.entity_id,),
        ))
        qs.append(Question(
            "q_unexplained_less_worst", "paise",
            f"Excluding {worst.entity_id}, what is the total unexplained amount "
            "across the remaining open items, in paise? Use absolute values.",
            sum(abs(e.residual_paise or 0) for e in result.exceptions)
            - abs(worst.residual_paise or 0),
            "paise", (worst.entity_id,),
        ))
    if len(with_residual) >= 2:
        a, b = with_residual[0], with_residual[1]
        qs.append(Question(
            "q_bigger_gap", "id",
            f"Which has the larger gap in absolute terms, {a.entity_id} or "
            f"{b.entity_id}? Answer with the id.",
            a.entity_id if abs(a.residual_paise or 0) >= abs(b.residual_paise or 0)
            else b.entity_id,
            "id", (a.entity_id, b.entity_id),
        ))

    # ── unanswerable: the hallucination probe ────────────────────────────
    qs.append(Question(
        "q_absent_entity", "decline",
        "By how many paise does bank credit bank_99999 differ from its batch?",
        None, "decline", expects_decline=True,
    ))
    qs.append(Question(
        "q_absent_field", "decline",
        "Which member of the finance team approved the largest settlement batch?",
        None, "decline", expects_decline=True,
    ))
    if with_residual:
        qs.append(Question(
            "q_absent_cause", "decline",
            f"What is the customer's email address on the payment that caused the "
            f"gap on {with_residual[0].entity_id}?",
            None, "decline", expects_decline=True, depends_on=(with_residual[0].entity_id,),
        ))

    return [q for q in qs if q.answer is not None or q.expects_decline]


def is_correct(question: Question, given: Any, tolerance_paise: Paise = 0) -> bool:
    """Exact comparison, typed. No partial credit and no fuzzy matching.

    A Q&A agent that is graded leniently reports a number nobody should act on.
    An operator either gets the right figure or does not.
    """
    if question.expects_decline:
        # Handled by the caller, which knows whether the agent declined.
        return False
    if given is None:
        return False
    try:
        if question.answer_type == "paise":
            return abs(int(given) - int(question.answer)) <= tolerance_paise
        if question.answer_type == "count":
            return int(given) == int(question.answer)
        if question.answer_type == "bool":
            if isinstance(given, str):
                given = given.strip().lower() in ("true", "yes", "1")
            return bool(given) == bool(question.answer)
        return str(given).strip() == str(question.answer).strip()
    except (TypeError, ValueError):
        return False
