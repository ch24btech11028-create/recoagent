"""The settlement Q&A agent, and the harness that scores it.

Ask it about a reconciliation run and it answers with a number or an id -- or
says it cannot. The design mirrors the reconciliation tier deliberately, because
the same failure is available here and it is the expensive one: a model that
answers confidently and wrongly is worse than a model that declines, since an
operator acts on the number either way.

So the lead metric is **wrong-answer rate**, not accuracy. Coverage is reported
second. That is the same ordering as false-match rate over match rate, for the
same reason.

Two things keep the numbers honest:

- **Facts are retrieved deterministically, not recalled.** The agent is handed a
  compact factsheet built from the run by code, and told to answer only from it.
  It has no memory of the batch and no way to invent one.
- **Answers are typed and compared exactly.** `bank.is_correct` does the
  grading, not a second model. A benchmark graded by a model is not a benchmark.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm import Chat, Usage
from ..money import format_inr
from ..schemas import LabelledBatch, ReconResult
from .bank import Question, is_correct

SYSTEM = """\
You answer questions about a completed payment reconciliation run for an Indian \
merchant. You are given a factsheet built from that run and one question.

Rules:
- Answer ONLY from the factsheet. It is the complete record; nothing else exists.
- All money is in integer paise. 100 paise = 1 rupee. Never convert to rupees.
- A negative gap means the bank credited LESS than the batch rows account for.
- If the factsheet does not contain what you need, say so. Declining is a good \
answer; a confident wrong number is the worst one available to you, because the \
operator will act on it either way.

Reply with a single JSON object and nothing else -- no prose, no markdown fence:
{"answer": <number, id string, or true/false>, "confidence": 0.0, "basis": "which fact you used"}
or
{"cannot_answer": "what is missing"}"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Answer:
    qid: str
    given: Any = None
    correct: bool = False
    declined: bool = False
    failed: bool = False
    confidence: float | None = None
    basis: str = ""
    detail: str = ""
    #: False when the question had no ground truth to check against. Such an
    #: answer is shown to the operator with its factsheet attached and is
    #: excluded from every rate this module reports.
    graded: bool = True
    usage: Usage = field(default_factory=Usage)


@dataclass
class QAReport:
    answers: list[Answer] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    seconds: float = 0.0

    @property
    def total(self) -> int:
        return len(self.scored)

    @property
    def scored(self) -> list[Answer]:
        """Only answers with a ground truth behind them. Every rate uses these."""
        return [a for a in self.answers if a.graded]

    @property
    def attempted(self) -> int:
        return sum(1 for a in self.scored if not a.declined and not a.failed)

    @property
    def correct(self) -> int:
        """Includes correctly declining an unanswerable question."""
        return sum(1 for a in self.scored if a.correct)

    @property
    def correct_answers(self) -> int:
        """Correct among questions it actually answered. Excludes right declines."""
        return sum(1 for a in self.scored if a.correct and not a.declined)

    @property
    def wrong(self) -> int:
        # Must be counted against answers given, not against `correct` -- a
        # correct decline raises `correct` without raising `attempted`, which
        # drove this negative (-6.45%) on the first run with decline probes.
        return self.attempted - self.correct_answers

    @property
    def declined(self) -> int:
        return sum(1 for a in self.scored if a.declined)

    @property
    def hallucinated(self) -> int:
        """Answered a question the factsheet cannot support. The worst outcome."""
        return sum(1 for a in self.scored if a.detail.startswith("HALLUCINATED"))

    @property
    def failed(self) -> int:
        return sum(1 for a in self.scored if a.failed)

    @property
    def wrong_answer_rate(self) -> float:
        """The metric that leads. Share of answers given that were wrong."""
        return self.wrong / self.attempted if self.attempted else 0.0

    @property
    def coverage(self) -> float:
        return self.attempted / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def factsheet(batch: LabelledBatch, result: ReconResult, question: Question) -> str:
    """Everything the agent may see, built by code from the run.

    Portfolio totals always; per-entity detail only for what the question names.
    Handing over the whole book would make the context enormous and would also
    stop measuring anything useful -- an agent that can see every row is being
    tested on arithmetic, not on retrieval plus reasoning.
    """
    sources = batch.sources
    matched = result.matches_for_leg(2)
    matched_ids = {m.left_ids[0] for m in matched}
    residual_excs = [
        e for e in result.exceptions
        if e.leg == 2 and e.entity_kind == "bank_line" and e.residual_paise is not None
    ]
    worst = max(residual_excs, key=lambda e: abs(e.residual_paise or 0), default=None)

    facts: dict[str, Any] = {
        "portfolio": {
            "bank_credits_in_statement": len(sources.bank_lines),
            "settlement_batches": len(sources.settlements),
            "payments": len(sources.payments),
            "credits_matched_to_a_batch": len(matched),
            "open_exception_items": len(result.exceptions),
            "unlinked_adjustment_rows": len(sources.unlinked_adjustments),
            "total_unexplained_paise": sum(
                abs(e.residual_paise or 0) for e in result.exceptions
            ),
        }
    }
    if worst is not None:
        facts["largest_unexplained_gap"] = {
            "bank_line_id": worst.entity_id,
            "gap_paise": worst.residual_paise,
        }

    # Pull in exactly the entities the question mentions.
    named = set(question.depends_on) | set(re.findall(r"\b(?:bank|setl)_\d+\b", question.text))
    if named:
        detail: dict[str, Any] = {}
        for line in sources.bank_lines:
            if line.bank_line_id in named:
                exc = next((e for e in residual_excs if e.entity_id == line.bank_line_id), None)
                match = next((m for m in matched if m.left_ids[0] == line.bank_line_id), None)
                detail[line.bank_line_id] = {
                    "credited_amount_paise": line.amount_paise,
                    "value_date": line.value_date.isoformat(),
                    "matched_to_settlement": match.right_ids[0] if match else None,
                    "is_matched": line.bank_line_id in matched_ids,
                    "gap_paise": exc.residual_paise if exc else None,
                    "status": "matched" if match else "open in the exception queue",
                }
        for s in sources.settlements:
            if s.settlement_id in named:
                members = sources.payments_by_settlement(s.settlement_id)
                detail[s.settlement_id] = {
                    "payments_in_batch": len(members),
                    "reported_net_paise": s.net_paise,
                    "settled_at": s.settled_at.isoformat(),
                    "status": s.status,
                }
        if detail:
            facts["entities"] = detail

    return json.dumps(facts, indent=2, sort_keys=True)


def _parse(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON.search(text)
        if not m:
            raise ValueError(f"no JSON object in reply: {text[:120]!r}")
        return json.loads(m.group(0))


def ask(
    chat: Chat,
    batch: LabelledBatch,
    result: ReconResult,
    question: Question,
    tolerance_paise: int = 0,
) -> Answer:
    """Ask one question. Never raises; a failed call is a recorded outcome."""
    ans = Answer(qid=question.qid)
    reply = chat.send(
        SYSTEM,
        json.dumps(
            {"factsheet": json.loads(factsheet(batch, result, question)),
             "question": question.text},
            indent=2,
        ),
        max_tokens=2000,
    )
    ans.usage.merge(reply.usage)
    if not reply.ok:
        ans.failed = True
        ans.detail = reply.error or "empty reply"
        return ans

    try:
        payload = _parse(reply.text)
    except (ValueError, json.JSONDecodeError) as exc:
        ans.failed = True
        ans.detail = f"malformed: {exc}"
        return ans

    if "cannot_answer" in payload:
        ans.declined = True
        ans.detail = str(payload["cannot_answer"])[:120]
        # Declining an unanswerable question is the right answer, not a gap in
        # coverage. Scoring it as a miss would push the agent toward guessing.
        ans.correct = question.expects_decline
        return ans

    ans.given = payload.get("answer")
    ans.basis = str(payload.get("basis", ""))[:160]
    try:
        ans.confidence = float(payload.get("confidence")) if payload.get("confidence") is not None else None
    except (TypeError, ValueError):
        ans.confidence = None
    if question.expects_decline:
        ans.correct = False
        ans.detail = (
            f"HALLUCINATED {ans.given!r} -- the factsheet does not contain this"
        )
        return ans
    if not question.graded:
        # No ground truth exists for this one. Say so rather than reporting a
        # `correct` of False, which reads as "the agent got it wrong".
        ans.graded = False
        ans.detail = "ungraded: asked live, no ground truth to check against"
        return ans
    ans.correct = is_correct(question, ans.given, tolerance_paise)
    if not ans.correct:
        ans.detail = f"said {ans.given!r}, expected {question.answer!r}"
    return ans


def render(report: QAReport, questions: list[Question]) -> str:
    by_id = {q.qid: q for q in questions}
    w = 72
    out = [
        "=" * w,
        "SETTLEMENT Q&A",
        "=" * w,
        "",
        f"  WRONG-ANSWER RATE   {report.wrong_answer_rate:>8.2%}   <- lead metric",
        f"  coverage            {report.coverage:>8.2%}   ({report.attempted} of {report.total} answered)",
        f"  accuracy            {report.accuracy:>8.2%}   (correct out of all questions)",
        "",
        f"  HALLUCINATED        {report.hallucinated:>8}   "
        "(answered a question the factsheet cannot support)",
        "",
        f"  correct {report.correct}   wrong {report.wrong}   "
        f"declined {report.declined}   call failed {report.failed}",
        f"  tokens {report.usage.input_tokens:,} in / {report.usage.output_tokens:,} out"
        f"   {report.seconds:.0f}s",
        "",
        "-" * w,
        "  WHAT IT GOT WRONG, DECLINED, OR FAILED",
        "-" * w,
    ]
    bad = [a for a in report.answers if not a.correct]
    if not bad:
        out.append("    nothing")
    for a in bad:
        q = by_id.get(a.qid)
        state = "declined" if a.declined else ("failed" if a.failed else "WRONG")
        out.append(f"    [{state}] {q.text[:60] if q else a.qid}")
        out.append(f"             {a.detail[:100]}")
    out += [
        "-" * w,
        "  A wrong answer is worse than a declined one: the operator acts on the",
        "  number either way. That is why the wrong-answer rate leads and why",
        "  declining is scored as a cost, not a failure.",
        "=" * w,
    ]
    return "\n".join(out)
