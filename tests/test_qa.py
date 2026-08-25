"""Settlement Q&A: the bank, the factsheet, and the grading.

Everything here runs against a scripted client. A test suite that needed an
endpoint would be skipped in CI and would stop protecting anything.
"""

import json

import pytest

from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.llm import ScriptedChat
from recoagent.pipeline import run_b2
from recoagent.qa import agent, bank


@pytest.fixture(scope="module")
def run():
    batch = generate(GeneratorConfig(n_orders=1500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    return batch, result, bank.build(batch, result)


# ── the bank ─────────────────────────────────────────────────────────────


def test_every_question_has_a_computed_answer(run):
    _, _, qs = run
    assert len(qs) >= 15
    answerable = [q for q in qs if not q.expects_decline]
    assert all(q.answer is not None for q in answerable)
    assert all(q.answer_type in {"paise", "count", "id", "bool"} for q in answerable)


def test_the_bank_contains_unanswerable_questions(run):
    """Without these the bank cannot detect hallucination at all.

    An agent that always answers scores perfectly on a bank where everything is
    answerable -- which is exactly the blind spot that lets a confident wrong
    number reach an operator.
    """
    _, _, qs = run
    probes = [q for q in qs if q.expects_decline]
    assert len(probes) >= 3
    assert all(q.answer is None for q in probes)


def test_the_bank_requires_arithmetic_not_just_lookup(run):
    """A bank of pure field-reads measures JSON parsing, not reasoning."""
    _, _, qs = run
    derived = {"q_unmatched_credits", "q_remaining_if_worst_fixed",
               "q_unexplained_less_worst", "q_bigger_gap"}
    assert derived <= {q.qid for q in qs}


def test_declining_an_unanswerable_question_is_correct(run):
    batch, result, qs = run
    probe = next(q for q in qs if q.expects_decline)
    a = agent.ask(_chat({"cannot_answer": "not in the factsheet"}), batch, result, probe)
    assert a.declined and a.correct


def test_answering_an_unanswerable_question_is_a_hallucination(run):
    batch, result, qs = run
    probe = next(q for q in qs if q.expects_decline)
    a = agent.ask(_chat({"answer": 12345, "confidence": 0.95, "basis": "made up"}),
                  batch, result, probe)
    assert not a.correct
    assert a.detail.startswith("HALLUCINATED")
    r = agent.QAReport(answers=[a])
    assert r.hallucinated == 1


def test_the_bank_is_deterministic(run):
    batch, result, qs = run
    again = bank.build(batch, result)
    assert [(q.qid, q.answer) for q in qs] == [(q.qid, q.answer) for q in again]


def test_answers_agree_with_the_run_they_came_from(run):
    """The bank must describe the actual run, not a remembered one."""
    _, result, qs = run
    by_id = {q.qid: q for q in qs}
    assert by_id["q_open_count"].answer == len(result.exceptions)
    assert by_id["q_matched_credits"].answer == len(result.matches_for_leg(2))


def test_grading_is_exact_not_lenient(run):
    _, _, qs = run
    q = next(q for q in qs if q.answer_type == "count")
    assert bank.is_correct(q, q.answer)
    assert not bank.is_correct(q, q.answer + 1)
    assert not bank.is_correct(q, None)
    assert not bank.is_correct(q, "about " + str(q.answer))


def test_paise_answers_accept_a_stated_tolerance_only(run):
    _, _, qs = run
    q = next(q for q in qs if q.answer_type == "paise" and q.answer)
    assert bank.is_correct(q, q.answer + 5, tolerance_paise=10)
    assert not bank.is_correct(q, q.answer + 5, tolerance_paise=0)


def test_bool_answers_accept_model_phrasing(run):
    _, _, qs = run
    q = next(q for q in qs if q.answer_type == "bool")
    assert bank.is_correct(q, q.answer)
    assert bank.is_correct(q, "true" if q.answer else "false")
    assert not bank.is_correct(q, not q.answer)


# ── the factsheet ────────────────────────────────────────────────────────


def test_factsheet_carries_no_ground_truth(run):
    """Same restriction as every matcher. An agent that can see the answer key
    is being tested on nothing."""
    batch, result, qs = run
    blob = " ".join(agent.factsheet(batch, result, q) for q in qs).lower()
    for leaked in ("defect", "injected", "ground_truth", "refund_netted", "fee_tax"):
        assert leaked not in blob, f"factsheet leaks {leaked!r}"


def test_factsheet_includes_the_entity_the_question_names(run):
    batch, result, qs = run
    q = next(q for q in qs if q.qid.startswith("q_gap_bank_"))
    sheet = json.loads(agent.factsheet(batch, result, q))
    assert q.depends_on[0] in sheet.get("entities", {})


def test_factsheet_stays_small(run):
    """Handing over the whole book would stop measuring retrieval."""
    batch, result, qs = run
    assert max(len(agent.factsheet(batch, result, q)) for q in qs) < 6000


# ── answering ────────────────────────────────────────────────────────────


def _chat(payload):
    return ScriptedChat(lambda s, u: json.dumps(payload))


def test_a_correct_answer_is_scored_correct(run):
    batch, result, qs = run
    q = next(q for q in qs if q.answer_type == "count")
    a = agent.ask(_chat({"answer": q.answer, "confidence": 0.9, "basis": "portfolio"}),
                  batch, result, q)
    assert a.correct and not a.declined and not a.failed


def test_a_confident_wrong_answer_is_scored_wrong(run):
    """The expensive failure. An operator acts on the number either way."""
    batch, result, qs = run
    q = next(q for q in qs if q.answer_type == "count")
    a = agent.ask(_chat({"answer": q.answer + 7, "confidence": 0.99, "basis": "x"}),
                  batch, result, q)
    assert not a.correct and not a.declined
    assert "expected" in a.detail


def test_declining_is_recorded_separately_from_being_wrong(run):
    batch, result, qs = run
    a = agent.ask(_chat({"cannot_answer": "not in the factsheet"}), batch, result, qs[0])
    assert a.declined and not a.correct and not a.failed


@pytest.mark.parametrize("reply", ["not json at all", "", "```json\n{oops}\n```"])
def test_malformed_replies_are_failures_not_guesses(run, reply):
    batch, result, qs = run
    a = agent.ask(ScriptedChat([reply]), batch, result, qs[0])
    assert a.failed and not a.correct


def test_transport_failure_is_caught(run):
    batch, result, qs = run

    class Boom:
        label = "boom"
        def send(self, system, user, **kw):
            from recoagent.llm import Reply
            return Reply(error="ConnectionError: endpoint unreachable")

    a = agent.ask(Boom(), batch, result, qs[0])
    assert a.failed and "unreachable" in a.detail


# ── the report ───────────────────────────────────────────────────────────


def test_wrong_answer_rate_ignores_declines(run):
    """Declining is a cost, not a wrong answer. Conflating them would make
    silence look like error and reward guessing."""
    r = agent.QAReport(answers=[
        agent.Answer("a", correct=True),
        agent.Answer("b", correct=False),
        agent.Answer("c", declined=True),
    ])
    assert r.attempted == 2
    assert r.wrong_answer_rate == 0.5
    assert r.coverage == pytest.approx(2 / 3)
    assert r.accuracy == pytest.approx(1 / 3)


def test_report_lists_everything_it_got_wrong(run):
    _, _, qs = run
    r = agent.QAReport(answers=[
        agent.Answer(qs[0].qid, correct=False, detail="said 5, expected 7"),
        agent.Answer(qs[1].qid, correct=True),
    ])
    text = agent.render(r, qs)
    assert "WRONG" in text and "said 5, expected 7" in text
    assert "WRONG-ANSWER RATE" in text
