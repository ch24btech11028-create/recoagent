"""The agent's tools, and the loop that drives them.

The loop is exercised with a fake client so every path -- a good investigation,
a malformed reply, a wandering agent that never concludes -- is a test rather
than an anecdote costing API calls.
"""

import json

import pytest

from recoagent.agent.agentic import AgenticProposer
from recoagent.agent.contracts import Hypothesis, ProposerError, Refusal
from recoagent.agent.evidence import build as build_packet
from recoagent.agent.tools import ToolContext, execute
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.money import FeeSchedule
from recoagent.pipeline import run_b2
from recoagent.validate import Tolerance


@pytest.fixture(scope="module")
def case():
    batch = generate(GeneratorConfig(n_orders=1500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    exc = next(
        e for e in result.exceptions
        if e.leg == 2 and e.entity_kind == "bank_line" and e.residual_paise is not None
    )
    line = next(b for b in batch.sources.bank_lines if b.bank_line_id == exc.entity_id)
    settlement = next(
        s for s in batch.sources.settlements if s.settlement_id == exc.related_id
    )
    ctx = ToolContext(
        sources=batch.sources, line=line, settlement=settlement,
        residual_paise=exc.residual_paise, tol=Tolerance.calibrated(),
        fees=FeeSchedule.default(),
    )
    packet = build_packet(
        batch.sources, line, settlement, exc.residual_paise, FeeSchedule.default()
    )
    return batch, ctx, packet


# ── tools ────────────────────────────────────────────────────────────────


def test_list_payments_filters_by_method(case):
    _, ctx, _ = case
    everything = execute(ctx, "list_payments", {})
    upi = execute(ctx, "list_payments", {"method": "upi"})
    assert everything["count"] >= upi["count"]
    assert all(p["method"] == "upi" for p in upi["payments"])


def test_check_hypothesis_reports_the_gap_without_committing(case):
    _, ctx, _ = case
    exact = execute(ctx, "check_hypothesis", {"rows": [{"amount_paise": ctx.residual_paise}]})
    assert exact["would_close"] is True
    assert exact["still_unexplained_paise"] == 0

    wrong = execute(ctx, "check_hypothesis",
                    {"rows": [{"amount_paise": ctx.residual_paise + 5000}]})
    assert wrong["would_close"] is False
    assert wrong["still_unexplained_paise"] == -5000


def test_check_hypothesis_rejects_fractional_amounts(case):
    _, ctx, _ = case
    out = execute(ctx, "check_hypothesis", {"rows": [{"amount_paise": 12.5}]})
    assert "error" in out


def test_compute_fee_scenario_is_a_calculator_not_an_oracle(case):
    """It answers 'what if the rate were X', not 'what was the rate'."""
    _, ctx, _ = case
    out = execute(ctx, "compute_fee_scenario", {"mdr_bps": 200})
    assert {"reported_fee_plus_tax_paise", "scenario_fee_plus_tax_paise",
            "delta_paise", "delta_matches_residual"} <= set(out)
    higher = execute(ctx, "compute_fee_scenario", {"mdr_bps": 400})
    assert higher["scenario_fee_plus_tax_paise"] > out["scenario_fee_plus_tax_paise"]


def test_tools_validate_their_arguments(case):
    _, ctx, _ = case
    assert "error" in execute(ctx, "compute_fee_scenario", {"mdr_bps": "two percent"})
    assert "error" in execute(ctx, "list_unlinked_rows", {"window_days": 999})
    assert "error" in execute(ctx, "check_hypothesis", {"rows": []})


def test_unknown_tool_returns_the_menu(case):
    _, ctx, _ = case
    out = execute(ctx, "book_the_match", {})
    assert "error" in out and "available" in out


def test_a_tool_that_explodes_does_not_kill_the_case(case):
    _, ctx, _ = case
    out = execute(ctx, "list_payments", {"min_gross_paise": object()})
    assert isinstance(out, dict)  # returned, not raised


# ── the loop ─────────────────────────────────────────────────────────────


class FakeClient:
    """Replays scripted assistant turns through the OpenAI response shape."""

    def __init__(self, turns):
        self._turns = list(turns)
        self.calls = 0
        self.chat = self  # client.chat.completions.create(...)
        self.completions = self

    def create(self, **kwargs):
        self.calls += 1
        text = self._turns.pop(0) if self._turns else '{"action": "decline", "reason": "x"}'

        class Msg:
            content = text
        class Choice:
            message = Msg(); finish_reason = "stop"
        class U:
            prompt_tokens = 100; completion_tokens = 20
        class R:
            choices = [Choice()]; usage = U()
        return R()


def _proposer(case, turns, **kw):
    batch, ctx, _ = case
    p = AgenticProposer(
        sources=batch.sources, tol=Tolerance.calibrated(),
        client=FakeClient(turns), **kw,
    )
    p.bind(ctx.line, ctx.settlement)
    return p


def test_the_agent_investigates_then_proposes(case):
    _, ctx, packet = case
    turns = [
        json.dumps({"tool": "list_payments", "args": {"method": "card_domestic"}}),
        json.dumps({"tool": "check_hypothesis",
                    "args": {"rows": [{"amount_paise": ctx.residual_paise}]}}),
        json.dumps({"action": "propose",
                    "rows": [{"label": "found", "amount_paise": ctx.residual_paise,
                              "rationale": "checked"}],
                    "reason": "verified with check_hypothesis", "confidence": 0.9}),
    ]
    p = _proposer(case, turns)
    proposal, usage = p.propose(packet)
    assert isinstance(proposal, Hypothesis)
    assert usage.calls == 3
    trace = p.transcripts[ctx.line.bank_line_id]
    assert [t.get("tool") for t in trace[:2]] == ["list_payments", "check_hypothesis"]


def test_tool_results_are_fed_back_to_the_model(case):
    """Without this the loop is not a loop -- the model would re-ask forever."""
    _, ctx, packet = case
    captured = {}

    class Recorder(FakeClient):
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            return super().create(**kwargs)

    batch, _, _ = case
    p = AgenticProposer(
        sources=batch.sources, tol=Tolerance.calibrated(),
        client=Recorder([
            json.dumps({"tool": "list_payments", "args": {}}),
            json.dumps({"action": "decline", "reason": "done"}),
        ]),
    )
    p.bind(ctx.line, ctx.settlement)
    p.propose(packet)
    assert any("tool_result" in str(m.get("content", "")) for m in captured["messages"])


def test_a_wandering_agent_declines_rather_than_guessing(case):
    _, ctx, packet = case
    forever = [json.dumps({"tool": "list_payments", "args": {}})] * 20
    p = _proposer(case, forever, max_turns=4)
    proposal, usage = p.propose(packet)
    assert isinstance(proposal, Refusal)
    assert "without reaching a conclusion" in proposal.reason
    assert usage.calls == 4


def test_malformed_turn_is_an_error_not_a_guess(case):
    _, ctx, packet = case
    p = _proposer(case, ["I think the answer is probably a refund of some kind."])
    proposal, _ = p.propose(packet)
    assert isinstance(proposal, ProposerError)
    assert proposal.kind == "malformed"


def test_unbound_proposer_refuses_to_run(case):
    """Tools scoped to the wrong case would investigate the wrong batch."""
    batch, _, packet = case
    p = AgenticProposer(
        sources=batch.sources, tol=Tolerance.calibrated(), client=FakeClient([]),
    )
    proposal, _ = p.propose(packet)
    assert isinstance(proposal, ProposerError)


def test_transport_failure_is_caught(case):
    _, ctx, packet = case

    class Boom(FakeClient):
        def create(self, **kwargs):
            raise ConnectionError("endpoint unreachable")

    batch, _, _ = case
    p = AgenticProposer(sources=batch.sources, tol=Tolerance.calibrated(), client=Boom([]))
    p.bind(ctx.line, ctx.settlement)
    proposal, _ = p.propose(packet)
    assert isinstance(proposal, ProposerError) and proposal.kind == "transport"
