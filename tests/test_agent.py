"""The agent tier, and every way a proposer can go wrong.

The load-bearing test in this file is
`test_a_confident_wrong_proposal_is_rejected`. Everything else in B3 is
plumbing; that one is the design. A model that can assert a match can assert a
wrong one convincingly, so the system is built so it can only ever offer
arithmetic that either closes or does not.
"""

import pytest

from recoagent.agent import (
    Hypothesis,
    NullProposer,
    ProposedRow,
    ProposerError,
    Refusal,
    ScriptedProposer,
    Usage,
    recover_with_agent,
)
from recoagent.agent.proposer import _parse_tool_call
from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b2, run_b3
from recoagent.validate import Tolerance


def _batch(n=1500, seed=7):
    return generate(GeneratorConfig(n_orders=n, seed=seed, mix=DefectMix.dev()))


def _first_target(batch):
    """One leg-2 residual exception that survived every deterministic tier."""
    result = run_b2(batch.sources)
    exc = next(
        e
        for e in result.exceptions
        if e.leg == 2 and e.entity_kind == "bank_line" and e.residual_paise is not None
    )
    return result, exc


def _hypothesis(amount, confidence=0.9):
    return Hypothesis(
        rows=(ProposedRow("inferred", amount, "because"),),
        reason="test hypothesis",
        confidence=confidence,
    )


# ── The core guarantee ───────────────────────────────────────────────────


def test_a_confident_wrong_proposal_is_rejected():
    """The whole reason the model proposes instead of deciding.

    A hypothesis stated at 0.99 confidence that does not close the arithmetic
    must not become a match, however plausible it reads.
    """
    batch = _batch()
    result, exc = _first_target(batch)
    wrong = exc.residual_paise + 5000  # confident, specific, and wrong

    # A callable, not a list: a list would run out mid-sweep and the tail of
    # the cases would report "failed" rather than "rejected", quietly weakening
    # the assertion this test exists to make.
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _hypothesis(wrong, confidence=0.99)),
    )
    assert report.rejected == report.attempted
    assert report.resolved == 0
    assert not [m for m in result.matches_for_leg(2) if m.tier == "T2"]


def test_a_correct_proposal_is_accepted():
    batch = _batch()
    result, exc = _first_target(batch)

    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise)),
    )
    assert report.resolved == report.attempted > 0
    booked = [m for m in result.matches_for_leg(2) if m.tier == "T2"]
    assert booked and all(m.proof.closes for m in booked)


def test_accepted_matches_are_still_correct_matches():
    """Resolving via the model must not book a credit against the wrong batch."""
    batch = _batch()
    result = run_b2(batch.sources)
    recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise)),
    )
    card = score(batch, result)
    assert card.overall_false_match_rate == 0.0
    assert card.mishandled_total == 0


# ── Failure modes ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "proposal, expected",
    [
        (ProposerError("timeout", "deadline exceeded"), "failed"),
        (ProposerError("malformed", "no tool call in response"), "failed"),
        (ProposerError("transport", "503"), "failed"),
        (Refusal("evidence does not support an explanation"), "refused"),
    ],
    ids=["timeout", "malformed", "transport", "refusal"],
)
def test_every_proposer_failure_lands_in_the_exception_queue(proposal, expected):
    batch = _batch()
    result, _ = _first_target(batch)
    before = len(result.exceptions)

    report = recover_with_agent(
        batch.sources, Tolerance.calibrated(), result, ScriptedProposer([proposal] * 40)
    )
    assert report.attempted > 0
    assert all(c.outcome == expected for c in report.cases)
    assert len(result.exceptions) == before  # nothing lost, nothing invented
    assert not [m for m in result.matches_for_leg(2) if m.tier == "T2"]


def test_low_confidence_is_not_even_checked():
    """A model that is unsure is telling you to escalate.

    Checking it anyway risks a coincidental close on a guess.
    """
    batch = _batch()
    result, exc = _first_target(batch)
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        # Correct arithmetic, but the model says it is unsure.
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise, confidence=0.2)),
    )
    assert all(c.outcome == "low_confidence" for c in report.cases)
    assert report.resolved == 0


def test_the_repair_loop_gets_a_second_chance_and_no_more():
    """Wrong once, right on the retry -> resolved. Wrong twice -> escalated."""
    batch = _batch()
    result, _ = _first_target(batch)

    # Attempt counting must be per case: the tier sweeps every surviving
    # exception, so a single shared counter would treat case 2's first attempt
    # as case 1's retry.
    seen: dict[str, int] = {}

    def script(packet):
        key = packet.bank_credit["bank_line_id"]
        seen[key] = seen.get(key, 0) + 1
        if seen[key] == 1:
            assert packet.repair_feedback is None
            return _hypothesis(packet.residual_paise + 999)
        assert packet.repair_feedback is not None, "retry must carry the residual back"
        assert "unexplained" in packet.repair_feedback
        return _hypothesis(packet.residual_paise)

    report = recover_with_agent(
        batch.sources, Tolerance.calibrated(), result, ScriptedProposer(script)
    )
    assert report.cases
    assert all(c.outcome == "resolved" for c in report.cases)
    assert all(c.attempts == 2 for c in report.cases)


def test_persistent_wrongness_stops_after_max_attempts():
    batch = _batch()
    result, _ = _first_target(batch)
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise + 12345)),
    )
    assert all(c.attempts == 2 for c in report.cases)
    assert report.resolved == 0


# ── The control ──────────────────────────────────────────────────────────


def test_null_proposer_reproduces_b2_exactly():
    """The tier must add nothing on its own.

    If B3-with-a-null-proposer differs from B2, then some part of the lift
    measured later would belong to plumbing rather than to the model.
    """
    batch = _batch()
    b2 = score(batch, run_b2(batch.sources))
    b3_result, report = run_b3(batch.sources, NullProposer())
    b3 = score(batch, b3_result)

    assert report.resolved == 0
    assert b3.legs[2].true_matches == b2.legs[2].true_matches
    assert b3.legs[2].exceptions == b2.legs[2].exceptions
    assert b3.value.share == b2.value.share


# ── Audit and accounting ─────────────────────────────────────────────────


def test_self_reported_confidence_is_capped_not_trusted():
    from recoagent.agent.tier import CONF_T2_CAP

    batch = _batch()
    result, _ = _first_target(batch)
    recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise, confidence=1.0)),
    )
    booked = [m for m in result.matches_for_leg(2) if m.tier == "T2"]
    assert booked
    assert all(m.confidence <= CONF_T2_CAP for m in booked)


def test_inferred_rows_never_look_like_reported_ones():
    batch = _batch()
    result, _ = _first_target(batch)
    recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise)),
    )
    booked = [m for m in result.matches_for_leg(2) if m.tier == "T2"]
    assert booked
    for m in booked:
        assert "hypothesised" in m.proof.expression


def test_usage_is_accounted_per_case_and_in_total():
    batch = _batch()
    result, _ = _first_target(batch)
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(
            lambda p: _hypothesis(p.residual_paise),
            usage_per_call=Usage(calls=1, input_tokens=1000, output_tokens=100),
        ),
    )
    assert report.usage.calls == sum(c.attempts for c in report.cases)
    assert report.usage.input_tokens == report.usage.calls * 1000
    assert report.cost_per_resolved(5.0, 25.0) > 0


# ── Evidence packet carries no labels ────────────────────────────────────


def test_evidence_packet_contains_no_ground_truth():
    """Belt and braces alongside the AST check in test_independence."""
    import json

    from recoagent.agent import evidence
    from recoagent.money import FeeSchedule

    batch = _batch()
    result, exc = _first_target(batch)
    line = next(b for b in batch.sources.bank_lines if b.bank_line_id == exc.entity_id)
    settlement = next(
        s for s in batch.sources.settlements if s.settlement_id == exc.related_id
    )
    packet = evidence.build(
        batch.sources, line, settlement, exc.residual_paise, FeeSchedule.default()
    )
    blob = json.dumps(packet.to_dict()).lower()
    for leaked in ("defect", "injected", "ground_truth", "truth", "label"):
        assert leaked not in blob, f"evidence packet leaks {leaked!r}"


# ── Tool-call parsing ────────────────────────────────────────────────────


def test_parse_rejects_fractional_paise():
    """Every amount in this system is whole paise. A float is malformed input,
    not something to silently truncate into the ledger."""
    with pytest.raises(ValueError):
        _parse_tool_call(
            "propose_hypothesis",
            {"rows": [{"label": "x", "amount_paise": 10.5, "rationale": "r"}],
             "reason": "r", "confidence": 0.8},
        )


def test_parse_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        _parse_tool_call(
            "propose_hypothesis",
            {"rows": [{"label": "x", "amount_paise": 10, "rationale": "r"}],
             "reason": "r", "confidence": 1.7},
        )


def test_parse_rejects_empty_rows():
    with pytest.raises(ValueError):
        _parse_tool_call(
            "propose_hypothesis", {"rows": [], "reason": "r", "confidence": 0.8}
        )


def test_parse_reads_a_refusal():
    assert isinstance(_parse_tool_call("flag_for_human", {"reason": "unclear"}), Refusal)


def test_parse_rejects_an_unknown_tool():
    with pytest.raises(ValueError):
        _parse_tool_call("book_the_match", {"settlement_id": "setl_0001"})


# ── concurrency ──────────────────────────────────────────────────────────


def test_parallel_and_serial_produce_identical_output():
    """Concurrency must be an execution detail, never a semantic one.

    Cases finish in whatever order the endpoint returns them, so if results
    were applied as they arrived, the exception queue and the audit log would
    reshuffle between runs and the determinism guarantee would quietly die.
    """
    from recoagent.pipeline import run_b2

    def snapshot(result):
        return (
            [(m.match_id, m.right_ids, round(m.confidence, 6)) for m in
             sorted(result.matches, key=lambda m: m.match_id)],
            [(e.exception_id, e.reason) for e in result.exceptions],
        )

    batch = _batch(n=1500, seed=7)

    serial = run_b2(batch.sources)
    recover_with_agent(
        batch.sources, Tolerance.calibrated(), serial,
        ScriptedProposer(lambda p: _hypothesis(p.residual_paise)),
    )

    parallel = run_b2(batch.sources)
    report = recover_with_agent(
        batch.sources, Tolerance.calibrated(), parallel,
        max_workers=8,
        proposer_factory=lambda: ScriptedProposer(
            lambda p: _hypothesis(p.residual_paise)
        ),
    )

    assert report.resolved > 0
    assert snapshot(serial) == snapshot(parallel)


def test_exception_order_survives_concurrency():
    from recoagent.pipeline import run_b2

    batch = _batch(n=1500, seed=7)
    serial = run_b2(batch.sources)
    recover_with_agent(
        batch.sources, Tolerance.calibrated(), serial,
        ScriptedProposer(lambda p: Refusal("no")),
    )
    parallel = run_b2(batch.sources)
    recover_with_agent(
        batch.sources, Tolerance.calibrated(), parallel,
        max_workers=8, proposer_factory=lambda: ScriptedProposer(lambda p: Refusal("no")),
    )
    assert [e.exception_id for e in serial.exceptions] == \
           [e.exception_id for e in parallel.exceptions]


def test_concurrency_without_a_factory_is_refused():
    """A proposer that investigates holds per-case state; sharing it would let
    two cases scribble over each other's context."""
    from recoagent.pipeline import run_b2

    batch = _batch(n=800, seed=7)
    with pytest.raises(ValueError, match="proposer_factory"):
        recover_with_agent(
            batch.sources, Tolerance.calibrated(), run_b2(batch.sources),
            ScriptedProposer(lambda p: _hypothesis(p.residual_paise)),
            max_workers=8,
        )


def test_each_worker_gets_its_own_proposer():
    from recoagent.pipeline import run_b2

    built = []

    def factory():
        p = ScriptedProposer(lambda pk: _hypothesis(pk.residual_paise))
        built.append(p)
        return p

    batch = _batch(n=1500, seed=7)
    recover_with_agent(
        batch.sources, Tolerance.calibrated(), run_b2(batch.sources),
        max_workers=4, proposer_factory=factory,
    )
    # One per worker thread, not one per case.
    assert 1 <= len(built) <= 4
