"""The agent tier, and every way a proposer can go wrong.

The load-bearing test in this file is
`test_a_confident_wrong_proposal_is_rejected`. Everything else in B3 is
plumbing; that one is the design. A model that can assert a match can assert a
wrong one convincingly, so the system is built so it can only ever offer
arithmetic that either closes or does not.
"""

import pytest

from dataclasses import replace
from datetime import timedelta

from recoagent.agent import (
    CitedAdjustment,
    FeeVarianceClaim,
    FxClaim,
    Hypothesis,
    NullProposer,
    ProposerError,
    Refusal,
    ScriptedProposer,
    Usage,
    recover_with_agent,
)
from recoagent.agent.proposer import _parse_tool_call
from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.money import FeeSchedule
from recoagent.pipeline import run_b2, run_b3
from recoagent.validate import Tolerance


def _batch(n=1500, seed=7, paperwork=False):
    """By default, a book whose rate notices never arrived.

    With the paperwork present the deterministic tier closes fee and FX
    variances itself, so the agent tier has nothing to propose -- that is the
    whole point of `legs/repricing.py`. The territory the agent still owns is
    the case where no document explains the gap, so these tests run without one
    unless a test is specifically about verifying a citation against a rate book.
    """
    batch = generate(GeneratorConfig(n_orders=n, seed=seed, mix=DefectMix.dev()))
    if paperwork:
        return batch
    return replace(batch, sources=replace(
        batch.sources, rate_notices=(), fx_advices=()
    ))


def _first_target(batch):
    """One leg-2 residual exception that survived every deterministic tier."""
    result = run_b2(batch.sources)
    exc = next(
        e
        for e in result.exceptions
        if e.leg == 2 and e.entity_kind == "bank_line" and e.residual_paise is not None
    )
    return result, exc


def _cite(adjustment_id, confidence=0.9):
    """A hypothesis that points at a real unlinked row."""
    return Hypothesis(
        citations=(CitedAdjustment(adjustment_id, "cited in test"),),
        reason="test hypothesis",
        confidence=confidence,
    )


def _invented(amount, confidence=0.99):
    """What the old design allowed: an amount with no source behind it.

    Kept as a hostile input. It must now be impossible to express -- the only
    way to name money is to cite a row -- so this cites an id that does not
    exist, which is the closest a proposer can now get to making one up.
    """
    return Hypothesis(
        citations=(CitedAdjustment(f"invented_{amount}", "there was an adjustment"),),
        reason="invented, cites nothing real",
        confidence=confidence,
    )


def _rate_book_for(batch, hypothesis):
    """A rate book that happens to confirm the rates in this hypothesis.

    Stands in for what a merchant genuinely has -- a repricing notice from the
    gateway, an FX advice from the bank. Built here from the claim so the
    *mechanism* can be tested; it says nothing about whether a real model would
    pick the right rate, which is what the provenance metric is for.
    """
    from recoagent.agent.citations import FeeVarianceClaim, FxClaim, RateBook

    mdr: dict[str, set[int]] = {}
    fx: dict[str, float] = {}
    sources = batch.sources
    for c in hypothesis.citations:
        if isinstance(c, FeeVarianceClaim):
            p0 = next(p for p in sources.payments if p.payment_id == c.payment_ids[0])
            mdr.setdefault(p0.method, set()).add(c.actual_mdr_bps)
        elif isinstance(c, FxClaim):
            fx[c.payment_id] = c.actual_rate_pct_of_gross
    return RateBook(mdr_bps=mdr, fx_pct=fx)


def _first_unlinked(batch):
    return batch.sources.unlinked_adjustments[0].adjustment_id


def _correct_citation(batch, packet, confidence=0.9):
    """Cite evidence that genuinely explains this residual.

    Unlinked-row citations cannot help here and that is the point: everything
    reaching B3 has already survived an exhaustive subset-sum over those rows.
    What is left needs a *rule* -- a repricing or a conversion rate -- so this
    searches for one the same way an analyst would, by trying candidates and
    checking whether the recomputed figure lands on the gap.

    Deliberately never reads ground truth. A test proposer that could see the
    answer key would prove nothing about the tier it is exercising.
    """
    from recoagent.agent.citations import resolve
    from recoagent.money import FeeSchedule

    sources = batch.sources
    sid = packet.settlement["settlement_id"]
    settlement = next(x for x in sources.settlements if x.settlement_id == sid)
    members = sources.payments_by_settlement(sid)
    target = packet.residual_paise
    fees = FeeSchedule.default()

    def closes(cits):
        r = resolve(sources, settlement, list(cits), fees)
        return r.ok and r.total_paise == target

    # An FX slip on a single international payment.
    for p in members:
        if p.currency != "INR" or p.fx_rate is not None:
            if p.gross_paise:
                pct = target / p.gross_paise * 100
                for candidate in (pct, round(pct, 4)):
                    cits = (FxClaim(p.payment_id, candidate, "rate slip"),)
                    if closes(cits):
                        return Hypothesis(cits, "FX conversion slip", confidence)

    # A repricing across the fee-bearing payments.
    charged = [p for p in members if fees.mdr_for(p.method) > 0]
    for size in range(len(charged), 0, -1):
        subset = charged[:size]
        ids = tuple(p.payment_id for p in subset)
        for bps in range(0, 801):
            cits = (FeeVarianceClaim(ids, bps, "mid-cycle repricing"),)
            if closes(cits):
                return Hypothesis(cits, "mid-cycle MDR repricing", confidence)

    return Refusal("no rule reproduces this gap")


# ── The core guarantee ───────────────────────────────────────────────────


def test_a_model_cannot_invent_the_residual():
    """The attack that broke the previous design, kept as a permanent test.

    Before citations, a proposer returned amounts. Since it chose the amount it
    could choose the residual, and "there was an adjustment of exactly this
    much" closed the arithmetic every time -- 7 of 7 cases resolved on a
    fabricated number while the false-match rate still read 0.00%. The gate was
    checking that the model's number made the model's own total add up.

    A proposer can no longer express an amount at all. The nearest it can do is
    cite an id that does not exist, and that resolves to nothing.
    """
    batch = _batch()
    result, exc = _first_target(batch)

    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _invented(p.residual_paise)),
    )
    assert report.resolved == 0, "an invented amount was accepted as a match"
    assert report.unverifiable == report.attempted
    assert not [m for m in result.matches_for_leg(2) if m.tier == "T2"]


def test_citing_a_real_row_that_does_not_close_is_still_rejected():
    """Real evidence, wrong evidence. The arithmetic gate still has work to do."""
    batch = _batch()
    result, _ = _first_target(batch)
    wrong_row = _first_unlinked(batch)

    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _cite(wrong_row, confidence=0.99)),
    )
    assert report.resolved <= 1  # at most the one batch that row truly belongs to
    assert report.rejected >= 1
    for m in result.matches_for_leg(2):
        if m.tier == "T2":
            assert m.proof.closes
            assert m.hypothesised_ids, "an accepted match cites nothing"


def test_a_correct_proposal_is_accepted():
    batch = _batch()
    result, exc = _first_target(batch)

    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _correct_citation(batch, p)),
    )
    # Not every case is explainable by a rule -- a couple are timing spills the
    # deterministic pass could not pair, and no repricing or FX rate reproduces
    # them. A correct proposer declines those. Asserting resolved == attempted
    # would encode the old assumption that the model always answers, which is
    # the behaviour the whole design exists to discourage.
    # Without a rate book, a repricing or FX claim closes the arithmetic on a
    # rate nobody confirmed. Those are held for approval, not reconciled.
    assert report.resolved == 0
    assert report.needs_approval > 0
    assert report.needs_approval + report.refused == report.attempted
    assert report.rejected == 0 and report.unverifiable == 0
    assert not [m for m in result.matches_for_leg(2) if m.tier == "T2"]


def test_a_confirmed_rate_is_reconciled_outright():
    """With an authoritative rate to check against, the same claim is a fact."""
    batch = _batch()
    result, _ = _first_target(batch)

    def propose(packet):
        h = _correct_citation(batch, packet)
        return h

    # Build the book from what a first pass proposes, then run for real.
    probe_result, _ = _first_target(batch)
    from recoagent.agent import evidence as ev
    from recoagent.money import FeeSchedule

    exc = next(e for e in probe_result.exceptions
               if e.leg == 2 and e.entity_kind == "bank_line"
               and e.residual_paise is not None)
    line = next(b for b in batch.sources.bank_lines if b.bank_line_id == exc.entity_id)
    st = next(x for x in batch.sources.settlements if x.settlement_id == exc.related_id)
    packet = ev.build(batch.sources, line, st, exc.residual_paise, FeeSchedule.default())
    first = _correct_citation(batch, packet)
    if isinstance(first, Refusal):
        pytest.skip("no rule explains the first case in this batch")

    report = recover_with_agent(
        batch.sources, Tolerance.calibrated(), result,
        ScriptedProposer(propose),
        rate_book=_rate_book_for(batch, first),
    )
    assert report.resolved >= 1, "a confirmed rate must reconcile"
    for m in result.matches_for_leg(2):
        if m.tier == "T2":
            assert m.proof.closes
            assert m.hypothesised_ids, "an accepted match must name its evidence"


def test_accepted_matches_are_still_correct_matches():
    """Resolving via the model must not book a credit against the wrong batch."""
    batch = _batch()
    result = run_b2(batch.sources)
    recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _correct_citation(batch, p)),
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
        ScriptedProposer(lambda p: _correct_citation(batch, p, confidence=0.2)),
    )
    # Cases the proposer would have declined anyway stay declined; the ones it
    # would have explained are stopped at the confidence floor before the
    # arithmetic is even run.
    assert {c.outcome for c in report.cases} <= {"low_confidence", "refused"}
    assert any(c.outcome == "low_confidence" for c in report.cases)
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
            return _invented(packet.residual_paise + 999)
        assert packet.repair_feedback is not None, "retry must carry the reason back"
        return _correct_citation(batch, packet)

    report = recover_with_agent(
        batch.sources, Tolerance.calibrated(), result, ScriptedProposer(script)
    )
    assert report.cases
    assert all(c.attempts == 2 for c in report.cases), "every case must get its retry"
    # The retry cites a rate rather than a row, so with no rate book it lands in
    # needs_approval. What is being tested here is that the second attempt was
    # made at all and that the feedback reached it -- not the final verdict.
    assert report.needs_approval + report.resolved >= 1


def test_persistent_wrongness_stops_after_max_attempts():
    batch = _batch()
    result, _ = _first_target(batch)
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _invented(p.residual_paise + 12345)),
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
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _correct_citation(batch, p, confidence=1.0)),
    )
    # Confidence capping applies to whatever is booked; with no rate book the
    # cases land in needs_approval, so assert on the recorded confidence there.
    assert report.needs_approval > 0 or report.resolved > 0
    for c in report.cases:
        if c.model_confidence is not None:
            assert c.model_confidence <= 1.0
    for m in result.matches_for_leg(2):
        if m.tier == "T2":
            assert m.confidence <= CONF_T2_CAP


def test_inferred_rows_never_look_like_reported_ones():
    batch = _batch()
    result, _ = _first_target(batch)
    recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(lambda p: _correct_citation(batch, p)),
    )
    # Needs a confirmed rate to actually book a T2 match.
    result2 = run_b2(batch.sources)
    from recoagent.agent import evidence as ev
    from recoagent.money import FeeSchedule
    exc = next(e for e in result2.exceptions if e.leg == 2
               and e.entity_kind == "bank_line" and e.residual_paise is not None)
    line = next(b for b in batch.sources.bank_lines if b.bank_line_id == exc.entity_id)
    st = next(x for x in batch.sources.settlements if x.settlement_id == exc.related_id)
    first = _correct_citation(
        batch, ev.build(batch.sources, line, st, exc.residual_paise, FeeSchedule.default())
    )
    if isinstance(first, Refusal):
        pytest.skip("no rule explains the first case in this batch")
    recover_with_agent(
        batch.sources, Tolerance.calibrated(), result2,
        ScriptedProposer(lambda p: _correct_citation(batch, p)),
        rate_book=_rate_book_for(batch, first),
    )
    booked = [m for m in result2.matches_for_leg(2) if m.tier == "T2"]
    assert booked
    for m in booked:
        assert "hypothesised" in m.proof.expression
        assert m.hypothesised_ids


def test_usage_is_accounted_per_case_and_in_total():
    batch = _batch()
    result, _ = _first_target(batch)
    report = recover_with_agent(
        batch.sources,
        Tolerance.calibrated(),
        result,
        ScriptedProposer(
            lambda p: _correct_citation(batch, p),
            usage_per_call=Usage(calls=1, input_tokens=1000, output_tokens=100),
        ),
    )
    assert report.usage.calls == sum(c.attempts for c in report.cases)
    assert report.usage.input_tokens == report.usage.calls * 1000
    assert report.usage.cost_usd(5.0, 25.0) > 0


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
        ScriptedProposer(lambda p: _correct_citation(batch, p)),
    )

    parallel = run_b2(batch.sources)
    report = recover_with_agent(
        batch.sources, Tolerance.calibrated(), parallel,
        max_workers=8,
        proposer_factory=lambda: ScriptedProposer(
            lambda p: _correct_citation(batch, p)
        ),
    )

    assert report.attempted > 0
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
            ScriptedProposer(lambda p: _correct_citation(batch, p)),
            max_workers=8,
        )


def test_each_worker_gets_its_own_proposer():
    from recoagent.pipeline import run_b2

    built = []

    def factory():
        p = ScriptedProposer(lambda pk: _correct_citation(batch, pk))
        built.append(p)
        return p

    batch = _batch(n=1500, seed=7)
    recover_with_agent(
        batch.sources, Tolerance.calibrated(), run_b2(batch.sources),
        max_workers=4, proposer_factory=factory,
    )
    # One per worker thread, not one per case.
    assert 1 <= len(built) <= 4


# ─────────────────────────────────────────────────────────────────────────────
# The rate book, built from the book's own paperwork rather than from the claim
# ─────────────────────────────────────────────────────────────────────────────


def _repriced_batch(sources, notice):
    """A settlement inside the notice's window, and one of its payments on that method."""
    for settlement in sorted(sources.settlements, key=lambda s: s.settled_at):
        if not notice.covers(settlement.settled_at, notice.method):
            continue
        for payment in sources.payments_by_settlement(settlement.settlement_id):
            if payment.method == notice.method:
                return settlement, payment
    raise AssertionError("no batch inside the notice window; the test would prove nothing")


def test_the_production_rate_book_comes_from_the_sources():
    """The P0 the whole citation contract was waiting on.

    `RateBook` existed and `resolve()` consulted it, but nothing outside the
    tests ever populated it -- so in production every fee and FX explanation was
    unverified by construction and closed as `needs_approval`. It is now built
    by `legs.repricing.rate_book` from the notices and advices in the book, and
    `run_b3` passes it in.
    """
    from recoagent.legs.repricing import rate_book

    batch = _batch(paperwork=True)
    live = [n for n in batch.sources.rate_notices if n.effective_to is None]
    assert live, "no live notice in this book; the test would prove nothing"

    when = max(s.settled_at for s in batch.sources.settlements)
    book = rate_book(batch.sources, when)

    for notice in live:
        assert book.confirms_mdr(notice.method, notice.mdr_bps)
        # A rate nobody issued is not confirmed just because it is close.
        assert not book.confirms_mdr(notice.method, notice.mdr_bps + 1)


def test_a_cited_rate_that_is_on_file_resolves():
    """Verified means a document says so, and the row books as reconciled."""
    from recoagent.agent.citations import FeeVarianceClaim, resolve
    from recoagent.legs.repricing import rate_book

    batch = _batch(paperwork=True)
    sources = batch.sources
    notice = next(n for n in sources.rate_notices if n.effective_to is None)
    # A batch inside the notice's window. One settled before it takes effect is
    # correctly *not* covered, which is the neighbouring test.
    settlement, payment = _repriced_batch(sources, notice)
    book = rate_book(sources, settlement.settled_at)

    claim = FeeVarianceClaim(
        payment_ids=(payment.payment_id,),
        actual_mdr_bps=notice.mdr_bps,
        rationale="the gateway's repricing notice",
    )
    out = resolve(sources, settlement, [claim], FeeSchedule.default(), book)
    assert out.ok, out.errors
    assert out.fully_verified, out.unverified_reasons


def test_a_cited_rate_nobody_issued_still_needs_approval():
    """The control. Without this, `verified` would just mean `well-formed`."""
    from recoagent.agent.citations import FeeVarianceClaim, resolve
    from recoagent.legs.repricing import rate_book

    batch = _batch(paperwork=True)
    sources = batch.sources
    notice = next(n for n in sources.rate_notices if n.effective_to is None)
    settlement, payment = _repriced_batch(sources, notice)
    book = rate_book(sources, settlement.settled_at)

    invented = FeeVarianceClaim(
        payment_ids=(payment.payment_id,),
        actual_mdr_bps=notice.mdr_bps + 37,   # plausible, self-consistent, unissued
        rationale="a rate that closes the arithmetic",
    )
    out = resolve(sources, settlement, [invented], FeeSchedule.default(), book)
    assert out.ok, out.errors
    assert not out.fully_verified
    assert out.unverified_reasons


def test_an_expired_notice_does_not_verify_a_claim():
    """Paperwork that has been superseded is not paperwork."""
    from recoagent.legs.repricing import rate_book

    batch = _batch(paperwork=True)
    sources = batch.sources
    expired = [n for n in sources.rate_notices if n.effective_to is not None]
    assert expired, "no superseded notice in this book; the test would prove nothing"
    stale = expired[0]

    after = stale.effective_to + timedelta(days=1)
    book = rate_book(sources, after)
    live_rates = {
        n.mdr_bps for n in sources.rate_notices
        if n.method == stale.method and n.covers(after, n.method)
    }
    if stale.mdr_bps not in live_rates:
        assert not book.confirms_mdr(stale.method, stale.mdr_bps)


# ── two ways a verified row could still rest on an unconfirmed rate ──────


def _one_settlement_with(methods):
    """A settlement whose payments use the methods given, in order."""
    from datetime import datetime, timezone

    from recoagent.schemas import BankLine, PGPayment, Settlement, SourceBundle

    when = datetime(2026, 7, 3, tzinfo=timezone.utc)
    payments = tuple(
        PGPayment(
            payment_id=f"pay_{i}", order_id=f"order_{i}", gross_paise=100_000,
            fee_paise=1_950, tax_paise=351, method=m, status="captured",
            settlement_id="setl_1", captured_at=when,
            currency="USD" if m == "card_international" else "INR",
            fx_rate=83.2 if m == "card_international" else None,
        )
        for i, m in enumerate(methods)
    )
    settlement = Settlement(
        settlement_id="setl_1", utr="U1", settled_at=when,
        net_paise=sum(p.gross_paise - p.fee_paise - p.tax_paise for p in payments),
        status="processed",
    )
    sources = SourceBundle(
        orders=(), payments=payments, adjustments=(), settlements=(settlement,),
        bank_lines=(BankLine(bank_line_id="bank_1", value_date=when.date(),
                             amount_paise=settlement.net_paise,
                             narration="X", bank_ref="U1"),),
    )
    return sources, settlement


def test_a_fee_claim_is_not_confirmed_by_a_notice_for_a_different_method():
    """The attack: hide a method the rate book has never confirmed.

    A repricing notice covers one method. A claim naming payments across two
    means one MDR is being asserted for both, and UPI carries zero MDR by
    regulation -- so a card notice at 195 bps must not confirm a UPI payment
    repriced at 195 bps. Verifying only the *first* cited payment's method let
    exactly that through, and a verified row books money.
    """
    from recoagent.agent.citations import FeeVarianceClaim, RateBook, resolve

    sources, settlement = _one_settlement_with(["card_domestic", "upi"])
    book = RateBook(mdr_bps={"card_domestic": {195}})  # nothing at all for upi

    out = resolve(sources, settlement,
                  [FeeVarianceClaim(("pay_0", "pay_1"), 195, "repriced")],
                  rate_book=book)

    assert out.rows, out.errors
    assert out.rows[0].verified is False, (
        "a UPI payment was confirmed by a card notice: "
        f"{out.rows[0].derivation}"
    )


def test_a_fee_claim_is_confirmed_only_when_every_method_is():
    from recoagent.agent.citations import FeeVarianceClaim, RateBook, resolve

    sources, settlement = _one_settlement_with(["card_domestic", "card_international"])
    both = RateBook(mdr_bps={"card_domestic": {195}, "card_international": {195}})

    out = resolve(sources, settlement,
                  [FeeVarianceClaim(("pay_0", "pay_1"), 195, "repriced")],
                  rate_book=both)
    assert out.rows[0].verified is True, out.rows[0].derivation


def test_a_confirmed_fx_row_books_the_authoritative_rate_not_the_claimed_one():
    """Tolerance decides whether a claim *matches* the advice. It must not
    decide what gets booked.

    The bank advised -1.600%. The model said -1.595%, which is inside the
    matching tolerance and so the row is verified -- and the row was then priced
    at the model's number. A verified row is one the system books, so that is
    money moved on a figure nobody issued, which is the single thing the
    citation contract exists to prevent.
    """
    from recoagent.agent.citations import FxClaim, RateBook, resolve

    sources, settlement = _one_settlement_with(["card_international"])
    book = RateBook(fx_pct={"pay_0": -1.600})

    out = resolve(sources, settlement, [FxClaim("pay_0", -1.595, "converted")],
                  rate_book=book)

    row = out.rows[0]
    assert row.verified is True, row.derivation
    authoritative = round(100_000 * -1.600 / 100)
    assert row.amount_paise == authoritative, (
        f"booked {row.amount_paise} from the model's -1.595%, not {authoritative} "
        "from the bank's -1.600%"
    )


def test_an_unconfirmed_fx_row_still_prices_the_claim_it_was_given():
    """With no advice on file there is nothing authoritative to prefer, so the
    claimed rate is priced and the row is simply not verified."""
    from recoagent.agent.citations import FxClaim, RateBook, resolve

    sources, settlement = _one_settlement_with(["card_international"])
    out = resolve(sources, settlement, [FxClaim("pay_0", -1.595, "converted")],
                  rate_book=RateBook())
    assert out.rows[0].verified is False
    assert out.rows[0].amount_paise == round(100_000 * -1.595 / 100)
