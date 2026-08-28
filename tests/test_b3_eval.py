"""The B3 measurement harness.

Every path here runs against a scripted proposer, so the suite exercises the
whole command -- including the parts that only matter when a model actually
answers -- without an API key and without a call.
"""

from __future__ import annotations

import pytest

from recoagent.agent.contracts import Hypothesis, ProposerError, Refusal, Usage
from recoagent.agent.citations import CitedAdjustment
from recoagent.eval.b3 import build, render, run, truth_ids_for


class _Scripted:
    """A proposer that always answers the same way."""

    def __init__(self, answer) -> None:
        self._answer = answer
        self.calls = 0

    def propose(self, packet, *_a, **_kw):
        self.calls += 1
        answer = self._answer() if callable(self._answer) else self._answer
        return answer, Usage()


# ── the population B3 is actually given ──────────────────────────────────


@pytest.mark.parametrize("profile", ["dev", "holdout"])
def test_with_the_paperwork_there_is_nothing_left_to_attempt(profile):
    """The measured consequence of building the repricing tier.

    On the published books every residual-bearing leg-2 item is now closed
    before the agent tier is reached, so it is never invoked at all. That is the
    result, not a setup failure -- and it is why the tier is measured a second
    time with the notices withheld.
    """
    called = _Scripted(Refusal(reason="should never be asked"))
    out = run(profile, n_orders=2000, paperwork=True,
              proposer_factory=lambda: called)
    assert out.open_before == 0
    assert out.report.attempted == 0
    assert called.calls == 0
    assert not out.recall_moved


def test_withholding_the_paperwork_gives_the_tier_something_to_do():
    """The control. If this were also empty, the harness would prove nothing."""
    batch = build("dev", 2000, paperwork=False)
    assert batch.sources.rate_notices == ()
    assert batch.sources.fx_advices == ()
    out = run("dev", n_orders=2000, paperwork=False,
              proposer_factory=lambda: _Scripted(Refusal(reason="no")))
    assert out.open_before > 0
    assert out.report.attempted == out.open_before


# ── the invariant, whatever the model says ───────────────────────────────


def test_a_refusing_model_changes_nothing_and_breaks_nothing():
    out = run("dev", n_orders=2000, paperwork=False,
              proposer_factory=lambda: _Scripted(Refusal(reason="cannot tell")))
    assert out.report.resolved == 0
    assert out.report.refused == out.report.attempted
    assert out.after.overall_false_match_rate == 0.0
    assert not out.recall_moved


def test_a_model_citing_nothing_real_cannot_move_the_false_match_rate():
    """The hostile case: confident, well-formed, and pointing at fiction."""
    invented = Hypothesis(
        citations=(CitedAdjustment("adj_does_not_exist", "there was an adjustment"),),
        reason="invented",
        confidence=0.99,
    )
    out = run("dev", n_orders=2000, paperwork=False,
              proposer_factory=lambda: _Scripted(invented))
    assert out.report.resolved == 0
    assert out.report.unverifiable + out.report.rejected == out.report.attempted
    assert out.after.overall_false_match_rate == 0.0
    assert out.after.mishandled_total == 0


def test_a_failing_transport_is_a_recorded_outcome_not_a_crash():
    """A dead endpoint is a queue item with a reason on it, not a stack trace.

    Note what is *not* tested here: a `proposer_factory` that raises. That is a
    startup failure -- no client could be built at all -- and it should reach
    the operator immediately rather than being recorded once per case.
    """
    dead = ProposerError(kind="transport", detail="connection reset")
    out = run("dev", n_orders=2000, paperwork=False,
              proposer_factory=lambda: _Scripted(dead))
    assert out.report.attempted > 0
    assert out.report.failed == out.report.attempted
    assert out.report.resolved == 0
    assert out.after.overall_false_match_rate == 0.0
    assert not out.recall_moved


# ── the report ───────────────────────────────────────────────────────────


def test_the_render_leads_with_resolved_and_survives_an_empty_run():
    out = run("dev", n_orders=2000, paperwork=True,
              proposer_factory=lambda: _Scripted(Refusal(reason="no")))
    text = render(out)
    assert "RESOLVED (source-backed)" in text
    assert "<- lead metric" in text
    assert "no residual-bearing leg-2 item survived the cheaper tiers" in text
    assert "nothing was resolved, so there is nothing to check" in text


def test_the_render_names_the_book_it_measured():
    with_paper = render(run("dev", n_orders=2000, paperwork=True,
                            proposer_factory=lambda: _Scripted(Refusal(reason="x"))))
    without = render(run("dev", n_orders=2000, paperwork=False,
                         proposer_factory=lambda: _Scripted(Refusal(reason="x"))))
    assert "with the merchant's paperwork" in with_paper
    assert "paperwork withheld" in without


# ── provenance ───────────────────────────────────────────────────────────


def test_provenance_maps_a_bank_line_to_evidence_from_its_own_batch():
    batch = build("dev", 2000, paperwork=False)
    mapping = truth_ids_for(batch)
    assert mapping, "no defect classes reached the provenance map"
    for line_id, expected in mapping.items():
        sid = batch.truth.leg2[line_id]
        members = {p.payment_id for p in batch.sources.payments_by_settlement(sid)}
        linked = {
            a.adjustment_id for a in batch.sources.adjustments
            if a.settlement_id == sid
        }
        assert expected <= (members | linked), line_id


def test_provenance_is_unmeasurable_when_nothing_resolves():
    out = run("dev", n_orders=2000, paperwork=False,
              proposer_factory=lambda: _Scripted(Refusal(reason="no")))
    assert out.provenance == (0, 0)


def test_a_model_string_picks_its_own_endpoint():
    """"Swapping the model is one string" has to be true on the tier as well.

    It was true for Q&A, which goes through `client_for`, and false here: the
    proposer built its own client with the endpoint and key variable defaulted
    to NIM, so `--model gemini/...` posted a Gemini model id to NVIDIA and
    demanded an NVIDIA key. Both paths now read the same provider table.
    """
    from recoagent.agent.openai_proposer import endpoint_for
    from recoagent.llm import PROVIDERS

    for provider, (url, env, keep_prefix) in PROVIDERS.items():
        if url is None:
            continue  # not an OpenAI-protocol host
        spec = f"{provider}/some-model"
        model, base_url, key_env = endpoint_for(spec, None, None)
        assert base_url == url
        assert key_env == env
        assert model == (spec if keep_prefix else "some-model")

    # A host the table has never heard of is the reason `--base-url` exists,
    # and passing one must not be overridden by a provider-shaped model name.
    assert endpoint_for("nvidia/x", "http://localhost:8000/v1", "MY_KEY") == (
        "nvidia/x", "http://localhost:8000/v1", "MY_KEY"
    )
