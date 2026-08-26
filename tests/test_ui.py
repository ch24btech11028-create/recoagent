"""The live console: request validation, payload shape, and the honesty rules.

The server is exercised over a real socket on an ephemeral port rather than by
calling the handlers directly, because the things most likely to break -- a bad
form value, a missing key, a route that does not exist -- only exist at that
boundary. Nothing here reaches a model: the Q&A paths run against a scripted
client, so the suite never needs an endpoint and never costs a call.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.llm import ScriptedChat
from recoagent.pipeline import run_b2
from recoagent.qa import bank as qa_bank
from recoagent.ui import Run, RunCache, RunKey, ask_payload, bank_payload, run_payload, serve


@pytest.fixture(scope="module")
def run():
    key = RunKey(n=1200, seed=7, profile="dev", rung="B2")
    batch = generate(GeneratorConfig(n_orders=key.n, seed=key.seed, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    from recoagent.eval.scorer import score
    return Run(key, batch, result, score(batch, result), 0.0)


class _Model:
    """Stands in for `ui.Model`, handing back a scripted client."""

    def __init__(self, payload) -> None:
        self.spec = "scripted"
        self._payload = payload

    def chat(self):
        return ScriptedChat(lambda s, u: json.dumps(
            self._payload(u) if callable(self._payload) else self._payload
        ))


# ── request validation ───────────────────────────────────────────────────


@pytest.mark.parametrize("payload, message", [
    ({"n": 99}, "between"),
    ({"n": 50_001}, "between"),
    ({"profile": "nope"}, "unknown profile"),
    ({"rung": "B9"}, "unknown rung"),
    ({"n": "abc"}, "whole numbers"),
    ({"seed": "x"}, "whole numbers"),
])
def test_bad_input_is_refused_with_a_reason(payload, message):
    with pytest.raises(ValueError, match=message):
        RunKey.parse(payload)


def test_defaults_are_the_published_ones():
    assert RunKey.parse({}) == RunKey(n=2000, seed=7, profile="dev", rung="B2")
    assert RunKey.parse({"profile": "holdout"}).seed == 21


def test_an_unbounded_n_cannot_be_requested():
    """A form field that generates 10m orders is a denial of service on the desk."""
    with pytest.raises(ValueError):
        RunKey.parse({"n": 10_000_000})


# ── payload shape and the ground-truth fence ─────────────────────────────


def test_run_payload_reports_the_measured_numbers(run):
    d = run_payload(run)
    assert d["headline"]["false_match_rate"] == 0.0
    assert 0 < d["headline"]["auto_match_rate"] <= 1
    assert d["verdict"]["mishandled_total"] == 0
    assert len(d["queue"]) == len(run.result.exceptions)
    assert {"leg", "population", "recall"} <= set(d["legs"][0])


def test_the_queue_never_carries_the_answer_key(run):
    """The operator's table is built from sources and result only.

    `suspected_class` is the system's own guess and belongs on the row. The
    injected truth does not, and a queue that quietly showed it would be a demo
    rather than a product.
    """
    d = run_payload(run)
    # An allowlist, not a banned-word scan: a new field that leaked the answer
    # key would have to be added here deliberately, which is the point.
    allowed = {
        "id", "leg", "kind", "residual_paise", "amount", "direction",
        "severity", "severity_hint", "suspected", "reason", "stopped_at",
    }
    for row in d["queue"]:
        assert set(row) == allowed, f"unexpected field on a queue row: {set(row) - allowed}"

    # `suspected` is the system's own guess. That it agrees with the injected
    # label wherever it fires is the classifier working, not an oracle: it is
    # set in legs/leg1.py and legs/leg2.py, which tests/test_independence.py
    # proves cannot import the generator or name a ground-truth type.
    #
    # What is checked here is the shape an oracle would not have. A queue fed by
    # the answer key would classify everything; this one leaves the cases with no
    # structural signal open, and it raises rows for entities that were never an
    # injection site at all.
    classified = [r for r in d["queue"] if r["suspected"] != "not classified"]
    unclassified = [r for r in d["queue"] if r["suspected"] == "not classified"]
    assert classified and unclassified, (
        "the queue either classified nothing or classified everything; "
        "an oracle would do the latter"
    )
    injected_ids = {x.entity_id for x in run.batch.truth.defects}
    queue_ids = {r["id"] for r in d["queue"]}
    assert queue_ids - injected_ids, (
        "every open item sat on an injection site; the queue is following the "
        "answer key rather than the evidence"
    )

    # The verification panel is where truth is allowed, and it is separate.
    assert "accounting" in d and d["accounting"]


def test_recovered_rows_all_carry_a_proof(run):
    for row in run_payload(run)["recovered"]:
        assert row["residual"] is not None
        assert row["tier"] in ("T0", "T1", "T2")


# ── the cache ────────────────────────────────────────────────────────────


def test_the_cache_returns_the_same_run():
    cache = RunCache()
    key = RunKey(n=300, seed=7, profile="dev", rung="B2")
    assert cache.get(key) is cache.get(key)


def test_the_cache_is_bounded():
    cache = RunCache()
    for seed in range(10):
        cache.get(RunKey(n=200, seed=seed, profile="dev", rung="B2"))
    assert len(cache._runs) <= 6


# ── the honesty rules on Q&A ─────────────────────────────────────────────


def test_a_live_question_comes_back_ungraded(run):
    """It has no ground truth, so it must not be reported as right or wrong."""
    out = ask_payload(_Model({"answer": 42, "confidence": 0.9, "basis": "the factsheet"}),
                      run, "how many open items are there?")
    assert out["graded"] is False
    assert out["answer"] == 42
    assert "correct" not in out


def test_a_live_answer_ships_with_the_factsheet_that_produced_it(run):
    """The operator's check on the model: everything it was allowed to see."""
    out = ask_payload(_Model({"answer": 1, "basis": "x"}), run, "what is open on bank_0034?")
    assert "portfolio" in out["factsheet"]
    assert isinstance(out["factsheet"], dict)


def test_an_ungraded_answer_cannot_move_a_measured_rate(run):
    """The attack this guards: asking easy questions in the box to lift the score."""
    from recoagent.qa.agent import QAReport, ask

    report = QAReport()
    graded = [q for q in qa_bank.build(run.batch, run.result) if not q.expects_decline][:3]
    chat = ScriptedChat(lambda s, u: json.dumps({"answer": -1, "basis": "wrong on purpose"}))
    report.answers = [ask(chat, run.batch, run.result, q) for q in graded]
    baseline = report.wrong_answer_rate
    assert baseline == 1.0

    live = qa_bank.Question("live", "freeform", "anything", None, "freeform", graded=False)
    report.answers.append(ask(chat, run.batch, run.result, live))
    assert report.wrong_answer_rate == baseline
    assert report.total == len(graded)


def test_the_scored_bank_still_grades_everything(run):
    out = bank_payload(_Model({"answer": 0, "basis": "always zero"}), run, limit=4, workers=2)
    assert out["total"] == 4
    assert all(a["expected"] is not None or a["expects_decline"] for a in out["answers"])
    assert 0.0 <= out["wrong_answer_rate"] <= 1.0


def test_declining_is_scored_as_a_cost_not_a_failure(run):
    out = bank_payload(_Model({"cannot_answer": "not in the factsheet"}), run, limit=6, workers=2)
    assert out["declined"] == 6
    assert out["attempted"] == 0
    # No answers given, so there is nothing to be wrong about.
    assert out["wrong_answer_rate"] == 0.0
    assert out["hallucinated"] == 0


# ── the server over a real socket ────────────────────────────────────────


@pytest.fixture(scope="module")
def server():
    httpd = serve("127.0.0.1", 0, "nvidia/nemotron-3-ultra-550b-a55b")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read())


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_page_serves(server):
    with urllib.request.urlopen(server + "/", timeout=30) as r:
        body = r.read().decode()
    assert r.status == 200
    assert "RecoAgent Console" in body
    # Self-contained apart from the font stylesheet: no script or data endpoint
    # outside this origin.
    assert "cdn" not in body.lower()


def test_an_unknown_route_is_a_clean_404(server):
    assert _post(server + "/api/nope", {})[0] == 404


def test_a_run_over_http_matches_the_published_shape(server):
    status, d = _post(server + "/api/run", {"n": 500, "seed": 7, "profile": "dev", "rung": "B2"})
    assert status == 200
    assert d["headline"]["false_match_rate"] == 0.0
    assert d["key"] == {"n": 500, "seed": 7, "profile": "dev", "rung": "B2"}


def test_bad_form_values_come_back_as_400(server):
    status, d = _post(server + "/api/run", {"n": 5})
    assert status == 400 and "between" in d["error"]


def test_qa_without_a_key_is_disabled_not_broken(server, monkeypatch):
    """The queue must keep working when no model is configured."""
    status, d = _get(server + "/api/model")
    if d["ready"]:
        pytest.skip("a live API key is configured in this environment")
    assert status == 200
    assert "NVIDIA_API_KEY" in d["problem"]
    for route in ("/api/ask", "/api/bank"):
        code, body = _post(server + route, {"question": "anything"})
        assert code == 503
        assert "NVIDIA_API_KEY" in body["error"]
    # ...and the reconciliation still runs.
    assert _post(server + "/api/run", {"n": 300})[0] == 200


def test_an_empty_question_is_refused(server):
    assert _post(server + "/api/ask", {"question": "   "})[0] == 400


def test_an_enormous_question_is_refused(server):
    assert _post(server + "/api/ask", {"question": "x" * 5000})[0] == 400
