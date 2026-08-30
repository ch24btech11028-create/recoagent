"""The live console: request validation, payload shape, and the honesty rules.

The server is exercised over a real socket on an ephemeral port rather than by
calling the handlers directly, because the things most likely to break -- a bad
form value, a missing key, a route that does not exist -- only exist at that
boundary. Nothing here reaches a model: the Q&A paths run against a scripted
client, so the suite never needs an endpoint and never costs a call.
"""

from __future__ import annotations

import json
import re
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
        # `xid` is the exception's own id, so a row can be deep-linked and its
        # case file fetched; `related` is the counterparty it was adjudicated
        # against. Both are the system's own output -- neither can be known
        # without running the matchers -- and both are needed by a screen that
        # opens an item rather than just listing it.
        "xid", "id", "leg", "kind", "related", "residual_paise", "amount",
        "direction", "severity", "severity_hint", "suspected", "reason",
        "stopped_at",
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
    # A refusal is a result here, not an exception: several of these tests are
    # about *which* 4xx a bad request earns.
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


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


def _text(server, path):
    with urllib.request.urlopen(server + path, timeout=30) as r:
        return r.status, r.headers.get("Content-Type"), r.read().decode()


def test_the_front_end_is_served_as_html_css_and_javascript(server):
    """The UI is web files, served as web files -- not strings built in Python."""
    status, content_type, body = _text(server, "/")
    assert status == 200 and content_type.startswith("text/html")
    assert "<title>RecoAgent</title>" in body
    # The shell and the container the router fills. Every screen renders into
    # this one element, so if it is missing the console is a blank page.
    assert 'id="view"' in body and 'class="app"' in body
    assert '<link rel="stylesheet" href="/app.css">' in body
    assert '<script src="/app.js" defer></script>' in body

    for path, expected in (("/base.css", "text/css"), ("/app.css", "text/css"),
                           ("/app.js", "text/javascript")):
        status, content_type, text = _text(server, path)
        assert status == 200 and content_type.startswith(expected), path
        assert text.strip(), path

    # Self-contained apart from the font stylesheet: no script or data endpoint
    # outside this origin.
    assert "cdn" not in body.lower()


def test_every_nav_link_has_a_view_behind_it(server):
    """A dead link in the sidebar is the one bug a screenshot will not show.

    The nav lives in the document and the views live in the script, so this is
    the seam where the two can drift apart without either file looking wrong.
    """
    _, _, html = _text(server, "/")
    _, _, js = _text(server, "/app.js")
    routes = set(re.findall(r'href="#/([a-z]+)"', html))
    assert len(routes) >= 8, routes
    for route in routes:
        assert f"VIEWS.{route} = " in js, f"{route} is in the nav with no view behind it"


def test_an_asset_that_is_not_on_the_list_is_not_served(server):
    """An allowlist, not a path join -- so there is no traversal to reason about."""
    for path in ("/../ui.py", "/views.py", "/web/app.js", "/nope.css"):
        try:
            with urllib.request.urlopen(server + path, timeout=30) as r:
                assert r.status == 404, path
        except urllib.error.HTTPError as exc:
            assert exc.code == 404, path


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


def test_qa_without_a_key_is_disabled_not_broken(server, monkeypatch, tmp_path):
    """The queue must keep working when no model is configured.

    The absence of a key is *constructed* rather than hoped for. This used to
    skip when the environment happened to have one, which meant it protected
    nothing on the machine of anyone who had configured a model -- and the
    behaviour it guards is exactly what a reader cloning the repository sees
    first. Unset the variable and move away from the `.env`, and the condition
    is the same for everyone.
    """
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # so `require_key` finds no .env either
    status, d = _get(server + "/api/model")
    assert d["ready"] is False, d
    assert status == 200
    # Two legitimate reasons the model is not ready -- no key, or no SDK -- and
    # a fresh clone hits the second one first. Assert the message is actionable
    # rather than that it names one specific cause; an earlier version pinned
    # NVIDIA_API_KEY and so failed for everyone who had simply not run pip yet.
    def actionable(text: str) -> bool:
        return "NVIDIA_API_KEY" in text or "pip install" in text

    assert actionable(d["problem"]), d["problem"]
    for route in ("/api/ask", "/api/bank"):
        code, body = _post(server + route, {"question": "anything"})
        assert code == 503
        assert actionable(body["error"]), body["error"]
    # ...and the reconciliation still runs.
    assert _post(server + "/api/run", {"n": 300})[0] == 200


def test_an_empty_question_is_refused(server):
    assert _post(server + "/api/ask", {"question": "   "})[0] == 400


def test_an_enormous_question_is_refused(server):
    assert _post(server + "/api/ask", {"question": "x" * 5000})[0] == 400


# ── the screens behind the queue ─────────────────────────────────────────


def test_a_case_file_opens_from_a_queue_row(server):
    """Every row the queue offers must lead somewhere. A dead row is a dead end."""
    _, run = _post(server + "/api/run", {"n": 600, "seed": 7, "profile": "dev", "rung": "B2"})
    assert run["queue"], "no exceptions to open on the dev mix"
    for row in run["queue"][:12]:
        status, d = _get(server + f"/api/exception?n=600&seed=7&profile=dev&rung=B2&id={row['xid']}")
        assert status == 200, row
        assert d["item"]["id"] == row["id"]
        assert d["case"]["shape"] != "bare", f"no case file for {row['id']} ({row['kind']})"


def test_an_unknown_case_file_is_a_404(server):
    assert _get(server + "/api/exception?n=300&id=not_a_real_exception")[0] == 404


def test_the_match_log_pages_and_filters(server):
    _post(server + "/api/run", {"n": 600, "seed": 7, "profile": "dev", "rung": "B2"})
    base = "/api/matches?n=600&seed=7&profile=dev&rung=B2"
    status, first = _get(server + base)
    assert status == 200 and first["rows"] and first["page"] == 1
    # Every accepted match carries the arithmetic that accepted it.
    assert all(r["proof"] is None or r["proof"]["closes"] for r in first["rows"])

    _, leg2 = _get(server + base + "&leg=2")
    assert leg2["rows"] and all(r["leg"] == 2 for r in leg2["rows"])
    assert leg2["total"] < first["total"]

    _, page2 = _get(server + base + "&page=2")
    assert page2["page"] == 2
    assert {r["match_id"] for r in page2["rows"]}.isdisjoint({r["match_id"] for r in first["rows"]})


def test_every_source_ledger_is_browsable(server):
    from recoagent.views import SOURCE_KINDS

    _post(server + "/api/run", {"n": 600, "seed": 7, "profile": "dev", "rung": "B2"})
    for kind in SOURCE_KINDS:
        status, d = _get(server + f"/api/source?n=600&seed=7&profile=dev&rung=B2&kind={kind}")
        assert status == 200, kind
        assert d["columns"] and d["blurb"]
        # Whatever the columns say they will show, the rows have to carry.
        for row in d["rows"][:5]:
            assert {c["key"] for c in d["columns"]} <= set(row), kind


def test_an_unknown_ledger_is_refused(server):
    status, d = _get(server + "/api/source?n=300&kind=../../etc/passwd")
    assert status == 400 and "unknown source" in d["error"]


def test_published_results_are_served_only_from_the_results_directory(server):
    """The name has to be one the index already offered."""
    status, index = _get(server + "/api/results")
    assert status == 200
    names = {f["name"] for f in index["files"]}
    if names:
        assert _get(server + "/api/results?file=" + sorted(names)[0])[0] == 200
    for escape in ("../STATE.md", "../.env", "/etc/passwd", "..%2f.env"):
        assert _get(server + "/api/results?file=" + escape)[0] == 404, escape


def test_a_case_file_opens_by_either_identifier():
    """A queue row carries two ids and the endpoint takes one of them.

    `id` on the row is the entity (`bank_0030`); `xid` is the exception
    (`x2_bank_0030`). The console sends the exception id under a parameter
    named `id`, so anyone reading the queue payload and asking for the field it
    calls `id` got a 404 that read as "this item no longer exists".
    """
    from recoagent import views
    from recoagent.generator import DefectMix, GeneratorConfig, generate
    from recoagent.pipeline import run_b2

    batch = generate(GeneratorConfig(n_orders=500, seed=7, mix=DefectMix.dev()))
    result = run_b2(batch.sources)
    row = views.queue(result)[0]

    assert row["id"] != row["xid"], "the two identifiers must stay distinguishable"
    by_exception = views.exception_case(batch.sources, result, row["xid"])
    by_entity = views.exception_case(batch.sources, result, row["id"])

    assert by_exception is not None
    assert by_entity is not None
    assert by_entity == by_exception

    # An id belonging to neither is still a miss, or the fallback would have
    # turned the endpoint into one that always finds something.
    assert views.exception_case(batch.sources, result, "no_such_row_anywhere") is None
