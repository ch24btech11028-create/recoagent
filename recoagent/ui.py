"""The live operator console: run a reconciliation, work the queue, ask about it.

`recoagent.report` writes a static page — the 9am snapshot, mailable, no server.
This is the same product with its hands free: change the run and watch the
numbers move, open an exception and see every tier that touched it, and ask the
settlement agent a question in English.

Three things are deliberate.

**The queue still cannot see the answer key.** It is built from `result` and
`sources` only, exactly like the static export and exactly like the matchers.
The verification panel is the one place ground truth appears, and it is fenced
off and labelled, because "what do I work on" and "should I believe this" are
different questions and mixing them is how a demo starts lying.

**A question typed into the box is answered but never scored.** There is no
ground truth for it, so it is returned marked *ungraded* with the factsheet the
model saw attached, and it cannot move any measured rate. The measured number
lives behind the second button, which runs the derived question bank — where
every answer does have a ground truth computed by code.

**The model only ever sees a factsheet.** Same retrieval path as the scored
harness: `qa.agent.factsheet` builds it from the run, the model answers from it
alone, and an operator can expand it and check the answer by hand.

Stdlib only, bound to loopback:

    python -m recoagent.ui
    python -m recoagent.ui --port 8080 --model anthropic/claude-opus-5
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .console_page import PAGE
from .eval.scorer import Scorecard, score
from .generator import DefectMix, GeneratorConfig, generate
from .money import format_inr
from .pipeline import run_b0, run_b2
from .qa import agent as qa_agent
from .qa import bank as qa_bank
from .report import RULE_LABEL, _severity
from .schemas import LabelledBatch, ReconResult
from .webstyle import CSS

MIXES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout), "clean": (7, DefectMix.clean)}

#: Runs are pure functions of these four numbers, so they cache perfectly. The
#: cap keeps a long demo session from holding every batch it ever generated.
CACHE_LIMIT = 6


@dataclass(frozen=True)
class RunKey:
    n: int
    seed: int
    profile: str
    rung: str

    @classmethod
    def parse(cls, payload: dict) -> "RunKey":
        profile = str(payload.get("profile", "dev"))
        if profile not in MIXES:
            raise ValueError(f"unknown profile {profile!r}")
        rung = str(payload.get("rung", "B2"))
        if rung not in ("B0", "B2"):
            raise ValueError(f"unknown rung {rung!r}")
        try:
            n = int(payload.get("n", 2000))
            seed = int(payload.get("seed", MIXES[profile][0]))
        except (TypeError, ValueError):
            raise ValueError("n and seed must be whole numbers")
        # An unbounded n from a form field is a denial of service against the
        # person running the console. 50k takes ~40s to generate; past that the
        # browser is waiting on nothing useful for a demo.
        if not 100 <= n <= 50_000:
            raise ValueError("n must be between 100 and 50,000")
        return cls(n=n, seed=seed, profile=profile, rung=rung)


@dataclass
class Run:
    key: RunKey
    batch: LabelledBatch
    result: ReconResult
    card: Scorecard
    seconds: float


class RunCache:
    """Memoised runs, so asking a question does not re-reconcile the book."""

    def __init__(self, limit: int = CACHE_LIMIT) -> None:
        self._runs: dict[RunKey, Run] = {}
        self._order: list[RunKey] = []
        self._lock = threading.Lock()

    def get(self, key: RunKey) -> Run:
        with self._lock:
            hit = self._runs.get(key)
        if hit is not None:
            return hit

        started = time.perf_counter()
        _, mix_factory = MIXES[key.profile]
        batch = generate(GeneratorConfig(n_orders=key.n, seed=key.seed, mix=mix_factory()))
        result = run_b0(batch.sources) if key.rung == "B0" else run_b2(batch.sources)
        card = score(batch, result)
        run = Run(key, batch, result, card, time.perf_counter() - started)

        with self._lock:
            self._runs[key] = run
            self._order.append(key)
            while len(self._order) > CACHE_LIMIT:
                self._runs.pop(self._order.pop(0), None)
        return run


# ── shaping a run for the browser ────────────────────────────────────────────

def _queue_payload(result: ReconResult) -> list[dict]:
    ordered = sorted(result.exceptions, key=lambda e: (-abs(e.residual_paise or 0), e.entity_id))
    out = []
    for exc in ordered:
        level, hint = _severity(exc.residual_paise)
        out.append({
            "id": exc.entity_id,
            "leg": exc.leg,
            "kind": exc.entity_kind,
            "residual_paise": exc.residual_paise,
            "amount": format_inr(exc.residual_paise) if exc.residual_paise is not None else "—",
            "direction": "" if not exc.residual_paise else ("short" if exc.residual_paise < 0 else "over"),
            "severity": level,
            "severity_hint": hint,
            "suspected": exc.suspected_class.value if exc.suspected_class else "not classified",
            "reason": exc.reason,
            "stopped_at": exc.escalated_from_tier or "T0",
        })
    return out


def _recovered_payload(result: ReconResult, limit: int = 40) -> list[dict]:
    leg2 = [m for m in result.matches_for_leg(2) if m.proof is not None]
    leg2.sort(key=lambda m: (m.tier, m.match_id))
    picked = [m for m in leg2 if m.tier != "T0"][:limit] or leg2[:limit]
    return [{
        "left": m.left_ids[0],
        "right": m.right_ids[0],
        "tier": m.tier,
        "rule": RULE_LABEL.get(m.rule_id, m.rule_id),
        "confidence": round(m.confidence, 2),
        "residual": format_inr(m.proof.residual_paise),
    } for m in picked]


def run_payload(run: Run) -> dict:
    card, result = run.card, run.result
    unexplained = sum(abs(e.residual_paise or 0) for e in result.exceptions)
    return {
        "key": {"n": run.key.n, "seed": run.key.seed, "profile": run.key.profile, "rung": run.key.rung},
        "seconds": round(run.seconds, 2),
        "counts": dict(run.batch.sources.counts),
        "headline": {
            "false_match_rate": card.overall_false_match_rate,
            "auto_match_rate": card.overall_auto_match_rate,
            "value_share": card.value.share if card.value else None,
            "value_matched": format_inr(card.value.matched_credit) if card.value else "—",
            "unexplained": format_inr(unexplained),
            "open_items": len(result.exceptions),
        },
        "legs": [{
            "leg": leg,
            "population": s.population,
            "attempted": s.attempted,
            "true_matches": s.true_matches,
            "false_matches": s.false_matches,
            "recall": s.recall,
            "exceptions": s.exceptions,
        } for leg, s in sorted(card.legs.items())],
        "accounting": [{
            "defect": a.defect.value,
            "injected": a.injected,
            "flagged": a.flagged,
            "resolved": a.resolved,
            "mishandled": a.mishandled,
        } for a in card.accounting],
        "verdict": {
            "fully_reconciles": card.fully_reconciles,
            "mishandled_total": card.mishandled_total,
            "unattributed": card.unattributed_exceptions,
            "injected_total": sum(a.injected for a in card.accounting),
        },
        "queue": _queue_payload(result),
        "recovered": _recovered_payload(result),
    }


# ── the Q&A endpoints ────────────────────────────────────────────────────────

class Model:
    """Lazily built chat clients, one per thread, plus a readable reason if none."""

    def __init__(self, spec: str) -> None:
        self.spec = spec
        self._local = threading.local()
        self._probe_lock = threading.Lock()
        self._problem: str | None = None
        self._probed = False

    def _build(self):
        from .llm import client_for
        return client_for(self.spec)

    def status(self) -> dict:
        with self._probe_lock:
            if not self._probed:
                try:
                    self._build()
                    self._problem = None
                except Exception as exc:  # noqa: BLE001 - shown to the operator verbatim
                    self._problem = str(exc)
                self._probed = True
        return {"model": self.spec, "ready": self._problem is None, "problem": self._problem}

    def chat(self):
        chat = getattr(self._local, "chat", None)
        if chat is None:
            chat = self._build()
            self._local.chat = chat
        return chat


def ask_payload(model: Model, run: Run, text: str) -> dict:
    question = qa_bank.Question(
        qid="live",
        kind="freeform",
        text=text,
        answer=None,
        answer_type="freeform",
        graded=False,
    )
    started = time.perf_counter()
    answer = qa_agent.ask(model.chat(), run.batch, run.result, question)
    return {
        "question": text,
        "answer": answer.given,
        "basis": answer.basis,
        "confidence": answer.confidence,
        "declined": answer.declined,
        "failed": answer.failed,
        "detail": answer.detail,
        "graded": answer.graded,
        # The operator's check on the model. Everything it was allowed to see.
        "factsheet": json.loads(qa_agent.factsheet(run.batch, run.result, question)),
        "seconds": round(time.perf_counter() - started, 1),
        "tokens_in": answer.usage.input_tokens,
        "tokens_out": answer.usage.output_tokens,
    }


def bank_payload(model: Model, run: Run, limit: int, workers: int = 6) -> dict:
    questions = qa_bank.build(run.batch, run.result)
    if limit:
        questions = questions[:limit]
    report = qa_agent.QAReport()
    lock = threading.Lock()
    started = time.perf_counter()

    def work(q):
        answer = qa_agent.ask(model.chat(), run.batch, run.result, q)
        with lock:
            report.usage.merge(answer.usage)
        return answer

    with ThreadPoolExecutor(max_workers=workers) as pool:
        report.answers = list(pool.map(work, questions))
    report.seconds = time.perf_counter() - started

    by_id = {q.qid: q for q in questions}
    return {
        "wrong_answer_rate": report.wrong_answer_rate,
        "coverage": report.coverage,
        "accuracy": report.accuracy,
        "hallucinated": report.hallucinated,
        "total": report.total,
        "attempted": report.attempted,
        "correct": report.correct,
        "wrong": report.wrong,
        "declined": report.declined,
        "failed": report.failed,
        "seconds": round(report.seconds, 1),
        "tokens_in": report.usage.input_tokens,
        "tokens_out": report.usage.output_tokens,
        "answers": [{
            "qid": a.qid,
            "text": by_id[a.qid].text if a.qid in by_id else a.qid,
            "expects_decline": by_id[a.qid].expects_decline if a.qid in by_id else False,
            "given": a.given,
            "expected": by_id[a.qid].answer if a.qid in by_id else None,
            "correct": a.correct,
            "declined": a.declined,
            "failed": a.failed,
            "detail": a.detail,
        } for a in report.answers],
    }


# ── the server ───────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    server_version = "recoagent"
    cache: RunCache
    model: Model

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Nothing here is meant to be embedded anywhere else.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json; charset=utf-8")

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0 or length > 1_000_000:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/model":
            self._json(self.model.status())
        elif path == "/api/run":
            query = parse_qs(urlparse(self.path).query)
            flat = {k: v[0] for k, v in query.items()}
            self._run(flat)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        payload = self._body()
        if path == "/api/run":
            self._run(payload)
        elif path == "/api/ask":
            self._ask(payload)
        elif path == "/api/bank":
            self._bank(payload)
        else:
            self._json({"error": "not found"}, 404)

    def _run(self, payload: dict) -> None:
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        self._json(run_payload(self.cache.get(key)))

    def _ask(self, payload: dict) -> None:
        text = str(payload.get("question", "")).strip()
        if not text:
            return self._json({"error": "ask something"}, 400)
        if len(text) > 2000:
            return self._json({"error": "question is too long"}, 400)
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        status = self.model.status()
        if not status["ready"]:
            return self._json({"error": status["problem"]}, 503)
        try:
            self._json(ask_payload(self.model, self.cache.get(key), text))
        except Exception as exc:  # noqa: BLE001 - a failed call is a result, not a crash
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 502)

    def _bank(self, payload: dict) -> None:
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        status = self.model.status()
        if not status["ready"]:
            return self._json({"error": status["problem"]}, 503)
        try:
            limit = int(payload.get("limit", 0))
        except (TypeError, ValueError):
            limit = 0
        try:
            self._json(bank_payload(self.model, self.cache.get(key), limit))
        except Exception as exc:  # noqa: BLE001
            self._json({"error": f"{type(exc).__name__}: {exc}"}, 502)


def serve(host: str, port: int, model_spec: str) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"cache": RunCache(), "model": Model(model_spec)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    from .llm import DEFAULT_MODEL

    ap = argparse.ArgumentParser(prog="recoagent.ui", description="Live operator console.")
    # Loopback by default and on purpose: this serves a generated book and holds
    # an API key. Binding it to 0.0.0.0 should be a decision, not a default.
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    args = ap.parse_args(argv)

    httpd = serve(args.host, args.port, args.model)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  RecoAgent console  {url}")
    print(f"  model: {args.model}")
    status = httpd.RequestHandlerClass.model.status()  # type: ignore[attr-defined]
    if status["ready"]:
        print("  model ready — the ask box and the scored bank are both live")
    else:
        print("  no model configured — the queue works, Q&A is disabled")
        print(f"    {status['problem'].splitlines()[0]}")
    print("  ctrl-c to stop\n")

    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
