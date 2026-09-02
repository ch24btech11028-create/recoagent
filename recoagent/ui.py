"""The operator console: run a reconciliation, work the queue, audit the result.

`recoagent.report` writes a static page — the 9am snapshot, mailable, no server.
This is the same product with its hands free, and it is a dashboard rather than
a page: an overview, the exception queue with a case file behind every row, the
match log, the four source ledgers, the agent, the published results, and the
one screen that uses the answer key.

The server is only routes and payload shaping. What each screen renders is
built in `views.py`; the front end itself is HTML, CSS and JavaScript under
`recoagent/web/`, served as-is by `webassets.py`.

Three things are deliberate.

**No operator screen can see the answer key.** The queue, the case files, the
match log and the source ledgers are shaped in `views.py`, which takes a
`SourceBundle` and a `ReconResult` and nothing else — exactly like the static
export and exactly like the matchers, and covered by the same test that proves
it (`tests/test_independence.py`). Ground truth appears on the Assurance screen
alone, fenced off and labelled, because "what do I work on" and "should I
believe this" are different questions and mixing them is how a demo starts
lying.

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
import os
import sys
import threading
import time
import webbrowser
from contextlib import contextmanager
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import plain, trace, views
from .worklist.store import (
    ALLOWED, ILLEGAL, Item, Worklist, WorklistError, fingerprint,
)

_log = trace.logger("ui")
from .eval.scorer import Scorecard, score
from .generator import DefectMix, GeneratorConfig, generate
from .money import format_inr
from .pipeline import run_b0, run_b2
from .qa import agent as qa_agent
from .qa import bank as qa_bank
from .schemas import LabelledBatch, ReconResult
from .views import RULE_LABEL
from .webassets import asset_for

#: Seed 21 for both held-out profiles, so `unknown` is `holdout` plus three
#: defect classes the engine has no tier for and nothing else. See
#: `recoagent.unknown`.
MIXES = {
    "dev": (7, DefectMix.dev),
    "holdout": (21, DefectMix.holdout),
    "clean": (7, DefectMix.clean),
    "unknown": (21, DefectMix.unknown),
}

#: Routes that either spend the operator's API key or write to their worklist.
#: On loopback that is the product. Facing the internet it is somebody else's
#: bill and somebody else's queue, so `--public` refuses them and says why.
GUARDED = {
    "/api/ask": "the model costs money per call",
    "/api/bank": "the scored bank costs money per question",
    "/api/worklist": "the worklist is this operator's record, not a shared one",
}

#: Runs are pure functions of these four numbers, so they cache perfectly. The
#: cap keeps a long demo session from holding every batch it ever generated.
CACHE_LIMIT = 6

#: Where the console keeps its queues. One database per book -- see `Desk`.
DEFAULT_WORKLIST = "data/worklist"


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


def _queue_order(result: ReconResult) -> list:
    """The exceptions `views.queue` returns, in its order.

    Zipping two independently sorted lists would pair the wrong fingerprint to
    the wrong row the first time either sort changed, so the order is taken
    from the view itself rather than re-derived here.
    """
    by_xid = {e.exception_id: e for e in result.exceptions}
    return [by_xid[row["xid"]] for row in views.queue(result)]


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
        # Classes with no tier behind them. Empty on every profile but
        # `unknown`, so the screen shows the panel only when there is a result
        # to show. See `recoagent.unknown`.
        "unknown_accounting": [{
            "defect": a.defect.value,
            "injected": a.injected,
            "contained": a.flagged,
            "absorbed": a.resolved,
            "mishandled": a.mishandled,
        } for a in card.unknown_accounting],
        "unknown": {
            "injected": card.unknown_injected,
            "contained": card.unknown_contained,
            "absorbed": card.unknown_absorbed,
            "mishandled": card.unknown_mishandled,
            "containment_rate": card.unknown_containment_rate,
            "holds": card.unknown_holds,
        },
        "verdict": {
            "fully_reconciles": card.fully_reconciles,
            "mishandled_total": card.mishandled_total,
            "unattributed": card.unattributed_exceptions,
            "injected_total": sum(a.injected for a in card.accounting),
        },
        # The worklist's own key, attached here rather than in `views` so that
        # module keeps taking a SourceBundle and a ReconResult and nothing else.
        "queue": [dict(row, fp=fingerprint(exc))
                  for row, exc in zip(views.queue(result, run.batch.sources), _queue_order(result))],
        "shape": views.shape(run.batch.sources, result),
        "recovered": _recovered_payload(result),
    }


# ── the desk: where a person actually acts on an item ────────────────────────

class Desk:
    """The console's link to the persistent queue in `recoagent.worklist`.

    Until this existed the console could show you an exception and nothing
    else: no way to take one, close one, or write one off, and nothing you did
    survived a refresh. The queue itself was already persistent and already
    knew how to carry an item forward -- what was missing was the screen.

    **One database per book.** The worklist keys on the business identity of the
    failing entity, and `order_00033` exists in the dev book and in the held-out
    book as two entirely different transactions. Pointing both at one queue
    would merge two unrelated problems on the strength of a shared id, so each
    (profile, seed, n, rung) gets its own file.

    **Opened per request, under a lock.** sqlite connections have thread
    affinity and this server is threaded. A reconciliation queue sees a click
    every few seconds, not a thousand a second, so the simple thing is also the
    correct one -- and it means a crash cannot leave a connection open.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def path_for(self, key: RunKey) -> Path:
        return self.directory / f"{key.profile}-seed{key.seed}-n{key.n}-{key.rung}.db"

    @contextmanager
    def _open(self, key: RunKey):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            book = Worklist(self.path_for(key))
            try:
                yield book
            finally:
                book.close()

    def sync(self, key: RunKey, run: Run) -> dict[str, int]:
        """Fold this book's exceptions into its queue.

        Called when the operator asks for a run, not when a page loads. That
        distinction is the whole of it: pressing Reconcile is a run and belongs
        in the `runs` table, while rendering a screen from a cached result is
        not and would inflate `carried_forward` -- the number an ops team
        actually feels -- into noise.

        Recording again is safe and is the point. `Worklist.record` updates the
        reason, the residual and the timestamps, and deliberately does not touch
        status, assignee or notes; an item that has since been matched closes
        itself. An earlier version of this method recorded only once per server
        lifetime, which meant pressing Reconcile never re-filed the queue and
        carry-forward could not happen at all.
        """
        with self._open(key) as book:
            changed = book.record(
                run.batch.sources, run.result,
                label=f"console {key.profile} seed={key.seed} n={key.n} {key.rung}",
            )
        trace.event(_log, "desk.sync", book=self.path_for(key).name, **changed)
        return changed

    def ensure(self, key: RunKey, run: Run) -> None:
        """File the book if it has never been filed, so an action has a target."""
        if self.path_for(key).exists():
            return
        self.sync(key, run)

    def snapshot(self, key: RunKey) -> dict:
        with self._open(key) as book:
            items = {i.fingerprint: _item_payload(i) for i in book.items()}
            counts = book.counts()
        return {"items": items, "counts": counts, "allowed": {k: list(v) for k, v in ALLOWED.items()}}

    def act(self, key: RunKey, fp: str, *, to: str | None, assignee: str | None,
            notes: str | None, actor: str, detail: str) -> dict:
        with self._open(key) as book:
            if assignee is not None or notes is not None:
                book.annotate(fp, assignee=assignee, notes=notes)
            if to:
                book.transition(fp, to, actor=actor or "analyst", detail=detail)
            return {"item": _item_payload(book.get(fp)), "history": book.history(fp),
                    "counts": book.counts()}

    def history(self, key: RunKey, fp: str) -> dict:
        with self._open(key) as book:
            return {"item": _item_payload(book.get(fp)), "history": book.history(fp)}


def _item_payload(item: Item) -> dict:
    return {
        "fp": item.fingerprint,
        "status": item.status,
        "assignee": item.assignee,
        "notes": item.notes,
        "entity_id": item.entity_id,
        "leg": item.leg,
        "first_seen_run": item.first_seen_run,
        "last_seen_run": item.last_seen_run,
        "closed_run": item.closed_run,
        "closed_reason": item.closed_reason,
        "age_in_runs": item.age_in_runs,
        "is_open": item.is_open,
        "can_move_to": list(ALLOWED.get(item.status, ())),
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
                    # `check_ready`, not construction: an OpenAI-compatible
                    # client is built lazily now, so constructing one proves
                    # nothing about whether a key is present.
                    self._build().check_ready()
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
        # The same answer as money, when it is money. The Q&A contract keeps
        # the model in integer paise so `qa.bank.is_correct` can grade by exact
        # comparison -- that rule is load-bearing for the wrong-answer rate and
        # is not being relaxed. It is converted here, at the edge, by the one
        # formatter this repository has. A merchant reading "-118632" off a
        # screen is a contract leaking into a product.
        "answer_money": plain.rupees(answer.given)
                        if isinstance(answer.given, int) and not isinstance(answer.given, bool)
                        else None,
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
    #: Empty unless an origin was named on the command line.
    allow_origin: str = ""
    #: True when this process is reachable by someone other than its operator.
    public: bool = False

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # Nothing here is meant to be embedded anywhere else.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        # Nothing this server sends is worth keeping. Every response describes
        # one run of a book that is regenerated on demand, and a cached page
        # after a restart shows an operator the previous version of the console
        # with none of the fixes in it.
        self.send_header("Cache-Control", "no-store")
        # A named origin or nothing. `*` would let any page anyone visits read
        # this book and, worse, spend the key behind /api/ask, so the origin has
        # to be typed on the command line by the person taking the risk.
        if self.allow_origin:
            self.send_header("Access-Control-Allow-Origin", self.allow_origin)
            self.send_header("Vary", "Origin")
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

    def do_OPTIONS(self) -> None:  # noqa: N802
        """The browser's preflight. Answered only when an origin was allowed."""
        if not self.allow_origin:
            return self._json({"error": "cross-origin requests are not enabled"}, 405)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", self.allow_origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.end_headers()

    def _refused_in_public(self, path: str) -> bool:
        """Say no, and say why, rather than failing in a way nobody can read."""
        if self.public and path in GUARDED:
            self._json({
                "error": f"{path} is disabled on a public deployment: {GUARDED[path]}",
                "public": True,
            }, 403)
            return True
        return False

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if self._refused_in_public(path):
            return
        asset = asset_for(path)
        if asset is not None:
            body, content_type = asset
            self._send(200, body.encode(), content_type)
        elif path == "/api/model":
            self._json(self.model.status())
        elif path in ("/api/run", "/api/exception", "/api/matches", "/api/source"):
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            if path == "/api/run":
                self._run(query)
            elif path == "/api/exception":
                self._exception(query)
            elif path == "/api/matches":
                self._matches(query)
            else:
                self._source(query)
        elif path == "/api/worklist":
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            self._worklist(query)
        elif path == "/api/results":
            query = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            self._results(query)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if self._refused_in_public(path):
            return
        payload = self._body()
        if path == "/api/run":
            self._run(payload, reconciling=True)
        elif path == "/api/ask":
            self._ask(payload)
        elif path == "/api/bank":
            self._bank(payload)
        elif path == "/api/worklist":
            self._act(payload)
        else:
            self._json({"error": "not found"}, 404)

    def _run(self, payload: dict, *, reconciling: bool = False) -> None:
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        run = self.cache.get(key)
        # A POST is the operator pressing Reconcile, which is a run. A GET is a
        # screen rendering itself, which is not.
        self.desk.sync(key, run) if reconciling else self.desk.ensure(key, run)
        self._json(run_payload(run))

    def _int(self, payload: dict, name: str, default: int) -> int:
        try:
            return int(payload.get(name, default))
        except (TypeError, ValueError):
            return default

    def _exception(self, payload: dict) -> None:
        """One item's case file. Built from the sources, like everything else."""
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        run = self.cache.get(key)
        found = views.exception_case(run.batch.sources, run.result, str(payload.get("id", "")))
        if found is None:
            return self._json({"error": "no such exception in this run"}, 404)
        self._json(found)

    def _worklist(self, payload: dict) -> None:
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        fp = str(payload.get("fp", ""))
        if fp:
            try:
                return self._json(self.desk.history(key, fp))
            except WorklistError as exc:
                return self._json({"error": str(exc)}, 404)
        self._json(self.desk.snapshot(key))

    def _act(self, payload: dict) -> None:
        """Take, resolve, write off, release -- or refuse, and say why.

        The store owns which moves are legal and the message explaining a
        refusal, so this returns that message verbatim at 409 rather than
        writing a second, drifting copy of the rule.
        """
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        fp = str(payload.get("fp", ""))
        if not fp:
            return self._json({"error": "which item? no fp given"}, 400)
        # The run has to be in the queue before anybody can act on it.
        self.desk.ensure(key, self.cache.get(key))
        try:
            self._json(self.desk.act(
                key, fp,
                to=(str(payload["to"]) if payload.get("to") else None),
                assignee=(str(payload["assignee"]) if "assignee" in payload else None),
                notes=(str(payload["notes"]) if "notes" in payload else None),
                actor=str(payload.get("actor") or "analyst"),
                detail=str(payload.get("detail") or ""),
            ))
        except WorklistError as exc:
            # 409, not 400: the request was well formed and the queue said no.
            self._json({"error": str(exc)}, 409 if ILLEGAL in str(exc) else 404)

    def _matches(self, payload: dict) -> None:
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        run = self.cache.get(key)
        self._json(views.matches(
            run.result,
            leg=str(payload.get("leg", "")),
            tier=str(payload.get("tier", "")),
            query=str(payload.get("q", ""))[:120],
            page=self._int(payload, "page", 1),
        ))

    def _source(self, payload: dict) -> None:
        try:
            key = RunKey.parse(payload)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        run = self.cache.get(key)
        try:
            self._json(views.source(
                run.batch.sources,
                str(payload.get("kind", "payments")),
                query=str(payload.get("q", ""))[:120],
                page=self._int(payload, "page", 1),
            ))
        except ValueError as exc:
            self._json({"error": str(exc)}, 400)

    def _results(self, payload: dict) -> None:
        """The published artefacts, served as they were written.

        Read-only and confined to the results directory: the name has to be one
        the index already offered, so a request cannot describe a path.
        """
        name = str(payload.get("file", ""))
        if not name:
            return self._json({"files": views.results_index()})
        text = views.result_file(name)
        if text is None:
            return self._json({"error": "no such result file"}, 404)
        self._json({"name": name, "text": text})

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


def serve(host: str, port: int, model_spec: str,
          worklist: str | Path = DEFAULT_WORKLIST,
          *, allow_origin: str = "", public: bool = False) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {
        "cache": RunCache(),
        "model": Model(model_spec),
        "desk": Desk(worklist),
        "allow_origin": allow_origin,
        "public": public,
    })
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    from .llm import DEFAULT_MODEL

    ap = argparse.ArgumentParser(prog="recoagent.ui", description="Live operator console.")
    # Loopback by default and on purpose: this serves a generated book and holds
    # an API key. Binding it to 0.0.0.0 should be a decision, not a default.
    ap.add_argument("--host", default="127.0.0.1")
    # Every host that runs a process for you names the port in the environment.
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    ap.add_argument(
        "--allow-origin", default="",
        help="allow browser requests from this exact origin, e.g. "
             "https://your-frontend.example. One origin, never '*'",
    )
    ap.add_argument(
        "--public", action="store_true",
        help="this process is reachable by strangers: refuse the routes that "
             "spend the API key or write the worklist",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-open", action="store_true", help="do not open a browser")
    ap.add_argument(
        "--worklist", default=DEFAULT_WORKLIST,
        help="directory for the persistent queues, one database per book",
    )
    trace.add_argument(ap)
    args = ap.parse_args(argv)
    trace.from_args(args)

    httpd = serve(args.host, args.port, args.model, args.worklist,
                  allow_origin=args.allow_origin, public=args.public)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  RecoAgent console  {url}")
    print(f"  model: {args.model}")
    print(f"  worklist: {args.worklist}/")
    if args.allow_origin:
        print(f"  cross-origin: {args.allow_origin}")
    if args.public:
        print("  public mode: /api/ask, /api/bank and /api/worklist are refused")
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        # Not an error -- there are good reasons to bind wider on a private
        # network -- but the operator should know what they just exposed.
        print(f"  WARNING: bound to {args.host}, not loopback, without --public.")
        print("           /api/ask and /api/bank will spend your API key for anyone who can reach this.")
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
