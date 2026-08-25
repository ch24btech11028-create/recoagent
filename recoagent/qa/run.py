"""CLI: score the Q&A agent over a generated question bank.

    python -m recoagent.qa.run --model nvidia/nemotron-3-ultra-550b-a55b
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from ..eval.scorer import score
from ..generator import DefectMix, GeneratorConfig, generate
from ..llm import DEFAULT_MODEL, client_for
from ..pipeline import run_b2
from . import agent, bank

MIXES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.qa.run")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="cap questions, 0 = all")
    args = ap.parse_args(argv)

    seed, mix = MIXES[args.profile]
    batch = generate(GeneratorConfig(n_orders=args.n, seed=seed, mix=mix()))
    result = run_b2(batch.sources)
    card = score(batch, result)
    questions = bank.build(batch, result)
    if args.limit:
        questions = questions[: args.limit]

    print(f"  run: {args.profile}, {args.n:,} orders, "
          f"leg-2 recall {card.legs[2].recall:.2%}, {len(result.exceptions)} open items")
    print(f"  asking {len(questions)} questions of {args.model} "
          f"with {args.workers} workers\n", flush=True)

    report = agent.QAReport()
    started = time.time()
    # One client per worker: the chat objects hold a connection, and a thread
    # only ever works one question at a time.
    import threading

    local = threading.local()
    lock = threading.Lock()

    def work(q):
        chat = getattr(local, "chat", None)
        if chat is None:
            chat = client_for(args.model)
            local.chat = chat
        answer = agent.ask(chat, batch, result, q)
        # Usage lives on the reply, so it has to be folded in here or the cost
        # line reports zero -- which it did on the first run.
        with lock:
            report.usage.merge(answer.usage)
        return answer

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        report.answers = list(pool.map(work, questions))
    report.seconds = time.time() - started

    print(agent.render(report, questions))
    return 0 if (report.wrong_answer_rate == 0 and report.hallucinated == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
