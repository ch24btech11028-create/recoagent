"""Reconcile a book that came from Razorpay rather than from us.

    python -m recoagent.razorpay.run pull --out data/razorpay/pull.json
    python -m recoagent.razorpay.run reconcile data/razorpay/pull.json --bank bank.csv
    python -m recoagent.razorpay.run serve --port 8000

`pull` needs `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET` in the environment or
an untracked `.env`, and refuses a live key. `reconcile` needs neither -- it
reads a recorded pull, so anybody can replay the exact book a published number
was computed on without an account and without network access.

There is no accuracy figure on this page, for the same reason `recoagent.ingest`
has none: a Razorpay pull carries no answer key. What it prints is coverage,
what tied each row out, and every exception -- plus, before any of that, a
readiness section saying which questions this particular book cannot answer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..ingest import IngestError, read_rows
from ..pipeline import run_b2
from .mapping import bundle_from_payload, readiness


def _bank_lines(path: Path | None, unit: str):
    if path is None:
        return ()
    return read_rows(path, "bank", unit=unit, overrides={})


def cmd_pull(args) -> int:
    from .api import Client, Recorder, pull

    out = Path(args.out)
    recorder = Recorder(directory=out.parent / "raw")
    client = Client(recorder=recorder)

    print("\n  pulling Razorpay test mode\n")
    payload = pull(
        client,
        days=args.days,
        account_number=args.account_number,
        limit=args.limit,
        on_progress=lambda msg: print(f"    {msg}"),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\n  wrote {out}")
    print(f"  {len(recorder.written)} raw responses under {recorder.directory}")
    print(
        "\n  The raw responses are the evidence. Keep them: a number computed on\n"
        "  a pull nobody kept is a number nobody can check.\n"
    )
    return 0


def cmd_reconcile(args) -> int:
    from ..ingest import report

    payload = json.loads(Path(args.pull).read_text(encoding="utf-8"))
    try:
        bank = _bank_lines(Path(args.bank) if args.bank else None, args.money)
    except IngestError as exc:
        print(f"\nerror: {exc}\n", file=sys.stderr)
        return 2

    sources = bundle_from_payload(payload, bank_lines=bank)

    notes = readiness(payload, sources)
    if notes:
        print("\n" + "=" * 72)
        print("  WHAT THIS BOOK CANNOT BE ASKED")
        print("=" * 72)
        for note in notes:
            print("\n  - " + note.replace(". ", ".\n    ", 1))
        print()

    result = run_b2(sources)
    print(report(sources, result, top=args.exceptions))

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "source": "razorpay-test-mode",
                    "pulled_at": payload.get("pulled_at"),
                    "readiness": notes,
                    "counts": dict(sources.counts),
                    "matches": len(result.matches),
                    "exceptions": len(result.exceptions),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {args.out}\n")
    return 0


def cmd_serve(args) -> int:
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from ..env import require_key
    from .webhook import SUBSCRIBED, EventLog, SignatureError

    secret = require_key("RAZORPAY_WEBHOOK_SECRET")
    log = EventLog(args.db)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802  (stdlib's spelling)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            signature = self.headers.get("X-Razorpay-Signature", "")
            try:
                event = log.record(body, signature, secret)
            except SignatureError as exc:
                # 401, not 400. A 400 says "fix your payload"; this payload is
                # not Razorpay's to fix.
                self._reply(401, {"error": str(exc)})
                print(f"    REJECTED  {exc}")
                return
            except (UnicodeDecodeError, ValueError) as exc:
                self._reply(400, {"error": f"body is not JSON: {exc}"})
                return

            mark = "replay " if event.replay else ""
            known = "" if event.event in SUBSCRIBED else "  (not subscribed)"
            print(f"    {mark}{event.event:<24}{event.event_id}{known}")
            # 200 even on a replay: Razorpay retries until it gets one, and a
            # non-2xx on an event we have already stored asks for the retry we
            # just proved we do not need.
            self._reply(200, {"status": "ok", "replay": event.replay})

        def _reply(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args) -> None:
            """Silence the default access log; the handler prints what matters."""

    server = HTTPServer(("", args.port), Handler)
    print(f"\n  listening on :{args.port}, storing to {args.db}")
    print("  every event is signature-checked before it is stored, and stored")
    print("  by event id, so a delivery retry books nothing twice.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\n  {log.count()} events from {log.deliveries()} deliveries\n")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.razorpay.run", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pull", help="fetch a book from Razorpay test mode")
    p.add_argument("--out", default="data/razorpay/pull.json")
    p.add_argument("--days", type=int, default=3, help="days of recon report to walk back")
    p.add_argument("--account-number", help="RazorpayX account, to include payouts")
    p.add_argument("--limit", type=int, help="stop after this many rows per entity")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("reconcile", help="reconcile a recorded pull")
    p.add_argument("pull")
    p.add_argument("--bank", help="bank statement CSV; without it Leg 2 cannot run")
    p.add_argument("--money", choices=("rupees", "paise"), default="rupees")
    p.add_argument("--exceptions", type=int, default=10)
    p.add_argument("--out", help="write a summary artifact here")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("serve", help="receive webhooks")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--db", default="data/razorpay/events.db")
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"\nerror: {exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
