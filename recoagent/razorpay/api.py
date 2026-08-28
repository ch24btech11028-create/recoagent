"""A Razorpay API client with no dependencies and no surprises.

`urllib` rather than `requests`, for the same reason the rest of the engine
has no third-party imports: CI runs the whole pipeline on a machine with
nothing installed, and one convenience import would end that.

Three behaviours are worth reading before trusting anything this returns.

**It records what it fetched.** Every response is written to disk verbatim
before it is parsed. A run that reconciles live API output and then throws the
output away cannot be re-checked by anyone, including its author; a recorded
pull can be replayed offline, diffed between runs, and handed to a reader who
has no key.

**It refuses live keys.** A `rzp_live_` id raises. This project has no reason
to touch a real merchant's money data, the difference between the two prefixes
is one character, and a mistake there is not recoverable by apologising.

**It backs off rather than hammering.** 429 and 5xx are retried with
exponential delay and a cap; everything else fails immediately with the body
Razorpay sent, because a 400 retried five times is still a 400 and the message
in it is the thing you actually needed to read.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..env import require_key

BASE_URL = "https://api.razorpay.com/v1"

#: Razorpay's collection endpoints page with `skip`/`count`, capped at 100.
PAGE_SIZE = 100

#: Retried. Everything else is reported as-is -- a 400 does not improve with
#: patience, and the body carries the reason.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

MAX_ATTEMPTS = 5


class RazorpayError(RuntimeError):
    """An API call that failed in a way the caller has to know about."""

    def __init__(self, status: int, body: str, url: str) -> None:
        # The URL is included but the credentials never are: they live in a
        # header, and this message ends up in logs and terminals.
        super().__init__(f"HTTP {status} from {url}\n{body.strip()[:800]}")
        self.status = status
        self.body = body


@dataclass
class Credentials:
    key_id: str
    key_secret: str

    @classmethod
    def from_env(cls) -> "Credentials":
        key_id = require_key("RAZORPAY_KEY_ID")
        secret = require_key("RAZORPAY_KEY_SECRET")
        if not key_id.startswith("rzp_test_"):
            raise RuntimeError(
                f"RAZORPAY_KEY_ID is {key_id[:9]}..., which is not a test key. "
                "This project reads test mode only -- a live key would pull a "
                "real merchant's transaction data onto this laptop and into "
                "recorded fixtures. Generate a test key from Dashboard -> "
                "Settings -> API Keys."
            )
        return cls(key_id=key_id, key_secret=secret)

    @property
    def header(self) -> str:
        raw = f"{self.key_id}:{self.key_secret}".encode()
        return "Basic " + base64.b64encode(raw).decode()


@dataclass
class Recorder:
    """Writes every response to disk before anybody gets to interpret it.

    Named by endpoint and page so a replay is deterministic and a diff between
    two pulls reads as a diff between two days of data, not between two
    orderings of the same data.
    """

    directory: Path
    written: list[Path] = field(default_factory=list)

    def save(self, name: str, page: int, payload: dict[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{name}.{page:03d}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self.written.append(path)
        return path


class Client:
    """Read-only access to Razorpay test mode.

    Read-only is a design decision, not an omission. Nothing here issues a
    POST: the data this project needs already exists in the account, and a
    reconciliation tool that can also move money is a reconciliation tool whose
    bugs cost money.
    """

    def __init__(
        self,
        creds: Credentials | None = None,
        *,
        base_url: str = BASE_URL,
        recorder: Recorder | None = None,
        timeout: float = 30.0,
        sleep=time.sleep,
    ) -> None:
        self.creds = creds or Credentials.from_env()
        self.base_url = base_url.rstrip("/")
        self.recorder = recorder
        self.timeout = timeout
        self._sleep = sleep

    # ── transport ───────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {k: v for k, v in (params or {}).items() if v is not None}
        )
        url = f"{self.base_url}{path}" + (f"?{query}" if query else "")

        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            request = urllib.request.Request(
                url,
                headers={
                    "Authorization": self.creds.header,
                    "Accept": "application/json",
                    "User-Agent": "recoagent/1.0 (+reconciliation research)",
                },
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")
                if exc.code not in RETRY_STATUS or attempt == MAX_ATTEMPTS:
                    raise RazorpayError(exc.code, body, url) from exc
                last = exc
            except urllib.error.URLError as exc:
                if attempt == MAX_ATTEMPTS:
                    raise RuntimeError(f"could not reach {url}: {exc.reason}") from exc
                last = exc
            # 1s, 2s, 4s, 8s. Razorpay documents per-endpoint rate limits and
            # the cheapest way to respect one is to stop asking for a while.
            self._sleep(2 ** (attempt - 1))

        raise RuntimeError(f"exhausted {MAX_ATTEMPTS} attempts against {url}") from last

    # ── collections ─────────────────────────────────────────────────────────

    def paginate(
        self,
        path: str,
        *,
        name: str,
        params: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every item across pages, recording each page as it arrives.

        Razorpay collections answer `{"entity": "collection", "count": n,
        "items": [...]}`. `count` is the size of *this page*, not the total, so
        the terminating condition is a short page -- there is no total to
        compare against and assuming otherwise silently truncates the pull.
        """
        skip = 0
        page = 0
        seen = 0
        while True:
            payload = self._get(path, {**(params or {}), "count": PAGE_SIZE, "skip": skip})
            if self.recorder is not None:
                self.recorder.save(name, page, payload)

            items = payload.get("items", [])
            for item in items:
                yield item
                seen += 1
                if limit is not None and seen >= limit:
                    return

            if len(items) < PAGE_SIZE:
                return
            skip += PAGE_SIZE
            page += 1

    # ── the four reads a reconciliation needs ───────────────────────────────

    def orders(self, **kw) -> list[dict[str, Any]]:
        return list(self.paginate("/orders", name="orders", **kw))

    def payments(self, **kw) -> list[dict[str, Any]]:
        return list(self.paginate("/payments", name="payments", **kw))

    def settlements(self, **kw) -> list[dict[str, Any]]:
        return list(self.paginate("/settlements", name="settlements", **kw))

    def refunds(self, **kw) -> list[dict[str, Any]]:
        return list(self.paginate("/refunds", name="refunds", **kw))

    def payouts(self, account_number: str, **kw) -> list[dict[str, Any]]:
        """RazorpayX payouts. Needs the X account number; there is no default."""
        params = dict(kw.pop("params", None) or {})
        params["account_number"] = account_number
        return list(self.paginate("/payouts", name="payouts", params=params, **kw))

    def settlement_recon(self, when: datetime) -> list[dict[str, Any]]:
        """The combined settlement recon report for one day.

        This is the endpoint that actually matters. `/settlements` returns
        batch headers -- an amount and a UTR -- and a header alone cannot be
        reconciled against anything, because Leg 2's entire problem is *which
        payments* went into the batch. The recon report answers that: one row
        per payment, refund, transfer or adjustment, each carrying the
        `settlement_id` it was netted into.
        """
        payload = self._get(
            "/settlements/recon/combined",
            {"year": when.year, "month": when.month, "day": when.day},
        )
        if self.recorder is not None:
            self.recorder.save(f"recon-{when:%Y%m%d}", 0, payload)
        return payload.get("items", [])


def pull(
    client: Client,
    *,
    days: int = 0,
    account_number: str | None = None,
    limit: int | None = None,
    on_progress=lambda _msg: None,
) -> dict[str, Any]:
    """Fetch one book. Returns the raw payload, unmapped and uninterpreted.

    Kept separate from the mapping on purpose: this function's output is what
    gets recorded and replayed, so it must contain everything Razorpay said and
    nothing this repository decided. The moment a pull starts summarising, a
    replay stops being evidence.
    """
    payload: dict[str, Any] = {
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "mode": "test",
    }

    for name, fetch in (
        ("orders", lambda: client.orders(limit=limit)),
        ("payments", lambda: client.payments(limit=limit)),
        ("refunds", lambda: client.refunds(limit=limit)),
        ("settlements", lambda: client.settlements(limit=limit)),
    ):
        rows = fetch()
        payload[name] = rows
        on_progress(f"{name:<14}{len(rows):>6}")

    if account_number:
        rows = client.payouts(account_number, limit=limit)
        payload["payouts"] = rows
        on_progress(f"{'payouts':<14}{len(rows):>6}")

    # The recon report is per-day, so it is only worth walking back as far as
    # the settlements we actually found.
    recon: list[dict[str, Any]] = []
    if days and payload["settlements"]:
        seen_days = sorted(
            {
                datetime.fromtimestamp(s["created_at"], timezone.utc).date()
                for s in payload["settlements"]
                if s.get("created_at")
            }
        )[-days:]
        for day in seen_days:
            recon.extend(
                client.settlement_recon(datetime(day.year, day.month, day.day))
            )
        on_progress(f"{'recon rows':<14}{len(recon):>6}")
    payload["recon"] = recon

    return payload
