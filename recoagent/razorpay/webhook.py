"""Webhook receipt: signature first, idempotency second, interpretation last.

Razorpay signs every webhook with HMAC-SHA256 over the **raw request body**,
using the secret you chose when you registered the endpoint, and sends the hex
digest in `X-Razorpay-Signature`.

Three details in that sentence are the entire security of this module, and all
three are easy to get wrong:

- **Raw body.** Not the parsed JSON re-serialised. `json.dumps(json.loads(b))`
  changes key order, whitespace and unicode escaping, and the digest of the
  round-tripped bytes is not the digest Razorpay computed. Verify, then parse.
- **Constant time.** `==` on digests leaks how many leading bytes matched, one
  request at a time. `hmac.compare_digest` does not.
- **Reject, don't repair.** An unsigned or wrongly-signed event is not a
  degraded event to be processed carefully; it is an event from someone who is
  not Razorpay, and the only correct handling is to drop it and say so.

Idempotency is separate and equally load-bearing. Razorpay retries a webhook
until it gets a 2xx, so an endpoint that books a payment twice on a delivery
retry has invented money. `EventLog` keys on the event id and stores the first
body it saw; a replay is recorded as a replay and changes nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIGNATURE_HEADER = "X-Razorpay-Signature"

#: Events worth storing. Anything else is acknowledged and dropped -- an
#: endpoint that persists every event type Razorpay might ever add is an
#: endpoint whose storage is defined by somebody else's roadmap.
SUBSCRIBED = frozenset({
    "payment.captured",
    "payment.failed",
    "payment.authorized",
    "order.paid",
    "refund.created",
    "refund.processed",
    "settlement.processed",
    "payout.processed",
    "payout.reversed",
})


class SignatureError(Exception):
    """The body did not come from someone holding the webhook secret."""


def verify_signature(body: bytes, signature: str, secret: str) -> None:
    """Raise unless `signature` is Razorpay's HMAC over exactly these bytes."""
    if not signature:
        raise SignatureError(f"missing {SIGNATURE_HEADER}")
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise SignatureError(
            "signature does not match the body. Either the secret is not the "
            "one this endpoint was registered with, or the body was modified "
            "in transit -- and a body that was modified in transit is not a "
            "body to process."
        )


@dataclass(frozen=True)
class StoredEvent:
    event_id: str
    event: str
    received_at: datetime
    payload: dict[str, Any]
    replay: bool


class EventLog:
    """An append-only record of every webhook that verified.

    SQLite because it is in the standard library. A queue and a Postgres
    instance would buy horizontal scale that a single-node reconciliation
    demo cannot use, at the cost of the property this project actually sells:
    that the whole thing runs on a machine with nothing installed.

    The raw body is stored, not a parsed summary. When a number later turns out
    to be wrong, the question is always "what did they actually send", and only
    the bytes answer it.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS raw_events (
        event_id     TEXT PRIMARY KEY,
        event        TEXT NOT NULL,
        received_at  TEXT NOT NULL,
        body         BLOB NOT NULL,
        signature    TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS deliveries (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id     TEXT NOT NULL,
        received_at  TEXT NOT NULL,
        replay       INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS deliveries_event ON deliveries(event_id);
    """

    def __init__(self, path: str | Path = "data/razorpay/events.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(self.SCHEMA)
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    def record(self, body: bytes, signature: str, secret: str) -> StoredEvent:
        """Verify, then store. Returns the event, flagged if it is a replay.

        Verification happens before anything is written, so a forged body never
        reaches storage and cannot be mistaken for history later.
        """
        verify_signature(body, signature, secret)
        payload = json.loads(body.decode("utf-8"))

        # Razorpay puts the delivery id in a header on newer integrations and
        # not on older ones; the payload always carries enough to key on.
        event_id = payload.get("id") or _derive_id(payload, body)
        event = payload.get("event", "unknown")
        now = datetime.now(timezone.utc)

        cursor = self.db.execute(
            "INSERT OR IGNORE INTO raw_events VALUES (?, ?, ?, ?, ?)",
            (event_id, event, now.isoformat(), body, signature),
        )
        replay = cursor.rowcount == 0
        self.db.execute(
            "INSERT INTO deliveries (event_id, received_at, replay) VALUES (?, ?, ?)",
            (event_id, now.isoformat(), int(replay)),
        )
        self.db.commit()

        return StoredEvent(
            event_id=event_id,
            event=event,
            received_at=now,
            payload=payload,
            replay=replay,
        )

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]

    def deliveries(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0]

    def payload(self) -> dict[str, list[dict[str, Any]]]:
        """Everything received, shaped like an `api.pull` result.

        The point of the webhook path is that it feeds the *same* reconciler as
        the polling path. Two ingestion routes that produce two different book
        shapes are two systems, and only one of them is tested.
        """
        out: dict[str, list[dict[str, Any]]] = {
            "orders": [], "payments": [], "refunds": [], "settlements": [], "recon": [],
        }
        seen: set[tuple[str, str]] = set()
        rows = self.db.execute(
            "SELECT event, body FROM raw_events ORDER BY received_at, event_id"
        )
        for event, body in rows:
            entities = (json.loads(body).get("payload") or {})
            for kind, holder in entities.items():
                entity = (holder or {}).get("entity")
                if not isinstance(entity, dict) or "id" not in entity:
                    continue
                bucket = _BUCKET.get(kind)
                if bucket is None:
                    continue
                key = (bucket, entity["id"])
                if key in seen:
                    continue
                seen.add(key)
                out[bucket].append(entity)
        return out


#: Webhook payloads nest entities under their type: `payload.payment.entity`.
_BUCKET = {
    "payment": "payments",
    "order": "orders",
    "refund": "refunds",
    "settlement": "settlements",
}


def _derive_id(payload: dict[str, Any], body: bytes) -> str:
    """A stable key for an event that did not carry one.

    Derived from the event name and the entity ids inside it, not from the
    bytes: two deliveries of the same event can differ in whitespace, and
    hashing the body would file them as two events and defeat the idempotency
    this exists to provide.
    """
    parts = [payload.get("event", "unknown")]
    for kind, holder in sorted((payload.get("payload") or {}).items()):
        entity = (holder or {}).get("entity") or {}
        parts.append(f"{kind}:{entity.get('id', '')}")
    if len(parts) == 1:
        parts.append(hashlib.sha256(body).hexdigest()[:16])
    return "evt_" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]
