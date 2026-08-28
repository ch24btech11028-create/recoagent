"""The Razorpay ingestion path, tested without a key and without a network.

Everything here runs against a recorded pull committed under
`tests/fixtures/razorpay/`. That is not a convenience: a test that needs an API
key is a test that does not run in CI, and an integration whose tests do not
run in CI is an integration nobody finds out has broken.

The transport tests drive `Client` through a fake opener rather than a mock
library, for the same reason the rest of the repo has no third-party imports.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import urllib.error
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pytest

from recoagent.ingest import read_rows
from recoagent.legs.leg1 import FUNDED_STATUSES
from recoagent.pipeline import run_b2
from recoagent.razorpay import api, mapping, webhook

FIXTURES = Path(__file__).parent / "fixtures" / "razorpay"


@pytest.fixture
def payload() -> dict:
    return json.loads((FIXTURES / "pull.json").read_text())


@pytest.fixture
def bank():
    return read_rows(FIXTURES / "bank.csv", "bank", unit="rupees", overrides={})


# ── the fee convention, which is the one that costs money to get wrong ──────


def test_razorpay_fee_is_inclusive_of_tax_and_is_split(payload):
    """`fee` contains `tax`; the engine's `fee_paise` must not.

    Razorpay's own recon report publishes `credit == amount - fee`. If the two
    fields were copied across unchanged, `net_paise` would subtract the GST a
    second time and every settlement would look short by exactly the tax in it.
    """
    row = next(p for p in payload["payments"] if p["id"] == "pay_QpA2bR7wLdY001")
    fee, tax = mapping.split_fee(row)

    assert row["fee"] == 5898 and row["tax"] == 900
    assert (fee, tax) == (4998, 900)

    payment = mapping.payment_from(row)
    credit = next(
        r for r in payload["recon"] if r["entity_id"] == "pay_QpA2bR7wLdY001"
    )["credit"]
    assert payment.net_paise == credit == row["amount"] - row["fee"]


def test_a_fee_smaller_than_its_own_tax_is_refused():
    with pytest.raises(ValueError, match="exceeds the inclusive fee"):
        mapping.split_fee({"id": "pay_x", "fee": 100, "tax": 400})


def test_the_whole_batch_reconstructs_from_the_split_fees(payload):
    """Sum the mapped rows and you get the settlement Razorpay reported.

    This is the arithmetic gate applied to the mapping itself. It would pass
    just as happily on a double-counted book if the fixture were also
    double-counted, so the number it is checked against is Razorpay's own
    settlement header, which the mapping never touches.
    """
    bundle = mapping.bundle_from_payload(payload)
    settlement = bundle.settlements[0]
    members = bundle.payments_by_settlement(settlement.settlement_id)
    adjustments = bundle.adjustments_by_settlement(settlement.settlement_id)

    derived = sum(p.net_paise for p in members) + sum(a.amount_paise for a in adjustments)
    assert derived == settlement.net_paise


# ── what the mapping refuses to invent ──────────────────────────────────────


def test_no_bank_side_is_invented(payload):
    """Leg 2 gets nothing unless a statement was supplied."""
    bundle = mapping.bundle_from_payload(payload)
    assert bundle.bank_lines == ()

    notes = mapping.readiness(payload, bundle)
    assert any("No bank statement" in n for n in notes)
    assert any("meaningless 100%" in n for n in notes)


def test_an_unknown_status_passes_through_rather_than_becoming_captured(payload):
    row = next(p for p in payload["payments"] if p["status"] == "pending_authorization")
    assert mapping.payment_from(row).status == "pending_authorization"


def test_amounts_must_be_integer_paise():
    with pytest.raises(ValueError, match="expected integer paise"):
        mapping.payment_from(
            {"id": "pay_x", "amount": 249.90, "status": "captured", "created_at": 1756290000}
        )


def test_a_partial_capture_needs_the_order_to_declare_it(payload):
    """`partially_captured` comes from `amount_paid < amount`, nothing looser."""
    bundle = mapping.bundle_from_payload(payload)
    partial = [p for p in bundle.payments if p.status == "partially_captured"]
    assert [p.payment_id for p in partial] == ["pay_QpA2bR7wLdY004"]

    # Strip the declaration and the same payment is an ordinary capture, which
    # Leg 1 will then refuse to match against a larger order.
    stripped = dict(payload)
    stripped["orders"] = [
        {**o, "amount_paid": o["amount"]} if o["id"] == "order_QpA1xK3mNvT004" else o
        for o in payload["orders"]
    ]
    assert not [
        p for p in mapping.bundle_from_payload(stripped).payments
        if p.status == "partially_captured"
    ]


def test_card_method_splits_domestic_from_international():
    base = {"id": "pay_x", "amount": 1000, "status": "captured", "created_at": 1, "method": "card"}
    assert mapping.method_of(base) == "card_domestic"
    assert mapping.method_of({**base, "international": True}) == "card_international"
    # A method the rate card has never heard of keeps its own name, so it
    # reaches a human instead of being priced at somebody else's rate.
    assert mapping.method_of({**base, "method": "paylater"}) == "paylater"


# ── the false match real data exposed ───────────────────────────────────────


def test_a_failed_payment_is_never_matched_to_its_order(payload, bank):
    """The bug generated books could not contain.

    `pay_...Y005` is a declined card carrying the full order amount. An exact
    join gated only on `gross == order.amount` matches it, and books revenue
    that never arrived.
    """
    assert "failed" not in FUNDED_STATUSES

    bundle = mapping.bundle_from_payload(payload, bank_lines=bank)
    result = run_b2(bundle)

    matched = {pid for m in result.matches_for_leg(1) for pid in m.right_ids}
    assert "pay_QpA2bR7wLdY005" not in matched

    exception = next(e for e in result.exceptions if e.entity_id == "order_QpA1xK3mNvT005")
    assert "none funded" in exception.reason


def test_a_retry_after_a_decline_is_not_ambiguous():
    """Two rows for one order, one of them dead, is a recovery not a conflict."""
    from recoagent.schemas import Order, PGPayment, ReconResult, SourceBundle
    from recoagent.validate import Tolerance
    from recoagent.legs import leg1

    when = datetime(2026, 8, 28, tzinfo=timezone.utc)
    order = Order("order_1", "cust", "INV-1", 100000, "INR", when)
    dead = PGPayment("pay_dead", "order_1", 100000, 0, 0, "card_domestic", "failed", None, when)
    good = PGPayment("pay_good", "order_1", 100000, 2000, 360, "card_domestic", "captured", None, when)

    result = ReconResult(rung="test")
    leg1.match(
        SourceBundle((order,), (dead, good), (), (), ()),
        Tolerance.calibrated(), result,
    )

    assert len(result.matches) == 1
    assert result.matches[0].right_ids == ("pay_good",)
    assert not result.exceptions


# ── signature verification ──────────────────────────────────────────────────


SECRET = "whsec_test_only_not_a_real_secret"
BODY = json.dumps(
    {
        "entity": "event",
        "event": "payment.captured",
        "contains": ["payment"],
        "payload": {"payment": {"entity": {"id": "pay_QpA2bR7wLdY001", "amount": 249900}}},
        "created_at": 1756290130,
    },
    separators=(",", ":"),
).encode()


def sign(body: bytes, secret: str = SECRET) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_a_valid_signature_verifies():
    webhook.verify_signature(BODY, sign(BODY), SECRET)


@pytest.mark.parametrize(
    "signature",
    ["", "deadbeef", sign(BODY, "the-wrong-secret")],
    ids=["missing", "garbage", "wrong-secret"],
)
def test_a_bad_signature_is_refused(signature):
    with pytest.raises(webhook.SignatureError):
        webhook.verify_signature(BODY, signature, SECRET)


def test_the_signature_covers_the_raw_bytes_not_the_reparsed_json():
    """Verifying a re-serialised body is the classic way to verify nothing.

    `json.dumps(json.loads(body))` changes separators and key order. If the
    implementation had round-tripped before hashing, this signature -- computed
    over the bytes Razorpay actually sent -- would fail to verify its own body.
    """
    reserialised = json.dumps(json.loads(BODY)).encode()
    assert reserialised != BODY
    webhook.verify_signature(BODY, sign(BODY), SECRET)
    with pytest.raises(webhook.SignatureError):
        webhook.verify_signature(reserialised, sign(BODY), SECRET)


# ── idempotency ─────────────────────────────────────────────────────────────


def test_a_redelivered_event_is_stored_once(tmp_path):
    log = webhook.EventLog(tmp_path / "events.db")
    first = log.record(BODY, sign(BODY), SECRET)
    second = log.record(BODY, sign(BODY), SECRET)

    assert first.replay is False and second.replay is True
    assert first.event_id == second.event_id
    assert log.count() == 1        # one event
    assert log.deliveries() == 2   # two deliveries of it
    log.close()


def test_a_forged_event_never_reaches_storage(tmp_path):
    log = webhook.EventLog(tmp_path / "events.db")
    with pytest.raises(webhook.SignatureError):
        log.record(BODY, "deadbeef", SECRET)
    assert log.count() == 0
    assert log.deliveries() == 0
    log.close()


def test_the_webhook_log_rebuilds_the_same_shape_as_a_pull(tmp_path):
    """Both ingestion routes have to produce one book, or only one is tested."""
    log = webhook.EventLog(tmp_path / "events.db")
    log.record(BODY, sign(BODY), SECRET)

    refund = json.dumps(
        {
            "event": "refund.processed",
            "payload": {"refund": {"entity": {
                "id": "rfnd_QpA3cS9xMeZ001", "amount": 640000,
                "payment_id": "pay_QpA2bR7wLdY006", "created_at": 1756303200,
            }}},
        },
        separators=(",", ":"),
    ).encode()
    log.record(refund, sign(refund), SECRET)

    rebuilt = log.payload()
    assert [p["id"] for p in rebuilt["payments"]] == ["pay_QpA2bR7wLdY001"]
    assert [r["id"] for r in rebuilt["refunds"]] == ["rfnd_QpA3cS9xMeZ001"]
    log.close()


# ── transport ───────────────────────────────────────────────────────────────


class FakeOpener:
    """Answers `urlopen` from a script of responses. Records every URL asked."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.urls: list[str] = []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        item = self.responses.pop(0)
        if isinstance(item, int):
            raise urllib.error.HTTPError(
                request.full_url, item, "boom", {}, BytesIO(b'{"error":"x"}')
            )
        body = json.dumps(item).encode()

        class Response:
            def read(self_inner):
                return body

            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_):
                return False

        return Response()


def client_with(monkeypatch, responses, **kw):
    opener = FakeOpener(responses)
    monkeypatch.setattr(api.urllib.request, "urlopen", opener)
    client = api.Client(
        api.Credentials("rzp_test_abc", "shh"),
        sleep=lambda _s: None,
        **kw,
    )
    return client, opener


def test_pagination_stops_on_a_short_page(monkeypatch):
    """`count` is this page's size, not the total. A short page is the end."""
    full = {"entity": "collection", "count": api.PAGE_SIZE,
            "items": [{"id": f"pay_{i}"} for i in range(api.PAGE_SIZE)]}
    short = {"entity": "collection", "count": 2, "items": [{"id": "pay_a"}, {"id": "pay_b"}]}
    client, opener = client_with(monkeypatch, [full, short])

    rows = client.payments()
    assert len(rows) == api.PAGE_SIZE + 2
    assert len(opener.urls) == 2
    assert "skip=100" in opener.urls[1]


def test_a_rate_limit_is_retried_and_a_bad_request_is_not(monkeypatch):
    client, opener = client_with(
        monkeypatch, [429, {"entity": "collection", "count": 0, "items": []}]
    )
    assert client.payments() == []
    assert len(opener.urls) == 2

    client, _ = client_with(monkeypatch, [400])
    with pytest.raises(api.RazorpayError) as exc:
        client.payments()
    assert exc.value.status == 400


def test_a_live_key_is_refused(monkeypatch):
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_live_abcdefgh")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "secret")
    with pytest.raises(RuntimeError, match="not a test key"):
        api.Credentials.from_env()


def test_credentials_never_appear_in_an_error_message(monkeypatch):
    client, _ = client_with(monkeypatch, [400])
    with pytest.raises(api.RazorpayError) as exc:
        client.payments()
    assert "shh" not in str(exc.value)
    assert "Basic" not in str(exc.value)


def test_every_response_is_recorded_before_it_is_parsed(monkeypatch, tmp_path):
    recorder = api.Recorder(directory=tmp_path / "raw")
    client, _ = client_with(
        monkeypatch,
        [{"entity": "collection", "count": 1, "items": [{"id": "pay_a"}]}],
        recorder=recorder,
    )
    client.payments()

    written = list((tmp_path / "raw").glob("*.json"))
    assert [p.name for p in written] == ["payments.000.json"]
    assert json.loads(written[0].read_text())["items"] == [{"id": "pay_a"}]


# ── end to end ──────────────────────────────────────────────────────────────


def test_a_recorded_pull_reconciles_against_a_bank_statement(payload, bank):
    bundle = mapping.bundle_from_payload(payload, bank_lines=bank)
    result = run_b2(bundle)

    assert len(result.matches_for_leg(2)) == 1
    assert result.matches_for_leg(2)[0].proof.residual_paise == 0

    # One exception, and it is the declined card -- not a silent match.
    assert [e.entity_id for e in result.exceptions] == ["order_QpA1xK3mNvT005"]

    # The under-capture closed, and the money it left behind is carried rather
    # than absorbed: Rs 1,500 between what was authorised and what was taken.
    carried = [m for m in result.matches if m.variance_paise]
    assert len(carried) == 1
    assert carried[0].variance_paise == -150000
