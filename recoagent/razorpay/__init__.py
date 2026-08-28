"""Razorpay test mode as a source of truth.

Every number this repository publishes so far was measured on a book it
generated itself or on BenchRec. Both are answers to "is the matching right?".
Neither answers "does it run on Razorpay's own data shapes?", and in front of
Razorpay's own judges that is the question that matters.

This package pulls real API responses from test mode -- real field names, real
paise integers, real unix timestamps, real `pay_`/`order_`/`setl_` id prefixes
-- and maps them onto the same `SourceBundle` the generator produces. Nothing
downstream of `SourceBundle` knows or cares where the rows came from, which is
the whole point of having drawn that line in the first place.

    recoagent.razorpay.api        the HTTP client: auth, pagination, backoff
    recoagent.razorpay.mapping    Razorpay JSON -> SourceBundle
    recoagent.razorpay.webhook    signature verification and the event log
    recoagent.razorpay.run        the CLI that pulls a book and reconciles it
"""

from .mapping import bundle_from_payload
from .webhook import EventLog, verify_signature

__all__ = ["bundle_from_payload", "EventLog", "verify_signature"]
