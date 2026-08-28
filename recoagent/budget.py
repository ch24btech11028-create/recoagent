"""Two wrappers around a `Chat`, for the constraint a free tier actually imposes.

The constraint is not cost. It is **requests per day**, and it is small enough
to change the design rather than the schedule. Measured on this project's own
key: Gemini 3.6 Flash allows 5 requests per minute and **20 per day**. The
categorisation tier has 20 rows to ask about. One run consumes the entire day's
quota, and the second run of the day -- the one after you fix a prompt -- gets
nothing but errors that look exactly like model failures.

That produced a genuinely misleading measurement before these existed: 19 of 20
rows came back `failed`, and the honest reading of the report was "the model
could not do this", when the true reading was "the model was never asked".

So:

**`Cached`** stores every reply on disk, keyed by the exact request. A re-run
after an unrelated change costs nothing, a published number can be regenerated
by someone else without a key, and -- the part that matters more -- a prompt
change is visibly a different key, so nobody can accidentally serve an old
answer for a new question.

**`Throttled`** spaces requests to a stated rate and stops the run when the
daily budget is gone, rather than converting the remaining rows into failures.
Stopping is the honest behaviour: a row nobody asked about is not a row the
model got wrong, and a report that cannot tell those apart is worthless.

Neither is specific to categorisation. Both take a `Chat` and return a `Chat`.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .llm import Chat, Reply, Usage

#: Free-tier limits observed on this project's own key, August 2026. Kept as
#: data rather than folded into a default: they are a property of somebody's
#: billing plan, not of the model, and they will be wrong for the next reader.
FREE_TIER = {
    "gemini-3.6-flash": (5, 20),
    "gemini-3.5-flash": (5, 20),
    "gemini-3-flash": (5, 20),
    "gemini-3.1-flash-lite": (15, 500),
    "gemini-3.5-flash-lite": (15, 500),
}


class BudgetExhausted(RuntimeError):
    """The daily request allowance is gone. Not a model failure."""


def limits_for(label: str, default: tuple[int, int] = (5, 20)) -> tuple[int, int]:
    """(requests per minute, requests per day) for a model label."""
    return FREE_TIER.get(label, default)


@dataclass
class Cached:
    """Serve from disk when the exact request has been asked before.

    The key covers the model label, the system prompt and the user message. It
    deliberately does not cover `max_tokens`: a reply that fit in a smaller
    budget is still the same reply, and including it would invalidate the cache
    on a change that cannot alter the answer.
    """

    inner: Chat
    directory: Path = Path("data/llm-cache")

    def __post_init__(self) -> None:
        self.label = self.inner.label
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def _path(self, system: str, user: str) -> Path:
        key = hashlib.sha256(
            json.dumps([self.label, system, user], sort_keys=True).encode()
        ).hexdigest()
        return self.directory / f"{key}.json"

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        path = self._path(system, user)
        if path.is_file():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                stored = None
            if stored is not None:
                self.hits += 1
                # Usage is zeroed on a hit rather than replayed. The tokens
                # were spent once; reporting them again would make a cached
                # run look as expensive as a live one and quietly inflate
                # every cost figure that is ever computed from a re-run.
                return Reply(text=stored["text"], usage=Usage())

        reply = self.inner.send(system, user, max_tokens=max_tokens)
        self.misses += 1
        # Only successes are cached. Caching an error would freeze a transient
        # rate limit into a permanent answer.
        if reply.ok:
            path.write_text(json.dumps({"text": reply.text}), encoding="utf-8")
        return reply


class Throttled:
    """Space requests to `rpm`, and refuse to exceed `rpd`.

    `stop_when_exhausted` is the important flag and it defaults to True. With
    it, the run ends and says how many rows were never asked about. Without it,
    every remaining row becomes a failure, and the report then reads as a
    verdict on the model rather than on the quota.
    """

    def __init__(
        self,
        inner: Chat,
        *,
        rpm: int = 5,
        rpd: int = 20,
        stop_when_exhausted: bool = True,
        sleep=time.sleep,
        now=time.monotonic,
    ) -> None:
        self.inner = inner
        self.label = inner.label
        self.rpm = max(1, rpm)
        self.rpd = max(0, rpd)
        self.stop_when_exhausted = stop_when_exhausted
        self._sleep = sleep
        self._now = now
        self._lock = threading.Lock()
        self._last = 0.0
        self.spent = 0

    @property
    def remaining(self) -> int:
        return max(0, self.rpd - self.spent)

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        with self._lock:
            if self.spent >= self.rpd:
                if self.stop_when_exhausted:
                    raise BudgetExhausted(
                        f"{self.label}: the daily allowance of {self.rpd} requests is "
                        "spent. The rows not yet asked about are unmeasured, not "
                        "unresolved -- rerun tomorrow, or point --model at a tier "
                        "with a larger daily quota."
                    )
                return Reply(error=f"daily budget of {self.rpd} requests exhausted")

            gap = 60.0 / self.rpm
            wait = gap - (self._now() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = self._now()
            self.spent += 1

        return self.inner.send(system, user, max_tokens=max_tokens)


def budgeted(
    chat: Chat,
    *,
    rpm: int | None = None,
    rpd: int | None = None,
    cache: Path | str | None = "data/llm-cache",
) -> Chat:
    """Cache outside, throttle inside.

    The ordering is the whole point: a cache hit must not consume a request
    from the daily budget, which it would if the throttle were on the outside.
    """
    default_rpm, default_rpd = limits_for(chat.label)
    throttled = Throttled(
        chat, rpm=rpm or default_rpm, rpd=default_rpd if rpd is None else rpd
    )
    if cache is None:
        return throttled
    return Cached(throttled, Path(cache))
