"""The request budget, which on a free tier is the binding constraint.

These matter because of a real misreading they prevent. A run of the
categorisation tier against a key allowing 20 requests a day reported 19 of 20
rows as `failed`, which reads as a verdict on the model. The model was never
asked: the quota was already spent. A report that cannot tell those apart is
worse than no report.
"""

from __future__ import annotations

import pytest

from recoagent.budget import BudgetExhausted, Cached, Throttled, budgeted, limits_for
from recoagent.categorize import rules
from recoagent.categorize.agent import ChatCategoriser, run_c2
from recoagent.llm import Reply, ScriptedChat
from recoagent.categorize.taxonomy import Category

from tests.test_categorize import reply, small_book


class Counting:
    """A Chat that answers everything and counts how often it was called."""

    label = "counting"

    def __init__(self, text: str = '{"category":"bank_charge","quote":"kind: mystery","confidence":0.9}'):
        self.text = text
        self.calls = 0

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        self.calls += 1
        return Reply(text=self.text)


def test_the_daily_budget_stops_the_run_rather_than_failing_the_rows():
    inner = Counting()
    chat = Throttled(inner, rpm=600, rpd=2, sleep=lambda _s: None)

    chat.send("s", "a")
    chat.send("s", "b")
    with pytest.raises(BudgetExhausted, match="unmeasured, not unresolved"):
        chat.send("s", "c")
    assert inner.calls == 2


def test_rows_past_the_budget_are_reported_as_never_asked():
    book = small_book()
    ledger = rules.Ledger()
    rules.run_c0(book, ledger)
    inner = Counting()
    report = run_c2(
        book, ledger,
        ChatCategoriser(Throttled(inner, rpm=600, rpd=1, sleep=lambda _s: None)),
    )

    assert report.not_asked == 1
    assert report.failed == 0          # never conflated
    assert inner.calls == 1
    # The unasked row still reaches a human, and says why.
    unasked = ledger.assignments["bl_1"]
    assert unasked.category is Category.NEEDS_REVIEW
    assert "budget spent" in unasked.evidence


def test_a_cache_hit_costs_no_request(tmp_path):
    """Cache outside, throttle inside: a repeat must not spend the quota."""
    inner = Counting()
    throttled = Throttled(inner, rpm=600, rpd=1, sleep=lambda _s: None)
    chat = Cached(throttled, tmp_path)

    first = chat.send("system", "user")
    second = chat.send("system", "user")

    assert first.text == second.text
    assert inner.calls == 1
    assert throttled.spent == 1        # the second never reached the throttle
    assert (chat.hits, chat.misses) == (1, 1)


def test_a_changed_prompt_is_a_different_key(tmp_path):
    inner = Counting()
    chat = Cached(Throttled(inner, rpm=600, rpd=10, sleep=lambda _s: None), tmp_path)
    chat.send("system v1", "user")
    chat.send("system v2", "user")
    assert inner.calls == 2


def test_a_cache_hit_reports_no_token_usage(tmp_path):
    """Replaying spent tokens would inflate every cost figure on a re-run."""
    chat = Cached(ScriptedChat(["hello", "hello"]), tmp_path)
    assert chat.send("s", "u").usage.calls == 1
    assert chat.send("s", "u").usage.calls == 0


def test_an_error_is_never_cached(tmp_path):
    class Flaky:
        label = "flaky"

        def __init__(self):
            self.calls = 0

        def send(self, system, user, *, max_tokens=4000):
            self.calls += 1
            return Reply(error="rate limited") if self.calls == 1 else Reply(text="ok")

    inner = Flaky()
    chat = Cached(inner, tmp_path)
    assert chat.send("s", "u").error == "rate limited"
    assert chat.send("s", "u").text == "ok"
    assert inner.calls == 2


def test_rate_limiting_spaces_requests():
    waits: list[float] = []
    clock = [0.0]
    chat = Throttled(
        Counting(), rpm=5, rpd=10,
        sleep=lambda s: (waits.append(s), clock.__setitem__(0, clock[0] + s)),
        now=lambda: clock[0],
    )
    for _ in range(3):
        chat.send("s", "u")
    # 5 per minute is one every twelve seconds.
    assert waits and all(abs(w - 12.0) < 0.01 for w in waits)


def test_known_free_tier_limits_are_data_not_guesses():
    assert limits_for("gemini-3.6-flash") == (5, 20)
    assert limits_for("gemini-3.5-flash-lite") == (15, 500)
    # An unknown model gets the cautious default rather than no limit at all.
    assert limits_for("something-nobody-has-measured") == (5, 20)


def test_budgeted_puts_the_cache_outside_the_throttle(tmp_path):
    chat = budgeted(Counting(), rpm=600, rpd=3, cache=tmp_path)
    assert isinstance(chat, Cached)
    assert isinstance(chat.inner, Throttled)
