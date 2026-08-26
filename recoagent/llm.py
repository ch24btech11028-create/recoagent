"""One place to choose a model.

Every LLM-backed part of this system -- the reconciliation proposer, the Q&A
agent -- takes a `Chat` and nothing else. Swapping models is a string:

    chat = client_for("nvidia/nemotron-3-ultra-550b-a55b")
    chat = client_for("anthropic/claude-opus-5")
    chat = client_for("openai-compatible:http://localhost:8000/v1:my-model")

Why a shim rather than using the SDKs directly at each call site: the two things
that differ between hosts are both small and both easy to get wrong in a way
that fails silently. Reasoning toggles live under different keys in
`chat_template_kwargs` per model family, and sending one family's flags to
another is worse than sending none. And structured output is not portable --
native tool calling is broken on NVIDIA NIM, which returns `tool_calls: None`
and writes malformed pseudo-JSON into the content instead. Every caller here
gets plain text back and validates it itself, because that is the only contract
all of these hosts actually honour.

Nothing in this module is imported unless an LLM tier actually runs. The
deterministic core has no dependencies and `tests/test_no_dependencies.py`
asserts it.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Protocol

from .env import require_key

DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

#: Reasoning toggles by model family. Absent means "send nothing", which is the
#: correct default for a host we have not characterised.
EXTRA_BODY_PRESETS: dict[str, dict] = {
    "nvidia/nemotron-3-super-120b-a12b": {"chat_template_kwargs": {"enable_thinking": True}},
    "nvidia/nemotron-3-ultra-550b-a55b": {"chat_template_kwargs": {"enable_thinking": True}},
    "deepseek-ai/deepseek-v4-flash-0731": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}

#: Endpoint defaults per provider prefix.
PROVIDERS = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "deepseek-ai": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    "anthropic": (None, "ANTHROPIC_API_KEY"),
}

_RETRYABLE = (
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "APIStatusError",
)


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, inp: int, out: int) -> None:
        self.calls += 1
        self.input_tokens += inp
        self.output_tokens += out

    def merge(self, other: Usage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def cost_usd(self, per_min: float, per_mout: float) -> float:
        return self.input_tokens / 1e6 * per_min + self.output_tokens / 1e6 * per_mout


@dataclass
class Reply:
    """What every backend returns. Text, or a reason it could not answer."""

    text: str = ""
    usage: Usage = field(default_factory=Usage)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


class Chat(Protocol):
    """Send a system prompt and a user message, get text back. Never raises."""

    label: str

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply: ...


def _is_retryable(exc: Exception) -> bool:
    if type(exc).__name__ in _RETRYABLE:
        return True
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


class OpenAICompatibleChat:
    """Any host speaking the OpenAI chat-completions protocol."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        *,
        temperature: float = 0.0,
        timeout: float = 300.0,
        max_retries: int = 3,
        extra_body: dict | None = None,
        client=None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            key = require_key(api_key_env)
            client = OpenAI(base_url=base_url, api_key=key, timeout=timeout)
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._extra_body = (
            extra_body if extra_body is not None else EXTRA_BODY_PRESETS.get(model)
        )
        self.label = model

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        usage = Usage()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        for attempt in range(self._max_retries + 1):
            try:
                r = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=self._temperature,
                    extra_body=self._extra_body,
                )
                break
            except Exception as exc:
                if attempt == self._max_retries or not _is_retryable(exc):
                    return Reply(usage=usage, error=f"{type(exc).__name__}: {exc}")
                # Jitter matters once callers run concurrently: without it, every
                # worker that hits the limit retries in lockstep and hits it again.
                time.sleep(min(2**attempt, 20) + random.uniform(0, 1.5))
        else:  # pragma: no cover - loop always breaks or returns
            return Reply(usage=usage, error="exhausted retries")

        if r.usage:
            usage.add(r.usage.prompt_tokens, r.usage.completion_tokens)
        choice = r.choices[0] if r.choices else None
        if choice is None:
            return Reply(usage=usage, error="no choices in response")
        text = choice.message.content or ""
        if not text.strip():
            # Reasoning models can spend the whole budget thinking and return
            # nothing. That is a failure to answer, not an empty answer.
            return Reply(
                usage=usage, error=f"empty content (finish={choice.finish_reason})"
            )
        return Reply(text=text, usage=usage)


class AnthropicChat:
    """Claude via the Anthropic SDK."""

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        effort: str = "high",
        timeout: float = 300.0,
        client=None,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout)
        self._client = client
        self._model = model
        self._effort = effort
        self.label = model

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        usage = Usage()
        try:
            r = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:
            return Reply(usage=usage, error=f"{type(exc).__name__}: {exc}")

        usage.add(r.usage.input_tokens, r.usage.output_tokens)
        if getattr(r, "stop_reason", None) == "refusal":
            return Reply(usage=usage, error="model declined for safety reasons")
        text = "".join(b.text for b in r.content if b.type == "text")
        if not text.strip():
            return Reply(usage=usage, error="no text in response")
        return Reply(text=text, usage=usage)


class ScriptedChat:
    """Replays canned replies. Tests only, so no path costs an API call."""

    def __init__(self, replies, label: str = "scripted") -> None:
        self._replies = replies
        self._i = 0
        self.label = label
        self.seen: list[tuple[str, str]] = []

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        self.seen.append((system, user))
        usage = Usage()
        usage.add(len(system) // 4, 50)
        if callable(self._replies):
            return Reply(text=self._replies(system, user), usage=usage)
        if self._i >= len(self._replies):
            return Reply(usage=usage, error="script exhausted")
        text = self._replies[self._i]
        self._i += 1
        return Reply(text=text, usage=usage)


def client_for(spec: str = DEFAULT_MODEL, **kw) -> Chat:
    """Build a chat client from a model string.

    Accepts `provider/model` for known providers, or an explicit
    `openai-compatible:<base_url>:<model>` for anything self-hosted.
    """
    if spec.startswith("openai-compatible:"):
        # rsplit, not split: the URL contains colons of its own (the scheme, and
        # a port). Splitting from the left turned
        # "openai-compatible:http://localhost:8000/v1:my-model" into base_url
        # "http" and model "//localhost:8000/v1:my-model", which then failed at
        # the endpoint rather than here.
        rest = spec[len("openai-compatible:"):]
        if ":" not in rest:
            raise ValueError(
                "expected openai-compatible:<base_url>:<model>, "
                f"got {spec!r} with no model after the URL"
            )
        base_url, model = rest.rsplit(":", 1)
        if not base_url or not model:
            raise ValueError(f"could not read a base_url and model from {spec!r}")
        return OpenAICompatibleChat(
            model, base_url, kw.pop("api_key_env", "OPENAI_API_KEY"), **kw
        )

    provider = spec.split("/", 1)[0]
    if provider == "anthropic":
        return AnthropicChat(spec.split("/", 1)[1], **kw)
    if provider in PROVIDERS:
        base_url, key_env = PROVIDERS[provider]
        return OpenAICompatibleChat(spec, base_url, key_env, **kw)

    raise ValueError(
        f"unknown provider in {spec!r}. Known: {', '.join(sorted(PROVIDERS))}, "
        "or use openai-compatible:<base_url>:<model>"
    )
