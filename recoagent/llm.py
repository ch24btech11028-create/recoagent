"""One place to choose a model.

Every LLM-backed part of this system -- the reconciliation proposer, the Q&A
agent -- takes a `Chat` and nothing else. Swapping models is a string:

    chat = client_for("nvidia/nemotron-3-ultra-550b-a55b")
    chat = client_for("anthropic/claude-opus-5")
    chat = client_for("gemini/gemini-3.6-flash")
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

from . import trace
from .env import require_key

_log = trace.logger("llm")

#: What runs when nobody says otherwise. `default_model()` is the thing to
#: call; this is only the last fallback.
FALLBACK_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

#: A merchant chooses a model once, not per command. `RECOAGENT_MODEL` in the
#: environment or in `.env` is read by every entry point, and an explicit
#: `--model` still wins over it -- configuration should be overridable from the
#: command line or it is a cage rather than a default.
MODEL_ENV = "RECOAGENT_MODEL"


def default_model(fallback: str | None = None) -> str:
    """The configured model: RECOAGENT_MODEL, else the caller's own default.

    Read at call time rather than import time. An argparse default evaluated
    when the module loads cannot see a `.env` that has not been read yet, which
    made the environment variable work from the shell and silently not from the
    file most people would actually put it in.
    """
    chosen = os.environ.get(MODEL_ENV)
    if not chosen:
        try:
            from .env import load_env

            chosen = load_env().get(MODEL_ENV) or os.environ.get(MODEL_ENV)
        except Exception:
            chosen = None
    return chosen or fallback or FALLBACK_MODEL


#: Kept so existing imports keep working. Prefer `default_model()`.
DEFAULT_MODEL = FALLBACK_MODEL

#: Reasoning toggles by model family. Absent means "send nothing", which is the
#: correct default for a host we have not characterised.
EXTRA_BODY_PRESETS: dict[str, dict] = {
    "nvidia/nemotron-3-super-120b-a12b": {"chat_template_kwargs": {"enable_thinking": True}},
    "nvidia/nemotron-3-ultra-550b-a55b": {"chat_template_kwargs": {"enable_thinking": True}},
    "deepseek-ai/deepseek-v4-flash-0731": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}

#: Endpoint defaults per provider prefix: (base_url, key env var, keep_prefix).
#:
#: `keep_prefix` is not a detail. On NIM the vendor prefix is genuinely part of
#: the model id -- `nvidia/nemotron-3-ultra-550b-a55b` is what the endpoint
#: expects -- while Google's OpenAI-compatible surface wants `gemini-3.6-flash`
#: and 404s on `gemini/gemini-3.6-flash`. Getting it wrong produces a not-found
#: from the far end, which reads like a typo in the model name rather than a
#: bug here.
PROVIDERS = {
    "nvidia": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", True),
    "deepseek-ai": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY", True),
    "gemini": (
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        False,
    ),
    "anthropic": (None, "ANTHROPIC_API_KEY", False),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", False),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", False),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", True),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY", True),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY", False),
    # Local hosts. `None` for the key variable means no credential is required
    # -- which is the point of them: a merchant's settlement book never leaves
    # the machine, and the agent tier still runs. Ports are each project's
    # documented default.
    "ollama": ("http://localhost:11434/v1", None, False),
    "lmstudio": ("http://localhost:1234/v1", None, False),
    "vllm": ("http://localhost:8000/v1", None, False),
}

#: Providers that need no API key. Used for reporting, not for control flow --
#: the control flow is `key_env is None`.
LOCAL_PROVIDERS = frozenset({"ollama", "lmstudio", "vllm"})

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


def is_retryable(exc: Exception) -> bool:
    """Shared with the agent tier, which used to keep its own narrower copy."""
    if type(exc).__name__ in _RETRYABLE:
        return True
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


def _openai_client(*, base_url: str, api_key_env: str, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        # "No module named 'openai'" is accurate and useless to someone who has
        # just cloned this. The deterministic rungs need no dependencies at all,
        # so nothing has told them the agent tier does -- say what to install
        # instead of leaking the traceback.
        raise RuntimeError(
            "The agent tier needs the OpenAI SDK, which is not installed. "
            "The deterministic rungs (B0, B2) do not need it, which is why "
            "it is absent by default.\n"
            "    pip install openai\n"
            "or, for every optional tier:\n"
            "    pip install -r requirements.txt"
        ) from exc

    # A local host wants no credential, but the SDK insists on a non-empty
    # string, so send a visibly fake one rather than leaving it unset. Passing
    # None here fails inside the SDK with an error about OPENAI_API_KEY, which
    # is exactly the wrong thing to tell someone running Ollama.
    key = require_key(api_key_env) if api_key_env else "no-key-required"
    return OpenAI(base_url=base_url, api_key=key, timeout=timeout)


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
        # Built on the first send, not here. A run whose every request is
        # already answered on disk sends nothing, and it should not need a
        # credential to replay answers it already has -- that is what makes a
        # published live-model result checkable by a reader who has no key.
        self._client = client
        self._build = None if client is not None else lambda: _openai_client(
            base_url=base_url, api_key_env=api_key_env, timeout=timeout
        )
        self._model = model
        self._temperature = temperature
        self._max_retries = max_retries
        self._extra_body = (
            extra_body if extra_body is not None else EXTRA_BODY_PRESETS.get(model)
        )
        self.label = model

    def check_ready(self) -> None:
        """Do now what the first send would do, so a caller can ask in advance.

        Construction is lazy so a cached run needs no key; that would otherwise
        turn "is a model configured?" into a question nothing could answer
        until the first question had already failed in front of an operator.
        """
        if self._client is None:
            self._client = self._build()

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        self.check_ready()
        started = time.perf_counter()
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
                if attempt == self._max_retries or not is_retryable(exc):
                    # A warning, not an event: this is the outcome somebody
                    # needs to see on a run they were not watching, and it is
                    # the difference between "the model could not answer" and
                    # "the model was never asked".
                    trace.problem(
                        _log, "model.failed", model=self._model,
                        error=type(exc).__name__, attempts=attempt + 1,
                        seconds=f"{time.perf_counter() - started:.1f}",
                    )
                    return Reply(usage=usage, error=f"{type(exc).__name__}: {exc}")
                trace.event(
                    _log, "model.retry", model=self._model,
                    error=type(exc).__name__, attempt=attempt + 1,
                )
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
        trace.event(
            _log, "model.call", model=self._model, outcome="ok",
            tokens_in=usage.input_tokens, tokens_out=usage.output_tokens,
            seconds=f"{time.perf_counter() - started:.1f}",
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
        # Deferred to the first send, for the same reason the OpenAI-compatible
        # path defers it: naming a model must not require the SDK that serves
        # it. Otherwise `recoagent.llm` cannot list Anthropic among the
        # providers it supports without the SDK installed, and a merchant
        # comparing their options gets an import error instead of a menu.
        self._client = client
        self._timeout = timeout
        self._build = None if client is not None else self._make_client
        self._model = model
        self._effort = effort
        self.label = model

    def _make_client(self):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "The Anthropic path needs the Anthropic SDK, which is not "
                "installed.\n    pip install anthropic"
            ) from exc
        return anthropic.Anthropic(timeout=self._timeout)

    def check_ready(self) -> None:
        """Build the SDK client now, so a caller can ask before it matters."""
        if self._client is None:
            self._client = self._build()

    def send(self, system: str, user: str, *, max_tokens: int = 4000) -> Reply:
        usage = Usage()
        try:
            self.check_ready()
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

    def check_ready(self) -> None:
        """Nothing deferred here: construction already decided it."""

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
        base_url, key_env, keep_prefix = PROVIDERS[provider]
        model = spec if keep_prefix else spec.split("/", 1)[1]
        return OpenAICompatibleChat(model, base_url, key_env, **kw)

    raise ValueError(
        f"unknown provider in {spec!r}. Known: {', '.join(sorted(PROVIDERS))}, "
        "or use openai-compatible:<base_url>:<model>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Choosing a model, from the command line
# ─────────────────────────────────────────────────────────────────────────────


def describe() -> str:
    """Every provider this build can reach, and what each one needs."""
    rows = [
        "=" * 72,
        "  CHOOSING A MODEL",
        "=" * 72,
        "",
        f"  currently configured   {default_model()}",
        f"  set it once            {MODEL_ENV}=<provider>/<model>  (shell or .env)",
        "  override per command   --model <provider>/<model>",
        "",
        "  The deterministic rungs need no model at all. Everything below is",
        "  only for the tiers that ask one to explain a residual.",
        "",
        "-" * 72,
        f"  {'provider':<14}{'needs':<24}{'example'}",
        "-" * 72,
    ]
    examples = {
        "nvidia": "nvidia/nemotron-3-ultra-550b-a55b",
        "deepseek-ai": "deepseek-ai/deepseek-v4-flash-0731",
        "gemini": "gemini/gemini-3.6-flash",
        "anthropic": "anthropic/claude-opus-5",
        "openai": "openai/gpt-5.1",
        "groq": "groq/llama-3.3-70b-versatile",
        "openrouter": "openrouter/meta-llama/llama-3.3-70b-instruct",
        "together": "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "mistral": "mistral/mistral-large-latest",
        "ollama": "ollama/llama3.1",
        "lmstudio": "lmstudio/local-model",
        "vllm": "vllm/my-model",
    }
    for name in sorted(PROVIDERS):
        _, key_env, _ = PROVIDERS[name]
        needs = key_env if key_env else "nothing (runs locally)"
        rows.append(f"  {name:<14}{needs:<24}{examples.get(name, '')}")
    rows += [
        "-" * 72,
        "",
        "  Anything else that speaks the OpenAI protocol:",
        "    openai-compatible:<base_url>:<model>",
        "",
        "  Local hosts need no API key and no network. That is the answer to",
        "  'I do not want a settlement book leaving this machine' -- the",
        "  reconciliation never calls out at all, and with a local model neither",
        "  does the tier that explains what it could not match.",
        "",
        "  Check the configured model actually answers:",
        "    python3 -m recoagent.llm --check",
        "=" * 72,
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="recoagent.llm",
        description="Show which models this build can use, and test one.",
    )
    ap.add_argument("--model", help=f"override {MODEL_ENV} for this check")
    ap.add_argument("--check", action="store_true",
                    help="send one short prompt and report what came back")
    args = ap.parse_args(argv)

    spec = args.model or default_model()
    if not args.check:
        print(describe())
        return 0

    print(f"  model   {spec}")
    try:
        chat = client_for(spec)
    except ValueError as exc:
        print(f"  FAILED  {exc}")
        return 2
    try:
        chat.check_ready()
    except Exception as exc:
        print(f"  FAILED  {type(exc).__name__}: {exc}")
        return 2

    try:
        reply = chat.send(
            "Answer with one word.",
            "Reply with the single word: ready",
            max_tokens=16,
        )
    except Exception as exc:
        # A wrong model name, a dead local server and an expired key all land
        # here, and the message from the far end is more useful than anything
        # this layer could invent.
        print(f"  FAILED  {type(exc).__name__}: {exc}")
        return 2

    # `send` reports failure in the reply rather than raising -- deliberately,
    # so a batch of questions is not abandoned because one endpoint blinked.
    # A check that ignores that field reports a dead local server as healthy,
    # which is the one answer a merchant must never be given here.
    if reply.error:
        print(f"  FAILED  {reply.error}")
        return 2
    if not reply.text.strip():
        print("  FAILED  the model answered with nothing")
        return 2

    print(f"  replied {reply.text.strip()[:60]!r}")
    print(f"  tokens  {reply.usage.input_tokens} in / {reply.usage.output_tokens} out")
    print("  OK -- this model is usable for the agent and Q&A tiers.")
    return 0


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(main())
