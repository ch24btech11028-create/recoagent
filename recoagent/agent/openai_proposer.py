"""A proposer for any OpenAI-compatible endpoint.

Written for NVIDIA NIM (`integrate.api.nvidia.com`) but the shape is generic:
anything speaking the OpenAI chat-completions protocol works by changing
`base_url` and `model`.

**Why JSON output rather than tool calling.** Tool calling is the cleaner
protocol and it is what `AnthropicProposer` uses. It does not work on this
endpoint: asked to call a function, the model returns `tool_calls: None` and
writes a malformed imitation into the content instead --

    [[ \\n  {\\n    "name": "propose", ...\\n  }\\n]

-- with mismatched brackets. Plain JSON mode is reliable on the same model, so
that is what this proposer uses.

That failure is worth keeping in mind rather than working around silently: an
endpoint whose structured-output path is broken is exactly the kind of thing
that produces confident nonsense downstream. Everything here is validated by
`_parse_tool_call`, the same strict parser the Anthropic path uses, and anything
that does not parse becomes a `ProposerError` rather than a guess.
"""

from __future__ import annotations

import json
import os
import random
import re
import time

from .contracts import Proposal, ProposerError, Usage
from .evidence import EvidencePacket
from .proposer import SYSTEM_PROMPT, _parse_tool_call

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

#: Reasoning toggles are not standardised across OpenAI-compatible hosts -- each
#: model family names them differently in `chat_template_kwargs`. Presets keep
#: that mess in one place instead of spreading it through call sites.
EXTRA_BODY_PRESETS: dict[str, dict] = {
    "nvidia/nemotron-3-super-120b-a12b": {
        "chat_template_kwargs": {"enable_thinking": True}
    },
    "nvidia/nemotron-3-ultra-550b-a55b": {
        "chat_template_kwargs": {"enable_thinking": True}
    },
    "deepseek-ai/deepseek-v4-flash-0731": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


def preset_for(model: str) -> dict | None:
    return EXTRA_BODY_PRESETS.get(model)

#: Appended to the shared system prompt. The response contract has to be stated
#: explicitly because there is no schema enforcement on this path -- the model
#: is being asked to honour a format rather than being constrained to one.
JSON_CONTRACT = """

Reply with a single JSON object and nothing else. No prose, no markdown fence.

To explain the residual:
{"action": "propose",
 "rows": [{"label": "...", "amount_paise": -123456, "rationale": "..."}],
 "reason": "...",
 "confidence": 0.0}

To decline:
{"action": "decline", "reason": "..."}

amount_paise must be a whole integer. Deductions are negative. The rows must \
sum to exactly the residual."""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict:
    """Pull the JSON object out of a response, tolerating fences and stray prose.

    Tolerant about *packaging*, strict about *content*: whatever comes out still
    goes through the same validation as the Anthropic path, so a well-formed
    wrapper around nonsense is rejected exactly as a malformed wrapper is.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if not match:
        raise ValueError(f"no JSON object in response: {text[:160]!r}")
    return json.loads(match.group(0))


#: Names rather than classes: the SDK exception hierarchy differs across
#: versions and hosts, and a proposer must never crash on an unexpected one.
_RETRYABLE = ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError")


def _is_retryable(exc: Exception) -> bool:
    if type(exc).__name__ in _RETRYABLE:
        return True
    status = getattr(exc, "status_code", None)
    return status == 429 or (isinstance(status, int) and status >= 500)


class OpenAICompatibleProposer:
    """Calls an OpenAI-compatible chat endpoint once per attempt.

    Every failure -- transport, timeout, unparseable output, a response that
    parses but violates the contract -- becomes a `ProposerError`. Nothing
    raises out of `propose`, because the tier treats a proposer that failed as
    an ordinary outcome rather than an exception to handle.
    """

    def __init__(
        self,
        client=None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "NVIDIA_API_KEY",
        max_tokens: int = 8000,
        temperature: float = 0.0,
        timeout: float = 240.0,
        extra_body: dict | None = None,
        max_retries: int = 3,
    ) -> None:
        if client is None:
            from openai import OpenAI  # lazy: the core stays dependency-free

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{api_key_env} is not set. Export it in your shell rather "
                    "than passing the key as a literal."
                )
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        # Explicit wins; otherwise fall back to the preset for this model, and
        # to nothing at all for a host we have no preset for. Sending another
        # family's reasoning flags is worse than sending none.
        self._extra_body = extra_body if extra_body is not None else preset_for(model)
        self._max_retries = max_retries

    @property
    def label(self) -> str:
        return self._model

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]:
        usage = Usage()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + JSON_CONTRACT},
            {
                "role": "user",
                "content": json.dumps(packet.to_dict(), indent=2, sort_keys=True),
            },
        ]

        # Running cases concurrently is what makes a shared free endpoint push
        # back, so the retry lives here rather than in the tier: rate limiting
        # is a property of the transport, and a 429 is not a failed case.
        last: Exception | None = None
        response = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                    extra_body=self._extra_body,
                )
                break
            except Exception as exc:
                last = exc
                if attempt == self._max_retries or not _is_retryable(exc):
                    return (
                        ProposerError("transport", f"{type(exc).__name__}: {exc}"),
                        usage,
                    )
                # Jittered backoff: without the jitter, eight workers that hit
                # the limit together would retry together and hit it again.
                time.sleep(min(2 ** attempt, 20) + random.uniform(0, 1.5))

        if response is None:
            return ProposerError("transport", f"exhausted retries: {last}"), usage

        if response.usage:
            usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)

        choice = response.choices[0] if response.choices else None
        if choice is None:
            return ProposerError("malformed", "no choices in response"), usage

        content = choice.message.content or ""
        if not content.strip():
            # Reasoning models can spend the whole budget thinking and return
            # nothing. That is a failure to answer, not an empty answer.
            return (
                ProposerError(
                    "malformed",
                    f"empty content (finish_reason={choice.finish_reason})",
                ),
                usage,
            )

        try:
            payload = _extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            return ProposerError("malformed", str(exc)), usage

        action = payload.get("action")
        if action == "decline":
            payload = {"reason": payload.get("reason", "declined")}
            return _parse_tool_call("flag_for_human", payload), usage
        if action != "propose":
            return (
                ProposerError("malformed", f"unknown action {action!r}"),
                usage,
            )

        try:
            return _parse_tool_call("propose_hypothesis", payload), usage
        except (KeyError, TypeError, ValueError) as exc:
            return ProposerError("malformed", f"{type(exc).__name__}: {exc}"), usage
