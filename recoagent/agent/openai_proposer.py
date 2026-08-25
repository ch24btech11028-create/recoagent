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
import re

from .contracts import Proposal, ProposerError, Usage
from .evidence import EvidencePacket
from .proposer import SYSTEM_PROMPT, _parse_tool_call

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"

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
        timeout: float = 120.0,
        enable_thinking: bool = True,
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
        self._enable_thinking = enable_thinking

    @property
    def label(self) -> str:
        return self._model

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]:
        usage = Usage()
        extra: dict = {}
        if self._enable_thinking:
            extra["chat_template_kwargs"] = {"enable_thinking": True}

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT + JSON_CONTRACT},
                    {
                        "role": "user",
                        "content": json.dumps(packet.to_dict(), indent=2, sort_keys=True),
                    },
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                extra_body=extra or None,
            )
        except Exception as exc:  # the SDK raises a wide family; all are the same to us
            return ProposerError("transport", f"{type(exc).__name__}: {exc}"), usage

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
