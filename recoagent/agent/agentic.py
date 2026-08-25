"""A proposer that investigates before it answers.

Multi-turn: the model calls tools, reads results, and decides when it has
enough to propose or to decline. The tier's contract is unchanged -- it still
receives one `Proposal` and gates it -- so the agent loop lives entirely here.

Protocol is JSON over plain chat completions rather than native tool calling,
because native tool calling is broken on the NVIDIA NIM endpoint this was built
against: asked to call a function, the model returns `tool_calls: None` and
writes malformed pseudo-JSON into the content. JSON mode is reliable on the same
model and works on every OpenAI-compatible host, so it is the portable choice.

Turn budget is deliberately small. Each turn is a full round-trip on a reasoning
model -- roughly 35 seconds against NIM -- so an agent allowed to wander costs
minutes per exception and the tier is supposed to be cheaper than a human
looking at it. If the model has not formed a theory in a handful of tool calls,
declining is the better answer anyway.
"""

from __future__ import annotations

import json

from ..money import FeeSchedule
from ..schemas import BankLine, Settlement, SourceBundle
from ..validate import Tolerance
from .contracts import Proposal, ProposerError, Refusal, Usage
from .evidence import EvidencePacket
from .openai_proposer import DEFAULT_BASE_URL, DEFAULT_MODEL, _extract_json
from .proposer import SYSTEM_PROMPT, _parse_tool_call
from .tools import TOOL_GUIDE, ToolContext, execute

MAX_TURNS = 6

AGENT_CONTRACT = f"""

You investigate before answering. Each turn, reply with a single JSON object and
nothing else -- no prose, no markdown fence.

{TOOL_GUIDE}
Work the evidence. A good sequence is: form a theory from the summary, test it
with compute_fee_scenario or list_unlinked_rows, confirm the numbers with
check_hypothesis, then propose. You have {MAX_TURNS} turns; declining early is
better than proposing something you have not checked."""


class AgenticProposer:
    """Runs a bounded tool-calling loop against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        sources: SourceBundle,
        tol: Tolerance,
        fees: FeeSchedule | None = None,
        client=None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key_env: str = "NVIDIA_API_KEY",
        max_turns: int = MAX_TURNS,
        max_tokens: int = 4000,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        if client is None:
            import os

            from openai import OpenAI

            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"{api_key_env} is not set. Export it in your shell rather "
                    "than passing the key as a literal."
                )
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

        self._client = client
        self._model = model
        self._sources = sources
        self._tol = tol
        self._fees = fees or FeeSchedule.default()
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._temperature = temperature

        # Filled in per case by the tier before each `propose` call.
        self._line: BankLine | None = None
        self._settlement: Settlement | None = None

        #: Tool-call transcripts, keyed by bank line. Kept for the audit trail:
        #: how the agent reached an answer is as reviewable as the answer.
        self.transcripts: dict[str, list[dict]] = {}

    @property
    def label(self) -> str:
        return f"{self._model} (agentic, <={self._max_turns} turns)"

    def bind(self, line: BankLine, settlement: Settlement) -> None:
        """Point the tools at the case about to be investigated."""
        self._line = line
        self._settlement = settlement

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]:
        usage = Usage()
        if self._line is None or self._settlement is None:
            return ProposerError("transport", "proposer was not bound to a case"), usage

        ctx = ToolContext(
            sources=self._sources,
            line=self._line,
            settlement=self._settlement,
            residual_paise=packet.residual_paise,
            tol=self._tol,
            fees=self._fees,
        )
        trace: list[dict] = []
        self.transcripts[self._line.bank_line_id] = trace

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + AGENT_CONTRACT},
            {
                "role": "user",
                "content": json.dumps(packet.to_dict(), indent=2, sort_keys=True),
            },
        ]

        for turn in range(1, self._max_turns + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
            except Exception as exc:
                return ProposerError("transport", f"{type(exc).__name__}: {exc}"), usage

            if response.usage:
                usage.add(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

            choice = response.choices[0] if response.choices else None
            content = (choice.message.content or "") if choice else ""
            if not content.strip():
                return (
                    ProposerError(
                        "malformed",
                        f"empty content on turn {turn} "
                        f"(finish_reason={choice.finish_reason if choice else 'none'})",
                    ),
                    usage,
                )

            try:
                payload = _extract_json(content)
            except (ValueError, json.JSONDecodeError) as exc:
                return ProposerError("malformed", f"turn {turn}: {exc}"), usage

            # ── terminal moves ────────────────────────────────────────────
            action = payload.get("action")
            if action == "decline":
                trace.append({"turn": turn, "decline": payload.get("reason", "")})
                return _parse_tool_call(
                    "flag_for_human", {"reason": payload.get("reason", "declined")}
                ), usage
            if action == "propose":
                trace.append({"turn": turn, "propose": payload.get("reason", "")})
                try:
                    return _parse_tool_call("propose_hypothesis", payload), usage
                except (KeyError, TypeError, ValueError) as exc:
                    return (
                        ProposerError("malformed", f"{type(exc).__name__}: {exc}"),
                        usage,
                    )

            # ── tool call ─────────────────────────────────────────────────
            name = payload.get("tool")
            if not name:
                return (
                    ProposerError(
                        "malformed",
                        f"turn {turn}: neither an action nor a tool: {list(payload)}",
                    ),
                    usage,
                )

            result = execute(ctx, name, payload.get("args"))
            trace.append({"turn": turn, "tool": name, "args": payload.get("args")})

            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": json.dumps({"tool_result": result}, sort_keys=True),
                }
            )

        # Budget spent without a verdict. Declining is the honest outcome: the
        # model investigated and did not reach a conclusion, which is different
        # from failing and different from having nothing to say.
        return (
            Refusal(
                f"investigated for {self._max_turns} turns without reaching a "
                "conclusion"
            ),
            usage,
        )
