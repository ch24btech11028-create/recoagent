"""Proposers: the swappable thing that turns evidence into a hypothesis.

Three implementations, one interface.

`ScriptedProposer` returns whatever a test tells it to, including malformed
output and timeouts. It exists so every failure path in the tier is exercised
without a network call, and so the gate's behaviour under a confident wrong
answer is a unit test rather than an anecdote.

`NullProposer` always refuses. Running B3 with it must produce exactly the B2
numbers -- a control that proves the tier adds nothing on its own, so any lift
measured later belongs to the model rather than to plumbing.

`AnthropicProposer` calls Claude. It is the only part of B3 that needs a key,
and it is deliberately the thinnest piece: two tools, one call, no agent loop.
The model gets one job -- explain this residual or decline -- because a wider
tool surface would mean a wider blast radius for a wrong answer, and the gate
downstream cannot tell an elaborate wrong answer from a simple one.
"""

from __future__ import annotations

import json
from typing import Callable, Protocol

from .citations import CitedAdjustment, FeeVarianceClaim, FxClaim
from .contracts import Hypothesis, Proposal, ProposerError, Refusal, Usage
from .evidence import EvidencePacket

DEFAULT_MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are a reconciliation analyst for an Indian payment gateway. A bank credit \
has been matched to a settlement batch, but the money that arrived does not \
equal the sum of the rows the gateway linked to that batch. You are given the \
batch, its payment rows, the fee schedule, and the unlinked rows nearby.

Your only job is to explain the residual, or to decline.

The residual is `bank_credit.amount_paise` minus the re-derived total of the \
batch. A negative residual means less money arrived than the rows account for; \
a positive residual means more arrived.

You cannot state an amount. You can only cite evidence, and the system computes \
the money from it:

- {"type": "adjustment", "adjustment_id": "..."} -- an unlinked row that belongs \
to this batch. It must appear in nearby_unlinked_rows. The amount comes from that \
row, not from you.
- {"type": "fee_variance", "payment_ids": [...], "actual_mdr_bps": 240} -- these \
payments were charged at a different MDR than reported. The system recomputes the \
variance from the fee schedule.
- {"type": "fx", "payment_id": "...", "rate_pct_of_gross": -1.6} -- an \
international payment converted at a rate the report does not carry.

Cite evidence that exists. A citation the system cannot resolve is rejected \
outright; it is not rounded into a partial answer.

Two explanations are common and neither can be found by arithmetic search alone:

1. A mid-cycle fee repricing. The settlement report shows the fee at the \
published schedule, but the deduction that actually happened used a different \
MDR. Compare `fee_paise` against `fee_at_schedule_paise` on each payment, and \
remember GST is charged on the fee. Only methods with a non-zero MDR can carry \
a fee variance -- UPI and RuPay debit are zero-rated by regulation.

2. An international payment converted at a rate the report does not carry. Look \
for payments with a non-INR currency or an `fx_rate`, and consider a small \
slippage against the reported figure.

Decline with flag_for_human when the evidence does not support a specific \
explanation. Declining is a good outcome. A confident wrong answer is the \
worst outcome available to you: your proposal is checked against the ledger \
arithmetic and a proposal that does not close is discarded, so guessing wastes \
the attempt and tells the operator nothing.

State your confidence honestly. It is recorded and audited against whether the \
arithmetic actually closed."""

PROPOSE_TOOL = {
    "name": "propose_hypothesis",
    "description": (
        "Offer a set of rows that explain the residual. The amounts must sum to "
        "the residual exactly. This is a proposal, not a decision: it is checked "
        "against the ledger and discarded if it does not close."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "citations": {
                "type": "array",
                "description": (
                    "Evidence that already exists. You cannot state an amount; the "
                    "system computes every rupee from what you cite."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["adjustment", "fee_variance", "fx"]},
                        "adjustment_id": {"type": "string"},
                        "payment_ids": {"type": "array", "items": {"type": "string"}},
                        "actual_mdr_bps": {"type": "integer"},
                        "payment_id": {"type": "string"},
                        "rate_pct_of_gross": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["type"],
                },
            },
            "reason": {
                "type": "string",
                "description": "One or two sentences an operator could act on.",
            },
            "confidence": {
                "type": "number",
                "description": "0.0 to 1.0. Honest, not optimistic.",
            },
        },
        "required": ["citations", "reason", "confidence"],
    },
}

FLAG_TOOL = {
    "name": "flag_for_human",
    "description": (
        "Decline to explain this residual. Use when the evidence does not "
        "support a specific explanation. This is a good outcome, not a failure."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reason": {
                "type": "string",
                "description": "What is missing or ambiguous, for the operator.",
            }
        },
        "required": ["reason"],
    },
}


class Proposer(Protocol):
    """Turn an evidence packet into a proposal. Must never raise."""

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]: ...


class NullProposer:
    """Always declines. The control condition for measuring B3's real lift."""

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]:
        return Refusal("no proposer configured"), Usage()


class ScriptedProposer:
    """Returns pre-set proposals. For tests, and only for tests.

    `script` may be a list consumed in order, or a callable taking the packet.
    Anything a real proposer could do -- including returning a confident answer
    that is wrong -- can be scripted here, which is the point.
    """

    def __init__(
        self,
        script: list[Proposal] | Callable[[EvidencePacket], Proposal],
        usage_per_call: Usage | None = None,
    ) -> None:
        self._script = script
        self._index = 0
        self._usage_per_call = usage_per_call or Usage(
            calls=1, input_tokens=2000, output_tokens=300
        )

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]:
        usage = Usage(
            calls=1,
            input_tokens=self._usage_per_call.input_tokens,
            output_tokens=self._usage_per_call.output_tokens,
        )
        if callable(self._script):
            return self._script(packet), usage
        if self._index >= len(self._script):
            return ProposerError("exhausted", "script ran out"), usage
        proposal = self._script[self._index]
        self._index += 1
        return proposal, usage


class AnthropicProposer:
    """Calls Claude once per attempt. Two tools, one of which must be chosen.

    Every failure mode -- timeout, transport, overload, malformed input -- is
    converted into a `ProposerError` rather than raised. The tier treats a model
    that failed and a model that declined as different outcomes but handles both
    the same way: the item goes to a human with the reason attached.
    """

    def __init__(
        self,
        client=None,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 4096,
        timeout: float = 60.0,
    ) -> None:
        if client is None:
            import anthropic  # imported lazily so the core stays dependency-free

            client = anthropic.Anthropic(timeout=timeout)
        self._client = client
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens

    def propose(self, packet: EvidencePacket) -> tuple[Proposal, Usage]:
        import anthropic

        usage = Usage()
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self._effort},
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": json.dumps(packet.to_dict(), indent=2, sort_keys=True),
                    }
                ],
                tools=[PROPOSE_TOOL, FLAG_TOOL],
                tool_choice={"type": "any"},
            )
        except anthropic.APITimeoutError as exc:
            return ProposerError("timeout", str(exc)), usage
        except anthropic.RateLimitError as exc:
            return ProposerError("overloaded", str(exc)), usage
        except anthropic.APIStatusError as exc:
            return ProposerError("transport", f"{exc.status_code}: {exc.message}"), usage
        except anthropic.APIConnectionError as exc:
            return ProposerError("transport", str(exc)), usage

        usage.add(response.usage.input_tokens, response.usage.output_tokens)

        if response.stop_reason == "refusal":
            return Refusal("model declined for safety reasons"), usage

        call = next((b for b in response.content if b.type == "tool_use"), None)
        if call is None:
            return ProposerError("malformed", "no tool call in response"), usage

        try:
            return _parse_tool_call(call.name, call.input), usage
        except (KeyError, TypeError, ValueError) as exc:
            return ProposerError("malformed", f"{type(exc).__name__}: {exc}"), usage


def _parse_tool_call(name: str, payload: dict) -> Proposal:
    """Convert a tool call into a proposal, or raise for the caller to catch."""
    if name == "flag_for_human":
        return Refusal(str(payload["reason"]))

    if name != "propose_hypothesis":
        raise ValueError(f"unknown tool {name!r}")

    cites = payload.get("citations")
    if not isinstance(cites, list) or not cites:
        raise ValueError("propose_hypothesis returned no citations")

    parsed: list = []
    for c in cites:
        kind = c.get("type")
        if kind == "adjustment":
            parsed.append(CitedAdjustment(
                adjustment_id=str(c["adjustment_id"]),
                rationale=str(c.get("rationale", "")),
            ))
        elif kind == "fee_variance":
            ids = c.get("payment_ids")
            if not isinstance(ids, list) or not ids:
                raise ValueError("fee_variance cited no payment_ids")
            parsed.append(FeeVarianceClaim(
                payment_ids=tuple(str(i) for i in ids),
                actual_mdr_bps=_strict_int(c["actual_mdr_bps"]),
                rationale=str(c.get("rationale", "")),
            ))
        elif kind == "fx":
            parsed.append(FxClaim(
                payment_id=str(c["payment_id"]),
                actual_rate_pct_of_gross=float(c["rate_pct_of_gross"]),
                rationale=str(c.get("rationale", "")),
            ))
        else:
            raise ValueError(f"unknown citation type {kind!r}")

    confidence = float(payload["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence {confidence} out of range")

    return Hypothesis(
        citations=tuple(parsed), reason=str(payload["reason"]), confidence=confidence
    )


def _strict_int(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"amount_paise must be a number, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"amount_paise must be whole paise, got {value!r}")
    return int(value)
