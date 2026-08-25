"""The LLM exception tier (B3).

Runs last, on residuals no deterministic tier could close. The model proposes;
`recoagent.validate` decides. See `tier.py` for why that asymmetry is the whole
design.
"""

from .contracts import (
    AgentReport,
    CaseOutcome,
    Hypothesis,
    Proposal,
    ProposedRow,
    ProposerError,
    Refusal,
    Usage,
)
from .agentic import AgenticProposer
from .openai_proposer import OpenAICompatibleProposer
from .proposer import AnthropicProposer, NullProposer, Proposer, ScriptedProposer
from .tier import recover_with_agent, render_report

__all__ = [
    "AgentReport", "CaseOutcome", "Hypothesis", "Proposal", "ProposedRow",
    "ProposerError", "Refusal", "Usage", "AnthropicProposer", "NullProposer",
    "AgenticProposer", "OpenAICompatibleProposer", "Proposer", "ScriptedProposer", "recover_with_agent", "render_report",
]
