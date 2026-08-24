"""What a proposer may say, and what it may never do.

The whole B3 design is in the shape of these types. A proposer returns a
`Hypothesis` -- a set of rows it believes explain a gap -- or it declines. It
cannot return a match, cannot name a settlement, and cannot express confidence
in anything except its own explanation. The decision to book a match is made
downstream by `validate.prove_leg2`, from arithmetic, every time.

That asymmetry is deliberate and it is the thing to defend on a panel call.
FinBalance (Tumpati et al., 2026) measured a 26-41 percentage-point gap between
the balance sheet a model *reports* and the one produced by replaying its own
entries through a ledger. A model that can assert a match is a model that can
assert a wrong one convincingly. This one can only ever offer arithmetic that
either closes or does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..money import Paise


@dataclass(frozen=True)
class ProposedRow:
    """One row a proposer believes was netted into a batch but not linked to it."""

    label: str
    amount_paise: Paise
    rationale: str


@dataclass(frozen=True)
class Hypothesis:
    """An explanation for a residual, offered for checking -- never a verdict."""

    rows: tuple[ProposedRow, ...]
    reason: str
    #: The proposer's own confidence. Recorded in the audit trail, but never
    #: trusted as a match confidence on its own -- see CONF_T2_CAP in `tier`.
    confidence: float

    @property
    def total_paise(self) -> Paise:
        return sum(r.amount_paise for r in self.rows)


@dataclass(frozen=True)
class Refusal:
    """The proposer declined to explain this gap. A legitimate, useful answer."""

    reason: str


@dataclass(frozen=True)
class ProposerError:
    """The proposer failed to answer at all.

    `kind` is one of: malformed | timeout | transport | overloaded. Kept as a
    first-class outcome rather than an exception because an exception queue
    that says *why* the automated tier gave up is more useful to an ops team
    than one that just says it did.
    """

    kind: str
    detail: str


Proposal = Hypothesis | Refusal | ProposerError


@dataclass
class Usage:
    """Token and call accounting, so cost per exception resolved is measurable."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def merge(self, other: Usage) -> None:
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens

    def cost_usd(self, input_per_mtok: float, output_per_mtok: float) -> float:
        return (
            self.input_tokens / 1_000_000 * input_per_mtok
            + self.output_tokens / 1_000_000 * output_per_mtok
        )


@dataclass
class CaseOutcome:
    """What happened to one exception the agent tier was given."""

    entity_id: str
    settlement_id: str
    residual_paise: Paise
    outcome: str  # resolved | rejected | refused | failed | low_confidence
    attempts: int = 0
    model_confidence: float | None = None
    detail: str = ""
    usage: Usage = field(default_factory=Usage)


@dataclass
class AgentReport:
    """Aggregate result of one B3 pass. Reported alongside the scorecard.

    `rejected` is the number that matters most for the argument. It counts
    hypotheses the model offered confidently and the arithmetic threw out --
    the gate doing exactly the job it exists to do.
    """

    cases: list[CaseOutcome] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def _count(self, outcome: str) -> int:
        return sum(1 for c in self.cases if c.outcome == outcome)

    @property
    def attempted(self) -> int:
        return len(self.cases)

    @property
    def resolved(self) -> int:
        return self._count("resolved")

    @property
    def rejected(self) -> int:
        return self._count("rejected")

    @property
    def refused(self) -> int:
        return self._count("refused")

    @property
    def failed(self) -> int:
        return self._count("failed")

    @property
    def low_confidence(self) -> int:
        return self._count("low_confidence")

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.attempted if self.attempted else 0.0

    def cost_per_resolved(self, input_per_mtok: float, output_per_mtok: float) -> float:
        if not self.resolved:
            return 0.0
        return self.usage.cost_usd(input_per_mtok, output_per_mtok) / self.resolved
