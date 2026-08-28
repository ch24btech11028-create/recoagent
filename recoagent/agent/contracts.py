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
from .citations import Citation


@dataclass(frozen=True)
class Hypothesis:
    """An explanation for a residual: citations, never amounts.

    The proposer used to return `(label, amount_paise)` pairs and the tier turned
    them into ledger rows. Because the proposer chose the amount it could always
    choose the residual, so "there was an adjustment of exactly this much" closed
    the arithmetic every time -- 7 of 7 cases, with the false-match rate still
    reporting 0.00%. The gate was checking that the model's number made the
    model's total add up.

    Now the proposer can only point at evidence. `recoagent.agent.citations`
    turns each pointer into money using the source rows and the fee schedule, so
    every rupee in a B3 match is computed by code from data that already
    existed.
    """

    citations: tuple[Citation, ...]
    reason: str
    #: The proposer's own confidence. Recorded in the audit trail, but never
    #: trusted as a match confidence on its own -- see CONF_T2_CAP in `tier`.
    confidence: float


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
    outcome: str  # resolved | needs_approval | rejected | unverifiable
    #                | refused | failed | low_confidence
    attempts: int = 0
    model_confidence: float | None = None
    #: For `failed` only: the `ProposerError.kind` behind it. A 429 from a
    #: shared endpoint and a model that answered with prose where JSON was
    #: required are both "the tier got nothing", and reporting them as one
    #: number puts an infrastructure limit in a column a reader will read as a
    #: property of the model.
    failure_kind: str = ""
    detail: str = ""
    usage: Usage = field(default_factory=Usage)
    #: Source ids the accepted explanation rests on. Empty unless resolved.
    cited_ids: tuple[str, ...] = ()


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
    def needs_approval(self) -> int:
        """Arithmetic closed, but on a rate the model chose rather than one any
        source confirms. Deliberately not counted as resolved."""
        return self._count("needs_approval")

    @property
    def unverifiable(self) -> int:
        """Cited evidence that does not exist, or does not belong to this batch.

        Distinct from `rejected`, which is a citation set that resolved cleanly
        and still did not close. This counts proposals that never earned an
        arithmetic check at all.
        """
        return self._count("unverifiable")

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
    def failed_transport(self) -> int:
        """Failures that never reached the model: rate limits, timeouts, 5xx."""
        return sum(
            1 for c in self.cases
            if c.outcome == "failed" and c.failure_kind in ("transport", "timeout", "overloaded")
        )

    @property
    def failed_malformed(self) -> int:
        """The model answered and the answer could not be used."""
        return sum(
            1 for c in self.cases
            if c.outcome == "failed" and c.failure_kind not in
            ("transport", "timeout", "overloaded")
        )

    @property
    def resolution_rate(self) -> float:
        return self.resolved / self.attempted if self.attempted else 0.0

    def provenance(self, truth_ids_for: dict[str, set[str]]) -> tuple[int, int]:
        """How often an accepted explanation cited the *right* evidence.

        Returns (correct, checked). The scorer's false-match rate cannot see
        this: a B3 match is graded on its bank-line -> settlement pairing, and
        that pairing comes from the UTR join rather than from the model. So an
        explanation can name the wrong rows, close the arithmetic, and still
        report a perfect false-match rate. This is the metric that notices.
        """
        correct = checked = 0
        for c in self.cases:
            if c.outcome != "resolved":
                continue
            expected = truth_ids_for.get(c.entity_id)
            if expected is None:
                continue
            checked += 1
            if set(c.cited_ids) & expected:
                correct += 1
        return correct, checked

    def cost_per_resolved(self, input_per_mtok: float, output_per_mtok: float) -> float:
        if not self.resolved:
            return 0.0
        return self.usage.cost_usd(input_per_mtok, output_per_mtok) / self.resolved
