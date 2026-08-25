# Architecture

## The problem is two problems

A merchant holds three files that never quite agree: their own order ledger,
the gateway's settlement report, and their bank statement. Reconciling them is
not one matching pass across three sources. It is two independent legs with
different cardinalities, different failure modes, and different algorithms.

**Leg 1 — order ↔ payment, 1:1.** A record-linkage problem. Join on
`payment_id`/`order_id`, then check the captured amount against what the order
said was owed.

**Leg 2 — settlement batch ↔ bank credit, N:1.** One bank credit is a whole
batch of payments, minus MDR, minus GST on MDR, minus refunds, minus
chargebacks, minus whatever else the gateway netted. This is not a join. It is
a set-partition problem with tolerances.

The two-leg framing follows Juspay's Hyperswitch, which runs it in production.
Using their vocabulary is deliberate: the goal is to be legible to someone who
has seen a real reconciliation stack, not to invent a private ontology.

## Leg 2 is a named research problem

Matching a subset of payments to a single credit within a tolerance is the
**Subset Sum Matching Problem** (Wu et al., 2025): given multisets *a* and *b*,
choose disjoint subsets with `|w·a − v·b| ≤ ε`, maximising matched pairs and
volume. Three solver families are known — MILP (optimal, intractable as ε
grows), DP-greedy (near-optimal, scales, tolerates large ε), and
meet-in-the-middle search (exponential, small instances only).

Rung B2 implements this, and **not** with the DP, which is the solver usually
recommended for reconciliation at scale. Two concrete reasons, both in
`legs/ssmp.py`:

- **S_max is enormous at paise granularity.** The DP is pseudo-polynomial in
  the maximum achievable sum, and settlement values run to crores — 10^9 paise.
  Discretising to rupees would make it tractable and would also discard the
  precision that the ROUNDING_DRIFT class exists to test.
- **The DP returns *a* solution, not *all* of them.** Uniqueness is a
  correctness requirement here, not a nicety: if two materially different sets
  of rows both close a gap, the honest verdict is "escalate", and a solver that
  hands back one answer cannot tell you that.

Candidate pools are small by construction — unlinked rows inside a date window
around one settlement, typically under thirty — so bounded exact enumeration is
complete *and* cheap, and `meet_in_middle` covers larger cases. That is the
third of Wu et al.'s three families, so the choice stays inside the literature
rather than around it.

Notably, the SSMP literature contains no work applying language models to it,
which is where the LLM tier sits rather than competing with the solver.

### Identity ambiguity is not amount ambiguity

The most interesting thing the solver found: every dispute fee is a flat
Rs 150, so a chargeback gap is closed by several subsets that name *different
rows containing identical amounts*. Refusing all of them leaves real money
unreconciled over a distinction that does not affect the total.

`SearchResult.value_equivalent` separates the two cases. Competing subsets with
the same multiset of amounts are financially identical and safe to act on;
competing subsets with different amounts are a genuine fork and stop the line.
The pairing being decided comes from the UTR join, so the subset only has to
prove the total — but fungible rows are interchangeable, not infinite, so the
tier tracks which rows it has spent and will not fund two batches with one
Rs 150 fee.

## Why the LLM is not a matcher

FinBalance (Tumpati et al., June 2026) evaluated six contemporary LLMs on
multi-document accounting reconciliation. Two findings shape this design:

- They reach **at most 46% exact accuracy**. An LLM asked to reconcile a batch
  end to end is not good enough to trust.
- There is a **26–41 percentage-point gap** between the balance sheet a model
  *reports* and the one produced by replaying its own entries through the
  ledger. Models do not merely make errors; they misreport the consequences of
  the entries they themselves wrote.

So the LLM never writes a match. Its entire privilege is to **propose rows**
that might explain a residual — an orphaned refund, a netted chargeback, a
manual adjustment. `validate.prove_leg2()` then sums the proposal on exactly
the same footing as the linked rows and applies exactly the same check. A
hypothesis that does not close is rejected. This is the ledger-replay
correction FinBalance found effective, made structural rather than optional.

The `hypothesised=` parameter on `prove_leg2` is already in place and tested
(`test_correct_hypothesis_closes_the_gap`,
`test_wrong_hypothesis_is_rejected`). Rung B3 fills it; nothing about the gate
changes when it arrives.

## Decisions worth defending

**Integer paise everywhere.** Floats are never used for money. `0.1 + 0.2` is a
cosmetic annoyance in a report and a false match in a reconciliation engine.

**Fees round per step, not per batch.** MDR is rounded to whole paise, then GST
is rounded on the rounded MDR. Real settlement reports do this, and it produces
genuine sub-rupee drift that exact equality cannot absorb — which is what makes
tolerance a design question rather than a fudge factor. `test_money.py` proves
per-step and single-step rounding actually diverge.

**UPI carries zero MDR.** Not a placeholder: it is zero by regulation in India,
and UPI is the majority of the book. A fee model that charges every method
reports most settlements as short — the single fastest way to generate a wall
of false exceptions.

**The settlement header is corroboration, never proof.** The proof re-derives
the batch total from the payment rows. The header is the gateway's *claim*
about what it paid; checking a claim against itself is not reconciliation.
`header_agrees()` exists as a diagnostic and is deliberately never used to
accept a match.

**Ambiguity is refused, not resolved.** Two payments claiming one order, or one
UTR on two statement lines, produce an exception rather than a guess. Refusing
both duplicate lines costs recall — it is visible in the 75% Leg 2 figure. The
alternative is booking the same money twice.

**On-hold settlements are correctly unmatched.** Money that never moved should
not be counted as a matching failure. They are labelled, not penalised.

**One failure, one exception.** A batch whose credit fails its proof is filed
once, against the credit. An earlier version also filed it against the
settlement, which doubled the apparent exception count and would have handed an
ops team the same item twice.

**Zero tolerance at B0; ten paise at B2, chosen against evidence.** This is the
one number in the system set by judgement, so it is the one number with a
measurement behind it (`python -m recoagent.eval.tolerance_sweep`).

The obvious way to pick it is to maximise recall, and that gives the wrong
answer. Leg 2 recall keeps climbing past 10 paise all the way to Rs 10, and
false-match rate stays flat at 0.00% the entire way — neither headline metric
objects to a window a thousand times too wide, because on Leg 2 the tolerance
never governs *which* batch a credit belongs to, only whether its explanation
may be approximate.

The per-class table is what decides it. At 10 paise every ROUNDING_DRIFT closes
and nothing else moves. At 50 the solver starts absorbing FX_CONVERSION; by
Rs 10 it is swallowing FEE_TAX_VARIANCE. Those are not recovered matches — they
are real differences in money reconciled green, a merchant silently short a few
hundred rupees. Ten paise is the largest window that absorbs rounding and only
rounding.

Leg 1 stays at zero: a capture that differs from its order by any amount is a
partial capture, not a rounding artifact.

## Why the evaluation is trustworthy

The generator emits a `LabelledBatch`; matchers receive only `batch.sources`.
Ground truth is structurally unreachable from `legs/`, `validate.py` and
`pipeline.py`, and `tests/test_independence.py` enforces it by AST inspection
in CI.

Defect counts are **exact, not sampled** — a rate of 0.045 over 164 settlements
injects exactly 7. This is what turns the scorer's reconciliation check from an
impression into an assertion: every injected defect must surface as an exception
on an entity it actually damaged, and `InjectedDefect.affected_ids` records the
collateral (a T+2 spill damages two batches; a duplicated UTR poisons both
lines).

Defect rates are calibrated to roughly 23% of batches and 5% of orders, because
production reconciliation runs at 85–95% straight-through. A synthetic set where
most batches are broken would not be a harder problem — it would be a different
one, and it would make the LLM tier look far more valuable than it is.

## The agent tier (B3)

Built, tested, and measured against live models — see `results/B3_*.txt`. The
numbers are directional, not conclusive: n is 7 on dev and 11 held out, and
three repeat runs resolved 2, 3 and 4 of 7, so the tier is reported as a range
rather than a point estimate.

The design is one asymmetry, and the first version of it was unsound. A proposer
used to return rows *with amounts*, so it could name the residual itself --
"there was an adjustment of exactly this much" closed the arithmetic every time,
7 of 7 cases, with the false-match rate still reading 0.00%. The gate was
checking that the model's number made the model's own total add up.

A proposer now returns **citations**: an unlinked row by id, or a rule to apply
to named payments. `recoagent/agent/citations.py` computes every rupee from the
source rows and the fee schedule. A citation that does not resolve is rejected
outright rather than partially credited, and accepted matches record the source
ids in `hypothesised_ids` so the audit trail names what the total rests on. It cannot return a match, cannot
name a settlement, and cannot mark anything resolved. `validate.prove_leg2` sums
its rows on exactly the same footing as rows a human reported and discards
anything that does not close. A rejection earns one retry with the residual fed
back; a second failure sends the item to a human with the reason attached.

That shape is a response to a measured failure. FinBalance found a 26–41
percentage-point gap between the balance sheet a model reports and the one
produced by replaying its own entries through a ledger. A model that can assert
a match can assert a wrong one convincingly; this one can only ever offer
arithmetic that either closes or does not.

**Confidence is recorded, not trusted.** The model's own number goes into the
audit record, but match confidence is capped at 0.70. Self-reported confidence
from an LLM is evidence about the model, not about the match.

**Inferred rows are marked.** `inferred:llm:*` and `inferred:spill:*` never look
like rows anyone reported.

**Three proposers, one interface.** `AnthropicProposer` calls Claude — two
tools, one call, no agent loop, because a wider tool surface is a wider blast
radius for a wrong answer and the gate cannot tell an elaborate wrong answer
from a simple one. `ScriptedProposer` returns whatever a test specifies,
including malformed output and timeouts, so every failure path is a unit test
rather than an anecdote. `NullProposer` always declines, and running B3 with it
reproduces B2 exactly — the control that makes any future lift attributable to
the model rather than to plumbing.

**The territory is deliberately narrow.** Seven exceptions of 129 defects on
dev. `TIMING_SPILL` was moved into the deterministic solver rather than left
here, because it is mechanically detectable and handing it over would have
inflated the model's apparent contribution.

## Not built yet

- **B1** — Splink / Fellegi-Sunter probabilistic linkage on Leg 1
- **An authoritative rate source.** Fee-variance and FX claims rest on a rate the
  model chooses. Code computes the money from it either way, but nothing
  confirms the rate itself, so those close as *needs approval* rather than
  reconciled. `RateBook` is the hook a real gateway repricing notice or bank FX
  advice would fill, turning the same claims into facts.
- **Provenance accuracy at scale.** `AgentReport.provenance()` checks whether an
  accepted explanation cited the right evidence, which the main scorer cannot
  see: it grades a B3 match on its bank-line → settlement pairing, and that
  pairing comes from the UTR join rather than from the model. So a wrong
  explanation can still report a perfect false-match rate.
- **BenchRec** — external validation on real labelled data. Until this lands,
  every number in the README is self-generated, and the independence and
  accounting checks are what stand in for external validity.
- **Confidence calibration** — B0 emits confidence 1.0 on every match, which is
  honest for exact keys but leaves nothing to calibrate. It becomes meaningful
  at B1 and necessary at B3.

## What mutation testing revealed

Two findings shaped the design more than any passing test did.

**Match-rate metrics are structurally blind to a broken gate.** Disabling the
arithmetic check entirely does not move false-match rate, because the Leg 2
pairing comes from the UTR join — a broken gate still picks the right batch, it
just accepts one whose money is wrong. The damage is a false *audit trail*, not
a false match. This is why defect accounting and an explicit attribution test
sit alongside the headline number, and it is the clearest argument for why this
system reports several numbers rather than one.

**The uniqueness guard is not exercised end-to-end by the current defect mix.**
Removing it changes nothing measurable, because the only ambiguity this data
produces is fungible — competing subsets that name different rows with
identical amounts. The guard is covered directly by unit test rather than by
the integration suite. Exercising it properly would need a defect class that
generates materially different competing explanations, which is a real gap and
is recorded as one rather than glossed.

## Known limitations

- Synthetic data only. The defect taxonomy is drawn from documented behaviour
  of Indian payment settlement, but the distributions are chosen, not measured.
- Leg 2 is scored over bank lines, which is the harder population: duplicated
  credits sit in the denominator with no correct answer available.
- The FX defect models a rate slip, not the full multi-currency settlement
  lifecycle.
- No persistence layer. Runs are in-memory and reproducible from a seed, which
  suits evaluation and would not suit production.
- The Leg 2 tolerance is calibrated against a ROUNDING_DRIFT class this
  repository also defines. The magnitude chosen (1-9 paise) is plausible for
  per-step rounding differences between a gateway and a merchant, but it is
  asserted rather than measured against a real settlement file. Real drift
  would move both the class and the tolerance together.
- The uniqueness guard's end-to-end coverage gap, above.
