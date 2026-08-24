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

Rung B2 will implement the DP-greedy solver. Notably, the SSMP literature
contains no work applying language models to it, which is where the LLM tier
sits rather than competing with the solver.

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

**Zero tolerance at B0.** Sub-rupee drift is real, but absorbing it requires an
explicit, defended tolerance. The baseline refuses to absorb anything, the drift
lands in the exception list, and a later rung has to earn the tolerance it takes.

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

## Not built yet

- **B1** — Splink / Fellegi-Sunter probabilistic linkage on Leg 1
- **B2** — SSMP DP-greedy solver and a defended tolerance on Leg 2
- **B3** — the LLM exception agent, its tool surface, and the repair loop
- **BenchRec** — external validation on real labelled data. Until this lands,
  every number in the README is self-generated, and the independence and
  accounting checks are what stand in for external validity.
- **Confidence calibration** — B0 emits confidence 1.0 on every match, which is
  honest for exact keys but leaves nothing to calibrate. It becomes meaningful
  at B1 and necessary at B3.

## Known limitations

- Synthetic data only. The defect taxonomy is drawn from documented behaviour
  of Indian payment settlement, but the distributions are chosen, not measured.
- Leg 2 is scored over bank lines, which is the harder population: duplicated
  credits sit in the denominator with no correct answer available.
- The FX defect models a rate slip, not the full multi-currency settlement
  lifecycle.
- No persistence layer. Runs are in-memory and reproducible from a seed, which
  suits evaluation and would not suit production.
