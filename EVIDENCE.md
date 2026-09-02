# Evidence

The working behind the numbers in [README.md](README.md): how each recovery
tier earns its place, what the adversarial audit found, how the ledger closes,
and the bugs that only showed up once real data arrived.

The README states the results. This states the reasoning, and it is deliberately
longer than a first read should be. Everything here is reproducible from the
commands in the README, and every figure quoted is in [`results/`](results/).

**Contents**

- [The documented capture, and why Leg 1 recall moves at all](#the-documented-capture-and-why-leg-1-recall-moves-at-all)
- [The fourth source, and what it did to B3](#the-fourth-source-and-what-it-did-to-b3)
- [Attacking the inputs: what the adversarial audit found](#attacking-the-inputs-what-the-adversarial-audit-found)
- [Throughput: the shape, not the peak](#throughput-the-shape-not-the-peak)
- [What code mutation testing found](#what-code-mutation-testing-found)
- [The clearing account is the whole argument](#the-clearing-account-is-the-whole-argument)
- [Three bugs a category scorecard could not see](#three-bugs-a-category-scorecard-could-not-see)
- [What is deliberately not posted](#what-is-deliberately-not-posted)

---

### The documented capture, and why Leg 1 recall moves at all

Leg 1 recall sat at 95.40% on every rung, because Leg 1 had no recovery tier and
its largest exception class did not need one. A partial capture is not a
discrepancy the system has to solve. It is a business event the gateway
*declares*: the order authorised one amount, the capture took a smaller one, and
the payment row says `partially_captured` on its face.

Refusing to match those conflates two different facts. The pairing is not in
doubt — the order id joins exactly, one payment, no ambiguity — and filing it
as "unmatched" says the system could not work out where the money went, when in
fact it knows precisely where it went and how much less of it there was.

So B2 matches them, and the gate is the same one used everywhere else in this
repository: **a status field is a claim, not a proof.** What earns the match is
that the fee and tax re-derive from the captured gross at a rate the merchant
has on file — the contracted rate card, or a repricing notice in force for that
method on that settlement date. A row stamped `partially_captured` whose fees do
not re-derive is a book that disagrees with its own rate card, which is a worse
problem than a short capture, and it is refused. `tests/test_leg1.py` runs that
attack, along with a capture larger than the authorisation and a shortfall with
no declaration behind it.

| | dev | held-out |
|---|---:|---:|
| `PARTIAL_CAPTURE` resolved | 56 of 56 | 32 of 32 |
| Leg 1 recall | 95.40% → **98.20%** | 95.40% → **97.00%** |
| Leg 1 exceptions | 92 → 36 | 92 → 60 |
| **False-match rate** | **0.00%** | **0.00%** |

**And the money does not disappear.** Every one of those matches carries a
`variance_paise` — Rs 4,17,173.79 across the dev book — reported on the run, in
the JSON artifact, and on its own line in the console. That is the whole reason
this is defensible: a reconciliation engine that raises its recall by
reclassifying a gap as a match, and stops showing the gap, has not reconciled
anything. The pairing is settled; the variance is not; the queue that an
operator works shrinks from 92 items to 36 and the under-captured rupees stay on
the screen.

What Leg 1 still refuses is what it always refused: two payment rows claiming one
order. There is no document that resolves that one, and picking a winner is a
coin flip with revenue.

---

### The fourth source, and what it did to B3

A fee variance used to be the agent tier's to explain. It is not any more.

The book now carries what a merchant actually receives: the gateway's
**repricing notice** and the bank's **FX advice**, alongside decoys — circulars
for other methods, schedules superseded months ago, advices for payments that
converted exactly as reported. `legs/repricing.py` reads the notice in force for
a method on a settlement date, re-derives the fee from the payment's own gross,
and checks. Every fee and FX defect on both published profiles now closes that
way, deterministically:

| | dev | held-out |
|---|---:|---:|
| `FEE_TAX_VARIANCE` resolved | 4 of 4 | 7 of 7 |
| `FX_CONVERSION` resolved | 3 of 3 | 2 of 2 |
| Leg 2 recall | 93.29% → **97.56%** | 93.29% → **98.78%** |
| **False-match rate** | **0.00%** | **0.00%** |

**This deliberately takes work away from the model.** The missing ingredient was
never reasoning — it was a document. Once the document is in the book the job is
a lookup and two multiplications, and that belongs in a tier that costs nothing
and replays identically, not one that costs a network call. It is the same call
already made for `TIMING_SPILL`, made for the same reason.

Where two notices are in force for one method on one day, the tier **refuses**.
That is a contradiction in the merchant's own file, and picking the newer one
would be exactly the guess the rest of this system exists to avoid.

**What B3 still owns.** The case where the paperwork is missing. A gap with no
document behind it is where a hypothesis is genuinely the best thing available,
and `RateBook` — now populated from the sources rather than left empty — is what
decides whether a cited rate turns out to be on file after all. A rate that is
confirmed books as `resolved`; one nobody issued still closes as
`needs_approval`, however well the arithmetic works out.

**And on the published books, there is nothing left for it to do.**

```bash
python3 -m recoagent.eval.b3 --profile dev
```

| | dev | held-out |
|---|---:|---:|
| Residual-bearing leg-2 items reaching B3 | **0** | **0** |
| Cases attempted | 0 | 0 |
| Leg 2 recall | 97.56% → 97.56% | 98.78% → 98.78% |

The agent tier is never invoked, because every residual the cheaper tiers can
account for, they now account for. That is the strongest form of the argument
this repository makes, and it cost the tier its job: measured against the real
book, the honest resolution rate of the LLM is not a number at all — there is no
denominator. Full reports in [`results/B3_dev.txt`](results/B3_dev.txt) and
[`results/B3_holdout.txt`](results/B3_holdout.txt).

Where it still has work is the book whose paperwork never arrived:

```bash
python3 -m recoagent.eval.b3 --profile dev --no-paperwork
```

That run needs an API key, and **it is now published.** Both profiles, live
against `nvidia/nemotron-3-ultra-550b-a55b`, with **zero endpoint failures** —
which is what makes them measurements rather than reports on an outage. Full
artifacts: [`results/B3_dev_nopaper.txt`](results/B3_dev_nopaper.txt) and
[`results/B3_holdout_nopaper.txt`](results/B3_holdout_nopaper.txt).

| | dev | held-out |
|---|---:|---:|
| Residual-bearing leg-2 items | 7 | 9 |
| Cases attempted | 7 of 7 | 9 of 9 |
| **RESOLVED (source-backed)** | **0** | **0** |
| needs approval | 4 | 5 |
| declined by the model | 2 | 3 |
| rejected by the gate | 0 | 0 |
| cited unverifiable evidence | 0 | 0 |
| malformed reply | 1 | 1 |
| **endpoint failed** | **0** | **0** |
| Leg 2 recall | 93.29% → 93.29% | 93.29% → 93.29% |
| **False-match rate** | **0.00%** | **0.00%** |
| Defects mishandled | 0 | 0 |
| Cost | $0.038, 428s | $0.041, 377s |

**Every case the model saw produced a worked account of the residual, and not
one of them could be verified, so nothing was booked.** That is the thesis
landing, not a disappointing result: the tier's job is to turn a bare number
into an explanation a human can act on, and the gate's job is to refuse to book
an explanation nothing confirms. Four cases on dev and five on held-out closed
the arithmetic on a rate no document in the book issues — `needs_approval`,
held, visible, not reconciled. Two and three respectively the model declined
outright, which is also the right answer.

**The zero is the number to read, and it is allowed to be zero.** An LLM tier
that resolved seven of seven here would mean the gate had been talked into
something, which is exactly the failure an earlier design shipped and a reviewer
caught (below).

**Both profiles land on 93.29%, and that is construction rather than a
copy-paste.** The two mixes hold each leg's *total* defect rate constant and
invert only the composition, so with the paperwork withheld both books lose the
same number of leg-2 credits. Where they differ is underneath: 15 leg-2
exceptions against 16, and 7 residual-bearing items against 9.

**The earlier `--no-paperwork` numbers stay void.** They were taken when fee and
FX cases still reached the model, so their denominators describe a tier that no
longer sees them; they remain quarantined in [`results/void/`](results/void/)
with a note rather than being quietly refreshed.

Note also what B3 has earned its place by surviving: a proposer
that cites a row which does not exist, one that refuses, and a dead endpoint all
leave the false-match rate at 0.00% and the book unchanged. Those are tests, not
claims — `tests/test_b3_eval.py` runs the whole command against scripted
proposers, so the paths that only matter when a model answers are exercised
without a key.

**An earlier design reported 95.73% recall here, and that number was wrong.**
The proposer could state amounts, so "there was an adjustment of exactly the
residual" closed the arithmetic every time — a reviewer found it, we reproduced
it (7 of 7 cases resolved on a fabricated number), and the fix is that the
proposer can now only cite evidence: an existing row by id, or a rule whose
money the code recomputes. The old runs are quarantined in
[`results/void/`](results/void/) with a note, because the before-and-after is
the most instructive thing in this repository. The attack is a permanent test.

**Malformed output is a first-class number here, not a footnote.** On the
published runs above it is **2 of 16 calls (12.5%)** — one on each profile, and
one of them again opens with "Let me…", a reasoning model leaking preamble where
JSON was required. The quarantined pre-RateBook run recorded 19% (3 of 16); that
figure belongs to the void run and is quoted only as history, but the failure
mode did not go away and is not rounded off.

The report counts a malformed reply and a rate-limited endpoint as separate
lines, because one is a property of the model and the other is a property of the
account, and a reader should never have to guess which produced a zero. On both
published runs the endpoint line is **0**, so the zero in the resolved column is
the model's answer rather than an outage.

**The control still holds:** running B3 with `NullProposer` reproduces B2
exactly, so whatever the tier contributes is the model's and not plumbing's.

**Why dev and held-out track each other.** By construction: the two mixes hold
each leg's *total* defect rate constant and invert the composition
(`PARTIAL_CAPTURE` 56→32, `DUPLICATE_PAYMENT` 36→60, `REFUND_NETTED` 7→3,
`FEE_TAX_VARIANCE` 4→7). Identical totals keep the comparison fair; the
per-class tables in `results/*.txt` are where the distributions actually
differ; after the repricing remodel the two mixes happen to land on the same
Leg 2 recall, which the per-class tables show is coincidence rather than
identical behaviour.

---

### Attacking the inputs: what the adversarial audit found

Everything above is measured on the generator, and the generator is friendly —
it injects the defects a real settlement book produces, at rates a real merchant
sees. That shows the matcher handles reality. It does not show the matcher
cannot be *made* to book money against the wrong transaction by someone trying.

`recoagent/audit/mutate.py` tries. It takes bank lines the matcher already
matched correctly, corrupts them in seventeen ways chosen to defeat a specific
part of the join, and scores every run for wrong matches — not just on the line
it attacked, but everywhere, so collateral damage counts.

```bash
python3 -m recoagent.audit.mutate --n 2000 --trials 25
```

| Family | Cases | Held | Refused | **Wrong** | Contained |
|---|---:|---:|---:|---:|---:|
| narration — clipped, decoy digits, transposition, unseen dialect, wrong batch's UTR | 122 | 58 | 64 | **0** | 100% |
| amount — one paisa, rupees-into-paise, swapped credits, sign flip | 100 | 24 | 76 | **0** | 100% |
| timing — credit or payout dated away from its cycle | 50 | 0 | 50 | **0** | 100% |
| structural — UTR collision, duplicate credit, orphan credit, stolen member, **forgery** | 148 | 32 | 99 | 17 | 89% |
| **Total** | **420** | **114** | **289** | **17** | **95.95%** |

Full scorecard in [`results/mutation_audit.json`](results/mutation_audit.json),
regenerated by the command above and diffed against a fresh run by
`tests/test_mutation_audit.py`.

**Four verdicts, not two.** A matcher that refuses everything under attack is
perfectly safe and completely useless, and a contained/not-contained taxonomy
scores it 100%. So *held* (the attack did not land, the correct pairing
survived) and *refused* (declined, exception filed) are reported separately.
Containment is their sum; safety is the absence of the other two.

**All seventeen wrong matches are one attack, and it is declared rather than
hidden.** `perfect_forgery` rewrites a credit to cite another batch's UTR *and*
carry that batch's exact net, then blinds the real credit so the duplicate-UTR
check cannot be what saves us. Every signal the matcher consults then agrees,
and every one of them is attacker-supplied.

Finding it exposed a genuine inconsistency: **Tier 1 had always required a
credit to fall within a day of its payout, and Tier 0 had not** — so the
exact-key path was the weaker of the two, because it had a UTR and trusted it.
One window, now read by both (`leg2.SETTLEMENT_WINDOW_DAYS`), and the result is
measured rather than assumed:

| Attack, 24 cases at n=2,000 | No date check | 1-day window |
|---|---:|---:|
| `perfect_forgery` | 16 wrong (66.7%) | **1 wrong (4.2%)** |
| `perfect_forgery_dated` — the forger also fakes the date | 16 wrong (66.7%) | 16 wrong (66.7%) |

**A sixteen-fold reduction, and not a fix.** The survivor is the case where both
payouts settle inside the same window, which with 164 settlements over 20
trading days is a coincidence that turns up. So both variants stay in
`audit.mutate.KNOWN_UNCONTAINED`, print on every run, and gate nothing — and a
test asserts they still land, so the disclaimer cannot outlive the weakness.

Leg 2's whole evidence base is the narration, the amount and the value date. An
adversary holding all three can manufacture a match, and no further arithmetic
recovers the difference. The defence is that a bank statement is not
attacker-controlled — a property of the world, not a rule in this repository,
and it is reported as such rather than rounded off into a containment figure.

The check costs nothing on an honest book: `results/B2_*.json` are byte-identical
with and without it, which is also why the deterministic corpus cannot exercise
it and the adversarial suite is the only thing that does.

---

### Throughput: the shape, not the peak

One timing at one size cannot tell you whether the next order costs the same as
the last, and a reconciliation engine has two things in it that go quadratic if
nobody watches — a per-settlement scan over payments, and a candidate pool drawn
from a date window. Both look fine at 2,000 rows.

```bash
python3 -m recoagent.eval.throughput
```

| Orders | Records | Batches | Seconds | Records/sec |
|---:|---:|---:|---:|---:|
| 500 | 1,111 | 44 | 0.009 | 129,601 |
| 2,000 | 4,412 | 164 | 0.035 | 126,589 |
| 5,000 | 11,029 | 418 | 0.091 | 121,542 |
| 10,000 | 22,068 | 846 | 0.199 | 110,999 |
| 25,000 | 55,080 | 2,083 | 0.734 | 75,034 |

**Across a 50× range throughput moves 1.7×** — effectively linear, single
process, single thread, standard library only. Timing excludes generating the
book: what is measured is the matcher, not the fixture. Full report in
[`results/throughput.txt`](results/throughput.txt).

**Two settlement densities are reported, and the second is the honest one.**
The table above is the generator's default, which holds batch size constant as
volume grows — so a 25,000-order book becomes 2,083 payouts, and the solver's
per-batch work is what bends the curve at the end. Real gateways settle on a
T+2 cycle: roughly one payout a day, with batches ten times bigger rather than
ten times as many. On that density the same code holds **115,365 records/sec at
25,000 orders across a 45× range, moving 1.1×** — flat. Reporting only the
flatter number would have been a choice, so both are in the artifact.

---

### What code mutation testing found

The suite is checked by breaking things on purpose. Three mutations are caught:
skipping the arithmetic gate, clipping narrations at a fixed column instead of
inside the UTR, and letting a matcher import the generator.

Two results are worth stating because they shaped the design:

- **Disabling the proof gate does not move false-match rate.** On Leg 2 the
  pairing comes from the UTR join, so a broken gate still picks the right
  batch — it just accepts one whose money is wrong. Match-rate metrics are
  structurally blind to this, which is why defect accounting and
  `test_the_explanation_is_right_not_merely_arithmetically_sufficient` exist
  alongside the headline number.
- **The uniqueness guard is not exercised end-to-end by the current mix.**
  Removing it changes nothing measurable, because the only ambiguity this data
  produces is *fungible* — every dispute fee is a flat Rs 150, so competing
  subsets name different rows but identical amounts. The guard is covered
  directly by unit test, not by the integration suite. Exercising it would need
  a defect class that generates materially different competing explanations.

---

---

### The clearing account is the whole argument

When a customer pays, the merchant does not have the money — the gateway does.
So a capture creates a **gateway receivable**, the fee and its tax and any
refund reduce it, and the settlement credit clears it into the bank. Which
means this:

```
gross - fee - tax - refunds - chargebacks - dispute fees - net credit = 0
```

is *both* the leg-2 identity the matcher proves *and* the statement that a
batch's receivable nets to zero. They are the same equation in two notations —
so the accounting output is tied numerically to the reconciliation rather than
sitting alongside it.

**The claim is not that the account empties.** It does not, and an engine that
said so would be hiding something. The claim is that every rupee left in it has
a name against it, and nothing is left over:

| Why a batch still has money in the clearing account | dev | held-out |
|---|---:|---:|
| The gateway has not paid this batch out | Rs 7,74,492.94 | Rs 8,21,970.91 |
| A payment in the batch never matched an order | -Rs 2,40,633.95 | -Rs 2,47,983.28 |
| A payment reported here was credited with another cycle | -Rs 1,27,594.82 | -Rs 1,02,722.31 |
| An FX or repricing difference against the reported figure | Rs 1,226.27 | Rs 1,980.33 |
| Sub-rupee rounding between the gateway and the bank | **Rs 0.03** | **Rs 0.01** |
| **Unattributed** | **Rs 0.00** | **Rs 0.00** |

Four of those five causes are read straight off **the matcher's own rule id** —
whatever Tier 1 had to do to close a credit is, by definition, why the books and
that batch disagree — so the table cannot drift away from the reconciliation it
describes. The fifth, rounding, is the leftover that no other cause claimed, and
a test fails if anything larger than sub-rupee drift hides in it. On a clean
book every batch clears to zero, which is what makes the dev numbers mean the
defects rather than the posting rules.

---

### Three bugs a category scorecard could not see

All three had the right label and the wrong number, which is exactly the shape
of error a categorisation accuracy figure is blind to. A clearing account either
empties or it does not, so it found them:

- **The settlement credit was booked at the gateway's declared net**, not the
  cash the bank actually sent. When a cutoff spill moves a credit, the gateway's
  row still states the original figure — so the bank account was being debited
  with money that never arrived.
- **An orphaned refund the solver had attributed to a batch, the books had
  attributed to nothing.** Tier 1's subset-sum records those on the match as
  `hypothesised_ids`, having required the subset to be the only one that closes;
  the journal was ignoring that link and the batch's credit then accounted for
  money nothing else did.
- **A manual adjustment with a positive amount was posted in the negative
  direction its category implies**, moving the receivable the wrong way by twice
  its value. The trial balance still balanced — a wrong direction is wrong on
  both sides at once — and it took the per-batch check to surface it. It is now
  posted as a contra entry *and* reported as a sign anomaly, because a refund
  carrying a positive amount is worth a human look either way.

---

### What is deliberately not posted

Nothing a model proposed. Every C2 assignment is marked unverified by design, so
it reaches the operator queue with its evidence attached and stops there — a
suggestion in the operator queue is useful, a suggestion in the general ledger
is a misstatement. A `NEEDS_REVIEW` row goes to **suspense**, which keeps the
books balanced while leaving the problem visible with a number against it, and
the suspense balance is printed at the top rather than buried.

The two structural errors [the taxonomy](#categorisation) exists to prevent are
checked here, where they would actually cost money: a settlement credit may
never touch an income account (booking a payout as revenue doubles declared
turnover), and GST on the MDR must debit an asset rather than an expense
(filed as a cost it understates the credit claimable in GSTR-3B).

---
