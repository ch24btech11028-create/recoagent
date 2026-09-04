# What this project is, in plain language

The README is written for someone who already knows what settlement
reconciliation is and wants the evidence fast. This page is for everyone else.
It assumes nothing.

---

## 1. The problem

A merchant sells something online for ₹1,000.

The customer pays ₹1,000. The merchant does **not** receive ₹1,000, and does not
receive it today. What actually happens:

1. The payment gateway (Razorpay, say) collects the ₹1,000.
2. It keeps a commission — the **MDR** — of maybe ₹20.
3. It charges **GST** on that commission, about ₹3.60.
4. Two days later it takes that payment, bundles it with a few hundred others,
   and sends the merchant **one lump sum** into their bank.

So the bank statement shows a single credit of, say, ₹8,42,317.44, covering 300
different sales, minus 300 different commissions, minus tax, minus any refunds
that happened in between, possibly minus a chargeback.

Now the merchant's finance team has to answer: **is that number right?** Which
sales are in it? Did the gateway charge the rate it promised? Did anything go
missing?

Today, someone does this by hand in a spreadsheet. Every month. It is slow, and
mistakes are invisible.

---

## 2. What this software does

It does that job automatically, in three stages.

**Stage one — reconcile.** Match every record to its counterpart:
- each *order* to the *payment* that paid it
- each *bank credit* to the *batch of payments* it represents

**Stage two — categorise.** Decide what each row is for accounting purposes:
this is sales revenue, that is a gateway fee, that is reclaimable GST, that is a
refund.

**Stage three — post.** Turn all of it into proper double-entry bookkeeping with
a trial balance, the way an accountant would.

Anything it cannot resolve goes onto an exception list with the reason it
stopped.

---

## 3. The one idea that shapes everything

Most reconciliation tools sell you a **match rate**: "we matched 95% of your
transactions!"

This project argues that is the wrong headline, and organises everything around
a different one.

Think about what a *wrong* match costs. If the software confidently pairs a bank
credit with the wrong batch, it books real money against the wrong transaction —
and then shows you a green tick. Nobody investigates a green tick. The error sits
in the books until an auditor finds it, or nobody ever does.

Whereas if the software **refuses** to match something and files it as an
exception, a human looks at it for five minutes and resolves it.

So:

> **A wrong match is worse than an admitted failure.**

The lead metric is therefore **false-match rate** — how often it matched
something incorrectly. This project's is **0.00%**. Match rate is reported
second. The list of things it couldn't do is published in full rather than
summarised away.

This isn't just an opinion. **BenchRec**, an academic benchmark from ICAIF 2023
for exactly this task, states the same principle: better to leave a transaction
unmatched for manual review than to match it incorrectly.

---

## 4. How it avoids fooling you (and itself)

### The ladder

Here is a trick that AI projects pull, often without meaning to:

> "Our AI categorises transactions with 95% accuracy!"

…when in fact the data already has a column called `status` that says
`refunded`, and the "AI" is reading it. The 95% is real. The claim is a lie.

To make that impossible here, every capability is built as a **ladder** — the
same job done with progressively more tools, so you can see exactly what each
addition bought.

**Reconciliation:**

| Rung | What it may use | Result |
|---|---|---|
| **B0** | exact key matching only | 93.85% auto-matched |
| **B2** | + arithmetic solvers | **98.15%** |
| **B3** | + an AI model on the remainder | 0 additional (see §6) |

**Categorisation:**

| Rung | What it may use | Coverage |
|---|---|---|
| **C0** | only fields the source already provides | **0.86%** |
| **C1** | + what the reconciliation proved | **94.86%** |
| **C2** | + an AI model that must quote its evidence | +0 booked |

**C0 exists purely to attack this project's own demo.** It scores 0.86%, which
proves the 94.86% came from the reconciliation and not from reading a status
field. No other entrant in the competition does this.

### Held-out testing

Numbers measured on the data you tuned against mean very little. So every result
is also reported on a **held-out** run that uses a different random seed *and* a
different mix of defects the matcher was never tuned on.

### The unknown-class test

Even held-out testing has a hole, and this project names it. The generator that
creates the test data and the matcher that solves it share an author. So a 0.00%
false-match rate over twelve kinds of defect is partly a statement about the
engine and partly a statement about the author's imagination.

So three *more* defect types were defined — all real events on Indian settlement
books — with **no handling for any of them anywhere in the code**:

- **TDS withholding** — 0.1% statutory tax withheld on gross and paid to the government
- **Bank charge debit** — the receiving bank takes its own fee out of the credit
- **Split payout** — one settlement arriving as two separate transfers

The question isn't whether it can solve these; it obviously can't. The question
is **what it does when it meets something it has no model for**. Does it file an
exception, or does it find some coincidental combination of numbers that adds up
and book a wrong match?

Result: **14 of 14 filed as exceptions. 0 wrong-matched. 0 silently absorbed.**
Recall fell 10.7 points, and reporting that cost is the point.

### Attacking it on purpose

There is a suite that deliberately corrupts bank statements — clipping reference
numbers, planting decoy digits, swapping amounts, forging references — to try to
trick the matcher into a wrong match. **17 attack types, 420 cases, 0 undeclared
wrong matches.**

Two attacks *do* work: a forged bank line carrying another batch's reference
number *and* that batch's exact amount, so the arithmetic closes perfectly.
These are **declared, printed on every run, and gate nothing** — with a test that
fails if they ever stop working, so the disclaimer can't outlive the weakness.

---

## 5. The books

The competition track is "AI Finance **Controller**". A controller produces
books, so the project produces books.

The elegant part is an account called **gateway receivable**. When a customer
pays, the merchant doesn't have the money — the gateway does. So:

- a payment **creates** a receivable (the gateway owes you)
- fees, tax and refunds **reduce** it
- the settlement credit **clears** it into the bank

Which means "this batch's receivable nets to zero" and "the reconciliation
arithmetic closes" are **the same statement in two different notations**. The
accounting is tied numerically to the matching rather than sitting beside it.

The claim is deliberately *not* "every batch clears" — it doesn't. The claim is
that **every rupee left over has a named cause and nothing is unexplained**:
Rs 0.00 unattributed, on both the dev and held-out books.

Building this found **three real bugs**, all of which a categorisation accuracy
score is blind to because in each case the *label* was right and the *number* was
wrong:

1. Bank credits were booked at the gateway's *claimed* amount rather than what
   the bank actually sent.
2. An orphaned refund the solver had linked to a batch, the books had linked to
   nothing.
3. An adjustment posted in the wrong direction — wrong by twice its value, while
   the trial balance still balanced, because a wrong direction is wrong on both
   sides at once.

---

## 6. What the AI actually does — and doesn't

This is the section most likely to disappoint, and it is measured rather than
argued.

The deterministic solvers close everything that is *arithmetic*: an orphaned
refund, a chargeback with its dispute fee, a manual adjustment, sub-rupee
rounding, a cutoff spill, a repricing on file. Every class that can be closed
mechanically **is** closed mechanically first.

That is deliberate, and it costs the demo something. Spill pairing would have
looked most impressive handed to an AI, and handing it over would have let the
model take credit for something arithmetic solves on its own.

So what reaches the model is only the cases that need a *reason* rather than a
sum. On the last live run: **7 cases attempted, 0 endpoint failures, 4 explained
but unverifiable, 0 booked, false-match rate unmoved.**

The model produced a worked explanation for every case it saw, and not one could
be verified against the merchant's own documents — so nothing was booked.

**That is the thesis landing, not failing.** The gate is arithmetic, not
confidence. A model that argues well does not get to move money.

---

## 7. The external check

Every settlement number above is measured on data this repository generated,
which is a real limitation. So the same discipline was run against **BenchRec** —
32,048 labelled bank statement lines from a real corporate cash ledger, produced
by researchers, not by anyone here. It ships with a language-model matcher's own
predictions, so the comparison is like-for-like on identical rows.

| | Coverage | Wrong-match rate |
|---|---:|---:|
| The matcher BenchRec ships with | 64.90% | **4.80%** |
| This project | **84.36%** | **0.28%** |

Higher coverage, **17× fewer wrong matches**, in half a second.

Worth stating honestly: BenchRec is a corporate cash ledger and is almost
entirely one-to-one, so it does **not** contain the many-to-one settlement
problem that is the harder half of this work. It is evidence that the matching
discipline transfers to data nobody here produced. It is not evidence that the
settlement problem is solved.

---

## 8. Engineering properties worth knowing

**Zero dependencies.** The whole deterministic core runs on a clean Python 3.11
using only the standard library — no pandas, no database, nothing installed.
This is verified rather than claimed: CI runs the full pipeline in a job with no
`pip install` anywhere in it.

**Deterministic.** The same seed produces a byte-identical run. CI diffs a repeat
run against itself, and diffs fresh runs against the committed results.

**Fast.** ~125,000 records/sec, single-threaded, effectively linear across a 50×
range of book sizes.

**The matcher cannot see the answer key.** A test parses every matching module
and fails if any of them imports the data generator or names a ground-truth
type — because a matcher that can reach the labels can be tuned against them,
accidentally or otherwise.

**443 tests**, all passing.

---

## 9. What is honestly not done

- **Not validated on a real merchant's full book.** One small recorded Razorpay
  pull already exposed two genuine bugs that synthetic data structurally could
  not contain. More real data would find more.
- **The journal and the work queue don't read your CSVs yet.** Reconciliation
  and categorisation do; posting and the persistent queue currently run only on
  generated books.
- **Two forgery attacks are uncontained**, declared rather than hidden.
- **The AI tier has resolved nothing** on the published run — which is the
  designed behaviour, but it means the model's value here is measured as a
  refusal rather than a contribution.

---

## Where to go next

- **[USAGE.md](USAGE.md)** — how to run it on your own data
- **[../README.md](../README.md)** — the evidence, with every number linked to
  the artifact that produced it
- **[../EVIDENCE.md](../EVIDENCE.md)** — how each tier earns its place
- **[../ARCHITECTURE.md](../ARCHITECTURE.md)** — design decisions and what is
  deliberately not built
