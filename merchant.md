# RecoAgent — a manual for merchants

This is written for the person who owns the books, not for the person who wrote
the code. If you accept payments online and a lump sum lands in your bank every
couple of days, this page is for you.

For the exhaustive column reference see [docs/USAGE.md](docs/USAGE.md). For the
design argument see the [README](README.md). This page is the practical one.

---

## 1. What this actually does for you

A customer pays you ₹1,000. You do not receive ₹1,000, and you do not receive it
today.

1. The gateway collects ₹1,000.
2. It keeps a commission (**MDR**) of roughly ₹20.
3. It charges **GST** on that commission, about ₹3.60.
4. Two days later it bundles your payment with a few hundred others and sends
   you **one credit** in your bank.

So your bank statement shows a single line — say ₹8,42,317.44 — standing for 300
sales, minus 300 commissions, minus tax, minus any refunds in between, possibly
minus a chargeback.

Somebody in your finance team then answers, by hand: **is that number right?
Which sales are inside it? Did the gateway charge the rate it promised? Did
anything go missing?**

This tool answers those questions automatically, and — this is the part that
matters — **it tells you what it could not answer instead of guessing.**

### The one promise it makes

A tool that matches everything and is occasionally wrong is worse than no tool.
It books money against the wrong sale and shows you a green tick, and nobody
investigates a green tick. So this one is built to stop and ask.

> **You will get a shorter matched list and an honest pile of exceptions,
> rather than a complete-looking list you cannot trust.**

---

## 2. What you need before you start

**Software:** Python 3.11 or newer. Nothing to install — no database, no
spreadsheet software, no cloud account.

**Data:** four exports covering the same period.

| Export | Where you get it | What it is |
|---|---|---|
| Orders | your own shop/ERP | what you asked customers to pay |
| Payments | gateway dashboard | what was actually captured, and what you were charged |
| Settlements | gateway dashboard | the payouts the gateway says it sent |
| Bank statement | your bank | the credits that actually arrived |

Refunds and chargebacks are optional but improve the result.

**Privacy:** everything runs on your own machine. Nothing is uploaded. The
reconciliation itself never contacts the internet.

---

## 3. Running it

Put the four files in a folder and run one command:

```bash
python3 -m recoagent.ingest --orders orders.csv --payments payments.csv --settlements settlements.csv --bank bank.csv
```

That is the whole thing.

**If you use Razorpay**, you can skip three of the four exports and pull them
straight from your account:

```bash
python3 -m recoagent.razorpay.run pull --out pull.json
python3 -m recoagent.razorpay.run reconcile pull.json --bank bank.csv
```

This needs your **test-mode** keys in a `.env` file. It refuses live keys on
purpose.

### You almost certainly do not need to rename your columns

Headers are matched loosely — `Order ID`, `order-id` and `ORDER_ID` all work —
and the names real exports use are already recognised: `MDR` and `commission`
for the fee, `GST` for the tax, `UTR` and `Chq/Ref No` for the bank reference,
`Particulars` and `Narration` for the description, `Txn Date` and `Value Date`
for the date. **Five real bank statement layouts (HDFC, ICICI, SBI, Axis,
Kotak) work without any configuration.**

If one column still is not found, it will tell you exactly which one. Rather
than editing your export, point at it:

```bash
python3 -m recoagent.ingest ... --map map.json
```

```json
{ "orders": { "amount_paise": "Gross Value" } }
```

### Money

Write rupees exactly as your export writes them — `12,345.67` is read correctly
down to the paisa. **Amounts are never converted to decimals internally**, so
nothing drifts by a rupee over ten thousand rows.

---

## 4. Reading what comes back

```
  Bank credits tied to a batch      160 of 164   (97.56%)
  Credit value cleared              Rs 2,49,17,275.49
  Outstanding                          Rs 4,60,814.88
  Open exceptions                        4
```

| Line | What it means for you |
|---|---|
| **Bank credits tied to a batch** | How many bank lines it explained completely |
| **Credit value cleared** | How much money is fully accounted for |
| **Outstanding** | Money it could not yet explain — *this is your work* |
| **Open exceptions** | How many items need a human |

Then it shows **what tied it out** — how each match was made. This matters: a
match made on an exact reference is stronger evidence than one recovered by
arithmetic, and they are listed separately rather than pooled into one number.

It may also show a **documented variance**. That is money on rows it *did*
match, where the amount still differs — most often the gateway captured less
than the order authorised. The pairing is certain, the shortfall is real, and
it is shown rather than quietly absorbed.

### The exception list is the product

Everything it refused is listed, biggest first, each with the reason it stopped.
That list is your queue for the day. Common entries in plain English:

| Reason | What to actually do |
|---|---|
| Bank credit does not match the batch total | Check for a refund or chargeback netted off |
| Two payments claim one order | A retry after a decline — confirm only one took money |
| The same reference appears twice | Ask the bank whether a credit was restated |
| No batch found for this credit | Money in from somewhere else, or a payout still in flight |
| Fee differs from the rate on file | Ask the gateway which rate applied |

Show more than the default ten:

```bash
python3 -m recoagent.ingest ... --exceptions 50
```

### If you prefer a screen to a terminal

```bash
python3 -m recoagent.ui
```

Every refusal opens into a case file: which checks were tried, why it stopped,
the credit exactly as your bank printed it, and the candidates a person would
check next.

---

## 5. The one thing it will never show you

**There is no accuracy percentage on your data, and that is deliberate.**

To say "97% accurate" a tool needs an answer key — a list of which rows truly
belong together, written by someone who already knows. Your exports do not come
with one.

So any tool that shows you an accuracy figure on *your* data is scoring itself
against its own opinion. This one refuses. You get coverage, the evidence behind
each match, and every item it declined — all of which you can check yourself.

---

## 6. Honest limits

- **Reconciliation and categorisation work on your files. The bookkeeping
  output does not, yet.** The double-entry journal and the persistent work queue
  currently run only on generated test books.
- **This has not been validated against a real merchant's full year.** One small
  real gateway pull already exposed two genuine bugs. Yours would likely find
  more — which is a reason to run it, not a reason not to.
- **A missing link gets a clear error, not a partial answer.** If your payments
  export has no column tying payments to payouts, it will say so rather than
  invent the connection.
- **It is not accounting advice.** It reconciles and reports; the judgement
  calls stay yours.

---

## 7. If something goes wrong

| What you see | What to do |
|---|---|
| `could not find a column for ...` | Name it in `--map`. The message says which file and field. |
| Almost everything is an exception | Usually the payments file has no settlement/payout id, or the bank reference column was not found. Check the "what tied it out" section. |
| Every amount looks 100× too big or small | Your files are already in paise — add `--money paise`. |
| `refusing a live key` | Intentional. Use test-mode credentials. |
| It matched fewer rows than another tool | Expected. Compare the exception lists, not the match rates — the refusals are where the difference is. |

---

## 8. A sensible first run

1. Export one month, not one year.
2. Run it and read the headline four lines.
3. Read the **whole** exception list, not just the top ten.
4. Work three or four exceptions by hand and check whether the stated reason was
   right.
5. If the reasons hold up on those, trust the rest of the list and scale up.

That fifth step is the point. This tool is asking you to check its refusals, not
to take its matches on faith.
