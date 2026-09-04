# Using this on your own data

Everything in the README is measured on data this repository generated, which
answers "is it right?". This page answers the other question: **"does it run on
mine?"**

There are two ways in. Pick the one that matches where your data lives.

---

## Door 1 — you have CSV exports

```bash
python3 -m recoagent.ingest \
    --orders orders.csv \
    --payments payments.csv \
    --settlements settlements.csv \
    --bank bank.csv
```

That is the whole command. Python 3.11 or newer, nothing to install.

### The four files, and what each one is

| File | What it is | Where it comes from |
|---|---|---|
| `--orders` | What you asked the customer to pay | your own system |
| `--payments` | What the gateway actually captured, and what it charged you | gateway dashboard export |
| `--settlements` | The payouts the gateway says it sent you | gateway dashboard export |
| `--bank` | The credits that actually landed | bank statement export |
| `--adjustments` | Refunds, chargebacks, dispute fees *(optional)* | gateway dashboard export |

### The columns each file needs

| File | Required columns |
|---|---|
| `--orders` | `order_id`, `customer_id`, `invoice_no`, `amount_paise`, `currency`, `created_at` |
| `--payments` | `payment_id`, `order_id`, `gross_paise`, `fee_paise`, `tax_paise`, `method`, `status`, `settlement_id`, `captured_at` |
| `--settlements` | `settlement_id`, `utr`, `settled_at`, `net_paise`, `status` |
| `--bank` | `bank_line_id`, `value_date`, `amount_paise`, `narration`, `bank_ref` |
| `--adjustments` | `adjustment_id`, `settlement_id`, `kind`, `payment_id`, `amount_paise`, `booked_at` |

`--payments` also accepts optional `currency` and `fx_rate`.

### You almost certainly do not need to rename anything

Header matching folds case and punctuation, so `Order ID`, `order-id` and
`ORDER_ID` all land on `order_id`. On top of that, the names real exports
actually use are recognised:

| Field it needs | Names it also accepts |
|---|---|
| `amount_paise` | amount, value, gross, gross_amount, order_amount |
| `gross_paise` | gross, amount, gross_amount, captured_amount |
| `fee_paise` | fee, **mdr**, commission |
| `tax_paise` | tax, **gst**, service_tax |
| `net_paise` | net, net_amount, settlement_amount, payout |
| `bank_ref` | **utr**, reference, ref, chq_ref_no, cheque_no, and the other shapes Indian bank statements use |
| `narration` | description, **particulars**, remarks, detail, transaction_remarks |
| `value_date` | date, txn_date, posting_date, credit_date, tran_date |
| `settled_at` | settlement_date, payout_date |
| `captured_at` | payment_date, txn_date, created_at |

Five real bank statement layouts — HDFC, ICICI, SBI, Axis and Kotak — are
covered by these aliases and tested.

### If a column still is not found

It fails with an error naming the column it could not find. It does **not**
guess, because a guess that silently maps the wrong column is worse than an
error message. Map it explicitly rather than editing your export:

```bash
python3 -m recoagent.ingest ... --map map.json
```

```json
{
  "orders":   { "amount_paise": "Gross Value" },
  "payments": { "fee_paise": "Commission Charged" },
  "bank":     { "narration": "Transaction Details" }
}
```

The shape is `{ file: { field_it_needs: your_column_name } }`.

### Money

Write rupees the way your export already writes them. `12,345.67` is read as a
string and becomes exactly `1234567` paise — floats never touch it, so nothing
drifts by a paisa over a large book.

If your files are already in the smallest unit, add `--money paise`.

### Useful flags

| Flag | What it does |
|---|---|
| `--exceptions 50` | show 50 unresolved items instead of the default 10 |
| `--out result.json` | write every match and exception to a file |
| `--rung B0` | run the plain exact-match baseline instead of the full solver |
| `--money paise` | your amounts are already in paise |
| `--map map.json` | column overrides |

---

## Door 2 — you use Razorpay directly

Skip the exports and pull the book from the API.

```bash
# 1. Fetch orders, payments, settlements and adjustments
python3 -m recoagent.razorpay.run pull --out data/razorpay/pull.json

# 2. Reconcile that against your bank statement
python3 -m recoagent.razorpay.run reconcile data/razorpay/pull.json --bank bank.csv
```

`pull` needs `RAZORPAY_KEY_ID` and `RAZORPAY_KEY_SECRET`, in your environment or
in an untracked `.env` file. **It refuses a live key** — test mode only, on
purpose.

`reconcile` needs no key and no network. It reads the recorded pull, which also
means anyone can replay the exact book a published number came from.

The only file you supply yourself is the bank statement, because Razorpay
provides the other three.

There is also a webhook receiver, if you want events as they happen:

```bash
python3 -m recoagent.razorpay.run serve --port 8000
```

It verifies the HMAC on the raw body and keeps an idempotent event log in
SQLite, so a replayed webhook cannot double-book anything.

---

## What you get back

Both doors print the same two things.

**Coverage** — how much of your credit tied out, and which tier tied it. A match
made by an exact reference is reported separately from one recovered by
arithmetic, because they are not equally strong.

**The exception list** — every item it would not resolve, each with the reason
it stopped. This is the part to read. It is your work queue.

To see it in a browser instead of a terminal:

```bash
python3 -m recoagent.ui
```

Every refusal opens into a case file: which tiers were tried, why it stopped,
the credit as the bank printed it, and the candidates an analyst would check
next.

---

## What it will not tell you, and why

**There is no accuracy percentage on your data.** That is deliberate, and it is
the most important sentence on this page.

A false-match rate needs an answer key — a list of which rows genuinely belong
together, written by someone who knows. The repository's own data has one,
because it was generated with one. BenchRec has one, because researchers
labelled it. **Your CSVs do not.**

So any tool that shows you an accuracy figure on your own unlabelled data is
scoring itself against its own opinion. This one refuses to. What it gives you
instead is coverage, the tier that made each match, and every item it declined —
which is information you can actually check.

---

## Honest limitations

**The journal and the work queue do not read your data yet.** `recoagent.journal`
(double-entry postings and the trial balance) and `recoagent.worklist` (the
persistent exception queue) currently run only on generated books, via
`--n/--seed/--profile`. Reconciliation and categorisation work on your CSVs;
posting and the queue do not, yet. Wiring them to the ingest path is
straightforward and simply has not been done.

**A poor export gets you a clear error, not a partial answer.** If your payments
file has no `settlement_id` linking payments to payouts, leg 2 has nothing to
work with and it will say so rather than inventing the link.

**Nothing here has been validated against a real merchant's full book.** One
small recorded Razorpay pull already exposed two genuine bugs that synthetic
data structurally could not contain. More real data would find more. That is an
argument for running it on yours, not against.

---

## Quick troubleshooting

| What you see | What it means |
|---|---|
| `could not find a column for X` | Name it in `--map`. The message says which file and which field. |
| Everything lands in exceptions | Usually `settlement_id` missing from payments, or the bank file's reference column not being found. Check the tier breakdown at the top of the output. |
| Amounts look 100× wrong | Your files are in paise already — add `--money paise`. |
| `refusing a live key` | By design. Use test-mode credentials. |
