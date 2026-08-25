"""Render a run as an exception queue an operator could actually work from.

The CLI prints what an engineer needs to check the system. This prints what a
finance-ops analyst needs at 9am after the nightly reconciliation ran: what
matched, what did not, how much money is sitting unexplained, and for each
unresolved item, what every tier already tried so nobody repeats the work by
hand.

Two deliberate separations:

- **The queue shows only what a real operator would have.** No defect labels,
  no ground truth -- the same restriction every matcher runs under. An ops
  screen that quietly displays the answer key would be a demo, not a product.
- **The verification panel is explicitly separate and explicitly labelled.** It
  uses ground truth, because it exists to answer "should you believe the
  numbers above", which is a different question from "what do I work on".

Standalone HTML: no server, no build step, no dependencies. `python -m
recoagent.report --out queue.html` and open the file.
"""

from __future__ import annotations

import argparse
import html
import sys
from datetime import datetime, timezone
from pathlib import Path

from .eval.scorer import Scorecard, score
from .generator import DefectMix, GeneratorConfig, generate
from .money import format_inr
from .pipeline import run_b0, run_b2
from .schemas import LabelledBatch, ReconResult

MIXES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout), "clean": (7, DefectMix.clean)}

#: Which tier each rule id belongs to, for the ladder shown on every row.
TIER_OF_RULE = {
    "leg1.t0.exact_order_id": "T0",
    "leg2.t0.exact_utr": "T0",
    "leg2.t1.amount_window": "T1",
    "leg2.t1.ssmp_residual": "T1",
    "leg2.t1.spill_pair": "T1",
    "leg2.t2.llm_hypothesis": "T2",
}

RULE_LABEL = {
    "leg1.t0.exact_order_id": "exact order id",
    "leg2.t0.exact_utr": "exact UTR",
    "leg2.t1.amount_window": "amount + date window",
    "leg2.t1.ssmp_residual": "subset-sum over unlinked rows",
    "leg2.t1.spill_pair": "cross-batch cutoff spill",
    "leg2.t2.llm_hypothesis": "model hypothesis, arithmetic verified",
}


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _severity(residual_paise: int | None) -> tuple[str, str]:
    """Form as well as number: how loud should this row be?"""
    if residual_paise is None:
        return "structural", "no amount in dispute"
    magnitude = abs(residual_paise)
    if magnitude >= 10_000_00:
        return "critical", "over Rs 10,000 unexplained"
    if magnitude >= 100_00:
        return "warn", "over Rs 100 unexplained"
    return "minor", "under Rs 100"


def _queue_rows(result: ReconResult) -> str:
    rows: list[str] = []
    ordered = sorted(
        result.exceptions,
        key=lambda e: (-abs(e.residual_paise or 0), e.entity_id),
    )
    for exc in ordered:
        level, hint = _severity(exc.residual_paise)
        amount = format_inr(exc.residual_paise) if exc.residual_paise is not None else "--"
        direction = ""
        if exc.residual_paise:
            direction = "short" if exc.residual_paise < 0 else "over"
        suspect = exc.suspected_class.value if exc.suspected_class else "not classified"
        stopped = exc.escalated_from_tier or "T0"
        rows.append(f"""
        <tr class="row row--{level}">
          <td class="cell-id"><span class="chip chip--{level}" title="{_esc(hint)}"></span>
              <code>{_esc(exc.entity_id)}</code></td>
          <td class="cell-leg"><span class="leg">Leg {exc.leg}</span></td>
          <td class="cell-amount num">{_esc(amount)}<span class="dir">{_esc(direction)}</span></td>
          <td class="cell-ladder">{_ladder(stopped)}</td>
          <td class="cell-class"><span class="tag">{_esc(suspect)}</span></td>
          <td class="cell-why">{_esc(exc.reason)}</td>
        </tr>""")
    return "".join(rows)


def _ladder(stopped_at: str) -> str:
    """Which tiers looked at this before giving up. True information, not decoration."""
    order = ["T0", "T1", "T2"]
    reached = order.index(stopped_at) if stopped_at in order else 0
    cells = []
    for i, tier in enumerate(order):
        state = "done" if i < reached else ("stop" if i == reached else "skip")
        cells.append(f'<span class="rung rung--{state}" title="{tier}">{tier}</span>')
    return f'<span class="ladder">{"".join(cells)}</span>'


def _matched_rows(result: ReconResult, limit: int = 12) -> str:
    leg2 = [m for m in result.matches_for_leg(2) if m.proof is not None]
    leg2.sort(key=lambda m: (m.tier, m.match_id))
    picked = [m for m in leg2 if m.tier != "T0"][:limit] or leg2[:limit]
    rows = []
    for m in picked:
        rows.append(f"""
        <tr>
          <td><code>{_esc(m.left_ids[0])}</code></td>
          <td><code>{_esc(m.right_ids[0])}</code></td>
          <td><span class="rung rung--done">{_esc(m.tier)}</span></td>
          <td>{_esc(RULE_LABEL.get(m.rule_id, m.rule_id))}</td>
          <td class="num">{m.confidence:.2f}</td>
          <td class="num proof">{_esc(format_inr(m.proof.residual_paise))}</td>
        </tr>""")
    return "".join(rows)


def _accounting_rows(card: Scorecard) -> str:
    rows = []
    for a in card.accounting:
        state = "ok" if a.mishandled == 0 else "bad"
        rows.append(f"""
        <tr>
          <td><code>{_esc(a.defect.value)}</code></td>
          <td class="num">{a.injected}</td>
          <td class="num">{a.flagged}</td>
          <td class="num">{a.resolved}</td>
          <td class="num verdict verdict--{state}">{a.mishandled}</td>
        </tr>""")
    return "".join(rows)


def render(batch: LabelledBatch, result: ReconResult, card: Scorecard) -> str:
    counts = batch.sources.counts
    unexplained = sum(abs(e.residual_paise or 0) for e in result.exceptions)
    leg2 = card.legs[2]
    generated = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    reconciles = "RECONCILES" if card.fully_reconciles else "DOES NOT RECONCILE"

    return f"""<title>Settlement Exception Queue</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;450;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>
:root {{
  --ground:#F4F6F8; --surface:#FFFFFF; --surface-2:#EBEEF2; --surface-3:#F8F9FB;
  --ink:#11151C; --ink-2:#3A4351; --muted:#5C6675;
  --rule:#DDE2E8; --rule-2:#C4CBD5;
  --ok:#1C5A4A; --ok-bg:#E3EEEA;
  --warn:#8A5214; --warn-bg:#F6EBDB;
  --crit:#8C2B26; --crit-bg:#F7E5E4;
  --flat:#4A5464; --flat-bg:#E9ECF1;
  --shadow:0 1px 2px rgba(17,21,28,.05),0 10px 26px -20px rgba(17,21,28,.3);
  --mono:"IBM Plex Mono",ui-monospace,Menlo,monospace;
  --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
  --serif:"Newsreader",ui-serif,Georgia,serif;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#0E1116; --surface:#151A21; --surface-2:#1D232C; --surface-3:#121720;
    --ink:#E7EBF1; --ink-2:#C0C8D4; --muted:#8F99A8;
    --rule:#252C36; --rule-2:#374050;
    --ok:#6FB49B; --ok-bg:#15271F;
    --warn:#D2954A; --warn-bg:#2A2013;
    --crit:#DE8079; --crit-bg:#2C1817;
    --flat:#98A2B0; --flat-bg:#1C222B;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 26px -20px rgba(0,0,0,.8);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#0E1116; --surface:#151A21; --surface-2:#1D232C; --surface-3:#121720;
  --ink:#E7EBF1; --ink-2:#C0C8D4; --muted:#8F99A8;
  --rule:#252C36; --rule-2:#374050;
  --ok:#6FB49B; --ok-bg:#15271F;
  --warn:#D2954A; --warn-bg:#2A2013;
  --crit:#DE8079; --crit-bg:#2C1817;
  --flat:#98A2B0; --flat-bg:#1C222B;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 26px -20px rgba(0,0,0,.8);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1220px;margin:0 auto;padding:0 22px 80px;display:flex;flex-direction:column;gap:26px}}
code{{font-family:var(--mono);font-size:.9em}}
.num{{font-variant-numeric:tabular-nums;font-family:var(--mono)}}

header.top{{padding:34px 0 0;display:flex;flex-direction:column;gap:6px;border-bottom:1px solid var(--rule);padding-bottom:20px}}
.eyebrow{{font-family:var(--mono);font-size:11px;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}}
h1{{font-family:var(--serif);font-weight:500;font-size:clamp(1.9rem,4vw,2.6rem);
    letter-spacing:-.02em;line-height:1.05;margin:0;text-wrap:balance}}
.sub{{color:var(--ink-2);max-width:70ch;margin:0}}

.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
        background:var(--rule);border:1px solid var(--rule);border-radius:5px;overflow:hidden}}
.stat{{background:var(--surface);padding:15px 17px;display:flex;flex-direction:column;gap:3px}}
.stat dt{{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}}
.stat dd{{margin:0;font-size:1.42rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.2}}
.stat dd small{{display:block;font-size:.7rem;font-weight:450;color:var(--muted)}}
.stat--hero dd{{color:var(--ok)}}
.stat--money dd{{color:var(--crit)}}

section{{display:flex;flex-direction:column;gap:12px}}
h2{{font-family:var(--serif);font-weight:500;font-size:1.4rem;margin:0;letter-spacing:-.01em}}
.note{{color:var(--muted);font-size:.88rem;max-width:78ch;margin:0}}

.panel{{border:1px solid var(--rule);border-radius:5px;background:var(--surface);
        overflow-x:auto;box-shadow:var(--shadow)}}
table{{width:100%;border-collapse:collapse;min-width:820px;font-size:.88rem}}
thead th{{position:sticky;top:0;background:var(--surface-2);text-align:left;
          font-family:var(--mono);font-size:10px;font-weight:500;letter-spacing:.09em;
          text-transform:uppercase;color:var(--muted);padding:10px 14px;
          border-bottom:1px solid var(--rule-2);white-space:nowrap}}
tbody td{{padding:11px 14px;border-bottom:1px solid var(--rule);vertical-align:top;color:var(--ink-2)}}
tbody tr:last-child td{{border-bottom:0}}
tbody tr:hover td{{background:var(--surface-3)}}
.cell-id code{{color:var(--ink);font-weight:500}}
.cell-amount{{white-space:nowrap;color:var(--ink);font-weight:500}}
.dir{{display:block;font-family:var(--sans);font-size:.7rem;font-weight:450;color:var(--muted);letter-spacing:.02em}}
.cell-why{{min-width:320px;font-size:.84rem;line-height:1.45}}

.chip{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:8px;vertical-align:middle}}
.chip--critical{{background:var(--crit)}} .chip--warn{{background:var(--warn)}}
.chip--minor{{background:var(--ok)}} .chip--structural{{background:var(--flat)}}
.leg{{font-family:var(--mono);font-size:11px;color:var(--muted)}}
.tag{{font-family:var(--mono);font-size:10.5px;letter-spacing:.03em;background:var(--surface-2);
      border:1px solid var(--rule);border-radius:3px;padding:2px 6px;color:var(--ink-2);white-space:nowrap}}

.ladder{{display:inline-flex;gap:3px}}
.rung{{font-family:var(--mono);font-size:10px;font-weight:500;padding:2px 5px;border-radius:3px;
       border:1px solid var(--rule-2);color:var(--muted);background:var(--surface-2)}}
.rung--done{{background:var(--ok-bg);border-color:var(--ok);color:var(--ok)}}
.rung--stop{{background:var(--warn-bg);border-color:var(--warn);color:var(--warn)}}
.rung--skip{{opacity:.42}}

.verdict{{font-weight:600}}
.verdict--ok{{color:var(--ok)}} .verdict--bad{{color:var(--crit)}}
.proof{{color:var(--ok)}}

.legend{{display:flex;gap:18px;flex-wrap:wrap;font-size:.8rem;color:var(--muted);
         font-family:var(--mono);letter-spacing:.02em}}
.callout{{border:1px solid var(--rule);border-left:2px solid var(--ok);border-radius:4px;
          background:var(--surface);padding:14px 17px;display:flex;flex-direction:column;gap:6px}}
.callout b{{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;color:var(--ok)}}
.callout p{{margin:0;font-size:.89rem;color:var(--ink-2);max-width:78ch}}

footer{{border-top:1px solid var(--rule);padding-top:16px;font-family:var(--mono);
        font-size:11px;color:var(--muted);display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<div class="wrap">
  <header class="top">
    <div class="eyebrow">Nightly reconciliation &middot; profile {_esc(card.profile)} &middot; seed {card.seed} &middot; rung {_esc(result.rung)}</div>
    <h1>Settlement Exception Queue</h1>
    <p class="sub">{counts['orders']:,} orders and {counts['payments']:,} payments consolidated into
    {counts['settlements']} settlement batches, checked against {counts['bank_lines']} bank credits.
    Everything below is what the automated tiers could not close.</p>
  </header>

  <dl class="stats">
    <div class="stat stat--hero"><dt>False-match rate</dt><dd>{card.overall_false_match_rate:.2%}<small>money filed against the wrong entry</small></dd></div>
    <div class="stat"><dt>Auto-matched</dt><dd>{card.overall_auto_match_rate:.1%}<small>{leg2.true_matches} of {leg2.population} credits</small></dd></div>
    <div class="stat"><dt>Credit value cleared</dt><dd>{card.value.share:.1%}<small>{_esc(format_inr(card.value.matched_credit))}</small></dd></div>
    <div class="stat stat--money"><dt>Unexplained</dt><dd>{_esc(format_inr(unexplained))}<small>across {len(result.exceptions)} open items</small></dd></div>
    <div class="stat"><dt>Needs a human</dt><dd>{len(result.exceptions)}<small>queue below</small></dd></div>
  </dl>

  <section>
    <h2>Open items</h2>
    <p class="note">Sorted by money at stake. The ladder shows which tiers examined the item and
    where it stopped &mdash; nobody needs to repeat work the solver already did.</p>
    <div class="legend">
      <span><span class="chip chip--critical"></span>over Rs 10,000</span>
      <span><span class="chip chip--warn"></span>over Rs 100</span>
      <span><span class="chip chip--minor"></span>under Rs 100</span>
      <span><span class="chip chip--structural"></span>no amount in dispute</span>
    </div>
    <div class="panel">
      <table>
        <thead><tr>
          <th>Item</th><th>Leg</th><th>Gap</th><th>Tiers tried</th><th>Suspected</th><th>What happened</th>
        </tr></thead>
        <tbody>{_queue_rows(result)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Recovered by the solver</h2>
    <p class="note">Credits that did not match on an exact key and were recovered anyway. Every one
    carries an arithmetic proof; the residual column is what remained after it closed.</p>
    <div class="panel">
      <table>
        <thead><tr>
          <th>Bank credit</th><th>Settlement</th><th>Tier</th><th>Rule</th><th>Confidence</th><th>Residual</th>
        </tr></thead>
        <tbody>{_matched_rows(result)}</tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Verification</h2>
    <p class="note">This panel is the only thing on the page that uses the answer key, and it exists
    to answer a different question from the queue above: should you believe any of it? Every injected
    defect must be either flagged for a human or correctly resolved. <strong>Mishandled must be zero</strong>
    &mdash; that column counts defects that produced a wrong match or that the system walked past.</p>
    <div class="panel">
      <table>
        <thead><tr>
          <th>Defect class</th><th>Injected</th><th>Flagged</th><th>Resolved</th><th>Mishandled</th>
        </tr></thead>
        <tbody>{_accounting_rows(card)}</tbody>
      </table>
    </div>
    <div class="callout">
      <b>Ground-truth accounting &mdash; {_esc(reconciles)}</b>
      <p>{sum(a.injected for a in card.accounting)} defects injected, {card.mishandled_total} mishandled,
      {card.unattributed_exceptions} exceptions raised against records with nothing wrong with them.
      A reconciliation engine that matches everything and is occasionally wrong is worse than useless:
      it files money against the wrong transaction and hides the error behind a green number.</p>
    </div>
  </section>

  <footer>
    <span>Generated {_esc(generated)}</span>
    <span>rung {_esc(result.rung)} &middot; profile {_esc(card.profile)} &middot; seed {card.seed}</span>
  </footer>
</div>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.report")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--profile", choices=sorted(MIXES), default="dev")
    ap.add_argument("--rung", choices=["B0", "B2"], default="B2")
    ap.add_argument("--out", default="queue.html")
    args = ap.parse_args(argv)

    seed, mix_factory = MIXES[args.profile]
    batch = generate(GeneratorConfig(n_orders=args.n, seed=seed, mix=mix_factory()))
    result = run_b0(batch.sources) if args.rung == "B0" else run_b2(batch.sources)
    card = score(batch, result)

    Path(args.out).write_text(render(batch, result, card))
    print(f"  wrote {args.out}  ({len(result.exceptions)} open items, "
          f"false-match rate {card.overall_false_match_rate:.2%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
