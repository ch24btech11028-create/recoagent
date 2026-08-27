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
from .webstyle import CSS
from .pipeline import run_b0, run_b2
from .schemas import LabelledBatch, ReconResult
# Tier labels, rule names and the severity bands live in `views.py`, which the
# live console reads from too. Two surfaces onto one run should not be able to
# disagree about what a rule is called or how loud a row is.
from .views import RULE_LABEL, severity as _severity

MIXES = {"dev": (7, DefectMix.dev), "holdout": (21, DefectMix.holdout), "clean": (7, DefectMix.clean)}


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


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
{CSS}</style>

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
