"""The console's single page: markup, the console-only styles, and the client.

Kept apart from `ui.py` so the server reads as a server. The shared palette and
table styles come from `webstyle.CSS`; everything added here is for controls
that only exist when there is something to click.
"""

from __future__ import annotations

from .webstyle import CSS

UI_CSS = r"""
.bar{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;padding:14px 16px;
     border:1px solid var(--rule);border-radius:5px;background:var(--surface);box-shadow:var(--shadow)}
.field{display:flex;flex-direction:column;gap:4px}
.field label{font-family:var(--mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.field input,.field select{font-family:var(--mono);font-size:13px;padding:7px 9px;border-radius:4px;
     border:1px solid var(--rule-2);background:var(--surface-3);color:var(--ink);min-width:96px}
.field input:focus,.field select:focus,button:focus-visible,textarea:focus-visible{
     outline:2px solid var(--ok);outline-offset:1px}
button{font-family:var(--sans);font-size:13px;font-weight:550;padding:8px 15px;border-radius:4px;
     border:1px solid var(--rule-2);background:var(--surface-2);color:var(--ink);cursor:pointer}
button:hover:not(:disabled){background:var(--surface-3);border-color:var(--muted)}
button:disabled{opacity:.45;cursor:not-allowed}
button.primary{background:var(--ok);border-color:var(--ok);color:#fff}
:root[data-theme="dark"] button.primary{color:#0E1116}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) button.primary{color:#0E1116}}
.spacer{flex:1}

.askbox{display:flex;gap:9px;align-items:stretch}
.askbox textarea{flex:1;font-family:var(--sans);font-size:14px;padding:10px 12px;border-radius:4px;
     border:1px solid var(--rule-2);background:var(--surface-3);color:var(--ink);resize:vertical;min-height:44px}
.chips{display:flex;gap:7px;flex-wrap:wrap}
.chips button{font-family:var(--mono);font-size:11px;padding:5px 9px;color:var(--ink-2)}

.card{border:1px solid var(--rule);border-radius:5px;background:var(--surface);
      padding:15px 17px;display:flex;flex-direction:column;gap:9px;box-shadow:var(--shadow)}
.card--warn{border-left:2px solid var(--warn)}
.card--crit{border-left:2px solid var(--crit)}
.card--ok{border-left:2px solid var(--ok)}
.answer{font-family:var(--mono);font-size:1.35rem;font-weight:600;color:var(--ink);
        font-variant-numeric:tabular-nums;word-break:break-word}
.meta{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;color:var(--muted);
      display:flex;gap:14px;flex-wrap:wrap}
.banner{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
        padding:3px 8px;border-radius:3px;align-self:flex-start}
.banner--ungraded{background:var(--warn-bg);color:var(--warn);border:1px solid var(--warn)}
.banner--measured{background:var(--ok-bg);color:var(--ok);border:1px solid var(--ok)}
details.sheet{border-top:1px solid var(--rule);padding-top:9px}
details.sheet summary{cursor:pointer;font-family:var(--mono);font-size:11px;color:var(--muted)}
details.sheet pre{margin:9px 0 0;padding:11px 13px;background:var(--surface-3);border:1px solid var(--rule);
     border-radius:4px;overflow-x:auto;font-family:var(--mono);font-size:11.5px;line-height:1.5;color:var(--ink-2)}

tbody tr.exp{cursor:pointer}
tbody tr.detail td{background:var(--surface-3);padding:0}
.detailwrap{padding:13px 16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.dkv{display:flex;flex-direction:column;gap:2px}
.dkv dt{font-family:var(--mono);font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted)}
.dkv dd{margin:0;font-size:.86rem;color:var(--ink)}
.dkv dd.wide{grid-column:1/-1}

.empty{padding:26px;text-align:center;color:var(--muted);font-size:.88rem}
.err{display:block;border-left:2px solid var(--crit);background:var(--crit-bg);color:var(--crit);
     padding:11px 14px;border-radius:4px;font-size:.86rem;white-space:pre-wrap;font-family:var(--mono);
     line-height:1.5;overflow-x:auto}
.load{display:inline-block;width:11px;height:11px;border:2px solid var(--rule-2);border-top-color:var(--ok);
      border-radius:50%;animation:spin .7s linear infinite;vertical-align:-1px;margin-right:7px}
@keyframes spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion:reduce){.load{animation:none}}
.qrow{display:flex;gap:10px;padding:7px 0;border-bottom:1px solid var(--rule);font-size:.83rem;align-items:baseline}
.qrow:last-child{border-bottom:0}
.qmark{font-family:var(--mono);font-size:10px;padding:1px 6px;border-radius:3px;white-space:nowrap}
.qmark--ok{background:var(--ok-bg);color:var(--ok)}
.qmark--bad{background:var(--crit-bg);color:var(--crit)}
.qmark--dec{background:var(--flat-bg);color:var(--flat)}
"""

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RecoAgent Console</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;450;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>__CSS__</style>
</head>
<body>
<div class="wrap">

  <header class="top">
    <div class="eyebrow" id="eyebrow">Multi-source settlement reconciliation</div>
    <h1>RecoAgent Console</h1>
    <p class="sub">Three sources that disagree — merchant orders, gateway settlements, bank credits —
    reconciled across two legs. Nothing below becomes a match because something believed it should;
    every one carries arithmetic that was re-derived from the payment rows.</p>
  </header>

  <div class="bar">
    <div class="field"><label for="n">Orders</label><input id="n" type="number" value="2000" min="100" max="50000" step="100"></div>
    <div class="field"><label for="seed">Seed</label><input id="seed" type="number" value="7"></div>
    <div class="field"><label for="profile">Defect mix</label>
      <select id="profile"><option value="dev">dev</option><option value="holdout">held-out</option><option value="clean">clean</option></select></div>
    <div class="field"><label for="rung">Rung</label>
      <select id="rung"><option value="B2">B2 — solver</option><option value="B0">B0 — exact keys</option></select></div>
    <button class="primary" id="run">Reconcile</button>
    <div class="spacer"></div>
    <div class="field"><label for="theme">Theme</label>
      <select id="theme"><option value="auto">auto</option><option value="light">light</option><option value="dark">dark</option></select></div>
  </div>

  <div id="err"></div>

  <dl class="stats" id="stats"></dl>

  <section>
    <h2>Ask the settlement agent</h2>
    <p class="note" id="asknote">The agent answers from a factsheet built from this run by code —
    it has no memory of the book and no way to invent one. An answer typed here has no ground truth,
    so it comes back <strong>ungraded</strong> with that factsheet attached for you to check.
    The measured number is the second button.</p>
    <div class="askbox">
      <textarea id="q" rows="2" placeholder="How much is sitting unexplained across the open items?"></textarea>
      <button class="primary" id="ask">Ask</button>
    </div>
    <div class="chips" id="chips"></div>
    <div id="answer"></div>
  </section>

  <section>
    <h2>Measured Q&amp;A accuracy</h2>
    <p class="note">The question bank is derived from this run, so every answer has a ground truth
    computed by code — including questions the factsheet deliberately cannot answer, where the only
    correct response is to decline. <strong>Wrong-answer rate leads</strong>, because an operator acts
    on a number either way and a confident wrong one is the expensive failure.</p>
    <div class="askbox">
      <button id="bank">Run the scored bank</button>
      <div class="field"><label for="limit">Questions (0 = all)</label><input id="limit" type="number" value="0" min="0" max="60"></div>
    </div>
    <div id="bankout"></div>
  </section>

  <section>
    <h2>Open items <span class="leg" id="qcount"></span></h2>
    <p class="note">Sorted by money at stake. Click a row for what every tier already tried, so nobody
    repeats the solver's work by hand. This table is built from the sources and the result only —
    the same restriction the matchers run under.</p>
    <div class="legend">
      <span><span class="chip chip--critical"></span>over Rs 10,000</span>
      <span><span class="chip chip--warn"></span>over Rs 100</span>
      <span><span class="chip chip--minor"></span>under Rs 100</span>
      <span><span class="chip chip--structural"></span>no amount in dispute</span>
    </div>
    <div class="panel"><table>
      <thead><tr><th>Item</th><th>Leg</th><th>Gap</th><th>Tiers tried</th><th>Suspected</th><th>What happened</th></tr></thead>
      <tbody id="queue"></tbody>
    </table></div>
  </section>

  <section>
    <h2>Recovered by the solver</h2>
    <p class="note">Credits that did not match on an exact key and were recovered anyway. The residual
    column is what was left after the arithmetic closed.</p>
    <div class="panel"><table>
      <thead><tr><th>Bank credit</th><th>Settlement</th><th>Tier</th><th>Rule</th><th>Confidence</th><th>Residual</th></tr></thead>
      <tbody id="recovered"></tbody>
    </table></div>
  </section>

  <section>
    <h2>Verification</h2>
    <p class="note">The only panel on this page that uses the answer key, and it answers a different
    question from the queue above: should you believe any of it? Every injected defect must be flagged
    for a human or correctly resolved. <strong>Mishandled must be zero.</strong></p>
    <div class="panel"><table>
      <thead><tr><th>Defect class</th><th>Injected</th><th>Flagged</th><th>Resolved</th><th>Mishandled</th></tr></thead>
      <tbody id="accounting"></tbody>
    </table></div>
    <div id="verdict"></div>
  </section>

  <footer>
    <span id="foot">—</span>
    <span id="modelfoot">—</span>
  </footer>
</div>

<script>
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(2) + "%");
let RUN = null, MODEL = null, busy = false;

function params() {
  return { n: +$("n").value, seed: +$("seed").value, profile: $("profile").value, rung: $("rung").value };
}
function showErr(msg) { $("err").innerHTML = msg ? `<div class="err">${esc(msg)}</div>` : ""; }

async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({ error: `${r.status} ${r.statusText}` }));
  if (!r.ok) throw new Error(data.error || `${r.status}`);
  return data;
}

const ladder = (stop) => ["T0","T1","T2"].map((t,i) => {
  const at = ["T0","T1","T2"].indexOf(stop);
  const state = i < at ? "done" : (i === at ? "stop" : "skip");
  return `<span class="rung rung--${state}">${t}</span>`;
}).join("");

function renderRun(d) {
  RUN = d;
  const h = d.headline;
  $("eyebrow").textContent = `${d.key.profile} mix · seed ${d.key.seed} · rung ${d.key.rung} · ${d.counts.orders.toLocaleString()} orders`;
  $("stats").innerHTML = `
    <div class="stat stat--hero"><dt>False-match rate</dt><dd>${pct(h.false_match_rate)}<small>money filed against the wrong entry</small></dd></div>
    <div class="stat"><dt>Auto-matched</dt><dd>${pct(h.auto_match_rate)}<small>across both legs</small></dd></div>
    <div class="stat"><dt>Credit value cleared</dt><dd>${pct(h.value_share)}<small>${esc(h.value_matched)}</small></dd></div>
    <div class="stat stat--money"><dt>Unexplained</dt><dd>${esc(h.unexplained)}<small>across ${h.open_items} open items</small></dd></div>
    <div class="stat"><dt>Leg 2 recall</dt><dd>${pct((d.legs.find(l=>l.leg===2)||{}).recall)}<small>credit → batch, the N:1 leg</small></dd></div>`;

  $("qcount").textContent = `${d.queue.length} items`;
  $("queue").innerHTML = d.queue.map((e, i) => `
    <tr class="row row--${e.severity} exp" data-i="${i}">
      <td class="cell-id"><span class="chip chip--${e.severity}" title="${esc(e.severity_hint)}"></span><code>${esc(e.id)}</code></td>
      <td class="cell-leg"><span class="leg">Leg ${e.leg}</span></td>
      <td class="cell-amount num">${esc(e.amount)}<span class="dir">${esc(e.direction)}</span></td>
      <td>${ladder(e.stopped_at)}</td>
      <td><span class="tag">${esc(e.suspected)}</span></td>
      <td class="cell-why">${esc(e.reason)}</td>
    </tr>`).join("") || `<tr><td colspan="6" class="empty">Nothing open. Every credit tied out.</td></tr>`;

  $("recovered").innerHTML = d.recovered.map(m => `
    <tr><td><code>${esc(m.left)}</code></td><td><code>${esc(m.right)}</code></td>
    <td><span class="rung rung--done">${esc(m.tier)}</span></td><td>${esc(m.rule)}</td>
    <td class="num">${m.confidence.toFixed(2)}</td><td class="num proof">${esc(m.residual)}</td></tr>`).join("")
    || `<tr><td colspan="6" class="empty">Every credit matched on an exact key.</td></tr>`;

  $("accounting").innerHTML = d.accounting.map(a => `
    <tr><td><code>${esc(a.defect)}</code></td><td class="num">${a.injected}</td><td class="num">${a.flagged}</td>
    <td class="num">${a.resolved}</td><td class="num verdict verdict--${a.mishandled === 0 ? "ok" : "bad"}">${a.mishandled}</td></tr>`).join("");

  const v = d.verdict;
  $("verdict").innerHTML = `<div class="callout"><b>Ground-truth accounting — ${v.fully_reconciles ? "RECONCILES" : "DOES NOT RECONCILE"}</b>
    <p>${v.injected_total} defects injected, ${v.mishandled_total} mishandled, ${v.unattributed} exceptions raised
    against records with nothing wrong with them. A reconciliation engine that matches everything and is
    occasionally wrong is worse than useless: it files money against the wrong transaction and hides the
    error behind a green number.</p></div>`;

  $("foot").textContent = `${d.counts.payments.toLocaleString()} payments · ${d.counts.settlements} batches · ${d.counts.bank_lines} bank credits · reconciled in ${d.seconds}s`;
  $("chips").innerHTML = suggestions(d).map(q => `<button class="sugg">${esc(q)}</button>`).join("");
  document.querySelectorAll(".sugg").forEach(b => b.onclick = () => { $("q").value = b.textContent; doAsk(); });
}

function suggestions(d) {
  const out = ["How many items are still open in the exception queue?",
               "What is the total unexplained amount across all open items, in paise?"];
  const first = d.queue.find(e => e.residual_paise != null);
  if (first) out.push(`What is the gap on ${first.id}, in paise?`);
  out.push("What is the merchant's registered GSTIN?");
  return out;
}

document.addEventListener("click", (ev) => {
  const tr = ev.target.closest("tr.exp");
  if (!tr || !RUN) return;
  const next = tr.nextElementSibling;
  if (next && next.classList.contains("detail")) { next.remove(); return; }
  const e = RUN.queue[+tr.dataset.i];
  const row = document.createElement("tr");
  row.className = "detail";
  row.innerHTML = `<td colspan="6"><div class="detailwrap">
    <div class="dkv"><dt>Entity</dt><dd><code>${esc(e.id)}</code> · ${esc(e.kind)}</dd></div>
    <div class="dkv"><dt>Residual</dt><dd class="num">${e.residual_paise == null ? "no amount in dispute" : e.residual_paise.toLocaleString() + " paise"}</dd></div>
    <div class="dkv"><dt>Stopped at</dt><dd>${ladder(e.stopped_at)}</dd></div>
    <div class="dkv"><dt>Suspected class</dt><dd><span class="tag">${esc(e.suspected)}</span></dd></div>
    <div class="dkv"><dt>Severity</dt><dd>${esc(e.severity_hint)}</dd></div>
    <div class="dkv"><dt>Full reason</dt><dd class="wide">${esc(e.reason)}</dd></div>
  </div></td>`;
  tr.after(row);
});

async function doRun() {
  if (busy) return;
  busy = true; showErr(""); $("run").disabled = true; $("run").innerHTML = `<span class="load"></span>Reconciling`;
  try { renderRun(await post("/api/run", params())); }
  catch (e) { showErr(e.message); }
  finally { busy = false; $("run").disabled = false; $("run").textContent = "Reconcile"; }
}

async function doAsk() {
  const text = $("q").value.trim();
  if (!text || !RUN) return;
  $("ask").disabled = true;
  $("answer").innerHTML = `<div class="card"><div class="meta"><span class="load"></span>asking ${esc(MODEL ? MODEL.model : "the model")}…</div></div>`;
  try {
    const a = await post("/api/ask", { ...params(), question: text });
    const tone = a.failed ? "crit" : (a.declined ? "warn" : "ok");
    const headline = a.failed ? "call failed"
      : (a.declined ? "Declined to answer" : (a.answer === null ? "—" : String(a.answer)));
    $("answer").innerHTML = `<div class="card card--${tone}">
      <span class="banner banner--ungraded">Ungraded — no ground truth for a live question</span>
      <div class="answer">${esc(headline)}</div>
      ${a.basis ? `<p class="note">${esc(a.basis)}</p>` : ""}
      ${a.detail && (a.failed || a.declined) ? `<p class="note">${esc(a.detail)}</p>` : ""}
      <div class="meta">
        <span>${esc(MODEL ? MODEL.model : "")}</span>
        ${a.confidence != null ? `<span>self-reported confidence ${a.confidence}</span>` : ""}
        <span>${a.seconds}s</span><span>${a.tokens_in.toLocaleString()} in / ${a.tokens_out.toLocaleString()} out</span>
      </div>
      <details class="sheet"><summary>Everything the model was allowed to see — check the answer against it</summary>
        <pre>${esc(JSON.stringify(a.factsheet, null, 2))}</pre></details>
    </div>`;
  } catch (e) { $("answer").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  finally { $("ask").disabled = false; }
}

async function doBank() {
  if (!RUN) return;
  $("bank").disabled = true;
  $("bankout").innerHTML = `<div class="card"><div class="meta"><span class="load"></span>asking the scored bank — this makes one call per question…</div></div>`;
  try {
    const b = await post("/api/bank", { ...params(), limit: +$("limit").value });
    const rows = b.answers.map(a => {
      const mark = a.correct ? "ok" : (a.declined ? "dec" : "bad");
      const label = a.correct ? (a.declined ? "right to decline" : "correct") : (a.failed ? "failed" : (a.declined ? "declined" : "WRONG"));
      const detail = a.correct ? "" : esc(a.detail || "");
      return `<div class="qrow"><span class="qmark qmark--${mark}">${label}</span>
              <span>${esc(a.text)}${detail ? ` <span class="leg">— ${detail}</span>` : ""}</span></div>`;
    }).join("");
    $("bankout").innerHTML = `<div class="card card--${b.wrong_answer_rate === 0 && b.hallucinated === 0 ? "ok" : "crit"}">
      <span class="banner banner--measured">Measured — every question graded by code</span>
      <dl class="stats">
        <div class="stat stat--hero"><dt>Wrong-answer rate</dt><dd>${pct(b.wrong_answer_rate)}<small>of ${b.attempted} answered</small></dd></div>
        <div class="stat"><dt>Coverage</dt><dd>${pct(b.coverage)}<small>${b.attempted} of ${b.total} attempted</small></dd></div>
        <div class="stat"><dt>Accuracy</dt><dd>${pct(b.accuracy)}<small>correct of all questions</small></dd></div>
        <div class="stat stat--money"><dt>Hallucinated</dt><dd>${b.hallucinated}<small>answered what the factsheet cannot support</small></dd></div>
      </dl>
      <div class="meta"><span>${b.correct} correct · ${b.wrong} wrong · ${b.declined} declined · ${b.failed} failed</span>
        <span>${b.seconds}s</span><span>${b.tokens_in.toLocaleString()} in / ${b.tokens_out.toLocaleString()} out</span></div>
      <details class="sheet" open><summary>Every question, and what it answered</summary><div>${rows}</div></details>
    </div>`;
  } catch (e) { $("bankout").innerHTML = `<div class="err">${esc(e.message)}</div>`; }
  finally { $("bank").disabled = false; }
}

$("run").onclick = doRun;
$("ask").onclick = doAsk;
$("bank").onclick = doBank;
$("q").addEventListener("keydown", e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) doAsk(); });
$("profile").onchange = () => { $("seed").value = $("profile").value === "holdout" ? 21 : 7; };
$("theme").onchange = () => {
  const v = $("theme").value;
  if (v === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", v);
  try { localStorage.setItem("recoagent-theme", v); } catch (_) {}
};
try {
  const saved = localStorage.getItem("recoagent-theme");
  if (saved) { $("theme").value = saved; $("theme").onchange(); }
} catch (_) {}

(async () => {
  try {
    MODEL = await (await fetch("/api/model")).json();
    $("modelfoot").textContent = MODEL.ready ? `model ${MODEL.model}` : "no model configured";
    if (!MODEL.ready) {
      $("ask").disabled = true; $("bank").disabled = true; $("q").disabled = true;
      const how = document.createElement("div");
      how.className = "err";
      how.textContent = MODEL.problem;
      $("asknote").replaceWith(how);
    }
  } catch (_) {}
  doRun();
})();
</script>
</body>
</html>
"""

PAGE = _HTML.replace("__CSS__", CSS + UI_CSS)
