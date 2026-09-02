/* The dashboard client: a router, eight views, and the fetches behind them.

   No framework and no build step, for the same reason the rest of the project
   has no dependencies it does not use: a reviewer should be able to read the
   whole surface without installing anything. The shape is deliberately boring --
   `state` holds the current run, a hash route picks a view, a view returns a
   string of markup and then wires its own handlers.

   The one rule the client enforces on itself: the queue, the match log and the
   source ledgers render only what the server sent from `views.py`, which cannot
   see the answer key. Ground truth appears on the Assurance screen, labelled. */

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, c =>
  ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const pct = (x) => (x == null ? "—" : (x * 100).toFixed(2) + "%");
const n = (x) => (x == null ? "—" : Number(x).toLocaleString());

const state = {
  run: null, model: null, busy: false, route: "overview", arg: "",
  filters: { sev: "", leg: "", text: "", status: "" },
  // Which reader the queue is written for. "desk" is the analyst's line --
  // ids, suspected class, the refusal as the matcher recorded it. "plain" is
  // the merchant's. Same rows, same order, same money; only the sentence
  // changes, and the case file shows both whichever is picked.
  register: "desk",
  worklist: null, desking: false,
  matches: { leg: "", tier: "", q: "", page: 1 },
  source: { kind: "payments", q: "", page: 1 },
  selected: null, answer: null, bank: null, resultFile: null,
};

function params() {
  return { n: +$("n").value, seed: +$("seed").value, profile: $("profile").value, rung: $("rung").value };
}
function qs(extra) {
  return new URLSearchParams({ ...params(), ...extra }).toString();
}
async function getJSON(path, extra) {
  const r = await fetch(path + "?" + qs(extra || {}));
  const d = await r.json().catch(() => ({ error: r.status + " " + r.statusText }));
  if (!r.ok) throw new Error(d.error || String(r.status));
  return d;
}
async function post(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({ error: r.status + " " + r.statusText }));
  if (!r.ok) throw new Error(d.error || String(r.status));
  return d;
}

const ladder = (stop) => `<span class="ladder">` + ["T0", "T1", "T2"].map((t, i) => {
  const at = ["T0", "T1", "T2"].indexOf(stop);
  return `<span class="rung rung--${i < at ? "done" : (i === at ? "stop" : "skip")}">${t}</span>`;
}).join("") + `</span>`;

const SEV_LABEL = { critical: "over Rs 10,000", warn: "over Rs 100", minor: "under Rs 100",
                    structural: "no amount in dispute" };

function distrow(label, value, total, tone) {
  const share = total ? (value / total) * 100 : 0;
  return `<div class="distrow">
    <span class="bar ${tone || ""}" style="width:${share.toFixed(1)}%"></span>
    <span class="lbl">${esc(label)}</span><span class="val">${n(value)}</span></div>`;
}

/* A heading, a hairline, and the thing itself. */
function block(title, body) {
  return `<div class="block"><h3>${esc(title)}</h3>${body}</div>`;
}

/* Reserved for something genuinely raised off the page: a model's answer, an
   error, a section of a case file. */
function card(title, body, tone) {
  return `<div class="card ${tone ? "card--" + tone : ""}">${
    title ? `<h3>${esc(title)}</h3>` : ""}${body}</div>`;
}

function table(head, rows, opts) {
  const o = opts || {};
  if (!rows.length) return `<div class="panel"><div class="empty">${esc(o.empty || "Nothing here.")}</div></div>`;
  return `<div class="panel"><table><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>
    <tbody>${rows.join("")}</tbody></table></div>`;
}

/* ── routing ──────────────────────────────────────────────────────────── */

const VIEWS = {};

function route() {
  const was = state.route;
  const raw = (location.hash || "#/overview").replace(/^#\/?/, "");
  const [name, ...rest] = raw.split("/");
  state.route = VIEWS[name] ? name : "overview";
  state.arg = decodeURIComponent(rest.join("/") || "");
  if (was !== state.route) window.scrollTo({ top: 0 });
  document.querySelectorAll(".nav a").forEach(a =>
    a.classList.toggle("on", a.getAttribute("href") === "#/" + state.route));
  render();
}

function render() {
  const view = VIEWS[state.route];
  const host = $("view");
  if (!state.run && view.needsRun) {
    host.innerHTML = `<div class="viewhead"><h1>Reconciling</h1>
      <p>Generating the book and running the tiers over it.</p></div>
      <div class="card"><div class="meta"><span class="load"></span>this takes a moment on a large run</div></div>`;
    return;
  }
  host.innerHTML = view.html();
  if (view.wire) view.wire();
}

/* ── overview ─────────────────────────────────────────────────────────── */

VIEWS.overview = {
  needsRun: true,
  html() {
    const d = state.run, h = d.headline, sh = d.shape, c = sh.credit;
    const leg2 = d.legs.find(l => l.leg === 2) || {};
    const openTotal = d.queue.length || 1;
    const matchTotal = sh.tiers.reduce((a, t) => a + t.count, 0) || 1;
    const cleared = (c.matched_share * 100).toFixed(1);
    const atRisk = d.queue.some(e => e.residual_paise);

    const top = d.queue.slice(0, 8).map(e => `
      <tr>
        <td class="cell-id"><span class="chip chip--${e.severity}"></span>
          <a href="#/exceptions/${encodeURIComponent(e.xid)}"><code>${esc(e.id)}</code></a></td>
        <td><span class="leg">Leg ${e.leg}</span></td>
        <td class="cell-amount num">${esc(e.amount)}<span class="dir">${esc(e.direction)}</span></td>
        <td><span class="tag">${esc(e.suspected)}</span></td>
        <td class="cell-why">${esc(e.reason)}</td></tr>`);

    return `
    <div class="viewhead">
      <h1>Nightly reconciliation</h1>
      <p>Three sources that disagree, plus the merchant's own paperwork, across two legs.</p>
    </div>

    <dl class="stats">
      <div class="stat stat--hero"><dt>False-match rate</dt><dd>${pct(h.false_match_rate)}
        <small>money filed against the wrong entry</small></dd></div>
      <div class="stat"><dt>Auto-matched</dt><dd>${pct(h.auto_match_rate)}</dd></div>
      <div class="stat"><dt>Leg 2 recall</dt><dd>${pct(leg2.recall)}</dd></div>
      <div class="stat"><dt>Value cleared</dt><dd>${pct(h.value_share)}</dd></div>
      <div class="stat ${atRisk ? "stat--money" : ""}"><dt>Unexplained</dt><dd>${esc(h.unexplained)}
        <small>${n(h.open_items)} open items</small></dd></div>
      <div class="stat"><dt>Reconciled in</dt><dd>${d.seconds}s</dd></div>
    </dl>

    <div class="cols-2">
      ${block("Where the money is", `
        <div class="split-bar">
          <span style="width:${cleared}%;background:var(--ok)"></span>
          <span style="width:${(100 - cleared).toFixed(1)}%;background:var(--crit)"></span>
        </div>
        <div class="splitkey">
          <span><i style="background:var(--ok)"></i>cleared ${esc(c.matched)}</span>
          <span><i style="background:var(--crit)"></i>outstanding ${esc(c.outstanding)}</span>
        </div>
        <p class="lede">${n(c.lines_matched)} of ${n(c.lines_total)} bank credits tie to a batch whose
        total was recomputed from its payment rows.</p>`)}

      ${block("Open items by money at stake", `<div class="dist">
        ${sh.severities.map(s => distrow(SEV_LABEL[s.level] || s.level, s.count, openTotal, s.level)).join("")}
      </div>`)}

      ${block("How the matches closed", `<div class="dist">
        ${sh.rules.slice(0, 7).map(r => distrow(r.label, r.count, matchTotal,
            r.tier === "T0" ? "flat" : (r.tier === "T1" ? "minor" : "warn"))).join("")}
      </div>`)}

      ${block("What the system suspects", `<div class="dist">
        ${sh.classes.slice(0, 7).map(k => distrow(k.name, k.count, openTotal, "flat")).join("")}
      </div>
      <p class="lede">The matcher's own read, made without the labels.</p>`)}
    </div>

    <div class="block">
      <h3>Largest open items</h3>
      ${table(["Item", "Leg", "Gap", "Suspected", "What happened"], top,
              { empty: "Nothing open. Every credit tied out." })}
    </div>

    <div class="block">
      <h3>The book this ran on</h3>
      <dl class="stats">
        ${Object.entries(d.counts).map(([k, v]) =>
          `<div class="stat"><dt>${esc(k.replace(/_/g, " "))}</dt><dd>${n(v)}</dd></div>`).join("")}
      </dl>
    </div>`;
  },
};

/* ── exceptions ───────────────────────────────────────────────────────── */

function filteredQueue() {
  const f = state.filters, needle = f.text.trim().toLowerCase();
  return state.run.queue.filter(e => {
    const it = wlItem(e.fp);
    const status = it ? it.status : "open";
    return (!f.sev || e.severity === f.sev) &&
      (!f.leg || String(e.leg) === f.leg) &&
      (!f.status || (f.status === "mine" ? (it && it.assignee) : status === f.status)) &&
      (!needle || (e.id + " " + e.reason + " " + e.suspected).toLowerCase().includes(needle));
  });
}

VIEWS.exceptions = {
  needsRun: true,
  html() {
    const rows = filteredQueue();
    const f = state.filters;
    const seg = (key, opts) => `<div class="seg">${opts.map(([v, label]) =>
      `<button data-f="${key}" data-v="${v}" class="${f[key] === v ? "on" : ""}">${esc(label)}</button>`).join("")}</div>`;

    return `
    <div class="viewhead">
      <h1>Exception queue</h1>
      <p>What the system refused to match, biggest money at stake first. Open an item for its case file.</p>
    </div>

    <div class="toolbar">
      <input class="search" id="qsearch" placeholder="Filter by id, reason or suspected class"
             value="${esc(f.text)}" autocomplete="off">
      ${seg("sev", [["", "all severities"], ["critical", "critical"], ["warn", "warn"], ["minor", "minor"], ["structural", "structural"]])}
      ${seg("leg", [["", "both legs"], ["1", "leg 1"], ["2", "leg 2"]])}
      ${seg("status", [["", "any state"], ["open", "open"], ["investigating", "taken"],
                       ["resolved", "resolved"], ["written_off", "written off"]])}
      <div class="seg" title="Who this list is written for. The rows and the money do not change.">
        <button data-reg="desk" class="${state.register === "desk" ? "on" : ""}">desk</button>
        <button data-reg="plain" class="${state.register === "plain" ? "on" : ""}">plain English</button>
      </div>
      <span class="ctx"><b>${n(rows.length)}</b> of ${n(state.run.queue.length)} shown${
        state.worklist && state.worklist.counts && state.worklist.counts.resolved != null
          ? ` <span class="sep">·</span> ${n(state.worklist.counts.open)} open, ${
              n(state.worklist.counts.investigating)} taken, ${
              n(state.worklist.counts.resolved + state.worklist.counts.written_off)} closed`
          : ""}</span>
    </div>

    <div class="split">
      <div class="qlist" id="qlist">${rows.length ? rows.map(e => `
        <button class="qitem ${state.selected === e.xid ? "on" : ""}" data-x="${esc(e.xid)}">
          <span class="chip chip--${e.severity}" title="${esc(e.severity_hint)}"></span>
          <span class="who">${esc(e.id)}</span>
          <span class="amt">${esc(e.amount)}</span>
          <span class="sub">${wlTag(e.fp)}${state.register === "plain" && e.plain
            ? esc(e.plain.headline)
            : `Leg ${e.leg} · ${esc(e.suspected)} — ${esc(e.reason)}`}</span>
        </button>`).join("") : `<div class="empty">Nothing matches this filter.</div>`}</div>

      <div class="detail" id="detail">${state.selected
        ? `<div class="card"><div class="meta"><span class="load"></span>opening the case file…</div></div>`
        : `<div class="card"><h3>No item selected</h3><p class="lede">Pick a row to open its case file:
           the credit, the batch it joined, every payment and adjustment inside that batch, and the
           arithmetic that failed to close. Use ↑ and ↓ to move through the queue.</p></div>`}</div>
    </div>`;
  },
  wire() {
    const search = $("qsearch");
    search.oninput = () => {
      state.filters.text = search.value;
      const at = search.selectionStart;
      render();
      const again = $("qsearch");
      again.focus();
      again.setSelectionRange(at, at);
    };
    document.querySelectorAll("[data-f]").forEach(b => b.onclick = () => {
      state.filters[b.dataset.f] = b.dataset.v;
      render();
    });
    document.querySelectorAll("[data-reg]").forEach(b => b.onclick = () => {
      state.register = b.dataset.reg;
      render();
    });
    document.querySelectorAll(".qitem").forEach(b => b.onclick = () => select(b.dataset.x));
    if (state.arg && state.arg !== state.selected) select(state.arg);
    else if (state.selected) openCase(state.selected);
  },
};

function select(xid) {
  state.selected = xid;
  history.replaceState(null, "", "#/exceptions/" + encodeURIComponent(xid));
  document.querySelectorAll(".qitem").forEach(b => b.classList.toggle("on", b.dataset.x === xid));
  // "nearest" is a no-op when the row is already on screen, so this only moves
  // the list for a row arrived at by link or by keyboard.
  const row = document.querySelector(`.qitem[data-x="${CSS.escape(xid)}"]`);
  if (row) row.scrollIntoView({ block: "nearest" });
  openCase(xid);
}

function moveSelection(step) {
  const rows = filteredQueue();
  if (!rows.length) return;
  const at = rows.findIndex(e => e.xid === state.selected);
  const next = rows[Math.min(rows.length - 1, Math.max(0, (at < 0 ? 0 : at + step)))];
  if (next) select(next.xid);
}

async function openCase(xid) {
  const host = $("detail");
  if (!host) return;
  host.innerHTML = `<div class="card"><div class="meta"><span class="load"></span>opening the case file…</div></div>`;
  try {
    const d = await getJSON("/api/exception", { id: xid });
    if (state.selected !== xid) return;
    const row = (state.run.queue || []).find(e => e.xid === xid);
    // The history is only fetched when an item is actually opened: the queue
    // snapshot carries state for every row, and every row's whole story would
    // be a great deal of it for a screen nobody has looked at yet.
    if (row && wlItem(row.fp) && !wlItem(row.fp).history) {
      try {
        const h = await getJSON("/api/worklist", { fp: row.fp });
        state.worklist.items[row.fp] = { ...h.item, history: h.history };
      } catch (e) { /* the case file is still worth showing without it */ }
    }
    if (state.selected !== xid) return;
    host.innerHTML = caseFile(d) + deskPanel(xid);
    wireDesk();
  } catch (e) {
    host.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

function payTable(rows) {
  return table(["Payment", "Method", "Status", "Gross", "Fee", "GST", "Net"],
    rows.map(p => `<tr><td class="cell-id"><code>${esc(p.payment_id)}</code></td>
      <td><span class="tag">${esc(p.method)}</span></td><td>${esc(p.status)}</td>
      <td class="num">${esc(p.gross)}</td><td class="num">${esc(p.fee)}</td>
      <td class="num">${esc(p.tax)}</td><td class="num">${esc(p.net)}</td></tr>`),
    { empty: "No payment rows are linked to this batch." });
}

function adjTable(rows, empty) {
  return table(["Adjustment", "Kind", "Payment", "Amount", "Booked"],
    rows.map(a => `<tr><td class="cell-id"><code>${esc(a.adjustment_id)}</code></td>
      <td><span class="tag">${esc(a.kind)}</span></td><td><code>${esc(a.payment_id || "—")}</code></td>
      <td class="num">${esc(a.amount != null ? a.amount : a.amount_paise + " paise")}</td>
      <td class="num">${esc(a.booked_at)}</td></tr>`), { empty });
}

/* The merchant's reading of the item, above the desk's.

   Deliberately first and deliberately plain: the person whose money this is
   should not have to parse `pay_00033` to find out what happened. Everything
   below it -- ids, paise, tiers -- stays exactly as it was, because the two
   registers are for two readers and neither is the other's summary. */
function plainPanel(p) {
  if (!p || !p.headline) return "";
  const body = (p.body || []).filter(Boolean).map(t => `<p>${esc(t)}</p>`).join("");
  return `
    <div class="card plainsay">
      <div class="eyebrow">In plain terms</div>
      <h3>${esc(p.headline)}</h3>
      ${body}
      <p class="said-status">${esc(p.status)}</p>
      <p class="said-next"><b>What to do:</b> ${esc(p.next_step)}</p>
    </div>`;
}

function caseFile(d) {
  const e = d.item, c = d.case;
  const head = plainPanel(e.plain) + `
    <div>
      <div class="casehead">
        <div style="flex:1;min-width:0">
          <h2>${esc(e.id)}</h2>
          <div class="meta"><span>Leg ${e.leg} · ${esc(e.kind)}</span><span>${esc(e.severity_hint)}</span>
            <span>stopped at ${esc(e.stopped_at)}</span></div>
        </div>
        <div style="text-align:right">${e.residual_paise == null
          ? `<div class="nogap">nothing in dispute</div>
             <div class="meta" style="justify-content:flex-end">structural</div>`
          : `<div class="bigamt ${esc(e.direction)}">${esc(e.amount)}</div>
             <div class="meta" style="justify-content:flex-end">${esc(e.direction)}</div>`}
        </div>
      </div>
      <div class="kv" style="margin-top:14px">
        <div><dt>Tiers tried</dt><dd>${ladder(e.stopped_at)}</dd></div>
        <div><dt>Suspected class</dt><dd><span class="tag">${esc(e.suspected)}</span></dd></div>
        <div><dt>Residual</dt><dd class="num">${e.residual_paise == null ? "no amount in dispute"
          : n(e.residual_paise) + " paise"}</dd></div>
        <div class="wide"><dt>Why it was refused</dt><dd>${esc(e.reason)}</dd></div>
      </div>
    </div>`;

  if (c.shape === "leg2_batch") {
    const a = c.arithmetic, sig = c.derived_signals;
    return head + `
      ${card("The arithmetic, re-derived from the rows", `
        <div class="ledger">
          <span class="lbl">${n(a.payments_count)} payments, net of fee and GST</span><span class="amt">${esc(a.payments_net)}</span>
          <span class="lbl">${n(a.adjustments_count)} linked adjustments</span><span class="amt">${esc(a.adjustments_net)}</span>
          <span class="rule"></span>
          <span class="lbl strong">derived from the rows</span><span class="amt strong">${esc(a.derived_net)}</span>
          <span class="lbl">bank credited</span><span class="amt">${esc(c.bank_credit.amount)}</span>
          <span class="rule"></span>
          <span class="lbl gap">unexplained</span><span class="amt gap">${esc(e.amount)}</span>
        </div>
        <p class="lede">The settlement header says ${esc(a.reported_net)}${a.header_agrees
          ? ", which agrees with the rows." : ` — ${esc(a.header_gap)} away from what the rows come to.`}
        The header is corroboration and is never the proof.</p>`, "crit")}

      ${card("Signals a human computes first", `<div class="kv">
        <div><dt>Residual as % of fee base</dt><dd class="num">${sig.residual_as_pct_of_fee_base ?? "—"}</dd></div>
        <div><dt>Residual as % of total fee</dt><dd class="num">${sig.residual_as_pct_of_total_fee ?? "—"}</dd></div>
        <div><dt>MDR-bearing rows</dt><dd class="num">${n(sig.mdr_bearing_payments)} of ${n(sig.payments_in_batch)}</dd></div>
        <div><dt>International rows</dt><dd class="num">${n(sig.international_payments.length)}</dd></div>
      </div><p class="lede">${esc(sig.note)}</p>`)}

      <div class="block"><h3>Payments in the batch</h3>${payTable(c.payments)}</div>
      <div class="block"><h3>Linked adjustments</h3>${adjTable(c.linked_adjustments, "Nothing was netted into this batch.")}</div>
      <div class="block"><h3>Unlinked rows booked nearby</h3>
        <p class="note">Netted by the gateway, linked to no batch. The solver already searched these.</p>
        ${adjTable(c.nearby_unlinked, "No unlinked rows anywhere near this batch.")}</div>
      ${card("Already ruled out", `<p class="lede">${esc(c.already_ruled_out)}</p>`)}`;
  }

  if (c.shape === "leg2_orphan") {
    return head + `
      ${card("The credit", `<div class="kv">
        <div><dt>Value date</dt><dd class="num">${esc(c.bank_credit.value_date)}</dd></div>
        <div><dt>Amount</dt><dd class="num">${esc(c.bank_credit.amount)}</dd></div>
        <div><dt>Bank reference</dt><dd class="num">${esc(c.bank_credit.bank_ref)}</dd></div>
        <div class="wide"><dt>Narration, as the bank printed it</dt>
          <dd><code>${esc(c.bank_credit.narration)}</code></dd></div>
      </div>`)}
      <div class="block"><h3>Nearest settlements by date, then amount</h3>
        <p class="note">Candidates, not matches — nothing here has closed any arithmetic.</p>
        ${table(["Batch", "UTR", "Settled", "Header says", "Gap to this credit", "Days apart"],
          c.candidates.map(s => `<tr><td class="cell-id"><code>${esc(s.settlement_id)}</code></td>
            <td><code>${esc(s.utr)}</code></td><td class="num">${esc(s.settled_at)}</td>
            <td class="num">${esc(s.reported_net)}</td><td class="num">${esc(s.gap)}</td>
            <td class="num">${s.days_apart}</td></tr>`), { empty: "No settlements in this book." })}</div>`;
  }

  if (c.shape === "settlement") {
    const a = c.arithmetic;
    return head + `
      ${card("The batch", `<div class="kv">
        <div><dt>UTR</dt><dd><code>${esc(c.settlement.utr)}</code></dd></div>
        <div><dt>Settled at</dt><dd class="num">${esc(c.settlement.settled_at)}</dd></div>
        <div><dt>Status</dt><dd><span class="tag">${esc(c.settlement.status)}</span></dd></div>
        <div><dt>Header says</dt><dd class="num">${esc(c.settlement.reported_net)}</dd></div>
      </div>
      <div class="ledger">
        <span class="lbl">${n(a.payments_count)} payments, net</span><span class="amt">${esc(a.payments_net)}</span>
        <span class="lbl">${n(a.adjustments_count)} adjustments</span><span class="amt">${esc(a.adjustments_net)}</span>
        <span class="rule"></span>
        <span class="lbl strong">derived from the rows</span><span class="amt strong">${esc(a.derived_net)}</span>
      </div>`)}
      <div class="block"><h3>Bank lines carrying this UTR</h3>
        ${table(["Credit", "Value date", "Amount", "Narration"],
          c.bank_lines_carrying_utr.map(b => `<tr><td class="cell-id"><code>${esc(b.bank_line_id)}</code></td>
            <td class="num">${esc(b.value_date)}</td><td class="num">${esc(b.amount)}</td>
            <td class="cell-why"><code>${esc(b.narration)}</code></td></tr>`),
          { empty: "No statement line mentions this UTR. The money has not arrived." })}</div>
      <div class="block"><h3>Payments in the batch</h3>${payTable(c.payments)}</div>
      <div class="block"><h3>Linked adjustments</h3>${adjTable(c.linked_adjustments, "Nothing was netted into this batch.")}</div>`;
  }

  if (c.shape === "leg1_order") {
    return head + `
      ${card("The order, as the merchant booked it", `<div class="kv">
        <div><dt>Amount</dt><dd class="num">${esc(c.order.amount)}</dd></div>
        <div><dt>Invoice</dt><dd><code>${esc(c.order.invoice_no)}</code></dd></div>
        <div><dt>Customer</dt><dd><code>${esc(c.order.customer_id)}</code></dd></div>
        <div><dt>Created</dt><dd class="num">${esc(c.order.created_at)}</dd></div>
      </div>`)}
      <div class="block"><h3>Payments claiming this order</h3>
        <p class="note">One order, one payment, exactly. The system refuses rather than picking.</p>
        ${table(["Payment", "Method", "Status", "Gross", "Gap to order", "Batch", "Captured"],
          c.claims.map(p => `<tr><td class="cell-id"><code>${esc(p.payment_id)}</code></td>
            <td><span class="tag">${esc(p.method)}</span></td><td>${esc(p.status)}</td>
            <td class="num">${esc(p.gross)}</td><td class="num">${esc(p.gap)}</td>
            <td><code>${esc(p.settlement_id || "—")}</code></td>
            <td class="num">${esc(p.captured_at)}</td></tr>`),
          { empty: "No payment row references this order at all." })}</div>
      <div class="block"><h3>Refunds and adjustments against those payments</h3>
        ${adjTable(c.refunds, "Nothing was booked against these payments.")}</div>`;
  }

  return head + `<div class="card"><p class="lede">No structured case file for this item kind.</p></div>`;
}

/* ── match log ────────────────────────────────────────────────────────── */

VIEWS.matches = {
  needsRun: true,
  html() {
    const m = state.matches;
    const seg = (key, opts) => `<div class="seg">${opts.map(([v, label]) =>
      `<button data-m="${key}" data-v="${v}" class="${m[key] === v ? "on" : ""}">${esc(label)}</button>`).join("")}</div>`;
    return `
    <div class="viewhead">
      <h1>Match log</h1>
      <p>Every accepted match and the arithmetic that accepted it. Any row opens to its proof.</p>
    </div>
    <div class="toolbar">
      <input class="search" id="msearch" placeholder="Find a match, a rule, or an entity id"
             value="${esc(m.q)}" autocomplete="off">
      ${seg("leg", [["", "both legs"], ["1", "leg 1"], ["2", "leg 2"]])}
      ${seg("tier", [["", "all tiers"], ["T0", "T0"], ["T1", "T1"], ["T2", "T2"]])}
    </div>
    <div id="mout"><div class="card"><div class="meta"><span class="load"></span>loading the log…</div></div></div>`;
  },
  wire() {
    const search = $("msearch");
    let timer = null;
    search.oninput = () => {
      clearTimeout(timer);
      timer = setTimeout(() => { state.matches.q = search.value; state.matches.page = 1; loadMatches(); }, 220);
    };
    document.querySelectorAll("[data-m]").forEach(b => b.onclick = () => {
      state.matches[b.dataset.m] = b.dataset.v;
      state.matches.page = 1;
      render();
    });
    loadMatches();
  },
};

// ── the desk ────────────────────────────────────────────────────────────────
// The queue is persistent and the engine is not. Everything below is about the
// half a person owns: who has an item, what they wrote on it, and whether it is
// still open. The pipeline never writes any of it.

const WL_LABEL = { open: "open", investigating: "taken", resolved: "resolved", written_off: "written off" };
const WL_ACTION = {
  investigating: ["Take", "primary"],
  resolved: ["Resolve", "primary"],
  written_off: ["Write off", "ghost"],
  open: ["Release", "ghost"],
};

async function loadWorklist() {
  try {
    state.worklist = await getJSON("/api/worklist");
  } catch (e) {
    // A queue that cannot be reached must not take the console down with it:
    // the reconciliation is still worth reading without it.
    state.worklist = { items: {}, counts: {}, unavailable: e.message };
  }
}

const wlItem = (fp) => (state.worklist && state.worklist.items[fp]) || null;

function wlTag(fp) {
  const it = wlItem(fp);
  if (!it || it.status === "open") return "";
  return `<span class="wl wl--${esc(it.status)}">${esc(WL_LABEL[it.status] || it.status)}${
    it.assignee ? " · " + esc(it.assignee) : ""}</span>`;
}

function deskPanel(xid) {
  const row = (state.run.queue || []).find(e => e.xid === xid);
  if (!row) return "";
  const it = wlItem(row.fp);
  if (!it) {
    return card("Ownership", `<p class="lede">This item is not in the queue yet. Reconcile the book
      to file it.</p>`);
  }
  const moves = (it.can_move_to || []).map(to => {
    const [label, kind] = WL_ACTION[to] || [to, "ghost"];
    return `<button class="${kind}" data-wl="${esc(to)}" data-fp="${esc(it.fp)}">${esc(label)}</button>`;
  }).join("");

  return `<div class="card">
    <div class="head"><div style="flex:1"><h3>Ownership and outcome</h3>
      <div class="meta"><span>seen in runs ${n(it.first_seen_run)}–${n(it.last_seen_run)}</span>
        <span>${it.age_in_runs === 0 ? "first run" : "open across " + n(it.age_in_runs + 1) + " runs"}</span></div></div>
      <span class="wl wl--${esc(it.status)} wl--big">${esc(WL_LABEL[it.status] || it.status)}</span>
    </div>
    ${it.closed_reason ? `<p class="lede">Closed: ${esc(it.closed_reason)}</p>` : ""}
    <div class="deskform">
      <label>Assignee<input id="wl-assignee" value="${esc(it.assignee)}" placeholder="nobody yet"
        autocomplete="off" ${it.is_open ? "" : "disabled"}></label>
      <label>Notes<textarea id="wl-notes" rows="2" placeholder="what you found, what you are waiting on"
        ${it.is_open ? "" : "disabled"}>${esc(it.notes)}</textarea></label>
    </div>
    ${it.is_open ? `<label class="deskwhy">Why<input id="wl-detail" placeholder="reason, kept on the record when this closes"
       autocomplete="off"></label>` : ""}
    <div class="chips" style="margin-top:12px">
      ${moves || `<span class="lede">Closed. A later run will not reopen it — somebody decided this.</span>`}
      ${it.is_open ? `<button class="ghost" data-wl-save="${esc(it.fp)}">Save note</button>` : ""}
    </div>
    <div id="wl-err"></div>
    ${it.history ? "" : ""}
  </div>
  ${deskHistory(it.history)}`;
}

function deskHistory(rows) {
  if (!rows || !rows.length) return "";
  return card("What has happened to this item", `<div class="wlhist">${rows.map(h => `
    <div><span class="when">${esc((h.at || "").replace("T", " ").slice(0, 16))}</span>
    <span class="what">${h.from_status ? esc(h.from_status) + " → " : "filed "}<b>${esc(h.to_status)}</b></span>
    <span class="who">${esc(h.actor)}</span>
    ${h.detail ? `<span class="why">${esc(h.detail)}</span>` : ""}</div>`).join("")}</div>`);
}

async function deskAct(fp, to) {
  if (state.desking) return;
  state.desking = true;
  const err = $("wl-err");
  if (err) err.innerHTML = `<div class="meta"><span class="load"></span>saving…</div>`;
  const a = $("wl-assignee"), nt = $("wl-notes"), why = $("wl-detail");
  try {
    const d = await post("/api/worklist", {
      ...params(), fp,
      ...(to ? { to } : {}),
      assignee: a ? a.value : undefined,
      notes: nt ? nt.value : undefined,
      actor: (a && a.value) || "analyst",
      detail: why ? why.value : "",
    });
    state.worklist.items[fp] = { ...d.item, history: d.history };
    state.worklist.counts = d.counts;
    render();
  } catch (e) {
    if (err) err.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    state.desking = false;
  }
}

function wireDesk() {
  document.querySelectorAll("[data-wl]").forEach(b =>
    b.onclick = () => deskAct(b.dataset.fp, b.dataset.wl));
  document.querySelectorAll("[data-wl-save]").forEach(b =>
    b.onclick = () => deskAct(b.dataset.wlSave, null));
}

async function loadMatches() {
  const host = $("mout");
  if (!host) return;
  try {
    const d = await getJSON("/api/matches", state.matches);
    const rows = d.rows.map((r, i) => `
      <tr class="exp" data-i="${i}">
        <td class="cell-id"><code>${esc(r.left.join(", "))}</code></td>
        <td class="cell-id"><code>${esc(r.right.join(", "))}</code></td>
        <td><span class="leg">Leg ${r.leg}</span></td>
        <td><span class="rung rung--done">${esc(r.tier)}</span></td>
        <td>${esc(r.rule)}</td>
        <td class="num">${r.confidence.toFixed(2)}</td>
        <td class="num proof">${esc(r.proof ? r.proof.residual : "—")}</td></tr>`);
    host.innerHTML = table(["Left", "Right", "Leg", "Tier", "Rule", "Confidence", "Residual"], rows,
      { empty: "No match fits this filter." }) + `
      <div class="pager"><span>${n(d.total)} matches · page ${d.page} of ${d.pages}</span>
        <button class="tiny" id="prev" ${d.page <= 1 ? "disabled" : ""}>← previous</button>
        <button class="tiny" id="next" ${d.page >= d.pages ? "disabled" : ""}>next →</button></div>`;
    const prev = $("prev"), next = $("next");
    if (prev) prev.onclick = () => { state.matches.page--; loadMatches(); };
    if (next) next.onclick = () => { state.matches.page++; loadMatches(); };
    host.querySelectorAll("tr.exp").forEach(tr => tr.onclick = () => {
      const after = tr.nextElementSibling;
      if (after && after.classList.contains("detailrow")) return after.remove();
      const r = d.rows[+tr.dataset.i];
      const p = r.proof;
      const row = document.createElement("tr");
      row.className = "detailrow";
      row.innerHTML = `<td colspan="7"><div class="detailwrap"><div class="kv">
        <div><dt>Match id</dt><dd><code>${esc(r.match_id)}</code></dd></div>
        <div><dt>Rule</dt><dd><code>${esc(r.rule_id)}</code></dd></div>
        <div><dt>Decided at</dt><dd class="num">${esc(r.created_at)}</dd></div>
        <div><dt>Input hash</dt><dd><code>${esc(r.input_hash)}</code></dd></div>
        ${p ? `<div class="wide"><dt>Proof</dt><dd><code>${esc(p.expression)}</code></dd></div>
        <div><dt>Left side</dt><dd class="num">${esc(p.lhs)}</dd></div>
        <div><dt>Right side</dt><dd class="num">${esc(p.rhs)}</dd></div>
        <div><dt>Residual</dt><dd class="num">${esc(p.residual)} (tolerance ${n(p.tolerance_paise)} paise)</dd></div>`
        : `<div class="wide"><dt>Proof</dt><dd>No arithmetic proof on this record.</dd></div>`}
        ${r.variance ? `<div class="wide"><dt>Variance carried on this match</dt>
          <dd class="num">${esc(r.variance)} — the pairing is settled, this gap is not</dd></div>` : ""}
        ${r.hypothesised.length ? `<div class="wide"><dt>Rows the source data did not link, needed to close it</dt>
          <dd><code>${esc(r.hypothesised.join(", "))}</code></dd></div>` : ""}
      </div></div></td>`;
      tr.after(row);
    });
  } catch (e) {
    host.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

/* ── source ledgers ───────────────────────────────────────────────────── */

const SOURCE_TABS = [
  ["orders", "Orders"], ["payments", "Payments"], ["adjustments", "Adjustments"],
  ["settlements", "Settlements"], ["bank_lines", "Bank credits"],
  ["rate_notices", "Rate notices"], ["fx_advices", "FX advices"],
];

VIEWS.sources = {
  needsRun: true,
  html() {
    const counts = state.run.counts;
    return `
    <div class="viewhead">
      <h1>Source ledgers</h1>
      <p>The sources exactly as the matcher receives them. Find the row somebody is asking about.</p>
    </div>
    <div class="tabs">${SOURCE_TABS.map(([k, label]) =>
      `<button data-s="${k}" class="${state.source.kind === k ? "on" : ""}">${esc(label)}
        <span class="n">${n(counts[k] ?? 0)}</span></button>`).join("")}</div>
    <div class="toolbar">
      <input class="search" id="ssearch" placeholder="Search every column of this ledger"
             value="${esc(state.source.q)}" autocomplete="off"></div>
    <div id="sout"><div class="card"><div class="meta"><span class="load"></span>loading…</div></div></div>`;
  },
  wire() {
    document.querySelectorAll("[data-s]").forEach(b => b.onclick = () => {
      state.source = { kind: b.dataset.s, q: "", page: 1 };
      render();
    });
    const search = $("ssearch");
    let timer = null;
    search.oninput = () => {
      clearTimeout(timer);
      timer = setTimeout(() => { state.source.q = search.value; state.source.page = 1; loadSource(); }, 220);
    };
    loadSource();
  },
};

async function loadSource() {
  const host = $("sout");
  if (!host) return;
  try {
    const d = await getJSON("/api/source", state.source);
    const cell = (col, row) => {
      const v = row[col.key];
      if (col.type === "id") return `<td class="cell-id"><code>${esc(v ?? "—")}</code></td>`;
      if (col.type === "tag") return `<td><span class="tag">${esc(v ?? "—")}</span></td>`;
      if (col.type === "money" || col.type === "num" || col.type === "when")
        return `<td class="num">${esc(v ?? "—")}</td>`;
      return `<td class="cell-why">${esc(v ?? "—")}</td>`;
    };
    const rows = d.rows.map(r => `<tr>${d.columns.map(c => cell(c, r)).join("")}
      ${r._flag ? `<td class="flagcell">${esc(r._flag)}</td>` : (d.kind === "settlements" ? "<td></td>" : "")}</tr>`);
    const head = d.columns.map(c => c.label).concat(d.kind === "settlements" ? [""] : []);
    host.innerHTML = `<p class="note">${esc(d.blurb)}</p>` +
      table(head, rows, { empty: "Nothing in this ledger matches that search." }) + `
      <div class="pager"><span>${n(d.total)} rows · page ${d.page} of ${d.pages}</span>
        <button class="tiny" id="sprev" ${d.page <= 1 ? "disabled" : ""}>← previous</button>
        <button class="tiny" id="snext" ${d.page >= d.pages ? "disabled" : ""}>next →</button></div>`;
    const prev = $("sprev"), next = $("snext");
    if (prev) prev.onclick = () => { state.source.page--; loadSource(); };
    if (next) next.onclick = () => { state.source.page++; loadSource(); };
  } catch (e) {
    host.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

/* ── ask the agent ────────────────────────────────────────────────────── */

function suggestions() {
  const out = ["How many items are still open in the exception queue?",
               "What is the total unexplained amount across all open items, in paise?"];
  const first = state.run.queue.find(e => e.residual_paise != null);
  if (first) out.push(`What is the gap on ${first.id}, in paise?`);
  out.push("What is the merchant's registered GSTIN?");
  return out;
}

VIEWS.agent = {
  needsRun: true,
  html() {
    const ready = state.model && state.model.ready;
    return `
    <div class="viewhead">
      <h1>Ask the settlement agent</h1>
      <p>Answered from a factsheet built by code. A question typed here has no ground truth, so it comes
      back ungraded with that factsheet attached.</p>
    </div>
    ${ready ? "" : `<div class="err">${esc((state.model && state.model.problem) || "No model configured.")}</div>`}
    <div class="askbox">
      <textarea id="q" rows="2" ${ready ? "" : "disabled"}
        placeholder="How much is sitting unexplained across the open items?"></textarea>
      <button class="primary" id="ask" ${ready ? "" : "disabled"}>Ask</button>
    </div>
    <div class="chips">${suggestions().map(s => `<button class="sugg">${esc(s)}</button>`).join("")}</div>
    <div id="answer">${state.answer || ""}</div>

    <div class="block">
      <h3>Measured Q&amp;A accuracy</h3>
      <p class="note">Every question has a ground truth computed by code, including ones the factsheet
      cannot answer, where the only correct response is to decline.</p>
      <div class="askbox">
        <button id="bank" ${ready ? "" : "disabled"}>Run the scored bank</button>
        <div class="field"><label for="limit">Questions (0 = all)</label>
          <input id="limit" type="number" value="0" min="0" max="60"></div>
      </div>
      <div id="bankout">${state.bank || ""}</div>
    </div>`;
  },
  wire() {
    document.querySelectorAll(".sugg").forEach(b => b.onclick = () => {
      $("q").value = b.textContent.trim();
      doAsk();
    });
    $("ask").onclick = doAsk;
    $("bank").onclick = doBank;
    $("q").addEventListener("keydown", e => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) doAsk();
    });
  },
};

async function doAsk() {
  const text = $("q").value.trim();
  if (!text || !state.run) return;
  $("ask").disabled = true;
  $("answer").innerHTML = `<div class="card"><div class="meta"><span class="load"></span>asking ${
    esc(state.model ? state.model.model : "the model")}…</div></div>`;
  try {
    const a = await post("/api/ask", { ...params(), question: text });
    const tone = a.failed ? "crit" : (a.declined ? "warn" : "ok");
    // Money is shown as money. The model answers in integer paise so the
    // graded bank can compare exactly; a person reading "-118632" off a screen
    // is that contract leaking out of the harness and into the product. The
    // raw figure stays underneath, because an operator checking the factsheet
    // needs the number the model actually gave.
    const headline = a.failed ? "call failed"
      : (a.declined ? "Declined to answer"
      : (a.answer === null ? "—" : String(a.answer_money || a.answer)));
    const raw = (a.answer_money && a.answer !== null)
      ? `<p class="note">answered <code>${esc(String(a.answer))}</code> paise</p>` : "";
    state.answer = `<div class="card card--${tone}">
      <span class="banner banner--ungraded">Ungraded — no ground truth for a live question</span>
      <div class="answer">${esc(headline)}</div>
      ${raw}
      ${a.basis ? `<p class="lede">${esc(a.basis)}</p>` : ""}
      ${a.detail && (a.failed || a.declined) ? `<p class="lede">${esc(a.detail)}</p>` : ""}
      <div class="meta"><span>${esc(state.model ? state.model.model : "")}</span>
        ${a.confidence != null ? `<span>self-reported confidence ${a.confidence}</span>` : ""}
        <span>${a.seconds}s</span><span>${n(a.tokens_in)} in / ${n(a.tokens_out)} out</span></div>
      <details class="sheet"><summary>Everything the model was allowed to see — check the answer against it</summary>
        <pre>${esc(JSON.stringify(a.factsheet, null, 2))}</pre></details></div>`;
    $("answer").innerHTML = state.answer;
  } catch (e) {
    $("answer").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    const b = $("ask"); if (b) b.disabled = false;
  }
}

async function doBank() {
  if (!state.run) return;
  $("bank").disabled = true;
  $("bankout").innerHTML = `<div class="card"><div class="meta"><span class="load"></span>asking the scored
    bank — this makes one call per question…</div></div>`;
  try {
    const b = await post("/api/bank", { ...params(), limit: +$("limit").value });
    const rows = b.answers.map(a => {
      const mark = a.correct ? "ok" : (a.declined ? "dec" : "bad");
      const label = a.correct ? (a.declined ? "right to decline" : "correct")
        : (a.failed ? "failed" : (a.declined ? "declined" : "WRONG"));
      const detail = a.correct ? "" : esc(a.detail || "");
      return `<div class="qrow"><span class="qmark qmark--${mark}">${label}</span>
        <span>${esc(a.text)}${detail ? ` <span class="leg">— ${detail}</span>` : ""}</span></div>`;
    }).join("");
    state.bank = `<div class="card card--${b.wrong_answer_rate === 0 && b.hallucinated === 0 ? "ok" : "crit"}">
      <span class="banner banner--measured">Measured — every question graded by code</span>
      <dl class="stats">
        <div class="stat stat--hero"><dt>Wrong-answer rate</dt><dd>${pct(b.wrong_answer_rate)}
          <small>of ${b.attempted} answered</small></dd></div>
        <div class="stat"><dt>Coverage</dt><dd>${pct(b.coverage)}<small>${b.attempted} of ${b.total} attempted</small></dd></div>
        <div class="stat"><dt>Accuracy</dt><dd>${pct(b.accuracy)}<small>correct of all questions</small></dd></div>
        <div class="stat stat--money"><dt>Hallucinated</dt><dd>${b.hallucinated}
          <small>answered what the factsheet cannot support</small></dd></div>
      </dl>
      <div class="meta"><span>${b.correct} correct · ${b.wrong} wrong · ${b.declined} declined · ${b.failed} failed</span>
        <span>${b.seconds}s</span><span>${n(b.tokens_in)} in / ${n(b.tokens_out)} out</span></div>
      <details class="sheet" open><summary>Every question, and what it answered</summary><div>${rows}</div></details></div>`;
    $("bankout").innerHTML = state.bank;
  } catch (e) {
    $("bankout").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    const b = $("bank"); if (b) b.disabled = false;
  }
}

/* ── assurance ────────────────────────────────────────────────────────── */

/* The classes the engine has no tier for. Rendered only on the `unknown`
   profile, because on every other book the honest answer is that this was not
   tested rather than that it passed. */
function unknownPanel(d) {
  const u = d.unknown, rows = d.unknown_accounting || [];
  if (!u || !u.injected) return "";
  const body = rows.map(a => `<tr>
    <td><code>${esc(a.defect)}</code></td><td class="num">${n(a.injected)}</td>
    <td class="num">${n(a.contained)}</td><td class="num">${n(a.absorbed)}</td>
    <td class="num verdict verdict--${a.mishandled === 0 ? "ok" : "bad"}">${a.mishandled}</td></tr>`);
  return `
    <div class="block"><h3>Defect classes the engine has no tier for</h3>
      <p class="note">Injected from a module no matcher may import. Nothing here has a rule, a
      tolerance or a solver behind it, so a recall of 0% is expected and is not the number to read.
      The question is whether an unexplainable gap gets filed — or gets closed by a coincidence.</p>
      <dl class="stats">
        <div class="stat stat--hero"><dt>Wrong or unnoticed</dt><dd>${u.mishandled}
          <small>booked against a coincidence, or sailed past</small></dd></div>
        <div class="stat"><dt>Contained</dt><dd>${pct(u.containment_rate)}
          <small>${n(u.contained)} of ${n(u.injected)} filed as exceptions</small></dd></div>
        <div class="stat"><dt>Absorbed</dt><dd>${n(u.absorbed)}
          <small>matched without the gap being noticed</small></dd></div>
        <div class="stat"><dt>Verdict</dt><dd style="font-size:1.05rem;color:var(--${u.holds ? "ok" : "crit"})">
          ${u.holds ? "HOLDS" : "DOES NOT HOLD"}<small>beyond the written taxonomy</small></dd></div>
      </dl>
      ${table(["Defect class", "Injected", "Contained", "Absorbed", "Wrong"], body)}</div>`;
}

VIEWS.assurance = {
  needsRun: true,
  html() {
    const d = state.run, v = d.verdict;
    const rows = d.accounting.map(a => `<tr>
      <td><code>${esc(a.defect)}</code></td><td class="num">${n(a.injected)}</td>
      <td class="num">${n(a.flagged)}</td><td class="num">${n(a.resolved)}</td>
      <td class="num verdict verdict--${a.mishandled === 0 ? "ok" : "bad"}">${a.mishandled}</td></tr>`);
    return `
    <div class="viewhead">
      <h1>Assurance</h1>
      <p>The only screen that uses the answer key. Every injected defect must be flagged or correctly
      resolved — mishandled must be zero.</p>
    </div>
    <span class="banner banner--truth">Ground truth — fenced off from every operator screen</span>
    <dl class="stats">
      <div class="stat stat--hero"><dt>Mishandled</dt><dd>${v.mishandled_total}
        <small>wrong match, or sailed past unnoticed</small></dd></div>
      <div class="stat"><dt>Defects injected</dt><dd>${n(v.injected_total)}<small>counted exactly, not sampled</small></dd></div>
      <div class="stat"><dt>Unattributed exceptions</dt><dd>${n(v.unattributed)}
        <small>raised against records with nothing wrong</small></dd></div>
      <div class="stat"><dt>Verdict</dt><dd style="font-size:1.05rem;color:var(--${v.fully_reconciles ? "ok" : "crit"})">
        ${v.fully_reconciles ? "RECONCILES" : "DOES NOT RECONCILE"}<small>injected equals accounted</small></dd></div>
    </dl>
    <div class="block"><h3>What became of each injected defect</h3>
      <p class="note">Flagged and resolved are both correct outcomes. A class migrating from flagged to
      resolved is a tier that learned to close it.</p>
      ${table(["Defect class", "Injected", "Flagged", "Resolved", "Mishandled"], rows)}</div>
    ${unknownPanel(d)}
    <div class="callout"><b>Why this panel is separate</b>
      <p>A reconciliation engine that matches everything and is occasionally wrong is worse than useless:
      it files money against the wrong transaction and hides the error behind a green number. So the
      operator screens are built from the sources and the result alone, and the labels live here, behind
      a heading that says so.</p></div>
    <div class="block"><h3>Per-leg detail</h3>
      ${table(["Leg", "Population", "Matched", "True", "False", "Recall", "Exceptions"],
        d.legs.map(l => `<tr><td><span class="leg">Leg ${l.leg}</span></td>
          <td class="num">${n(l.population)}</td><td class="num">${n(l.attempted)}</td>
          <td class="num">${n(l.true_matches)}</td>
          <td class="num verdict verdict--${l.false_matches === 0 ? "ok" : "bad"}">${l.false_matches}</td>
          <td class="num">${pct(l.recall)}</td><td class="num">${n(l.exceptions)}</td></tr>`))}</div>`;
  },
};

/* ── published results ────────────────────────────────────────────────── */

VIEWS.results = {
  needsRun: false,
  html() {
    return `
    <div class="viewhead">
      <h1>Published results</h1>
      <p>The artefacts in <code>results/</code>, as the eval commands wrote them. Nothing here is
      computed live.</p>
    </div>
    <div class="split">
      <div id="rlist"><div class="card"><div class="meta"><span class="load"></span>listing…</div></div></div>
      <div id="rfile"><div class="card"><h3>Nothing open</h3>
        <p class="lede">Pick a file. <code>benchrec_recoagent</code> is the external one: RecoAgent scored
        on BenchRec, the industry benchmark, against the matcher that ships with it.</p></div></div>
    </div>`;
  },
  wire() { loadResults(); },
};

async function loadResults() {
  const host = $("rlist");
  try {
    const d = await getJSON("/api/results", {});
    host.innerHTML = `<div class="qlist">${d.files.map(f => `
      <button class="qitem" data-r="${esc(f.name)}">
        <span class="chip chip--${f.blurb ? "minor" : "structural"}"></span>
        <span class="who">${esc(f.stem)}</span>
        <span class="amt">${(f.bytes / 1024).toFixed(1)}k</span>
        ${f.blurb ? `<span class="sub">${esc(f.blurb)}</span>` : ""}
      </button>`).join("") || `<div class="empty">No result files on disk.</div>`}</div>`;
    host.querySelectorAll("[data-r]").forEach(b => b.onclick = async () => {
      host.querySelectorAll(".qitem").forEach(x => x.classList.remove("on"));
      b.classList.add("on");
      const out = $("rfile");
      out.innerHTML = `<div class="card"><div class="meta"><span class="load"></span>reading…</div></div>`;
      try {
        const f = await getJSON("/api/results", { file: b.dataset.r });
        out.innerHTML = `<div class="card"><h3>${esc(f.name)}</h3><pre class="raw">${esc(f.text)}</pre></div>`;
      } catch (e) { out.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
    });
  } catch (e) {
    host.innerHTML = `<div class="err">${esc(e.message)}</div>`;
  }
}

/* ── method ───────────────────────────────────────────────────────────── */

VIEWS.method = {
  needsRun: false,
  html() {
    /* Prose set as prose. Six equal boxes made six unequal ideas look like a
       feature list, which is the opposite of what they are. */
    const rule = (h, body) => `<div class="rule-item"><h3>${h}</h3><p>${body}</p></div>`;
    return `
    <div class="viewhead">
      <h1>How to read this</h1>
      <p>Every screen here is one view of one run. What each is allowed to know is the design, not an
      implementation detail.</p>
    </div>
    <div class="prose">
      ${rule("The gate is the thesis", `Nothing becomes a match because something believed it should.
        Leg 2 re-derives every batch total from the payment rows, never from the settlement header, and a
        credit only books when that arithmetic closes inside a stated tolerance. The header is
        corroboration. It is never the proof.`)}
      ${rule("Two legs, two problems", `Leg 1 is order to payment, 1:1, record linkage. Leg 2 is
        settlement batch to bank credit, N:1 — the subset-sum matching problem. They are scored
        separately because they fail differently, and Leg 2 is scored over bank lines, the harder
        population.`)}
      ${rule("False-match rate leads", `Match rate is reported second on purpose. An engine that matches
        everything and is occasionally wrong books money against the wrong transaction and hides it
        behind a green number. Refusing costs a queue item; guessing costs a correction, and somebody
        has to find it first.`)}
      ${rule("The operator screens cannot see the labels", `The queue, the match log and the source
        ledgers are built from the sources and the result only — the same restriction every matcher runs
        under, enforced by a test that reads the imports. Ground truth appears on Assurance, labelled.`)}
      ${rule("Ambiguity is refused, not resolved", `Two payments claiming one order, a UTR on two
        statement lines, two rate notices covering the same day: each is a contradiction in the
        merchant's own book. The system files it rather than picking, and the reason it files says which
        contradiction it hit.`)}
      ${rule("A model may cite, never assert", `The agent tier cannot express an amount. It cites an
        adjustment row, or a rate, and code recomputes the consequence. An unverified rate closes as
        <em>needs approval</em> with the working attached — never as resolved. That rule exists because
        an earlier design let a proposer choose the residual, and it closed every case.`)}
      ${rule("Integer paise, everywhere", `Floats never touch money here. Fees round per step — MDR, then
        GST on the rounded MDR — because real settlement reports do, and the sub-rupee drift that
        produces is a defect class of its own rather than something a tolerance quietly absorbs.`)}
    </div>`;
  },
};

/* ── the run ──────────────────────────────────────────────────────────── */

async function doRun() {
  if (state.busy) return;
  state.busy = true;
  $("go").disabled = true;
  $("go").innerHTML = `<span class="load"></span>Reconciling`;
  $("err").innerHTML = "";
  try {
    state.run = await post("/api/run", params());
    await loadWorklist();
    state.selected = null;
    state.answer = state.bank = null;
    state.matches.page = 1;
    state.source.page = 1;
    paintContext();
    render();
  } catch (e) {
    $("err").innerHTML = `<div class="err">${esc(e.message)}</div>`;
  } finally {
    state.busy = false;
    $("go").disabled = false;
    $("go").textContent = "Reconcile";
  }
}

function paintContext() {
  const d = state.run;
  if (!d) return;
  $("ctx").innerHTML = `<b>${esc(d.key.profile)}</b> mix <span class="sep">·</span> seed ${d.key.seed}
    <span class="sep">·</span> rung <b>${esc(d.key.rung)}</b> <span class="sep">·</span>
    ${n(d.counts.orders)} orders <span class="sep">·</span> ${d.seconds}s`;
  $("navq").textContent = n(d.queue.length);
  $("navm").textContent = n(d.shape.tiers.reduce((a, t) => a + t.count, 0));
  $("foot").innerHTML = `<span class="dot ${state.model && state.model.ready ? "live" : ""}"></span>${
    esc(state.model && state.model.ready ? state.model.model : "no model configured")}<br>
    ${n(d.counts.payments)} payments · ${n(d.counts.settlements)} batches · ${n(d.counts.bank_lines)} credits`;
}

/* ── boot ─────────────────────────────────────────────────────────────── */

$("go").onclick = doRun;
$("toggle").onclick = () => $("runbar").classList.toggle("open");
$("profile").onchange = () => { $("seed").value = $("profile").value === "holdout" ? 21 : 7; };
window.addEventListener("hashchange", route);
document.addEventListener("keydown", (e) => {
  // `e.target` is not always an Element -- a key delivered to the document
  // itself has no `matches`, and calling it there throws inside the listener,
  // where the failure is swallowed and the shortcut just silently stops working.
  const el = e.target;
  if (el instanceof Element && el.closest("input, textarea, select")) return;
  if (state.route !== "exceptions") return;
  if (e.key === "ArrowDown" || e.key === "j") { e.preventDefault(); moveSelection(1); }
  if (e.key === "ArrowUp" || e.key === "k") { e.preventDefault(); moveSelection(-1); }
});

(async () => {
  try {
    state.model = await (await fetch("/api/model")).json();
  } catch (_) { state.model = { ready: false, problem: "model status unavailable" }; }
  route();
  await doRun();
})();
