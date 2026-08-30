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
  filters: { sev: "", leg: "", text: "" },
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

function meter(label, value, total, tone) {
  const share = total ? (value / total) * 100 : 0;
  return `<div class="meter"><span class="lbl">${esc(label)}</span>
    <span class="track"><span class="fill ${tone || ""}" style="width:${share.toFixed(1)}%"></span></span>
    <span class="val">${n(value)}</span></div>`;
}

function card(title, body, tone) {
  return `<div class="card ${tone ? "card--" + tone : ""}"><h3>${esc(title)}</h3>${body}</div>`;
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
    const matchedPct = (c.matched_share * 100).toFixed(1);

    const top = d.queue.slice(0, 6).map(e => `
      <tr class="row row--${e.severity}">
        <td class="cell-id"><span class="chip chip--${e.severity}"></span>
          <a href="#/exceptions/${encodeURIComponent(e.xid)}"><code>${esc(e.id)}</code></a></td>
        <td class="cell-amount num">${esc(e.amount)}<span class="dir">${esc(e.direction)}</span></td>
        <td><span class="tag">${esc(e.suspected)}</span></td>
        <td class="cell-why">${esc(e.reason)}</td></tr>`);

    return `
    <div class="viewhead">
      <h1>Nightly reconciliation</h1>
      <p>Three sources that disagree — merchant orders, the gateway settlement report, the bank
      statement — plus the merchant's own paperwork, reconciled across two legs. Nothing below became
      a match because something believed it should: every one carries arithmetic re-derived from the
      payment rows.</p>
    </div>

    <dl class="stats">
      <div class="stat stat--hero"><dt>False-match rate</dt><dd>${pct(h.false_match_rate)}
        <small>money filed against the wrong entry</small></dd></div>
      <div class="stat"><dt>Auto-matched</dt><dd>${pct(h.auto_match_rate)}<small>across both legs</small></dd></div>
      <div class="stat"><dt>Credit value cleared</dt><dd>${pct(h.value_share)}<small>${esc(h.value_matched)}</small></dd></div>
      <div class="stat stat--money"><dt>Unexplained</dt><dd>${esc(h.unexplained)}
        <small>across ${n(h.open_items)} open items</small></dd></div>
      <div class="stat"><dt>Leg 2 recall</dt><dd>${pct(leg2.recall)}<small>credit → batch, the N:1 leg</small></dd></div>
      ${sh.variance && sh.variance.count ? `<div class="stat stat--money"><dt>Documented variance</dt>
        <dd>${esc(sh.variance.total)}<small>matched, not reconciled away — ${n(sh.variance.count)} declared gaps</small></dd></div>` : ""}
    </dl>

    <div class="cols-2">
      ${card("Where the money is", `
        <div class="stack">
          <span style="width:${matchedPct}%;background:var(--ok)"></span>
          <span style="width:${(100 - matchedPct).toFixed(1)}%;background:var(--crit)"></span>
        </div>
        <div class="stackkey">
          <span><i style="background:var(--ok)"></i>cleared ${esc(c.matched)}</span>
          <span><i style="background:var(--crit)"></i>outstanding ${esc(c.outstanding)}</span>
        </div>
        <p class="lede">${n(c.lines_matched)} of ${n(c.lines_total)} bank credits are tied to a batch whose
        total was recomputed from its payment rows. The settlement header was corroboration, never proof.</p>`)}

      ${card("Open items by money at stake", `<div class="meters">
        ${sh.severities.map(s => meter(SEV_LABEL[s.level] || s.level, s.count, openTotal, s.level)).join("")}
      </div>
      <p class="lede">Severity is the sort order as well as the colour. ${esc(sh.severities[0].at_risk)} of the
      unexplained total sits in the first band.</p>
      <div><a class="tiny" href="#/exceptions"><button class="tiny">Work the queue →</button></a></div>`)}
    </div>

    <div class="cols-2">
      ${card("How the matches closed", `<div class="meters">
        ${sh.rules.slice(0, 7).map(r => meter(r.label, r.count, matchTotal,
            r.tier === "T0" ? "flat" : (r.tier === "T1" ? "minor" : "warn"))).join("")}
      </div>
      <p class="lede">T0 is exact keys and pure bookkeeping. Everything below it is what the recovery
      tiers earned, and each one still had to close the arithmetic to count.</p>`)}

      ${card("What the system suspects", `<div class="meters">
        ${sh.classes.slice(0, 7).map(k => meter(k.name, k.count, openTotal, "flat")).join("")}
      </div>
      <p class="lede">The matcher's own read of what went wrong, made without the labels. How often that
      guess is right is measured separately, on the Assurance screen.</p>`)}
    </div>

    <section>
      <h2>Biggest open items</h2>
      ${table(["Item", "Gap", "Suspected", "What happened"], top, { empty: "Nothing open. Every credit tied out." })}
    </section>

    <section>
      <h2>The book this ran on</h2>
      <dl class="stats">
        ${Object.entries(d.counts).map(([k, v]) =>
          `<div class="stat"><dt>${esc(k.replace(/_/g, " "))}</dt><dd>${n(v)}</dd></div>`).join("")}
      </dl>
      <p class="note">Generated from seed ${d.key.seed} on the ${esc(d.key.profile)} defect mix and
      reconciled in ${d.seconds}s. Same seed, same book, every time —
      <a href="#/sources">browse the source ledgers</a>.</p>
    </section>`;
  },
};

/* ── exceptions ───────────────────────────────────────────────────────── */

function filteredQueue() {
  const f = state.filters, needle = f.text.trim().toLowerCase();
  return state.run.queue.filter(e =>
    (!f.sev || e.severity === f.sev) &&
    (!f.leg || String(e.leg) === f.leg) &&
    (!needle || (e.id + " " + e.reason + " " + e.suspected).toLowerCase().includes(needle)));
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
      <p>What the system refused to match, biggest money at stake first. Built from the sources and the
      result only — the same restriction the matchers run under. Open an item to see everything a tier
      already tried, so nobody repeats the solver's work by hand.</p>
    </div>

    <div class="toolbar">
      <input class="search" id="qsearch" placeholder="Filter by id, reason or suspected class"
             value="${esc(f.text)}" autocomplete="off">
      ${seg("sev", [["", "all severities"], ["critical", "critical"], ["warn", "warn"], ["minor", "minor"], ["structural", "structural"]])}
      ${seg("leg", [["", "both legs"], ["1", "leg 1"], ["2", "leg 2"]])}
      <span class="ctx"><b>${n(rows.length)}</b> of ${n(state.run.queue.length)} shown</span>
    </div>

    <div class="split">
      <div class="qlist" id="qlist">${rows.length ? rows.map(e => `
        <button class="qitem ${state.selected === e.xid ? "on" : ""}" data-x="${esc(e.xid)}">
          <span class="chip chip--${e.severity}" title="${esc(e.severity_hint)}"></span>
          <span class="who">${esc(e.id)}</span>
          <span class="amt">${esc(e.amount)}</span>
          <span class="sub">Leg ${e.leg} · ${esc(e.suspected)} — ${esc(e.reason)}</span>
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
    host.innerHTML = caseFile(d);
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

function caseFile(d) {
  const e = d.item, c = d.case;
  const head = `
    <div class="card">
      <div class="head">
        <div style="flex:1;min-width:0">
          <h2>${esc(e.id)}</h2>
          <div class="meta"><span>Leg ${e.leg} · ${esc(e.kind)}</span><span>${esc(e.severity_hint)}</span>
            <span>stopped at ${esc(e.stopped_at)}</span></div>
        </div>
        <div style="text-align:right">
          <div class="bigamt ${esc(e.direction)}">${esc(e.amount)}</div>
          <div class="meta" style="justify-content:flex-end">${esc(e.direction || "structural")}</div>
        </div>
      </div>
      <div class="kv">
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

      <section><h2>Payments in the batch</h2>${payTable(c.payments)}</section>
      <section><h2>Linked adjustments</h2>${adjTable(c.linked_adjustments, "Nothing was netted into this batch.")}</section>
      <section><h2>Unlinked rows booked nearby</h2>
        <p class="note">Rows the gateway netted but linked to no batch, within a week of this settlement.
        The solver has already searched combinations of these.</p>
        ${adjTable(c.nearby_unlinked, "No unlinked rows anywhere near this batch.")}</section>
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
      <section><h2>Nearest settlements by date, then amount</h2>
        <p class="note">Candidates, not matches. Nothing here has closed any arithmetic; this is the
        lookup an analyst does next, done for them.</p>
        ${table(["Batch", "UTR", "Settled", "Header says", "Gap to this credit", "Days apart"],
          c.candidates.map(s => `<tr><td class="cell-id"><code>${esc(s.settlement_id)}</code></td>
            <td><code>${esc(s.utr)}</code></td><td class="num">${esc(s.settled_at)}</td>
            <td class="num">${esc(s.reported_net)}</td><td class="num">${esc(s.gap)}</td>
            <td class="num">${s.days_apart}</td></tr>`), { empty: "No settlements in this book." })}</section>`;
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
      <section><h2>Bank lines carrying this UTR</h2>
        ${table(["Credit", "Value date", "Amount", "Narration"],
          c.bank_lines_carrying_utr.map(b => `<tr><td class="cell-id"><code>${esc(b.bank_line_id)}</code></td>
            <td class="num">${esc(b.value_date)}</td><td class="num">${esc(b.amount)}</td>
            <td class="cell-why"><code>${esc(b.narration)}</code></td></tr>`),
          { empty: "No statement line mentions this UTR. The money has not arrived." })}</section>
      <section><h2>Payments in the batch</h2>${payTable(c.payments)}</section>
      <section><h2>Linked adjustments</h2>${adjTable(c.linked_adjustments, "Nothing was netted into this batch.")}</section>`;
  }

  if (c.shape === "leg1_order") {
    return head + `
      ${card("The order, as the merchant booked it", `<div class="kv">
        <div><dt>Amount</dt><dd class="num">${esc(c.order.amount)}</dd></div>
        <div><dt>Invoice</dt><dd><code>${esc(c.order.invoice_no)}</code></dd></div>
        <div><dt>Customer</dt><dd><code>${esc(c.order.customer_id)}</code></dd></div>
        <div><dt>Created</dt><dd class="num">${esc(c.order.created_at)}</dd></div>
      </div>`)}
      <section><h2>Payments claiming this order</h2>
        <p class="note">One order, one payment, exactly. Two rows claiming it is a duplicate; a gross
        that differs from the order is a partial capture. Either way the system refuses rather than
        picking.</p>
        ${table(["Payment", "Method", "Status", "Gross", "Gap to order", "Batch", "Captured"],
          c.claims.map(p => `<tr><td class="cell-id"><code>${esc(p.payment_id)}</code></td>
            <td><span class="tag">${esc(p.method)}</span></td><td>${esc(p.status)}</td>
            <td class="num">${esc(p.gross)}</td><td class="num">${esc(p.gap)}</td>
            <td><code>${esc(p.settlement_id || "—")}</code></td>
            <td class="num">${esc(p.captured_at)}</td></tr>`),
          { empty: "No payment row references this order at all." })}</section>
      <section><h2>Refunds and adjustments against those payments</h2>
        ${adjTable(c.refunds, "Nothing was booked against these payments.")}</section>`;
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
      <p>Every accepted match and the arithmetic that accepted it. A match without a proof that can be
      re-checked from the record alone is a claim, not a reconciliation — click any row to see the
      expression, the two sides, the tolerance it had to fit inside, and the hash of the inputs it was
      decided from.</p>
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
      <p>The four sources exactly as the matcher receives them — no labels, no answer key, nothing
      cleaned up on the way in. This is the look-up screen: find the row somebody is asking about.</p>
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
      <p>The agent answers from a factsheet built from this run by code — it has no memory of the book
      and no way to invent one. An answer typed here has no ground truth, so it comes back
      <strong>ungraded</strong> with that factsheet attached for you to check. The measured number is
      the scored bank below.</p>
    </div>
    ${ready ? "" : `<div class="err">${esc((state.model && state.model.problem) || "No model configured.")}</div>`}
    <div class="askbox">
      <textarea id="q" rows="2" ${ready ? "" : "disabled"}
        placeholder="How much is sitting unexplained across the open items?"></textarea>
      <button class="primary" id="ask" ${ready ? "" : "disabled"}>Ask</button>
    </div>
    <div class="chips">${suggestions().map(s => `<button class="sugg">${esc(s)}</button>`).join("")}</div>
    <div id="answer">${state.answer || ""}</div>

    <section>
      <h2>Measured Q&amp;A accuracy</h2>
      <p class="note">The question bank is derived from this run, so every answer has a ground truth
      computed by code — including questions the factsheet deliberately cannot answer, where the only
      correct response is to decline. <strong>Wrong-answer rate leads</strong>, because an operator acts
      on a number either way and a confident wrong one is the expensive failure.</p>
      <div class="askbox">
        <button id="bank" ${ready ? "" : "disabled"}>Run the scored bank</button>
        <div class="field"><label for="limit">Questions (0 = all)</label>
          <input id="limit" type="number" value="0" min="0" max="60"></div>
      </div>
      <div id="bankout">${state.bank || ""}</div>
    </section>`;
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
    const headline = a.failed ? "call failed"
      : (a.declined ? "Declined to answer" : (a.answer === null ? "—" : String(a.answer)));
    state.answer = `<div class="card card--${tone}">
      <span class="banner banner--ungraded">Ungraded — no ground truth for a live question</span>
      <div class="answer">${esc(headline)}</div>
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
      <p>The only screen here that uses the answer key, and it answers a different question from the
      queue: should you believe any of it? Every defect the generator injected must be either flagged
      for a human or correctly resolved. <strong>Mishandled must be zero.</strong></p>
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
    <section><h2>What became of each injected defect</h2>
      <p class="note">Flagged and resolved are both correct outcomes; which one it is shows how far the
      ladder has come. A class migrating from flagged to resolved is a tier that learned to close it.</p>
      ${table(["Defect class", "Injected", "Flagged", "Resolved", "Mishandled"], rows)}</section>
    <div class="callout"><b>Why this panel is separate</b>
      <p>A reconciliation engine that matches everything and is occasionally wrong is worse than useless:
      it files money against the wrong transaction and hides the error behind a green number. So the
      operator screens are built from the sources and the result alone, and the labels live here, behind
      a heading that says so.</p></div>
    <section><h2>Per-leg detail</h2>
      ${table(["Leg", "Population", "Matched", "True", "False", "Recall", "Exceptions"],
        d.legs.map(l => `<tr><td><span class="leg">Leg ${l.leg}</span></td>
          <td class="num">${n(l.population)}</td><td class="num">${n(l.attempted)}</td>
          <td class="num">${n(l.true_matches)}</td>
          <td class="num verdict verdict--${l.false_matches === 0 ? "ok" : "bad"}">${l.false_matches}</td>
          <td class="num">${pct(l.recall)}</td><td class="num">${n(l.exceptions)}</td></tr>`))}</section>`;
  },
};

/* ── published results ────────────────────────────────────────────────── */

VIEWS.results = {
  needsRun: false,
  html() {
    return `
    <div class="viewhead">
      <h1>Published results</h1>
      <p>The artefacts in <code>results/</code>, served exactly as they were written by the eval
      commands. Nothing on this screen is computed live — these are the runs the README quotes, so a
      reader can check the claim against the file that produced it.</p>
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
    return `
    <div class="viewhead">
      <h1>How to read this</h1>
      <p>Every screen here is one view of one run. What each is allowed to know is not a detail of the
      implementation — it is the design.</p>
    </div>
    <div class="cols">
      ${card("The gate is the thesis", `<p class="lede">Nothing becomes a match because something believed
        it should. Leg 2 re-derives every batch total from the payment rows, never from the settlement
        header, and a credit only books when that arithmetic closes inside a stated tolerance. The
        header is corroboration. It is never the proof.</p>`)}
      ${card("Two legs, two problems", `<p class="lede">Leg 1 is order ↔ payment, 1:1, record linkage.
        Leg 2 is settlement batch ↔ bank credit, N:1 — the subset-sum matching problem. They are scored
        separately because they fail differently, and Leg 2 is scored over bank lines, which is the
        harder population.</p>`)}
      ${card("False-match rate leads", `<p class="lede">Match rate is reported second on purpose. An
        engine that matches everything and is occasionally wrong books money against the wrong
        transaction and hides it behind a green number. Refusing costs a queue item; guessing costs a
        correction, and somebody has to find it first.</p>`)}
      ${card("The operator screens cannot see the labels", `<p class="lede">The queue, the match log and
        the source ledgers are built from the sources and the result only — the same restriction every
        matcher runs under, enforced by a test that reads the imports. Ground truth appears on
        Assurance, labelled.</p>`)}
      ${card("Ambiguity is refused, not resolved", `<p class="lede">Two payments claiming one order, a
        UTR on two statement lines, two rate notices covering the same day: each is a contradiction in
        the merchant's own book. The system files it rather than picking, and the reason it files says
        which contradiction it hit.</p>`)}
      ${card("A model may cite, never assert", `<p class="lede">The agent tier cannot express an amount.
        It cites an adjustment row, or a rate, and code recomputes the consequence. An unverified rate
        closes as <em>needs approval</em> with the working attached — never as resolved. That rule exists
        because the earlier design let a proposer choose the residual, and it closed every case.</p>`)}
    </div>
    <div class="callout"><b>Integer paise, everywhere</b>
      <p>Floats never touch money here. Fees round per step — MDR, then GST on the rounded MDR — because
      real settlement reports do, and the sub-rupee drift that produces is a defect class of its own
      rather than something a tolerance quietly absorbs.</p></div>`;
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
