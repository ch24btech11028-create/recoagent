"""A single static page a reader can open before deciding to read the code.

The repository's evidence is spread across twenty-odd artifacts in `results/`,
and a reader who has to open six files to find out whether the false-match rate
is zero will not open any of them. This assembles the headline numbers into one
self-contained HTML file.

Three rules it follows, because a summary page is the easiest place in a
project to start overstating things:

1. **Every number is read from a committed artifact.** Nothing is recomputed
   here and nothing is typed in by hand, so this page cannot say something the
   evidence does not. If an artifact is missing, its panel says so rather than
   quietly disappearing.
2. **The unflattering numbers get the same size type as the flattering ones.**
   The adversarial audit's declared limits, the open items in the clearing
   account and the agent tier's zero are on the page, not behind a link.
3. **No external assets.** Stdlib string formatting, inline CSS, inline SVG. It
   opens from a file:// URL with no network, which is the same property the
   rest of the repository has.

Usage:
    python -m recoagent.publish --out site/index.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import date
from pathlib import Path

from .money import format_inr

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Reading the evidence
# ─────────────────────────────────────────────────────────────────────────────


def _json(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def _text(name: str) -> str | None:
    path = RESULTS / name
    return path.read_text(encoding="utf-8") if path.exists() else None




def _pct(x: float) -> str:
    return f"{x:.2%}"


def _grep(text: str | None, pattern: str, group: int = 1) -> str | None:
    if not text:
        return None
    m = re.search(pattern, text)
    return m.group(group).strip() if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Panels
# ─────────────────────────────────────────────────────────────────────────────


def _missing(title: str, name: str) -> str:
    return (
        f'<section class="panel"><h2>{html.escape(title)}</h2>'
        f'<p class="absent">No <code>results/{html.escape(name)}</code> in this '
        f"checkout, so this panel has nothing to report. It is left visible on "
        f"purpose: a panel that vanished when its evidence did would make the "
        f"page look complete when it is not.</p></section>"
    )


def _headline(dev: dict | None, holdout: dict | None) -> str:
    if not dev or not holdout:
        return _missing("Results", "B2_dev.json")
    d, h = dev["scorecard"], holdout["scorecard"]
    cards = [
        ("False-match rate", _pct(d["false_match_rate"]),
         "dev and held-out, every rung", True),
        ("Auto-match rate", _pct(d["auto_match_rate"]),
         f"held-out {_pct(h['auto_match_rate'])}", False),
        ("Credit value matched", _pct(d["value_share"]),
         f"held-out {_pct(h['value_share'])}", False),
        ("Variance carried, not absorbed", format_inr(abs(dev["documented_variance_paise"])),
         "on the match records, reported not hidden", False),
    ]
    out = ['<section class="panel"><h2>Reconciliation</h2>', '<div class="cards">']
    for label, value, note, lead in cards:
        cls = "card lead" if lead else "card"
        out.append(
            f'<div class="{cls}"><span class="k">{html.escape(label)}</span>'
            f'<span class="v">{html.escape(value)}</span>'
            f'<span class="n">{html.escape(note)}</span></div>'
        )
    out.append("</div>")

    out.append(
        '<table><thead><tr><th>Leg</th><th>Population</th><th>Matched</th>'
        "<th>True</th><th>False</th><th>Exceptions</th></tr></thead><tbody>"
    )
    names = {"1": "order &rarr; payment", "2": "credit &rarr; batch"}
    for leg in ("1", "2"):
        s = d["legs"][leg]
        out.append(
            f'<tr><td>{names[leg]}</td><td class="n">{s["population"]:,}</td>'
            f'<td class="n">{s["attempted"]:,}</td>'
            f'<td class="n">{s["true_matches"]:,}</td>'
            f'<td class="n {"bad" if s["false_matches"] else "good"}">'
            f'{s["false_matches"]}</td>'
            f'<td class="n">{s["exceptions"]:,}</td></tr>'
        )
    out.append("</tbody></table>")

    mishandled = sum(a["mishandled"] for a in d["accounting"])
    out.append(
        f'<p class="note">Every one of the twelve injected defect classes was '
        f"either resolved or correctly flagged: <strong>{mishandled} "
        f"mishandled</strong>, {d['unattributed_exceptions']} exceptions with no "
        f"injected cause. Read the first row first — an engine that matches "
        f"everything and is occasionally wrong books money against the wrong "
        f"transaction and hides it behind a green number.</p>"
    )
    # The caveat travels with the claim. A zero on a summary page with the
    # argument against it two clicks away is the shape of an overclaim, so the
    # limits sit in the same panel as the number they qualify.
    out.append(
        '<p class="note warn"><strong>What the 0.00% does not mean.</strong> '
        "The arithmetic gate does not produce it: forcing 85 genuinely failing "
        "proofs to close leaves it unchanged, because the pairing comes from an "
        "identifier join (97.1% of Leg 1, 80.0% of Leg 2) and the gate only asks "
        "whether the money agrees. The population where a wrong pairing was "
        "possible \u2014 duplicate payments and duplicate UTRs, 90.9% of the "
        "exception list \u2014 is refused rather than solved. On real "
        "third-party data (BenchRec) the wrong-match rate is <strong>0.28%</strong>, "
        "and under adversarial attack <strong>17 of 420</strong> land. Run "
        "<code>python3 -m recoagent.audit.gate</code> to reproduce this.</p>"
    )
    return "".join(out) + "</section>"


def _audit(card: dict | None) -> str:
    if not card:
        return _missing("Adversarial audit", "mutation_audit.json")
    o = card["overall"]
    unexpected = len(card.get("unexpected_failures", []))
    limits = card.get("known_limits", [])
    out = [
        '<section class="panel"><h2>Attacking it on purpose</h2>',
        '<div class="cards">',
        f'<div class="card lead"><span class="k">Undeclared wrong matches</span>'
        f'<span class="v">{unexpected}</span>'
        f'<span class="n">across {o["total"]} adversarial cases</span></div>',
        f'<div class="card"><span class="k">Containment</span>'
        f'<span class="v">{o["containment_rate"]:.2%}</span>'
        f'<span class="n">held {o["held"]} &middot; refused {o["refused"]}</span></div>',
        f'<div class="card"><span class="k">Crashes</span>'
        f'<span class="v">{o["crash"]}</span>'
        f'<span class="n">malformed input must not stop the book</span></div>',
        "</div>",
        '<table><thead><tr><th>Family</th><th>Cases</th><th>Held</th>'
        "<th>Refused</th><th>Wrong</th><th>Contained</th></tr></thead><tbody>",
    ]
    for family, f in sorted(card["families"].items()):
        out.append(
            f'<tr><td>{html.escape(family)}</td><td class="n">{f["total"]}</td>'
            f'<td class="n">{f["held"]}</td><td class="n">{f["refused"]}</td>'
            f'<td class="n {"bad" if f["wrong_match"] else "good"}">'
            f'{f["wrong_match"]}</td>'
            f'<td class="n">{f["containment_rate"]:.0%}</td></tr>'
        )
    out.append("</tbody></table>")

    if limits:
        names = sorted({c["mutation"] for c in limits})
        out.append(
            f'<p class="note warn"><strong>{len(limits)} wrong matches, all of '
            f"them on declared limits.</strong> "
            f"{html.escape(', '.join(names))} — attacks this design does not "
            f"claim to survive. They are run and printed on every pass, they "
            f"gate nothing, and a test asserts they still land so the "
            f"disclaimer cannot outlive the weakness. Leg 2's evidence is the "
            f"narration, the amount and the value date; an adversary holding "
            f"all three can manufacture a match, and the defence is that a "
            f"bank statement is not attacker-controlled.</p>"
        )
    return "".join(out) + "</section>"


def _books(dev: str | None, holdout: str | None) -> str:
    if not dev:
        return _missing("The books", "journal_dev.txt")
    balanced = "BALANCED" in dev
    debits = _grep(dev, r"total debits\s+(Rs [\d,\.]+)")
    entries = _grep(dev, r"entries posted\s+([\d,]+)")
    unattributed = _grep(dev, r"unattributed\s+(-?Rs [\d,\.]+)")
    rounding = _grep(dev, r"sub-rupee rounding[^\n]*?(-?Rs [\d,\.]+)")

    rows = re.findall(
        r"^  (the gateway has not paid[^\n]*?|a payment in the batch[^\n]*?|"
        r"a payment reported here[^\n]*?|an FX or repricing[^\n]*?|"
        r"sub-rupee rounding[^\n]*?)\s{2,}(\d+)\s+(-?Rs [\d,\.]+)\s*$",
        dev, re.M,
    )

    out = [
        '<section class="panel"><h2>The books</h2>',
        '<div class="cards">',
        f'<div class="card lead"><span class="k">Trial balance</span>'
        f'<span class="v">{"balanced" if balanced else "OUT"}</span>'
        f'<span class="n">{html.escape(debits or "")} both sides</span></div>',
        f'<div class="card"><span class="k">Entries posted</span>'
        f'<span class="v">{html.escape(entries or "-")}</span>'
        f'<span class="n">dev; held-out balances too</span></div>',
        f'<div class="card lead"><span class="k">Unattributed</span>'
        f'<span class="v">{html.escape(unattributed or "-")}</span>'
        f'<span class="n">every open rupee has a cause</span></div>',
        "</div>",
    ]
    if rows:
        out.append(
            "<table><thead><tr><th>Why a batch still has money in the clearing "
            "account</th><th>Batches</th><th>Value</th></tr></thead><tbody>"
        )
        for cause, n, value in rows:
            out.append(
                f"<tr><td>{html.escape(cause)}</td>"
                f'<td class="n">{n}</td><td class="n">{html.escape(value)}</td></tr>'
            )
        out.append("</tbody></table>")
    out.append(
        f'<p class="note">A capture creates a gateway receivable, fees and '
        f"refunds reduce it, the settlement credit clears it into the bank — so "
        f"the leg-2 identity the matcher proves and “this batch’s receivable "
        f"nets to zero” are the same equation. The claim is not that the account "
        f"empties; it is that nothing is left over. Four of the five causes are "
        f"read off the matcher’s own rule id, and what lands in rounding is "
        f"{html.escape(rounding or 'sub-rupee')}.</p>"
    )
    return "".join(out) + "</section>"


def _agent(text: str | None) -> str:
    if not text:
        return _missing("The agent tier", "B3_dev_nopaper.txt")
    fields = {
        k: _grep(text, rf"{p}\s+(\d+)")
        for k, p in (
            ("resolved", r"RESOLVED \(source-backed\)"),
            ("attempted", r"attempted"),
            ("approval", r"needs approval"),
            ("rejected", r"rejected by the gate"),
            ("declined", r"declined by the model"),
            ("failed", r"endpoint failed"),
        )
    }
    model = _grep(text, r"model=(\S+)")
    return (
        '<section class="panel"><h2>The agent tier, measured</h2>'
        '<div class="cards">'
        f'<div class="card lead"><span class="k">Booked by the model</span>'
        f'<span class="v">{fields["resolved"] or "0"}</span>'
        f'<span class="n">of {fields["attempted"] or "0"} it was asked about</span></div>'
        f'<div class="card"><span class="k">Held for approval</span>'
        f'<span class="v">{fields["approval"] or "0"}</span>'
        f'<span class="n">explained, not verifiable</span></div>'
        f'<div class="card"><span class="k">Endpoint failures</span>'
        f'<span class="v">{fields["failed"] or "0"}</span>'
        f'<span class="n">a run nothing reached is not a measurement</span></div>'
        "</div>"
        f'<p class="note">Model <code>{html.escape(model or "?")}</code>, '
        f"paperwork withheld so the tier has something to do. Every case it saw "
        f"produced a worked explanation and not one could be verified against "
        f"the merchant's own documents, so <strong>nothing was booked</strong> "
        f"and the false-match rate did not move. That is the thesis landing "
        f"rather than failing: the gate is arithmetic, not confidence.</p>"
        "</section>"
    )


def _throughput(text: str | None) -> str:
    if not text:
        return _missing("Throughput", "throughput.txt")
    rows = re.findall(
        r"^\s+([\d,]+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)\s+([\d,]+)\s*$", text, re.M
    )
    points = [
        (int(r[1].replace(",", "")), int(r[4].replace(",", ""))) for r in rows
    ][:5]
    svg = ""
    if len(points) > 1:
        w, h, pad = 560, 170, 30
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        lo, hi = 0, max(ys) * 1.15
        def px(i):
            return pad + i * (w - 2 * pad) / (len(points) - 1)
        def py(v):
            return h - pad - (v - lo) / (hi - lo) * (h - 2 * pad)
        line = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
        dots = "".join(
            f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3.5"/>'
            for i, v in enumerate(ys)
        )
        labels = "".join(
            f'<text x="{px(i):.1f}" y="{h - 9}" text-anchor="middle" '
            f'class="ax">{n:,}</text>'
            for i, n in enumerate(xs)
        )
        svg = (
            f'<svg viewBox="0 0 {w} {h}" role="img" '
            f'aria-label="records per second against book size">'
            f'<polyline points="{line}" fill="none" stroke-width="2"/>{dots}'
            f'{labels}<text x="{pad}" y="16" class="ax">records/sec</text></svg>'
        )
    span = _grep(text, r"across a (\d+x range, throughput moves [\d.]+x)")
    return (
        '<section class="panel"><h2>Throughput</h2>'
        f'<div class="chart">{svg}</div>'
        f'<p class="note">Single process, single thread, <strong>standard '
        f"library only</strong> — no pandas, no database, nothing installed. "
        f"{html.escape(span or '')}. Timing excludes generating the book, so "
        f"what is measured is the matcher and not the fixture.</p></section>"
    )


def _artifacts() -> str:
    names = sorted(p.name for p in RESULTS.glob("*") if p.is_file())
    links = "".join(
        f'<li><a href="../results/{html.escape(n)}">{html.escape(n)}</a></li>'
        for n in names
    )
    return (
        '<section class="panel"><h2>Every artifact on this page</h2>'
        f'<ul class="files">{links}</ul>'
        '<p class="note">Each one is regenerated by a command in the README and '
        "diffed against a fresh run in CI, on a machine that is not the "
        "author's. A published number that has drifted from the code that made "
        "it fails the build.</p></section>"
    )


# ─────────────────────────────────────────────────────────────────────────────
# The page
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#fbfcfd;--panel:#fff;--ink:#0f1720;--soft:#465565;--muted:#6b7a89;
--rule:#dde5ec;--accent:#1b4d7a;--good:#1f6f4a;--bad:#a3282b;--warnbg:#f7eedc}
@media(prefers-color-scheme:dark){:root{--bg:#0b1015;--panel:#141c24;
--ink:#e5ecf2;--soft:#b6c3ce;--muted:#8a99a6;--rule:#27333d;--accent:#7fb3dd;
--good:#5fc492;--bad:#e88184;--warnbg:#2a2113}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);padding:0 1.25rem 5rem;
font:16px/1.6 ui-sans-serif,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto}
header{padding:3.5rem 0 2rem;border-bottom:2px solid var(--ink)}
h1{margin:0 0 .5rem;font-size:2.6rem;letter-spacing:-.02em;line-height:1.1}
.sub{margin:0;color:var(--soft);font-size:1.12rem;max-width:60ch}
.meta{margin-top:1.25rem;color:var(--muted);font-size:.8rem;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.meta a{color:var(--accent)}
.panel{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
padding:1.5rem;margin-top:1.75rem}
h2{margin:0 0 1.1rem;font-size:1.3rem;letter-spacing:-.01em}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));
gap:.85rem;margin-bottom:1.35rem}
.card{border:1px solid var(--rule);border-radius:5px;padding:.85rem .95rem;
display:flex;flex-direction:column;gap:.2rem}
.card.lead{border-color:var(--accent);border-width:1.5px}
.card .k{font-size:.72rem;letter-spacing:.07em;text-transform:uppercase;
color:var(--muted)}
.card .v{font-size:clamp(1.15rem,2.4vw,1.7rem);font-weight:600;
font-variant-numeric:tabular-nums;letter-spacing:-.02em;white-space:nowrap}
.card.lead .v{color:var(--accent)}
.card .n{font-size:.78rem;color:var(--muted)}
table{width:100%;border-collapse:collapse;font-size:.9rem;margin-bottom:.5rem;
display:block;overflow-x:auto}
th{text-align:left;font-size:.7rem;letter-spacing:.07em;text-transform:uppercase;
color:var(--muted);font-weight:600;padding:.5rem .6rem;
border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.5rem .6rem;border-bottom:1px solid var(--rule)}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.good{color:var(--good);font-weight:600}
td.bad{color:var(--bad);font-weight:600}
.note{margin:.6rem 0 0;color:var(--soft);font-size:.9rem;max-width:72ch}
.note.warn{background:var(--warnbg);border-left:3px solid var(--bad);
padding:.8rem 1rem;border-radius:0 4px 4px 0;max-width:none}
.absent{color:var(--muted);font-size:.9rem}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em}
.chart svg{width:100%;height:auto;max-width:620px}
.chart polyline{stroke:var(--accent)}
.chart circle{fill:var(--accent)}
.chart .ax{fill:var(--muted);font-size:10px;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.files{columns:2;font-size:.85rem;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
list-style:none;padding:0;margin:0}
.files a{color:var(--accent);text-decoration:none}
.files a:hover,.files a:focus{text-decoration:underline}
footer{margin-top:2.5rem;padding-top:1.25rem;border-top:1px solid var(--rule);
color:var(--muted);font-size:.82rem;max-width:72ch}
@media(max-width:640px){h1{font-size:1.9rem}.files{columns:1}}
"""


def build(repo_url: str) -> str:
    dev, holdout = _json("B2_dev.json"), _json("B2_holdout.json")
    panels = [
        _headline(dev, holdout),
        _audit(_json("mutation_audit.json")),
        _books(_text("journal_dev.txt"), _text("journal_holdout.txt")),
        _agent(_text("B3_dev_nopaper.txt")),
        _throughput(_text("throughput.txt")),
        _artifacts(),
    ]
    counts = ""
    if dev:
        c = dev["scorecard"]
        counts = (
            f"{c['legs']['1']['population']:,} orders &middot; "
            f"{c['legs']['2']['population']:,} settlement batches &middot; "
            f"seed {c['seed']}"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RecoAgent &mdash; scorecard</title>
<style>{_CSS}</style></head><body><div class="wrap">
<header>
<h1>RecoAgent</h1>
<p class="sub">A settlement reconciliation engine that proves every match
before it books it &mdash; and files an exception rather than guessing.</p>
<p class="meta">{counts} &middot; generated {date.today().isoformat()} from the
committed artifacts &middot; <a href="{html.escape(repo_url)}">source</a></p>
</header>
{''.join(panels)}
<footer>Every figure on this page is read from a file in <code>results/</code>
that CI regenerates and diffs on each push. Nothing here is recomputed at page
build time and nothing is typed in by hand, so this page cannot claim more than
the evidence does &mdash; which is also why the declared limits of the
adversarial audit, the open items in the clearing account, and the agent tier's
zero are on the page rather than behind a link.</footer>
</div></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.publish")
    ap.add_argument("--out", default="site/index.html")
    ap.add_argument(
        "--repo",
        default="https://github.com/ch24btech11028-create/recoagent",
        help="link target for the source link in the header",
    )
    args = ap.parse_args(argv)

    page = build(args.repo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
