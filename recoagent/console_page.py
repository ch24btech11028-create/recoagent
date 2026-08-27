"""The console shell: the document every view is rendered into.

Kept apart from `ui.py` so the server reads as a server, and apart from
`console_js.py` so the markup can be read as markup. What lives here is only
what is true for every screen -- the sidebar, the run bar, and the empty
container the router fills.

The shell holds the run controls because changing the run is not a screen. It
is a thing you do *to* whichever screen you are looking at, and moving it into
one of them would mean the others could not be trusted to say which book they
were describing.
"""

from __future__ import annotations

from .console_css import APP_CSS
from .console_js import APP_JS
from .webstyle import CSS

#: One line each, drawn rather than named, because a nav that is all words is a
#: nav people read every time instead of aiming at.
_ICONS = {
    "overview": '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
    "exceptions": '<path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>',
    "matches": '<path d="M4 7h7"/><path d="M13 7h7"/><path d="M4 17h7"/><path d="M13 17h7"/>'
                '<circle cx="11.5" cy="7" r="2"/><circle cx="12.5" cy="17" r="2"/>',
    "sources": '<path d="M4 6h16"/><path d="M4 12h16"/><path d="M4 18h16"/>'
               '<circle cx="8" cy="6" r="1.4"/><circle cx="15" cy="12" r="1.4"/><circle cx="10" cy="18" r="1.4"/>',
    "agent": '<path d="M4 5h16v11H8l-4 4V5z"/><path d="M9 10h.01"/><path d="M13 10h.01"/>',
    "assurance": '<path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z"/><path d="M9 12l2 2 4-4"/>',
    "results": '<path d="M6 3h9l5 5v13H6z"/><path d="M15 3v5h5"/><path d="M9 13h7"/><path d="M9 17h5"/>',
    "method": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
}


def _nav(route: str, label: str, badge: str | None = None) -> str:
    icon = f'<svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round">{_ICONS[route]}</svg>'
    count = f'<span class="count" id="{badge}">—</span>' if badge else ""
    return f'<a href="#/{route}">{icon}<span>{label}</span>{count}</a>'


_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RecoAgent</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;450;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap">
<style>__CSS__</style>
</head>
<body>
<div class="app">

  <aside class="side">
    <div class="brand">
      <span class="mark">R</span>
      <div><b>RecoAgent</b><small>settlement control</small></div>
    </div>

    <div class="navgroup">Operations</div>
    <nav class="nav">
      __NAV_OPS__
    </nav>

    <div class="navgroup">Evidence</div>
    <nav class="nav">
      __NAV_EVIDENCE__
    </nav>

    <div class="sidefoot" id="foot">—</div>
  </aside>

  <main class="main">
    <div class="topbar">
      <span class="ctx" id="ctx">no run yet</span>
      <span class="spacer"></span>
      <button class="ghost" id="toggle">Run settings</button>
      <button class="primary" id="go">Reconcile</button>
    </div>

    <div class="runbar" id="runbar">
      <div class="field"><label for="n">Orders</label>
        <input id="n" type="number" value="2000" min="100" max="50000" step="100"></div>
      <div class="field"><label for="seed">Seed</label><input id="seed" type="number" value="7"></div>
      <div class="field"><label for="profile">Defect mix</label>
        <select id="profile">
          <option value="dev">dev</option>
          <option value="holdout">held-out</option>
          <option value="clean">clean</option>
        </select></div>
      <div class="field"><label for="rung">Rung</label>
        <select id="rung">
          <option value="B2">B2 — solver</option>
          <option value="B0">B0 — exact keys</option>
        </select></div>
      <span class="spacer"></span>
      <div class="field"><label for="theme">Theme</label>
        <select id="theme">
          <option value="auto">auto</option>
          <option value="light">light</option>
          <option value="dark">dark</option>
        </select></div>
    </div>

    <div id="err"></div>
    <div class="view" id="view"></div>
  </main>
</div>
<script>__JS__</script>
</body>
</html>
"""

PAGE = (
    _HTML
    .replace("__CSS__", CSS + APP_CSS)
    .replace("__NAV_OPS__", "\n      ".join([
        _nav("overview", "Overview"),
        _nav("exceptions", "Exception queue", "navq"),
        _nav("agent", "Ask the agent"),
    ]))
    .replace("__NAV_EVIDENCE__", "\n      ".join([
        _nav("matches", "Match log", "navm"),
        _nav("sources", "Source ledgers"),
        _nav("assurance", "Assurance"),
        _nav("results", "Published results"),
        _nav("method", "How to read this"),
    ]))
    .replace("__JS__", APP_JS)
)
