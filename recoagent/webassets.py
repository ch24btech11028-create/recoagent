"""Where the console's front end lives, and how the server hands it over.

The UI is HTML, CSS and JavaScript, and it is written in HTML, CSS and
JavaScript -- `recoagent/web/`. It used to be Python string constants, which
bought nothing and cost everything an editor does for a web file: no
highlighting, no linting, no formatter, no way to open it as what it is.

This module is the whole of the Python side: it locates those files, says what
each one's content type is, and refuses to serve anything not on the list.

Files are read per request. A loopback console serving four files does not need
a cache, and reading each time means editing `app.js` and reloading the browser
is the entire development loop -- no restart.
"""

from __future__ import annotations

import pathlib

WEB_DIR = pathlib.Path(__file__).resolve().parent / "web"

#: The only files this server will hand out, and what they are. An allowlist
#: rather than a path join: a request names one of these or it gets nothing, so
#: there is no traversal to reason about.
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/base.css": ("base.css", "text/css; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


def read(name: str) -> str:
    """One file out of the web directory, by name."""
    return (WEB_DIR / name).read_text(encoding="utf-8")


def asset_for(path: str) -> tuple[str, str] | None:
    """The (body, content type) for a request path, or None if it is not ours."""
    entry = ASSETS.get(path)
    if entry is None:
        return None
    name, content_type = entry
    return read(name), content_type
