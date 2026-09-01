"""Structured events, so a run that behaved oddly leaves a trail.

This repository had no logging at all. That is defensible for a pure function --
same sources, same result, byte for byte, so a rerun *is* the diagnostic -- and
it stops being defensible the moment anything talks to a network, waits on a
rate limit, retries, or is driven by a person clicking. Those runs are not
reproducible from a seed, and when one behaves strangely there is currently
nothing to read afterwards.

So: standard-library `logging`, one line per event, key=value pairs.

    run.complete rung=B2 profile=dev n=2000 matches=2151 exceptions=40 seconds=0.148
    model.call model=gemini-3.5-flash-lite outcome=ok tokens_in=329 tokens_out=128 seconds=2.0
    worklist.transition fp=1f3c9a status="open -> resolved" actor=asha

Three decisions worth stating.

**Quiet by default.** The level starts at WARNING, so a CLI that prints a
scorecard prints a scorecard. Nothing here appears unless somebody asks for it
with `-v` or `RECOAGENT_LOG=info`.

**stderr, never stdout.** `python -m recoagent.run --out x.json` writes an
artifact and CI diffs it; a log line on stdout would corrupt exactly the
property this project is built on.

**key=value, not JSON.** It greps, it reads in a terminal, and it needs no
dependency to write or to parse. A value containing a space is quoted, which is
the only escaping rule there is.
"""

from __future__ import annotations

import logging
import os
import sys

#: Set to a level name -- `RECOAGENT_LOG=info` -- to turn events on without a
#: flag. Useful for the console and for CI, which have no argv to add to.
ENV_VAR = "RECOAGENT_LOG"

_ROOT = "recoagent"
_configured = False


def logger(name: str) -> logging.Logger:
    """The logger for one subsystem: `trace.logger("pipeline")`."""
    return logging.getLogger(f"{_ROOT}.{name}")


def _render(value: object) -> str:
    text = str(value)
    if text == "":
        return '""'
    return f'"{text}"' if any(c in text for c in ' ="') else text


def event(log: logging.Logger, name: str, **fields: object) -> None:
    """One event, one line.

    Guarded on the level before anything is formatted, so an event on a hot
    path costs an attribute lookup when logging is off -- which is the default.
    """
    if not log.isEnabledFor(logging.INFO):
        return
    if fields:
        log.info("%s %s", name, " ".join(f"{k}={_render(v)}" for k, v in fields.items()))
    else:
        log.info("%s", name)


def problem(log: logging.Logger, name: str, **fields: object) -> None:
    """An event that is worth seeing even when nobody asked for events.

    Reserved for things a person needs to know about a run they are not
    watching: a call that never reached the endpoint, a queue action refused.
    """
    if fields:
        log.warning("%s %s", name, " ".join(f"{k}={_render(v)}" for k, v in fields.items()))
    else:
        log.warning("%s", name)


def configure(level: str | int | None = None, *, stream=None, force: bool = False) -> None:
    """Attach one stderr handler to the `recoagent` logger. Idempotent.

    Explicit `level` wins over the environment. Absent both, the level is
    WARNING: the events are there, and nothing prints them.
    """
    global _configured
    if _configured and not force:
        if level is not None:
            logging.getLogger(_ROOT).setLevel(_level(level))
        return

    log = logging.getLogger(_ROOT)
    for old in list(log.handlers):
        log.removeHandler(old)
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s", "%H:%M:%S"))
    log.addHandler(handler)
    log.setLevel(_level(level if level is not None else os.environ.get(ENV_VAR, "warning")))
    # A library that also propagates to the root logger prints twice the moment
    # anything else calls basicConfig.
    log.propagate = False
    _configured = True


def _level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    resolved = logging.getLevelName(str(value).strip().upper())
    return resolved if isinstance(resolved, int) else logging.WARNING


def add_argument(parser) -> None:
    """The one flag, spelled the same way on every CLI that offers it."""
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="log what the run is doing to stderr (-vv for debug)",
    )


def from_args(args) -> None:
    """Configure from a parsed `-v` count, letting the environment fill in."""
    count = getattr(args, "verbose", 0) or 0
    if count >= 2:
        configure(logging.DEBUG)
    elif count == 1:
        configure(logging.INFO)
    else:
        configure()
