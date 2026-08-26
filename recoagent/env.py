"""Read API keys from a local `.env`, without a dependency and without surprises.

The keys live in the environment; this only fills gaps in it. Two rules make
that safe to reason about:

- **The shell always wins.** A variable already set in `os.environ` is never
  overwritten by the file, so `NVIDIA_API_KEY=... python -m ...` behaves the way
  anyone would expect and a stale `.env` cannot silently shadow it.
- **The file is never tracked.** `.gitignore` carries `.env` and every `.env.*`
  except `.env.example`. `require_key` refuses to read a key out of a file that
  git is tracking, because a key in a tracked file is a key that has been
  published.

Format is the boring subset: `KEY=value`, `#` comments, blank lines, optional
surrounding quotes, and an optional `export ` prefix so a file can be both
`source`d and parsed. No interpolation, no multi-line values -- an API key needs
none of it, and every extra rule is another way to mis-read a secret.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: Loaded at most once per process, so repeated `require_key` calls are cheap
#: and a file edited mid-run cannot change an answer half way through.
_loaded: set[Path] = set()


def _parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip one matching pair of quotes; leave anything else alone so a key
        # containing a quote survives intact.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def _is_tracked(path: Path) -> bool:
    """True if git is tracking this file. Best-effort: no git, no objection."""
    try:
        done = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=path.parent if path.parent.exists() else None,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def load_env(path: str | os.PathLike[str] = ".env", *, override: bool = False) -> dict[str, str]:
    """Merge `path` into `os.environ`. Returns what the file defined.

    Existing environment variables are left alone unless `override` is set.
    A missing file is not an error -- exporting in the shell is still the
    documented path, and the file is a convenience on top of it.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError):
        return {}
    except OSError:
        return {}

    values = _parse(text)
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value
    _loaded.add(p.resolve())
    return values


def require_key(env_var: str, *, dotenv: str | os.PathLike[str] = ".env") -> str:
    """The one place a missing API key turns into an error message.

    Checks the environment, then `.env`, and if neither has it explains both
    ways to fix it rather than naming only the one the caller happened to miss.
    """
    key = os.environ.get(env_var)
    if key:
        return key

    p = Path(dotenv)
    if p.is_file():
        if _is_tracked(p):
            raise RuntimeError(
                f"{p} is tracked by git, so anything in it is published. Refusing "
                f"to read {env_var} from it. Run `git rm --cached {p}`, rotate the "
                "key, and put the new one in an untracked .env."
            )
        load_env(p)
        key = os.environ.get(env_var)
        if key:
            return key

    raise RuntimeError(
        f"{env_var} is not set. Either export it in your shell:\n"
        f"    export {env_var}=...\n"
        f"or put it in a .env file next to this repo (untracked, gitignored):\n"
        f"    {env_var}=...\n"
        "Never pass the key as a literal on the command line -- it lands in your "
        "shell history."
    )
