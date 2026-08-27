"""No key-shaped string may reach a tracked file.

This exists because one did. A real NVIDIA key was pasted into `.env.example`
-- the template, which is deliberately *not* gitignored so people can copy it --
and it was committed and pushed to a public repository. The mistake is easy and
silent: `.env` is ignored, `.env.example` is not, and they differ by one word.

Rotation is the only real remedy once a key is pushed, so the value of this test
is entirely in catching the next one before it ships.
"""

import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Prefixes long enough to be unambiguous. Deliberately not a general
#: high-entropy scan -- this repo is full of hashes and ids, and a checker that
#: cries wolf gets disabled.
PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg",
    ".sh", ".html", ".js", ".example", ".env",
}


def _tracked_files() -> list[pathlib.Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    return [ROOT / p for p in out.split("\0") if p]


def test_no_api_key_in_any_tracked_file():
    offenders: list[str] = []
    for path in _tracked_files():
        if path.suffix not in TEXT_SUFFIXES and path.name != ".env.example":
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for pattern in PATTERNS:
            hit = pattern.search(text)
            if hit:
                # Report the prefix only. A failing test that prints the secret
                # copies it into CI logs, which is a second leak.
                offenders.append(
                    f"{path.relative_to(ROOT)}: {hit.group(0)[:12]}...(redacted)"
                )
    assert not offenders, "API key in a tracked file:\n  " + "\n  ".join(offenders)


def test_env_example_ships_empty_values():
    """The template must be a template.

    It is committed on purpose so people can copy it, which is exactly why a
    value in it is worse than a value in any other file here.
    """
    example = ROOT / ".env.example"
    assert example.exists(), ".env.example is missing; contributors need the template"
    for line in example.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert "=" in line, f"unparseable line in .env.example: {line!r}"
        _, _, value = line.partition("=")
        assert not value.strip().strip('"').strip("'"), (
            f"{line.split('=')[0]} has a value in .env.example; it must be blank"
        )


def test_dotenv_is_ignored():
    result = subprocess.run(
        ["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, ".env is not gitignored"
