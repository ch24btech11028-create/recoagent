"""The deterministic core must run with nothing installed.

The README leads with "zero third-party dependencies", and a claim that lives
only in prose drifts the moment someone adds a convenience import. This runs
the real pipeline in a subprocess with every third-party package blocked at the
import hook, so the claim fails loudly rather than quietly becoming untrue.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED = (
    "pandas", "numpy", "scipy", "sklearn", "splink", "duckdb",
    "anthropic", "openai", "pydantic", "rich", "httpx", "requests",
)

BLOCKER = f"""
import sys
BLOCKED = {BLOCKED!r}
class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError('BLOCKED third-party import: ' + name)
        return None
sys.meta_path.insert(0, Blocker())
import runpy
sys.argv = ['recoagent.run', '--n', '600', '--seed', '7', '--rung', {{rung!r}}]
runpy.run_module('recoagent.run', run_name='__main__')
"""


def _run_blocked(rung: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCKER.format(rung=rung)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_b0_runs_with_every_third_party_package_blocked():
    proc = _run_blocked("B0")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "FALSE-MATCH RATE" in proc.stdout


def test_b2_runs_with_every_third_party_package_blocked():
    """B2 includes the SSMP solver and spill pairing -- still pure stdlib."""
    proc = _run_blocked("B2")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "FALSE-MATCH RATE" in proc.stdout


def test_the_blocker_actually_blocks():
    """Guard against the test passing because the blocker silently does nothing."""
    proc = subprocess.run(
        [sys.executable, "-c", BLOCKER.format(rung="B0").replace(
            "runpy.run_module('recoagent.run', run_name='__main__')", "import pandas"
        )],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "BLOCKED third-party import" in proc.stderr


def test_agent_tier_imports_are_lazy():
    """Importing the agent package must not require an LLM SDK.

    B0 and B2 users install neither SDK; if the import moved to module scope
    the whole package would stop importing for them.
    """
    proc = subprocess.run(
        [sys.executable, "-c", BLOCKER.format(rung="B0").replace(
            "runpy.run_module('recoagent.run', run_name='__main__')",
            "import recoagent.agent; from recoagent.agent import ScriptedProposer, NullProposer; "
            "print('agent package imported without any SDK')",
        )],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "imported without any SDK" in proc.stdout
