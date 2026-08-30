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


def test_ingest_runs_with_every_third_party_package_blocked(tmp_path):
    """The CSV door is stdlib too.

    It is the entry point a reader is most likely to try on their own machine
    before installing anything, so a pandas import sneaking in here would break
    the claim exactly where it is first tested by someone else.
    """
    csv_maker = """
import csv, datetime
rows = {
  'orders': (['order_id','customer_id','invoice_no','amount_paise','currency','created_at'],
             [['o1','c1','i1','100.00','INR','2026-07-01T10:00:00']]),
  'payments': (['payment_id','order_id','gross_paise','fee_paise','tax_paise','method','status','settlement_id','captured_at'],
               [['p1','o1','100.00','2.00','0.36','card_domestic','captured','s1','2026-07-01T10:05:00']]),
  'settlements': (['settlement_id','utr','settled_at','net_paise','status'],
                  [['s1','UTR1','2026-07-03T10:00:00','97.64','processed']]),
  'bank': (['bank_line_id','value_date','amount_paise','narration','bank_ref'],
           [['b1','2026-07-03','97.64','NEFT UTR1','UTR1']]),
}
import sys
out = sys.argv[1]
for name, (head, body) in rows.items():
    with open(f'{out}/{name}.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(head); w.writerows(body)
"""
    subprocess.run([sys.executable, "-c", csv_maker, str(tmp_path)], check=True)

    script = BLOCKER.format(rung="B2").replace(
        "sys.argv = ['recoagent.run', '--n', '600', '--seed', '7', '--rung', 'B2']",
        "sys.argv = ['recoagent.ingest', '--orders', %r, '--payments', %r, "
        "'--settlements', %r, '--bank', %r]"
        % tuple(str(tmp_path / f"{n}.csv") for n in
                ("orders", "payments", "settlements", "bank")),
    ).replace("runpy.run_module('recoagent.run'", "runpy.run_module('recoagent.ingest'")

    proc = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "Credit value cleared" in proc.stdout


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


def test_the_readme_states_the_real_test_count(request):
    """A number in a document that nothing checks is a number that goes stale.

    It already had: the README said 298 while the suite ran 310. Nobody lies on
    purpose about a test count, which is exactly why it needs a machine to
    notice -- every other figure the README publishes is regenerated by a
    command, and this one was typed.

    Skipped when a subset was run, because `testscollected` then describes the
    subset and would fail for the wrong reason.
    """
    import re
    from pathlib import Path

    collected = request.session.testscollected
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    match = re.search(r"tests/\s+(\d[\d,]*) tests", readme)
    assert match, "the README no longer states a test count in its Layout block"
    claimed = int(match.group(1).replace(",", ""))

    if collected < claimed:
        import pytest
        pytest.skip(f"only {collected} tests in this run; the claim is about the full suite")

    assert collected == claimed, (
        f"README says {claimed} tests, the suite collects {collected}. "
        "Update the Layout block."
    )
