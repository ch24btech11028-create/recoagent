"""Audit records must replay to the same verdict from their own contents."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b0
from recoagent.schemas import ArithmeticProof

ROOT = Path(__file__).resolve().parents[1]


def test_every_accepted_match_carries_a_closing_proof():
    batch = generate(GeneratorConfig(n_orders=800, seed=5, mix=DefectMix.dev()))
    result = run_b0(batch.sources)
    assert result.matches
    for m in result.matches:
        assert m.proof is not None, f"{m.match_id} accepted with no proof"
        assert m.proof.closes


def test_proof_replays_from_recorded_numbers_alone():
    """Reconstruct the verdict from the audit record without the source data."""
    batch = generate(GeneratorConfig(n_orders=800, seed=5, mix=DefectMix.dev()))
    for m in run_b0(batch.sources).matches:
        p = m.proof
        replayed = ArithmeticProof(
            expression=p.expression,
            lhs_paise=p.lhs_paise,
            rhs_paise=p.rhs_paise,
            tolerance_paise=p.tolerance_paise,
        )
        assert replayed.closes == p.closes
        assert replayed.residual_paise == p.residual_paise


def test_input_hash_is_stable_across_runs():
    a = {m.match_id: m.input_hash for m in run_b0(
        generate(GeneratorConfig(n_orders=500, seed=9)).sources).matches}
    b = {m.match_id: m.input_hash for m in run_b0(
        generate(GeneratorConfig(n_orders=500, seed=9)).sources).matches}
    assert a == b and a


def test_cli_output_is_byte_identical_across_runs():
    """End-to-end determinism, exercised the way the README tells a reader to."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for name in ("a.json", "b.json"):
            out = Path(tmp) / name
            subprocess.run(
                [sys.executable, "-m", "recoagent.run",
                 "--n", "600", "--seed", "13", "--out", str(out)],
                cwd=ROOT, check=True, capture_output=True,
            )
            paths.append(out)
        assert json.loads(paths[0].read_text()) == json.loads(paths[1].read_text())
        assert paths[0].read_bytes() == paths[1].read_bytes()
