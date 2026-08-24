"""The matcher must not be able to reach the answer key.

This is the test that makes every reported number worth reading. A matcher
that can import the generator can be tuned -- accidentally or otherwise --
against the labels it is being scored on. Enforcing the boundary in CI is
cheap; discovering it was violated after publishing metrics is not.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "recoagent"

#: Modules that do the matching. None of these may see the generator.
MATCHER_MODULES = [
    ROOT / "legs" / "leg1.py",
    ROOT / "legs" / "leg2.py",
    ROOT / "legs" / "leg2_t1.py",
    ROOT / "legs" / "ssmp.py",
    ROOT / "legs" / "__init__.py",
    ROOT / "validate.py",
    ROOT / "pipeline.py",
    # The agent tier is under the same restriction as every other matcher. A
    # proposer that could reach ground truth would make the B3 numbers
    # worthless in exactly the way this test exists to prevent.
    ROOT / "agent" / "contracts.py",
    ROOT / "agent" / "evidence.py",
    ROOT / "agent" / "proposer.py",
    ROOT / "agent" / "tier.py",
]

FORBIDDEN = "generator"


def _imported_names(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(a.name for a in node.names)
    return names


def test_matchers_cannot_import_the_generator():
    for path in MATCHER_MODULES:
        assert path.exists(), f"missing matcher module {path}"
        offending = {n for n in _imported_names(path) if FORBIDDEN in n}
        assert not offending, (
            f"{path.name} imports {offending}; the matcher must never be able to "
            "reach ground truth"
        )


def test_matchers_never_mention_ground_truth_types():
    """A weaker but broader net: no textual reference to the label types."""
    for path in MATCHER_MODULES:
        src = path.read_text()
        for banned in ("GroundTruth", "LabelledBatch", "InjectedDefect"):
            assert banned not in src, f"{path.name} references {banned}"
