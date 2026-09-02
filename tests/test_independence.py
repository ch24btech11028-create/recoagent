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
    # The tier that reads the merchant's paperwork is under the same rule. It
    # applies a rate from a source document; being able to see the injected
    # rate instead would make every fee variance it closes meaningless.
    ROOT / "legs" / "repricing.py",
    ROOT / "legs" / "__init__.py",
    ROOT / "validate.py",
    ROOT / "pipeline.py",
    # The door someone else's data comes in through. It builds a SourceBundle
    # from files this repository has never seen, and it must not acquire a
    # taste for the shapes the generator happens to produce.
    ROOT / "ingest.py",
    # The other door, and the one with the strongest pull towards cheating:
    # the generator and the Razorpay mapping both produce SourceBundles, so it
    # would be effortless to reach across and borrow a shape. A translation
    # layer that knew what our synthetic books look like would quietly make
    # real data resemble them.
    ROOT / "razorpay" / "mapping.py",
    ROOT / "razorpay" / "api.py",
    ROOT / "razorpay" / "webhook.py",
    # The categoriser is graded against an answer key on `GroundTruth`, so it
    # is under exactly the same rule as the matchers: a rung that could read
    # `truth.categories` would score 100% and mean nothing.
    ROOT / "categorize" / "rules.py",
    ROOT / "categorize" / "agent.py",
    ROOT / "categorize" / "taxonomy.py",
    # The agent tier is under the same restriction as every other matcher. A
    # proposer that could reach ground truth would make the B3 numbers
    # worthless in exactly the way this test exists to prevent.
    ROOT / "agent" / "contracts.py",
    ROOT / "agent" / "evidence.py",
    ROOT / "agent" / "proposer.py",
    ROOT / "agent" / "tier.py",
    ROOT / "agent" / "tools.py",
    ROOT / "agent" / "agentic.py",
    # Every operator screen -- the queue, the case files, the match log, the
    # source ledgers -- is shaped here. A console that could reach the labels
    # would show an analyst answers no real book comes with, and the difference
    # is invisible from a screenshot.
    ROOT / "views.py",
    # The merchant-register renderer. It exists to say what happened in plain
    # language, which makes it the module where a leaked label would be least
    # visible and most convincing: a fluent sentence carrying a fact no real
    # book comes with reads exactly like the ones that do not.
    ROOT / "plain.py",
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
