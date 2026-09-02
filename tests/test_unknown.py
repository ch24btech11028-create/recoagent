"""What happens when the engine meets a defect nobody wrote code for.

`tests/test_pipeline.py` and `tests/test_b2.py` already assert the safety
invariants on the `unknown` profile, because they parametrise over it. This
file asserts the things that are specific to it: that the taxonomy really is
out of the matcher's reach, that the comparison against `holdout` is a
controlled one, and that the containment result is what the README says it is.
"""

import ast
import pathlib

import pytest

from recoagent.defects import DefectClass
from recoagent.eval.scorer import score
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.pipeline import run_b0, run_b2
from recoagent.unknown import UNKNOWN_INJECTORS, UNKNOWN_SPECS, UnknownDefectClass

ROOT = pathlib.Path(__file__).resolve().parents[1] / "recoagent"

N = 2000
SEED = 7


def _run(rung=run_b2, mix=None, n=N, seed=SEED):
    batch = generate(GeneratorConfig(n_orders=n, seed=seed, mix=mix or DefectMix.unknown()))
    return batch, score(batch, rung(batch.sources))


# ── The fence ────────────────────────────────────────────────────────────


def _imported(path: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
            names.update(a.name for a in node.names)
    return names


def test_no_matcher_can_import_the_unknown_taxonomy():
    """The measurement is only worth reading while the engine cannot see this.

    Deliberately imports the list from the ground-truth fence rather than
    restating it, so a matcher added there is covered here on the same day.
    """
    from test_independence import MATCHER_MODULES

    for path in MATCHER_MODULES:
        offending = {n for n in _imported(path) if "unknown" in n.split(".")}
        assert not offending, (
            f"{path.name} imports {offending}; a defect class the matcher can "
            "see is not an unknown one, and the containment number would mean "
            "nothing"
        )


def test_no_tier_names_an_unknown_class():
    """A weaker, broader net: no handling anywhere, not even by string.

    Three modules are exempt, and each is on the far side of the fence from
    the matchers: the taxonomy itself, the generator that injects it, and the
    scorer that grades it. All three already see ground truth. Every other file
    in the package -- every tier, solver, tolerance and report -- must not
    contain these strings at all.
    """
    exempt = {
        ROOT / "unknown.py",
        ROOT / "generator.py",
        ROOT / "eval" / "scorer.py",
    }
    names = [c.value for c in UnknownDefectClass]
    for path in sorted(ROOT.rglob("*.py")):
        if path in exempt:
            continue
        src = path.read_text()
        for name in names:
            assert name not in src, (
                f"{path.relative_to(ROOT)} mentions {name}; if a tier now "
                "handles this class it belongs in defects.py, and a genuinely "
                "unhandled class belongs here in its place"
            )


def test_every_unknown_class_is_injectable_and_documented():
    for cls in UnknownDefectClass:
        assert cls in UNKNOWN_INJECTORS, f"{cls.value} has no injector"
        spec = UNKNOWN_SPECS[cls]
        assert spec.description and spec.why_unseen
        assert spec.code == cls.value


# ── The controlled comparison ────────────────────────────────────────────


def test_unknown_differs_from_holdout_by_exactly_the_unknown_classes():
    """Same book, same known defects, plus three the engine has never seen.

    If the known counts drifted, a recall drop on this profile could be the
    mix moving rather than the new classes landing, and the comparison would
    prove nothing.
    """
    def counts(mix):
        batch = generate(GeneratorConfig(n_orders=N, seed=SEED, mix=mix))
        known, unknown = {}, {}
        for d in batch.truth.defects:
            bucket = known if isinstance(d.defect, DefectClass) else unknown
            bucket[d.defect] = bucket.get(d.defect, 0) + 1
        return known, unknown

    holdout_known, holdout_unknown = counts(DefectMix.holdout())
    unknown_known, unknown_unknown = counts(DefectMix.unknown())

    assert holdout_known == unknown_known
    assert holdout_unknown == {}
    assert set(unknown_unknown) == set(UnknownDefectClass)


# ── The result ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("rung", [run_b0, run_b2], ids=["B0", "B2"])
def test_an_unexplainable_defect_is_filed_not_guessed(rung):
    """The claim the profile exists to test.

    Every injected class here is money that does not add up for a reason the
    engine has no model of. It must file each one as an exception. Booking a
    match anyway -- because some subset of rows happened to close the gap --
    is the failure this whole design is meant to make impossible, and it is a
    live risk rather than a theoretical one: `legs.ssmp` will search against a
    residual whatever produced it.
    """
    _, card = _run(rung)
    assert card.unknown_injected > 0, "the profile injected nothing to test"
    assert card.unknown_mishandled == 0
    assert card.unknown_holds
    assert card.overall_false_match_rate == 0.0


def test_nothing_unexplainable_is_quietly_absorbed():
    """Tolerance must not swallow a discrepancy it has no account of.

    An absorbed defect is matched to the *right* settlement, so the false-match
    rate cannot see it. It is still money moving through a gate unexplained.
    """
    _, card = _run()
    assert card.unknown_absorbed == 0


def test_recall_pays_for_it_and_the_scorecard_says_so():
    """Containment is not free, and a report that hid the cost would be lying.

    Leg 2 recall must fall against `holdout` on the same book: these lines
    genuinely cannot be matched, and an engine that kept its recall here would
    be closing them on something other than proof.
    """
    _, unknown = _run()
    _, holdout = _run(mix=DefectMix.holdout())
    assert unknown.legs[2].recall < holdout.legs[2].recall
    assert unknown.legs[2].false_matches == 0


def test_the_section_is_rendered_only_when_it_has_something_to_say():
    from recoagent.eval.scorer import render

    _, unknown = _run()
    _, dev = _run(mix=DefectMix.dev())

    body = render(unknown)
    assert "DEFECT CLASSES THE ENGINE HAS NO TIER FOR" in body
    assert "HOLDS" in body
    for cls in UnknownDefectClass:
        assert cls.value in body

    assert "DEFECT CLASSES THE ENGINE HAS NO TIER FOR" not in render(dev)
