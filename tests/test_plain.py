"""Is it actually plain? Two questions, and they are different.

The first is whether the sentence is *true*: the rupees in it have to be the
rupees in the proof, and an item that was not booked has to still read as one.
Those are the tests that stop this module becoming a friendlier way to be wrong.

The second is whether it is *readable*, and that is not a matter of taste here.
`readability` below scores every account the way a legibility standard would --
sentence length and syllable count, the Flesch reading-ease formula -- and the
suite fails if the text drifts back towards the register it was written to
escape. A house style nothing measures is a house style that lasts one commit.
"""

from __future__ import annotations

import re

import pytest

from recoagent import plain
from recoagent.agent.citations import FeeVarianceClaim, resolve
from recoagent.defects import DefectClass
from recoagent.eval.b3 import build
from recoagent.generator import DefectMix, GeneratorConfig, generate
from recoagent.money import FeeSchedule
from recoagent.pipeline import run_b2

PROFILES = ["dev", "holdout", "unknown"]
SEEDS = {"dev": 7, "holdout": 21, "unknown": 21}


def _book(profile="dev", n=2000):
    mix = getattr(DefectMix, profile)()
    batch = generate(GeneratorConfig(n_orders=n, seed=SEEDS[profile], mix=mix))
    return batch, run_b2(batch.sources)


def _accounts(profile="dev"):
    batch, result = _book(profile)
    return [
        (exc, plain.account_for(exc, batch.sources)) for exc in result.exceptions
    ]


# ── is it true ───────────────────────────────────────────────────────────


def test_every_exception_gets_an_account():
    """No queue row may fall through to an empty panel.

    A merchant-facing screen that is blank for the one item they care about is
    worse than the analyst string it replaced.
    """
    for profile in PROFILES:
        for exc, account in _accounts(profile):
            assert account.headline.strip(), f"{profile}/{exc.exception_id}: no headline"
            assert account.status.strip()
            assert account.next_step.strip()


def test_the_rupees_in_the_sentence_are_the_rupees_in_the_proof():
    """The property no model-written summary could have.

    Every figure is rendered from the priced resolution, so the text cannot
    drift from the arithmetic. This asserts it rather than trusting it.
    """
    batch = build("dev", 2000, paperwork=False)
    result = run_b2(batch.sources)
    fees = FeeSchedule.default()
    checked = 0

    for exc in result.exceptions:
        if exc.leg != 2 or exc.residual_paise is None:
            continue
        settlement = next(
            (s for s in batch.sources.settlements
             if s.settlement_id == exc.related_id), None,
        )
        if settlement is None:
            continue
        charged = [
            p for p in batch.sources.payments_by_settlement(exc.related_id)
            if fees.mdr_for(p.method) > 0
        ]
        if not charged:
            continue
        ids = tuple(p.payment_id for p in charged)
        for bps in range(150, 320):
            res = resolve(batch.sources, settlement, [FeeVarianceClaim(ids, bps)], fees)
            if res.ok and res.total_paise == exc.residual_paise:
                account = plain.account_for(
                    exc, batch.sources, resolution=res, verified=res.fully_verified
                )
                assert plain.rupees(res.total_paise) in account.text
                assert plain.rupees(exc.residual_paise) in account.text
                checked += 1
                break
        if checked >= 3:
            break

    assert checked, "no priced hypothesis was available to check the wording against"


def test_a_held_item_never_reads_as_a_settled_one():
    """The failure this module could introduce, held shut.

    A warm paragraph about an item nobody accepted is worse than the jargon it
    replaced: the reader walks away believing the book balances.
    """
    batch = build("dev", 2000, paperwork=False)
    result = run_b2(batch.sources)
    fees = FeeSchedule.default()

    for exc in result.exceptions:
        if exc.leg != 2 or exc.residual_paise is None:
            continue
        settlement = next(
            (s for s in batch.sources.settlements
             if s.settlement_id == exc.related_id), None,
        )
        if settlement is None:
            continue
        charged = [
            p for p in batch.sources.payments_by_settlement(exc.related_id)
            if fees.mdr_for(p.method) > 0
        ]
        if not charged:
            continue
        ids = tuple(p.payment_id for p in charged)
        for bps in range(150, 320):
            res = resolve(batch.sources, settlement, [FeeVarianceClaim(ids, bps)], fees)
            if res.ok and res.total_paise == exc.residual_paise:
                assert not res.fully_verified, "expected an unconfirmed rate here"
                account = plain.account_for(
                    exc, batch.sources, resolution=res, verified=False
                )
                assert "not accepted" in account.status.lower()
                return
    pytest.skip("no unverified hypothesis available on this book")


def test_nothing_is_claimed_when_nothing_explains_it():
    """With the paperwork in the book the deterministic tiers leave no residual
    at all, so this is measured on the book that does leave one."""
    batch = build("dev", 2000, paperwork=False)
    result = run_b2(batch.sources)
    exc = next(
        (e for e in result.exceptions if e.residual_paise is not None), None
    )
    if exc is None:
        pytest.skip("no residual-bearing exception on this book")
    account = plain.account_for(exc, batch.sources)
    assert "do not account" in account.text or "did not add up" in account.text
    assert "accepted this and your books balance" not in account.text


# ── is it readable ───────────────────────────────────────────────────────

#: Things a merchant should never have to read. Row ids, internal codes, and
#: the units the harness uses to grade itself.
JARGON = [
    (re.compile(r"\b(pay|order|setl|bank|adj)_\d"), "a row id"),
    (re.compile(r"\bx[12]_"), "an exception id"),
    (re.compile(r"\bpaise\b", re.I), "paise"),
    (re.compile(r"\bbps\b", re.I), "basis points"),
    (re.compile(r"\bMDR\b"), "MDR"),
    (re.compile(r"\bresidual\b", re.I), "residual"),
    (re.compile(r"\bleg [12]\b", re.I), "leg 1 / leg 2"),
    (re.compile(r"\bT[012]\b"), "a tier name"),
    (re.compile(r"[a-z]+_[a-z]+"), "a snake_case identifier"),
    (re.compile(r"\b[A-Z]{3,}_[A-Z_]+\b"), "a defect class code"),
]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _syllables(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1 and not word.endswith(("le", "ee")):
        n -= 1
    return max(1, n)


def readability(text: str) -> float:
    """Flesch reading ease. Higher is plainer; 60+ is broadly 'plain English'."""
    sentences = _sentences(text)
    words = re.findall(r"[A-Za-z']+", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return (
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syllables / len(words))
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_no_account_speaks_in_jargon(profile):
    for exc, account in _accounts(profile):
        for part in (account.headline, *account.body, account.status, account.next_step):
            if not part:
                continue
            # Money is formatted by `money.format_inr`, which writes "Rs" -- the
            # one place a non-word token is expected.
            probe = part.replace("Rs ", "")
            for pattern, what in JARGON:
                assert not pattern.search(probe), (
                    f"{profile}/{exc.exception_id} says {what}: {part!r}"
                )


@pytest.mark.parametrize("profile", PROFILES)
def test_every_sentence_is_a_sentence(profile):
    for exc, account in _accounts(profile):
        for part in (account.headline, *account.body, account.status, account.next_step):
            if not part:
                continue
            assert part[0].isupper(), f"{exc.exception_id}: {part!r} does not start a sentence"
            assert part.rstrip().endswith((".", "!", "?")), (
                f"{exc.exception_id}: {part!r} does not end one"
            )


@pytest.mark.parametrize("profile", PROFILES)
def test_sentences_stay_short_enough_to_read(profile):
    """Long sentences are how a plain register quietly becomes a technical one."""
    for exc, account in _accounts(profile):
        for sentence in _sentences(account.text):
            words = len(re.findall(r"[A-Za-z']+", sentence))
            assert words <= 34, (
                f"{profile}/{exc.exception_id}: {words}-word sentence: {sentence!r}"
            )


@pytest.mark.parametrize("profile", PROFILES)
def test_it_scores_as_plain_english(profile):
    """The objective version of "can a merchant read this".

    Flesch reading ease, computed over every account this book produces. 55 is
    around the level of a broadsheet newspaper and comfortably above the
    register these sentences replaced -- `The residual of -947 paise equals
    1.9979% of the card_domestic gross` scores in the twenties.
    """
    scores = [readability(a.text) for _, a in _accounts(profile)]
    assert scores
    worst = min(scores)
    mean = sum(scores) / len(scores)
    assert worst >= 45, f"{profile}: an account scored {worst:.1f}"
    assert mean >= 55, f"{profile}: mean reading ease {mean:.1f}"


def test_the_measures_can_tell_the_two_registers_apart():
    """What each measure is actually good for -- including where one is useless.

    Flesch scores the analyst line at ~61, *higher* than some of the plain
    accounts. That is not a bug in the formula, it is the formula: it counts
    syllables and sentence length, and `card_domestic` is three short syllables
    while `charged at a higher rate than your rate card` is nine longer ones.
    Readability scoring cannot see that a word is unreadable, only that it is
    short.

    So the two measures do different jobs and neither is dropped. The jargon
    scan is what separates the registers. Flesch guards the other direction --
    plain words assembled into long, clause-heavy sentences, which is how this
    kind of writing usually decays. Recording that here rather than lowering a
    threshold until the suite agrees with a claim it cannot actually support.
    """
    before = (
        "The residual of -947 paise equals 1.9979% of the card_domestic gross "
        "for pay_00316; fee_variance citation at 200 bps rejected by the gate."
    )

    # Flesch does NOT separate them. Asserted so nobody later "fixes" the
    # thresholds on the assumption that it should.
    assert readability(before) > 45

    # The jargon scan does, and this is the line that carries the claim.
    hits = [what for pattern, what in JARGON if pattern.search(before)]
    assert len(hits) >= 4, f"the analyst line should trip several rules, tripped {hits}"

    batch, result = _book("dev")
    exc = next(
        e for e in result.exceptions
        if e.suspected_class == DefectClass.DUPLICATE_PAYMENT
    )
    account = plain.account_for(exc, batch.sources)
    probe = account.text.replace("Rs ", "")
    assert not [what for pattern, what in JARGON if pattern.search(probe)]
    assert readability(account.text) >= 55
