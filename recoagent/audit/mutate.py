"""Adversarial mutation audit -- attacking our own matcher on purpose.

A false-match rate of 0.00% is only as strong as the inputs it was measured on.
Every number in `results/B2_*.json` comes from the generator, and the generator
is friendly: it injects the twelve defect classes a real settlement book
actually produces, at rates a real merchant actually sees. That measures
whether the matcher handles reality. It does not measure whether the matcher
can be *made* to book money against the wrong transaction by someone trying.

This file tries. It takes bank lines the matcher already matched correctly,
corrupts them in ways chosen to defeat a specific part of the join, and asks
one question:

    Under adversarial input, does the system produce more exceptions, or does
    it produce a confident wrong answer?

An exception is a success here. Coverage loss is a cost, not a failure. The
only true failure is a match that contradicts ground truth -- because that is
the one outcome a green dashboard hides.

Four verdicts, not two
----------------------
The obvious taxonomy is contained/not-contained, and it hides the thing worth
knowing. A matcher that refuses everything under attack is perfectly safe and
completely useless, and it would score 100% containment. So refusal and
survival are reported separately:

    HELD         The attack did not land. The matcher still produced the
                 correct pairing -- either the corruption did not reach the
                 evidence it relies on, or a later tier recovered it.
    REFUSED      The matcher declined and filed an exception. Safe. For a
                 mutation that destroys the evidence, this is the *right*
                 answer. For one that leaves it intact, it is a coverage cost.
    WRONG_MATCH  A match was accepted whose pairing contradicts ground truth.
                 The only real failure.
    CRASH        The pass raised. Also a failure -- a matcher that dies on
                 malformed input is a matcher that stops mid-book.

Containment is HELD + REFUSED. Safety is the absence of WRONG_MATCH and CRASH.
The two are reported side by side and never averaged into one number.

Collateral damage counts
------------------------
A mutation targets one bank line, but the verdict is scored over *every* match
in the run. Corrupting one credit and thereby causing a different credit to be
booked against the wrong batch is a wrong match, and it is exactly the kind of
failure a target-only check would miss.

Reproducing a failure
---------------------
Every case is fully determined by its mutation name and seed. Any WRONG_MATCH
prints its own replay line, and `--replay <name> <seed>` re-runs that single
case with the mutation described.

Usage:
    python -m recoagent.audit.mutate --trials 20
    python -m recoagent.audit.mutate --trials 20 --out results/mutation_audit.json
    python -m recoagent.audit.mutate --replay perfect_forgery 100042
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from ..generator import DefectMix, GeneratorConfig, generate
from ..pipeline import run_b2
from ..schemas import BankLine, GroundTruth, ReconResult, Settlement, SourceBundle

HELD = "held"
REFUSED = "refused"
WRONG_MATCH = "wrong_match"
CRASH = "crash"

#: Verdicts that mean the attack did not book money against the wrong batch.
CONTAINED = (HELD, REFUSED)


# ─────────────────────────────────────────────────────────────────────────────
# What a mutation is
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Mutant:
    """A corrupted book, plus what to look at afterwards."""

    sources: SourceBundle
    #: The bank line ids whose verdict is being judged. Wrong matches are
    #: looked for everywhere; HELD/REFUSED is decided on these.
    targets: tuple[str, ...]
    note: str


MutationFn = Callable[[SourceBundle, GroundTruth, random.Random], "Mutant | None"]


@dataclass(frozen=True)
class Mutation:
    name: str
    family: str
    #: True when the correct pairing still exists and is still correct after
    #: the mutation -- so HELD is attainable and REFUSED is a coverage cost.
    #: False when the mutation destroys or contradicts the evidence, making
    #: REFUSED the only defensible answer.
    recoverable: bool
    description: str
    apply: MutationFn


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _lines_by_id(sources: SourceBundle) -> dict[str, BankLine]:
    return {b.bank_line_id: b for b in sources.bank_lines}


def _settlements_by_id(sources: SourceBundle) -> dict[str, Settlement]:
    return {s.settlement_id: s for s in sources.settlements}


def _swap_line(sources: SourceBundle, line: BankLine, **changes) -> SourceBundle:
    """Return a bundle with one bank line replaced. Order is preserved."""
    mutated = replace(line, **changes)
    return replace(
        sources,
        bank_lines=tuple(
            mutated if b.bank_line_id == line.bank_line_id else b
            for b in sources.bank_lines
        ),
    )


def _swap_settlement(sources: SourceBundle, s: Settlement, **changes) -> SourceBundle:
    mutated = replace(s, **changes)
    return replace(
        sources,
        settlements=tuple(
            mutated if x.settlement_id == s.settlement_id else x
            for x in sources.settlements
        ),
    )


def _clean_pairs(
    sources: SourceBundle, truth: GroundTruth, baseline: ReconResult
) -> list[tuple[BankLine, Settlement]]:
    """Bank lines the *unmutated* run matched, correctly, to their true batch.

    Attacking a line that was already an exception proves nothing -- it was
    going to be refused either way. Every mutation starts from a case the
    matcher currently gets right, so any change in verdict is attributable.
    """
    lines = _lines_by_id(sources)
    settlements = _settlements_by_id(sources)
    out = []
    for m in baseline.matches_for_leg(2):
        line_id, sid = m.left_ids[0], m.right_ids[0]
        if truth.leg2.get(line_id) != sid:
            continue  # never build an attack on top of an existing wrong match
        if line_id in lines and sid in settlements:
            out.append((lines[line_id], settlements[sid]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Narration family -- attack the UTR the leg-2 join reads
# ─────────────────────────────────────────────────────────────────────────────


def _m_utr_clipped(sources, truth, rng):
    """The bank clipped the narration mid-UTR. The commonest real corruption."""
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, s = rng.choice(pairs)
    pos = line.narration.find(s.utr)
    if pos < 0:
        return None
    cut = line.narration[: pos + rng.randint(3, 9)]
    return Mutant(
        _swap_line(sources, line, narration=cut),
        (line.bank_line_id,),
        f"narration cut to {cut!r}",
    )


def _m_utr_decoy_digits(sources, truth, rng):
    """A second twelve-digit run in the narration -- an account or cheque number.

    The extractor refuses on more than one candidate rather than picking. This
    checks that it still refuses when the decoy looks exactly like a UTR.
    """
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, s = rng.choice(pairs)
    decoy = "".join(str(rng.randint(0, 9)) for _ in range(12))
    if decoy == s.utr:
        return None
    text = f"{line.narration} A/C {decoy}"
    return Mutant(
        _swap_line(sources, line, narration=text),
        (line.bank_line_id,),
        f"decoy 12-digit run {decoy} appended",
    )


def _m_utr_digit_transposed(sources, truth, rng):
    """Two adjacent digits swapped -- a keying error that keeps the shape."""
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, s = rng.choice(pairs)
    pos = line.narration.find(s.utr)
    if pos < 0:
        return None
    i = rng.randrange(len(s.utr) - 1)
    if s.utr[i] == s.utr[i + 1]:
        return None
    scrambled = s.utr[:i] + s.utr[i + 1] + s.utr[i] + s.utr[i + 2 :]
    text = line.narration.replace(s.utr, scrambled, 1)
    return Mutant(
        _swap_line(sources, line, narration=text),
        (line.bank_line_id,),
        f"UTR {s.utr} keyed as {scrambled}",
    )


def _m_narration_dialect(sources, truth, rng):
    """A statement format nobody wrote a rule for.

    Two real dialects that break a bare twelve-digit regex: one that groups the
    reference the way a cheque number is grouped, and one that runs it into an
    alphanumeric bank reference with no word boundary in front of it.
    """
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, s = rng.choice(pairs)
    dialect = rng.choice(
        [
            f"BY CLG/{s.utr[:6]} {s.utr[6:]}/RZPY SETTLE",
            f"TRF FROM RAZORPAYSOFT REF{s.utr}CR",
            f"CR-SETTLEMENT-{s.utr[:4]}-{s.utr[4:8]}-{s.utr[8:]}",
        ]
    )
    return Mutant(
        _swap_line(sources, line, narration=dialect),
        (line.bank_line_id,),
        f"reformatted as {dialect!r}",
    )


def _m_utr_of_another_settlement(sources, truth, rng):
    """The narration names a real UTR -- belonging to a different batch.

    Nothing about this line is malformed. The evidence is well-formed and
    false, which is the case a shape check cannot catch and the arithmetic
    has to.
    """
    pairs = _clean_pairs_cached(sources, truth)
    if len(pairs) < 2:
        return None
    (line, mine), (_, theirs) = rng.sample(pairs, 2)
    if mine.utr == theirs.utr:
        return None
    text = line.narration.replace(mine.utr, theirs.utr, 1)
    if text == line.narration:
        return None
    return Mutant(
        _swap_line(sources, line, narration=text),
        (line.bank_line_id,),
        f"narration now cites {theirs.settlement_id}'s UTR {theirs.utr}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Amount family -- attack the arithmetic gate
# ─────────────────────────────────────────────────────────────────────────────


def _m_credit_off_by_one_paisa(sources, truth, rng):
    """One paisa out. Inside tolerance is a pass; the point is it is not silent."""
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, _ = rng.choice(pairs)
    delta = rng.choice([-1, 1])
    return Mutant(
        _swap_line(sources, line, amount_paise=line.amount_paise + delta),
        (line.bank_line_id,),
        f"credit moved by {delta} paisa",
    )


def _m_credit_rupees_not_paise(sources, truth, rng):
    """The classic integration bug: a credit posted in rupees into a paise field."""
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, _ = rng.choice(pairs)
    if line.amount_paise < 100:
        return None
    return Mutant(
        _swap_line(sources, line, amount_paise=line.amount_paise // 100),
        (line.bank_line_id,),
        f"credit {line.amount_paise} posted as {line.amount_paise // 100}",
    )


def _m_credits_swapped(sources, truth, rng):
    """Two credits exchange amounts. Both proofs must fail; neither may re-pair."""
    pairs = _clean_pairs_cached(sources, truth)
    if len(pairs) < 2:
        return None
    (a, _), (b, _) = rng.sample(pairs, 2)
    if a.amount_paise == b.amount_paise:
        return None
    mutated = _swap_line(sources, a, amount_paise=b.amount_paise)
    mutated = _swap_line(mutated, b, amount_paise=a.amount_paise)
    return Mutant(
        mutated,
        (a.bank_line_id, b.bank_line_id),
        f"{a.bank_line_id} and {b.bank_line_id} exchanged amounts",
    )


def _m_credit_negated(sources, truth, rng):
    """A credit posted as a debit. The magnitude still 'matches' the batch."""
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, _ = rng.choice(pairs)
    if line.amount_paise == 0:
        return None
    return Mutant(
        _swap_line(sources, line, amount_paise=-line.amount_paise),
        (line.bank_line_id,),
        "credit sign flipped",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Timing family
# ─────────────────────────────────────────────────────────────────────────────


def _m_value_date_pushed(sources, truth, rng):
    """The credit is dated weeks from the payout it supposedly settles."""
    from datetime import timedelta

    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, _ = rng.choice(pairs)
    # Probes both sides of the settlement window rather than only far from it:
    # a bound is only meaningful if something tests where it actually sits.
    days = rng.choice([-30, -7, -2, 2, 7, 30])
    return Mutant(
        _swap_line(sources, line, value_date=line.value_date + timedelta(days=days)),
        (line.bank_line_id,),
        f"value date moved {days:+d} days",
    )


def _m_settled_at_pushed(sources, truth, rng):
    """The same gap, opened from the gateway's side instead of the bank's."""
    from datetime import timedelta

    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, s = rng.choice(pairs)
    days = rng.choice([-60, -45, 45, 60])
    return Mutant(
        _swap_settlement(sources, s, settled_at=s.settled_at + timedelta(days=days)),
        (line.bank_line_id,),
        f"{s.settlement_id} settled_at moved {days:+d} days",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Structural family -- attack identity and membership
# ─────────────────────────────────────────────────────────────────────────────


def _m_settlement_utr_collision(sources, truth, rng):
    """Two payouts claim the same UTR. Neither line has an unambiguous answer."""
    pairs = _clean_pairs_cached(sources, truth)
    if len(pairs) < 2:
        return None
    (line, mine), (_, theirs) = rng.sample(pairs, 2)
    if mine.utr == theirs.utr:
        return None
    return Mutant(
        _swap_settlement(sources, theirs, utr=mine.utr),
        (line.bank_line_id,),
        f"{theirs.settlement_id} restamped with {mine.settlement_id}'s UTR",
    )


def _m_duplicate_credit_line(sources, truth, rng):
    """The same credit restated on a second statement line.

    The added line has no entry in ground truth, so booking either one against
    the batch is a wrong match by construction -- which is the correct reading:
    the matcher cannot know which of two identical credits is the real one.
    """
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    line, _ = rng.choice(pairs)
    clone = replace(line, bank_line_id=f"{line.bank_line_id}_dup")
    return Mutant(
        replace(sources, bank_lines=sources.bank_lines + (clone,)),
        (line.bank_line_id, clone.bank_line_id),
        f"{line.bank_line_id} restated as {clone.bank_line_id}",
    )


def _m_orphan_credit(sources, truth, rng):
    """A credit that belongs to no batch in this book, wearing a valid UTR."""
    known = {s.utr for s in sources.settlements}
    for _ in range(20):
        utr = "".join(str(rng.randint(0, 9)) for _ in range(12))
        if utr not in known:
            break
    else:
        return None
    pairs = _clean_pairs_cached(sources, truth)
    if not pairs:
        return None
    template, _ = rng.choice(pairs)
    orphan = replace(
        template,
        bank_line_id=f"{template.bank_line_id}_orphan",
        narration=f"NEFT CR-HDFC0000123-RAZORPAY SOFTWARE PVT LTD-{utr}",
    )
    return Mutant(
        replace(sources, bank_lines=sources.bank_lines + (orphan,)),
        (orphan.bank_line_id,),
        f"orphan credit carrying unknown UTR {utr}",
    )


def _m_member_moved(sources, truth, rng):
    """A payment reassigned to a neighbouring batch behind the matcher's back.

    Both batches now disagree with their credit. Neither proof may close, and
    -- the part that matters -- neither credit may be re-paired to the batch
    whose total it now happens to resemble.
    """
    pairs = _clean_pairs_cached(sources, truth)
    if len(pairs) < 2:
        return None
    (line_a, sa), (line_b, sb) = rng.sample(pairs, 2)
    members = sources.payments_by_settlement(sa.settlement_id)
    if not members:
        return None
    victim = rng.choice(members)
    moved = replace(victim, settlement_id=sb.settlement_id)
    return Mutant(
        replace(
            sources,
            payments=tuple(
                moved if p.payment_id == victim.payment_id else p
                for p in sources.payments
            ),
        ),
        (line_a.bank_line_id, line_b.bank_line_id),
        f"{victim.payment_id} moved {sa.settlement_id} -> {sb.settlement_id}",
    )


def _m_perfect_forgery(sources, truth, rng):
    """The smart lie: a wrong pairing whose arithmetic closes exactly.

    This is the hardest attack in the file and the only one built to defeat the
    gate rather than the join. Credit L1 (truly batch S1) is rewritten to cite
    S2's UTR *and* carry S2's exact net, and S2's own credit is stripped of its
    UTR so the duplicate-UTR check cannot be what saves us.

    L1 now joins S2 uniquely and its arithmetic replays perfectly. Every signal
    the matcher consults agrees, and every one of them is wrong. If anything
    contains this, it is a check that does not appear in the residual at all.
    """
    pairs = _clean_pairs_cached(sources, truth)
    if len(pairs) < 2:
        return None
    (l1, s1), (l2, s2) = rng.sample(pairs, 2)
    if s1.utr == s2.utr or l1.amount_paise == s2.net_paise:
        return None
    forged = l1.narration.replace(s1.utr, s2.utr, 1)
    if forged == l1.narration:
        return None
    mutated = _swap_line(
        sources, l1, narration=forged, amount_paise=s2.net_paise
    )
    # Strip the real credit's UTR so the forgery is the only line citing S2.
    mutated = _swap_line(
        mutated, l2, narration=l2.narration.replace(s2.utr, "XXXXXXXXXXXX", 1)
    )
    return Mutant(
        mutated,
        (l1.bank_line_id, l2.bank_line_id),
        f"{l1.bank_line_id} forged onto {s2.settlement_id} "
        f"at its exact net {s2.net_paise}, {l2.bank_line_id} blinded",
    )


def _m_perfect_forgery_dated(sources, truth, rng):
    """The forgery again, by someone who also thought about the date.

    `perfect_forgery` is contained by the settlement window, and it is worth
    being precise about *why*: not because the matcher detected a lie, but
    because that particular forger left one field alone. This variant does not.
    It moves the credit's value date onto the target payout's date as well, so
    the key, the arithmetic and the calendar all agree.

    Nothing in this repository contains it, and the audit says so. Leg 2's
    entire evidence base is the narration, the amount and the date; an
    adversary holding all three can manufacture a match, and no further
    arithmetic recovers the difference. The defence is not a rule the matcher
    is missing -- it is that a bank statement is not attacker-controlled. That
    boundary is worth stating plainly rather than papering over with a check
    tuned until this one case passes.
    """
    from datetime import timedelta  # noqa: F401  (kept local, as siblings do)

    pairs = _clean_pairs_cached(sources, truth)
    if len(pairs) < 2:
        return None
    (l1, s1), (l2, s2) = rng.sample(pairs, 2)
    if s1.utr == s2.utr or l1.amount_paise == s2.net_paise:
        return None
    forged = l1.narration.replace(s1.utr, s2.utr, 1)
    if forged == l1.narration:
        return None
    mutated = _swap_line(
        sources,
        l1,
        narration=forged,
        amount_paise=s2.net_paise,
        value_date=s2.settled_at.date(),
    )
    mutated = _swap_line(
        mutated, l2, narration=l2.narration.replace(s2.utr, "XXXXXXXXXXXX", 1)
    )
    return Mutant(
        mutated,
        (l1.bank_line_id, l2.bank_line_id),
        f"{l1.bank_line_id} forged onto {s2.settlement_id} on every axis "
        f"-- UTR, net {s2.net_paise}, and value date",
    )


MUTATIONS: tuple[Mutation, ...] = (
    Mutation("utr_clipped", "narration", True,
             "narration cut mid-UTR", _m_utr_clipped),
    Mutation("utr_decoy_digits", "narration", True,
             "a second 12-digit run appended", _m_utr_decoy_digits),
    Mutation("utr_digit_transposed", "narration", True,
             "two adjacent UTR digits swapped", _m_utr_digit_transposed),
    Mutation("narration_dialect", "narration", True,
             "a statement format no rule was written for", _m_narration_dialect),
    Mutation("utr_of_another_settlement", "narration", False,
             "well-formed narration citing the wrong batch",
             _m_utr_of_another_settlement),
    Mutation("credit_off_by_one_paisa", "amount", True,
             "credit moved by one paisa", _m_credit_off_by_one_paisa),
    Mutation("credit_rupees_not_paise", "amount", True,
             "credit posted in rupees into a paise field",
             _m_credit_rupees_not_paise),
    Mutation("credits_swapped", "amount", True,
             "two credits exchange amounts", _m_credits_swapped),
    Mutation("credit_negated", "amount", True,
             "credit posted as a debit", _m_credit_negated),
    Mutation("value_date_pushed", "timing", True,
             "credit dated weeks from its payout", _m_value_date_pushed),
    Mutation("settled_at_pushed", "timing", True,
             "payout dated weeks from its credit", _m_settled_at_pushed),
    Mutation("settlement_utr_collision", "structural", False,
             "two payouts claiming one UTR", _m_settlement_utr_collision),
    Mutation("duplicate_credit_line", "structural", False,
             "the same credit on two statement lines", _m_duplicate_credit_line),
    Mutation("orphan_credit", "structural", False,
             "a credit belonging to no batch here", _m_orphan_credit),
    Mutation("member_moved", "structural", True,
             "a payment reassigned between batches", _m_member_moved),
    Mutation("perfect_forgery", "structural", False,
             "a wrong pairing whose arithmetic closes exactly",
             _m_perfect_forgery),
    Mutation("perfect_forgery_dated", "structural", False,
             "the same forgery, with the calendar faked too",
             _m_perfect_forgery_dated),
)

#: Attacks this design does not claim to survive. They are run, scored and
#: printed like everything else -- listing one here changes the report, never
#: the verdict, and never the exit code's view of the others. The point is that
#: a known limit stated up front is worth more than a suite quietly pruned until
#: it reports 100%.
#:
#: Both forgeries are here, and the honest summary is that the settlement window
#: narrowed one of them rather than closing it. Measured at n=2,000, 24 cases:
#:
#:     attack                  no date check      1-day window
#:     perfect_forgery         16/24  (66.7%)     1/24  (4.2%)
#:     perfect_forgery_dated   16/24  (66.7%)    16/24  (66.7%)
#:
#: A sixteen-fold reduction is worth having and is not a fix. What survives is
#: the case where the two payouts happen to settle inside the same window, which
#: on a book with 164 settlements over 20 days is not rare. Reporting 4.2% as
#: "contained" would be the exact dishonesty this file exists to catch.
KNOWN_UNCONTAINED = frozenset({"perfect_forgery", "perfect_forgery_dated"})


# ─────────────────────────────────────────────────────────────────────────────
# Running one case
# ─────────────────────────────────────────────────────────────────────────────

#: `_clean_pairs` is the same for every mutation of a given book, and rebuilding
#: it per mutation dominated the runtime. Keyed by id() of the bundle, which is
#: safe because the base bundle is held alive for the length of the audit.
_PAIR_CACHE: dict[int, list[tuple[BankLine, Settlement]]] = {}


def _clean_pairs_cached(sources, truth):
    return _PAIR_CACHE.get(id(sources), [])


@dataclass(frozen=True)
class Case:
    mutation: str
    family: str
    seed: int
    verdict: str
    note: str
    detail: str = ""


def _case_json(c: "Case") -> dict:
    return {
        "mutation": c.mutation,
        "seed": c.seed,
        "verdict": c.verdict,
        "note": c.note,
        "detail": c.detail,
        "replay": f"python -m recoagent.audit.mutate --replay {c.mutation} {c.seed}",
    }


@dataclass
class Scorecard:
    n_orders: int
    profile: str
    trials: int
    cases: list[Case] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.cases)

    def _count(self, verdict: str, cases=None) -> int:
        return sum(1 for c in (cases or self.cases) if c.verdict == verdict)

    def containment_rate(self, cases=None) -> float:
        pool = cases if cases is not None else self.cases
        if not pool:
            return 0.0
        return sum(1 for c in pool if c.verdict in CONTAINED) / len(pool)

    @property
    def wrong_matches(self) -> list[Case]:
        return [c for c in self.cases if c.verdict == WRONG_MATCH]

    @property
    def crashes(self) -> list[Case]:
        return [c for c in self.cases if c.verdict == CRASH]

    @property
    def unexpected(self) -> list[Case]:
        """Failures on attacks this design does claim to survive.

        These are the ones that fail the command. A documented limit is
        reported every run and gates nothing; an undocumented one gates
        everything.
        """
        return [
            c
            for c in self.cases
            if c.verdict not in CONTAINED and c.mutation not in KNOWN_UNCONTAINED
        ]

    @property
    def known_limits(self) -> list[Case]:
        return [
            c
            for c in self.cases
            if c.verdict not in CONTAINED and c.mutation in KNOWN_UNCONTAINED
        ]

    def by(self, key: str) -> dict[str, list[Case]]:
        out: dict[str, list[Case]] = {}
        for c in self.cases:
            out.setdefault(getattr(c, key), []).append(c)
        return out

    def to_json(self) -> dict:
        def block(cases: list[Case]) -> dict:
            return {
                "total": len(cases),
                "held": self._count(HELD, cases),
                "refused": self._count(REFUSED, cases),
                "wrong_match": self._count(WRONG_MATCH, cases),
                "crash": self._count(CRASH, cases),
                "containment_rate": round(self.containment_rate(cases), 6),
            }

        return {
            "n_orders": self.n_orders,
            "profile": self.profile,
            "trials_per_mutation": self.trials,
            "overall": block(self.cases),
            "families": {k: block(v) for k, v in sorted(self.by("family").items())},
            "mutations": {
                k: {
                    **block(v),
                    # Whether a correct pairing still existed after the attack.
                    # Where it did not, `refused` is the right answer and
                    # `held` is unreachable -- so the two columns are not
                    # comparable across this flag, and the flag has to travel
                    # with them.
                    "correct_answer_still_available": next(
                        m.recoverable for m in MUTATIONS if m.name == k
                    ),
                }
                for k, v in sorted(self.by("mutation").items())
            },
            "unexpected_failures": [_case_json(c) for c in self.unexpected],
            "known_limits": [_case_json(c) for c in self.known_limits],
            "declared_uncontained": sorted(KNOWN_UNCONTAINED),
        }


def _judge(mutant: Mutant, truth: GroundTruth, result: ReconResult) -> tuple[str, str]:
    """Read a run for wrong matches first, then for what happened to the target."""
    for m in result.matches:
        left = m.left_ids[0]
        expected = (truth.leg1 if m.leg == 1 else truth.leg2).get(left)
        if expected != m.right_ids[0]:
            return WRONG_MATCH, (
                f"leg {m.leg}: {left} booked to {m.right_ids[0]}, "
                f"truth says {expected}"
            )

    matched_targets = {
        m.left_ids[0] for m in result.matches_for_leg(2)
        if m.left_ids[0] in mutant.targets
    }
    if matched_targets:
        return HELD, f"still matched: {', '.join(sorted(matched_targets))}"
    return REFUSED, "routed to the exception list"


def _run_case(mutation: Mutation, base, truth, seed: int) -> Case | None:
    rng = random.Random(seed)
    try:
        mutant = mutation.apply(base, truth, rng)
    except Exception:
        return Case(mutation.name, mutation.family, seed, CRASH,
                    "mutation could not be built", traceback.format_exc(limit=3))
    if mutant is None:
        return None  # nothing eligible at this seed; not a result either way

    try:
        result = run_b2(mutant.sources)
    except Exception:
        return Case(mutation.name, mutation.family, seed, CRASH,
                    mutant.note, traceback.format_exc(limit=3))

    verdict, detail = _judge(mutant, truth, result)
    return Case(mutation.name, mutation.family, seed, verdict, mutant.note, detail)


def audit(
    *,
    n_orders: int = 800,
    profile: str = "dev",
    trials: int = 15,
    base_seed: int = 100_000,
    only: str | None = None,
    progress=None,
) -> Scorecard:
    mix = {"dev": DefectMix.dev, "holdout": DefectMix.holdout}[profile]
    batch = generate(GeneratorConfig(n_orders=n_orders, seed=7, mix=mix()))
    baseline = run_b2(batch.sources)

    _PAIR_CACHE[id(batch.sources)] = _clean_pairs(batch.sources, batch.truth, baseline)

    card = Scorecard(n_orders=n_orders, profile=profile, trials=trials)
    chosen = [m for m in MUTATIONS if only in (None, m.name)]
    if not chosen:
        raise SystemExit(f"no mutation named {only!r}")

    for mutation in chosen:
        for i in range(trials):
            case = _run_case(mutation, batch.sources, batch.truth, base_seed + i)
            if case is not None:
                card.cases.append(case)
        if progress:
            progress(mutation, card)

    _PAIR_CACHE.clear()
    return card


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def render(card: Scorecard) -> str:
    out = [
        "=" * 72,
        f"ADVERSARIAL MUTATION AUDIT   profile={card.profile}  "
        f"n={card.n_orders:,}  trials={card.trials}",
        "=" * 72,
        "",
        f"  cases run                    {card.total}",
        f"  UNEXPECTED WRONG MATCHES     {len(card.unexpected)}"
        f"   <- lead metric, must be 0",
        f"  crashes                      {len(card.crashes)}",
        f"  containment                  {card.containment_rate():.2%}",
        f"  on declared limits           {len(card.known_limits)}"
        f"   (attacks this design does not claim to survive)",
        "",
        f"  held (attack did not land)   {card._count(HELD)}",
        f"  refused (filed an exception) {card._count(REFUSED)}",
        "",
        "-" * 72,
        "  BY FAMILY",
        "-" * 72,
        f"  {'family':<14}{'cases':>7}{'held':>7}{'refused':>9}"
        f"{'wrong':>7}{'crash':>7}{'contained':>11}",
    ]
    for name, cases in sorted(card.by("family").items()):
        out.append(
            f"  {name:<14}{len(cases):>7}{card._count(HELD, cases):>7}"
            f"{card._count(REFUSED, cases):>9}{card._count(WRONG_MATCH, cases):>7}"
            f"{card._count(CRASH, cases):>7}{card.containment_rate(cases):>10.0%} "
        )

    out += ["", "-" * 72, "  BY MUTATION", "-" * 72]
    for m in MUTATIONS:
        cases = card.by("mutation").get(m.name, [])
        if not cases:
            out.append(f"  {m.name:<28}{'-- nothing eligible':>40}")
            continue
        flag = ""
        if card._count(WRONG_MATCH, cases):
            flag = ("   <- declared limit, see below" if m.name in KNOWN_UNCONTAINED
                    else "   <- BOOKED THE WRONG BATCH")
        elif card._count(CRASH, cases):
            flag = "   <- CRASHED"
        out.append(
            f"  {m.name:<28}{len(cases):>5} cases  "
            f"held {card._count(HELD, cases):>3}  "
            f"refused {card._count(REFUSED, cases):>3}  "
            f"wrong {card._count(WRONG_MATCH, cases):>3}{flag}"
        )

    if card.unexpected:
        out += ["", "-" * 72, "  FAILURES, EVERY ONE REPRODUCIBLE", "-" * 72]
        for c in card.unexpected:
            out += [
                f"  [{c.verdict}] {c.mutation}  seed={c.seed}",
                f"      {c.note}",
                f"      {c.detail.strip().splitlines()[-1] if c.detail else ''}",
                f"      python -m recoagent.audit.mutate --replay {c.mutation} {c.seed}",
            ]
    else:
        out += [
            "",
            "  No attack this design claims to survive booked money against the",
            "  wrong batch. Refusals are the expected outcome, not a shortfall:",
            "  the mutations marked non-recoverable destroy the evidence a correct",
            "  answer would need, and an exception is the only defensible reading.",
        ]

    if card.known_limits:
        seen = sorted({c.mutation for c in card.known_limits})
        out += ["", "-" * 72, "  DECLARED LIMITS -- attacks that do land", "-" * 72]
        for name in seen:
            hit = [c for c in card.known_limits if c.mutation == name]
            spec = next(m for m in MUTATIONS if m.name == name)
            out += [
                f"  {name}: {len(hit)} of "
                f"{len(card.by('mutation').get(name, []))} cases booked the wrong batch",
                f"      {spec.description}",
                f"      {(spec.apply.__doc__ or '').strip().splitlines()[0]}",
                f"      python -m recoagent.audit.mutate --replay {name} {hit[0].seed}",
            ]
        out += [
            "",
            "  These are printed every run and gate nothing. Leg 2's evidence is",
            "  the narration, the amount and the value date; an adversary holding",
            "  all three can manufacture a match, and no further arithmetic",
            "  recovers the difference. The defence is that a bank statement is",
            "  not attacker-controlled -- which is a property of the world, not a",
            "  rule in this repository, and is reported as such.",
        ]

    out += [
        "",
        "  Containment is held + refused. It is not an accuracy figure and it",
        "  is not comparable to the match rate -- these inputs were built to",
        "  break the matcher, not to resemble a merchant's book.",
        "=" * 72,
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="recoagent.audit.mutate")
    ap.add_argument("--n", type=int, default=800, help="orders in the book under attack")
    ap.add_argument("--profile", choices=["dev", "holdout"], default="dev")
    ap.add_argument("--trials", type=int, default=15, help="seeds per mutation")
    ap.add_argument("--base-seed", type=int, default=100_000)
    ap.add_argument("--out", help="write the JSON scorecard here")
    ap.add_argument("--json", action="store_true", help="print JSON instead of the report")
    ap.add_argument(
        "--replay",
        nargs=2,
        metavar=("MUTATION", "SEED"),
        help="re-run one case: a mutation name and the seed it failed on",
    )
    args = ap.parse_args(argv)

    if args.replay:
        name, seed = args.replay[0], int(args.replay[1])
        card = audit(
            n_orders=args.n, profile=args.profile, trials=1,
            base_seed=seed, only=name,
        )
        if not card.cases:
            print(f"{name} had nothing eligible to mutate at seed {seed}")
            return 1
        c = card.cases[0]
        print(f"{c.mutation}  seed={c.seed}")
        print(f"  mutation : {c.note}")
        print(f"  verdict  : {c.verdict.upper()}")
        print(f"  detail   : {c.detail}")
        return 0 if (c.verdict in CONTAINED or c.mutation in KNOWN_UNCONTAINED) else 1

    def progress(mutation, card):
        cases = card.by("mutation").get(mutation.name, [])
        print(
            f"    {mutation.name:<28} {len(cases):>3} cases  "
            f"contained {card.containment_rate(cases):>4.0%}",
            flush=True,
        )

    # With --json, stdout is the artifact: nothing but the document, so it can
    # be redirected straight into a file or a diff without being sliced apart.
    if not args.json:
        print(f">>> attacking a {args.n:,}-order book, {args.trials} seeds per mutation")
    card = audit(
        n_orders=args.n, profile=args.profile,
        trials=args.trials, base_seed=args.base_seed,
        progress=None if args.json else progress,
    )

    if args.json:
        print(json.dumps(card.to_json(), indent=2))
    else:
        print()
        print(render(card))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(card.to_json(), indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {path}", file=sys.stderr if args.json else sys.stdout)

    # An undeclared wrong match or a crash fails the command, so CI can gate on
    # it. A declared limit is printed above and deliberately does not.
    return 1 if (card.unexpected or card.crashes) else 0


if __name__ == "__main__":
    sys.exit(main())
