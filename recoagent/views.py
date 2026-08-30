"""Everything the dashboard renders, shaped from the sources and the result.

The console is more than one screen now -- a queue to work, a match log to
audit, the four source ledgers to look things up in -- and each screen needs the
same run shaped a different way. That shaping lives here rather than in `ui.py`
so the server stays a server, and so the one rule that matters can be enforced
by a test rather than by care:

**Nothing in this module can see the answer key.** Every function takes a
`SourceBundle` and a `ReconResult` and nothing else, which is the same
restriction every matcher in `recoagent.legs` runs under. `tests/test_
independence.py` covers this file for the same reason it covers the agent tier:
an operator screen that quietly displayed the injected label would be a demo,
not a product, and the difference is invisible from a screenshot.

Ground truth appears on exactly one screen -- Assurance -- and it is built in
`ui.py` from the scorecard, fenced off and labelled, because "what do I work on"
and "should I believe this" are different questions.
"""

from __future__ import annotations

import pathlib
from datetime import date, datetime
from typing import Any, Iterable

from .agent import evidence
from .money import FeeSchedule, Paise, format_inr
from .schemas import ReconException, ReconResult, SourceBundle

#: Which tier each rule id belongs to, for the ladder shown on every row.
TIER_OF_RULE = {
    "leg1.t0.exact_order_id": "T0",
    "leg1.t1.documented_partial_capture": "T1",
    "leg2.t0.exact_utr": "T0",
    "leg2.t1.rate_notice": "T1",
    "leg2.t1.amount_window": "T1",
    "leg2.t1.ssmp_residual": "T1",
    "leg2.t1.spill_pair": "T1",
    "leg2.t2.llm_hypothesis": "T2",
}

RULE_LABEL = {
    "leg1.t0.exact_order_id": "exact order id",
    "leg1.t1.documented_partial_capture": "declared partial capture, fees re-derived",
    "leg2.t0.exact_utr": "exact UTR",
    "leg2.t1.rate_notice": "gateway repricing notice, fees re-derived",
    "leg2.t1.amount_window": "amount + date window",
    "leg2.t1.ssmp_residual": "subset-sum over unlinked rows",
    "leg2.t1.spill_pair": "cross-batch cutoff spill",
    "leg2.t2.llm_hypothesis": "model hypothesis, arithmetic verified",
}

PAGE_SIZE = 60

#: Where the published result artefacts live. Read-only, and only these.
RESULTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"


def severity(residual_paise: Paise | None) -> tuple[str, str]:
    """Form as well as number: how loud should this row be?"""
    if residual_paise is None:
        return "structural", "no amount in dispute"
    magnitude = abs(residual_paise)
    if magnitude >= 10_000_00:
        return "critical", "over Rs 10,000 unexplained"
    if magnitude >= 100_00:
        return "warn", "over Rs 100 unexplained"
    return "minor", "under Rs 100"


def money(paise: Paise | None) -> str:
    return format_inr(paise) if paise is not None else "—"


def _when(value: datetime | date | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="minutes")
    return value.isoformat()


# ── the exception queue ──────────────────────────────────────────────────────


def queue_row(exc: ReconException) -> dict[str, Any]:
    """One row of the operator's work queue.

    Every field here is either the system's own output or arithmetic over the
    sources. `suspected` is the matcher's guess at what went wrong, made without
    the labels; the scorer reports separately how often that guess is right.
    """
    level, hint = severity(exc.residual_paise)
    return {
        # The exception's own id, so a row can be deep-linked and its case file
        # fetched. Neither this nor `related` is ground truth: both are the
        # system's own output, and the fence is the point of this module.
        "xid": exc.exception_id,
        "id": exc.entity_id,
        "leg": exc.leg,
        "kind": exc.entity_kind,
        "related": exc.related_id,
        "residual_paise": exc.residual_paise,
        "amount": money(exc.residual_paise),
        "direction": "" if not exc.residual_paise else ("short" if exc.residual_paise < 0 else "over"),
        "severity": level,
        "severity_hint": hint,
        "suspected": exc.suspected_class.value if exc.suspected_class else "not classified",
        "reason": exc.reason,
        "stopped_at": exc.escalated_from_tier or "T0",
    }


def queue(result: ReconResult) -> list[dict[str, Any]]:
    """The whole queue, biggest money at stake first.

    Sorted, not paged: an analyst deciding what to work on wants the shape of
    the whole day, and the filtering that follows is a question about this list
    rather than a new request to the server.
    """
    ordered = sorted(result.exceptions, key=lambda e: (-abs(e.residual_paise or 0), e.entity_id))
    return [queue_row(e) for e in ordered]


# ── the case file behind one queue row ───────────────────────────────────────


def _payment_rows(payments: Iterable) -> list[dict[str, Any]]:
    return [{
        "payment_id": p.payment_id,
        "order_id": p.order_id,
        "method": p.method,
        "status": p.status,
        "currency": p.currency,
        "fx_rate": p.fx_rate,
        "gross": money(p.gross_paise),
        "fee": money(p.fee_paise),
        "tax": money(p.tax_paise),
        "net": money(p.net_paise),
        "net_paise": p.net_paise,
        "settlement_id": p.settlement_id,
        "captured_at": _when(p.captured_at),
    } for p in payments]


def _adjustment_rows(adjustments: Iterable) -> list[dict[str, Any]]:
    return [{
        "adjustment_id": a.adjustment_id,
        "kind": a.kind,
        "payment_id": a.payment_id,
        "settlement_id": a.settlement_id,
        "amount": money(a.amount_paise),
        "amount_paise": a.amount_paise,
        "booked_at": _when(a.booked_at),
    } for a in adjustments]


def _batch_arithmetic(sources: SourceBundle, settlement) -> dict[str, Any]:
    """Re-derive the batch total from the rows, never from the header.

    This is the thesis of the whole system rendered as four numbers: what the
    payments net to, what the adjustments move it by, what that comes to, and
    what the gateway said. The last one is corroboration and is never the proof.
    """
    members = sources.payments_by_settlement(settlement.settlement_id)
    linked = sources.adjustments_by_settlement(settlement.settlement_id)
    payments_net = sum(p.net_paise for p in members)
    adjustments_net = sum(a.amount_paise for a in linked)
    derived = payments_net + adjustments_net
    return {
        "payments_net": money(payments_net),
        "payments_count": len(members),
        "adjustments_net": money(adjustments_net),
        "adjustments_count": len(linked),
        "derived_net": money(derived),
        "derived_net_paise": derived,
        "reported_net": money(settlement.net_paise),
        "header_gap": money(derived - settlement.net_paise),
        "header_agrees": derived == settlement.net_paise,
    }


def exception_case(
    sources: SourceBundle, result: ReconResult, exception_id: str
) -> dict[str, Any] | None:
    """Everything an analyst needs to work one item, and nothing they would not have.

    The leg-2 batch case is the same `EvidencePacket` the model tier is handed,
    for a reason worth stating: if the packet is enough for a proposer to reason
    from, it is enough for a human to check the proposer with, and building a
    second, prettier version of it for the screen would let the two drift.

    Accepts either identifier. A queue row carries both -- `id` is the entity
    (`bank_0030`) and `xid` is the exception (`x2_bank_0030`) -- and the console
    happens to send the second one under a parameter called `id`. Anyone reading
    the queue payload and asking for the row it calls `id` got a 404 that looked
    like the item had vanished. Resolving the entity too costs one fallback scan
    and removes a trap; where an entity has more than one exception the first is
    returned, since that is the one the queue shows.
    """
    exc = next((e for e in result.exceptions if e.exception_id == exception_id), None)
    if exc is None:
        exc = next((e for e in result.exceptions if e.entity_id == exception_id), None)
    if exc is None:
        return None

    case: dict[str, Any] = {"shape": "bare"}
    lines = {b.bank_line_id: b for b in sources.bank_lines}
    settlements = {s.settlement_id: s for s in sources.settlements}

    if exc.entity_kind == "bank_line":
        line = lines.get(exc.entity_id)
        settlement = settlements.get(exc.related_id or "")
        if line is not None and settlement is not None:
            packet = evidence.build(
                sources, line, settlement, exc.residual_paise or 0, FeeSchedule.default()
            ).to_dict()
            case = {
                "shape": "leg2_batch",
                "arithmetic": _batch_arithmetic(sources, settlement),
                # Money is formatted once, here. A client that formats rupees
                # itself is a second opinion about what a number means.
                "bank_credit": dict(packet["bank_credit"], amount=money(line.amount_paise)),
                "settlement": packet["settlement"],
                "derived_signals": packet["derived_signals"],
                "payments": _payment_rows(sources.payments_by_settlement(settlement.settlement_id)),
                "linked_adjustments": _adjustment_rows(
                    sources.adjustments_by_settlement(settlement.settlement_id)
                ),
                "nearby_unlinked": packet["nearby_unlinked_rows"],
                "fee_schedule": packet["fee_schedule"],
                "already_ruled_out": packet["already_ruled_out"],
            }
        elif line is not None:
            # No settlement joined it. What an analyst does next is look for one
            # by amount and date, so the screen does that lookup for them --
            # and calls them candidates, because none of them is a match.
            near = sorted(
                sources.settlements,
                key=lambda s: (
                    abs((s.settled_at.date() - line.value_date).days),
                    abs(s.net_paise - line.amount_paise),
                ),
            )[:8]
            case = {
                "shape": "leg2_orphan",
                "bank_credit": {
                    "bank_line_id": line.bank_line_id,
                    "value_date": line.value_date.isoformat(),
                    "amount": money(line.amount_paise),
                    "narration": line.narration,
                    "bank_ref": line.bank_ref,
                },
                "candidates": [{
                    "settlement_id": s.settlement_id,
                    "utr": s.utr,
                    "settled_at": _when(s.settled_at),
                    "reported_net": money(s.net_paise),
                    "gap": money(line.amount_paise - s.net_paise),
                    "status": s.status,
                    "days_apart": (s.settled_at.date() - line.value_date).days,
                } for s in near],
            }

    elif exc.entity_kind == "settlement":
        settlement = settlements.get(exc.entity_id)
        if settlement is not None:
            case = {
                "shape": "settlement",
                "settlement": {
                    "settlement_id": settlement.settlement_id,
                    "utr": settlement.utr,
                    "settled_at": _when(settlement.settled_at),
                    "reported_net": money(settlement.net_paise),
                    "status": settlement.status,
                },
                "arithmetic": _batch_arithmetic(sources, settlement),
                "payments": _payment_rows(sources.payments_by_settlement(settlement.settlement_id)),
                "linked_adjustments": _adjustment_rows(
                    sources.adjustments_by_settlement(settlement.settlement_id)
                ),
                "bank_lines_carrying_utr": [{
                    "bank_line_id": b.bank_line_id,
                    "value_date": b.value_date.isoformat(),
                    "amount": money(b.amount_paise),
                    "narration": b.narration,
                } for b in sources.bank_lines if settlement.utr in b.narration],
            }

    elif exc.entity_kind == "order":
        order = next((o for o in sources.orders if o.order_id == exc.entity_id), None)
        if order is not None:
            claims = [p for p in sources.payments if p.order_id == order.order_id]
            case = {
                "shape": "leg1_order",
                "order": {
                    "order_id": order.order_id,
                    "customer_id": order.customer_id,
                    "invoice_no": order.invoice_no,
                    "amount": money(order.amount_paise),
                    "amount_paise": order.amount_paise,
                    "currency": order.currency,
                    "created_at": _when(order.created_at),
                },
                "claims": [dict(
                    row,
                    gap=money(p.gross_paise - order.amount_paise),
                ) for row, p in zip(_payment_rows(claims), claims)],
                "refunds": _adjustment_rows(
                    a for a in sources.adjustments
                    if a.payment_id in {p.payment_id for p in claims}
                ),
            }

    return {"item": queue_row(exc), "exception_id": exc.exception_id, "case": case}


# ── the match log ────────────────────────────────────────────────────────────


def matches(
    result: ReconResult,
    *,
    leg: str = "",
    tier: str = "",
    query: str = "",
    page: int = 1,
    size: int = PAGE_SIZE,
) -> dict[str, Any]:
    """Every accepted match, with the arithmetic that accepted it.

    Paged on the server because a 50,000-order book produces 50,000 of these and
    an audit log nobody can open is not an audit log.
    """
    rows = list(result.matches)
    if leg in ("1", "2"):
        rows = [m for m in rows if m.leg == int(leg)]
    if tier in ("T0", "T1", "T2"):
        rows = [m for m in rows if m.tier == tier]
    needle = query.strip().lower()
    if needle:
        rows = [m for m in rows if needle in " ".join(
            (m.match_id, m.rule_id, *m.left_ids, *m.right_ids, *m.hypothesised_ids)
        ).lower()]
    rows.sort(key=lambda m: (m.leg, m.tier, m.match_id))

    total = len(rows)
    page = max(1, page)
    window = rows[(page - 1) * size: page * size]
    return {
        "total": total,
        "page": page,
        "pages": max(1, (total + size - 1) // size),
        "rows": [{
            "match_id": m.match_id,
            "leg": m.leg,
            "tier": m.tier,
            "rule": RULE_LABEL.get(m.rule_id, m.rule_id),
            "rule_id": m.rule_id,
            "left": list(m.left_ids),
            "right": list(m.right_ids),
            "confidence": round(m.confidence, 2),
            "hypothesised": list(m.hypothesised_ids),
            "variance": money(m.variance_paise) if m.variance_paise else "",
            "variance_paise": m.variance_paise,
            "input_hash": m.input_hash,
            "created_at": _when(m.created_at),
            "proof": None if m.proof is None else {
                "expression": m.proof.expression,
                "lhs": money(m.proof.lhs_paise),
                "rhs": money(m.proof.rhs_paise),
                "lhs_paise": m.proof.lhs_paise,
                "rhs_paise": m.proof.rhs_paise,
                "tolerance_paise": m.proof.tolerance_paise,
                "residual": money(m.proof.residual_paise),
                "residual_paise": m.proof.residual_paise,
                "closes": m.proof.closes,
            },
        } for m in window],
    }


# ── the source ledgers ───────────────────────────────────────────────────────

#: Column definitions per ledger. The client renders whatever it is given, so a
#: new source becomes a new entry here and nothing else.
SOURCE_COLUMNS: dict[str, list[dict[str, str]]] = {
    "orders": [
        {"key": "order_id", "label": "Order", "type": "id"},
        {"key": "invoice_no", "label": "Invoice", "type": "id"},
        {"key": "customer_id", "label": "Customer", "type": "id"},
        {"key": "amount", "label": "Amount", "type": "money"},
        {"key": "currency", "label": "Ccy", "type": "text"},
        {"key": "created_at", "label": "Created", "type": "when"},
    ],
    "payments": [
        {"key": "payment_id", "label": "Payment", "type": "id"},
        {"key": "order_id", "label": "Order", "type": "id"},
        {"key": "method", "label": "Method", "type": "tag"},
        {"key": "status", "label": "Status", "type": "tag"},
        {"key": "gross", "label": "Gross", "type": "money"},
        {"key": "fee", "label": "Fee", "type": "money"},
        {"key": "tax", "label": "GST", "type": "money"},
        {"key": "net", "label": "Net", "type": "money"},
        {"key": "settlement_id", "label": "Batch", "type": "id"},
        {"key": "captured_at", "label": "Captured", "type": "when"},
    ],
    "adjustments": [
        {"key": "adjustment_id", "label": "Adjustment", "type": "id"},
        {"key": "kind", "label": "Kind", "type": "tag"},
        {"key": "payment_id", "label": "Payment", "type": "id"},
        {"key": "settlement_id", "label": "Batch", "type": "id"},
        {"key": "amount", "label": "Amount", "type": "money"},
        {"key": "booked_at", "label": "Booked", "type": "when"},
    ],
    "settlements": [
        {"key": "settlement_id", "label": "Batch", "type": "id"},
        {"key": "utr", "label": "UTR", "type": "id"},
        {"key": "status", "label": "Status", "type": "tag"},
        {"key": "payments", "label": "Rows", "type": "num"},
        {"key": "derived_net", "label": "Derived from rows", "type": "money"},
        {"key": "net", "label": "Header says", "type": "money"},
        {"key": "settled_at", "label": "Settled", "type": "when"},
    ],
    "bank_lines": [
        {"key": "bank_line_id", "label": "Credit", "type": "id"},
        {"key": "value_date", "label": "Value date", "type": "when"},
        {"key": "amount", "label": "Amount", "type": "money"},
        {"key": "bank_ref", "label": "Bank ref", "type": "id"},
        {"key": "narration", "label": "Narration", "type": "text"},
    ],
    "rate_notices": [
        {"key": "notice_id", "label": "Notice", "type": "id"},
        {"key": "method", "label": "Method", "type": "tag"},
        {"key": "mdr_bps", "label": "MDR (bps)", "type": "num"},
        {"key": "effective_from", "label": "From", "type": "when"},
        {"key": "effective_to", "label": "Until", "type": "when"},
        {"key": "reference", "label": "Reference", "type": "text"},
    ],
    "fx_advices": [
        {"key": "advice_id", "label": "Advice", "type": "id"},
        {"key": "payment_id", "label": "Payment", "type": "id"},
        {"key": "rate_pct_of_gross", "label": "Rate (% of gross)", "type": "num"},
        {"key": "advised_at", "label": "Advised", "type": "when"},
        {"key": "reference", "label": "Reference", "type": "text"},
    ],
}

SOURCE_KINDS = tuple(SOURCE_COLUMNS)

SOURCE_BLURB = {
    "orders": "The merchant's own ledger. The claim: this is what was sold.",
    "payments": "The gateway's settlement report, one row per transaction, with the fee and GST it says it took.",
    "adjustments": "Non-payment rows netted into a batch — refunds, chargebacks, dispute fees. Rows with no batch are the solver's haystack.",
    "settlements": "The batch header the gateway reports. Corroboration, never proof: the derived column is re-computed from the payment rows.",
    "bank_lines": "What actually landed, as the bank printed it. The narration is raw — extracting a usable UTR from it is part of the problem.",
    "rate_notices": "The fourth source. A gateway repricing notice turns a fee variance from a hypothesis into something code can check.",
    "fx_advices": "The bank's conversion advice: what the conversion actually cost, against the rate the gateway report expected.",
}


def _source_rows(sources: SourceBundle, kind: str) -> list[dict[str, Any]]:
    if kind == "orders":
        return [{
            "order_id": o.order_id, "invoice_no": o.invoice_no, "customer_id": o.customer_id,
            "amount": money(o.amount_paise), "currency": o.currency,
            "created_at": _when(o.created_at),
        } for o in sources.orders]
    if kind == "payments":
        return _payment_rows(sources.payments)
    if kind == "adjustments":
        return _adjustment_rows(sources.adjustments)
    if kind == "settlements":
        rows = []
        for s in sources.settlements:
            members = sources.payments_by_settlement(s.settlement_id)
            derived = (sum(p.net_paise for p in members)
                       + sum(a.amount_paise for a in sources.adjustments_by_settlement(s.settlement_id)))
            rows.append({
                "settlement_id": s.settlement_id, "utr": s.utr, "status": s.status,
                "payments": len(members), "derived_net": money(derived),
                "net": money(s.net_paise), "settled_at": _when(s.settled_at),
                "_flag": "" if derived == s.net_paise else "header disagrees with the rows",
            })
        return rows
    if kind == "bank_lines":
        return [{
            "bank_line_id": b.bank_line_id, "value_date": b.value_date.isoformat(),
            "amount": money(b.amount_paise), "bank_ref": b.bank_ref, "narration": b.narration,
        } for b in sources.bank_lines]
    if kind == "rate_notices":
        return [{
            "notice_id": n.notice_id, "method": n.method, "mdr_bps": n.mdr_bps,
            "effective_from": _when(n.effective_from),
            "effective_to": _when(n.effective_to) if n.effective_to else "still in force",
            "reference": n.reference,
        } for n in sources.rate_notices]
    if kind == "fx_advices":
        return [{
            "advice_id": a.advice_id, "payment_id": a.payment_id,
            "rate_pct_of_gross": a.rate_pct_of_gross, "advised_at": _when(a.advised_at),
            "reference": a.reference,
        } for a in sources.fx_advices]
    raise ValueError(f"unknown source {kind!r}")


def source(
    sources: SourceBundle, kind: str, *, query: str = "", page: int = 1, size: int = PAGE_SIZE
) -> dict[str, Any]:
    """One source ledger, searched and paged. Look-up, not analysis."""
    if kind not in SOURCE_COLUMNS:
        raise ValueError(f"unknown source {kind!r}")
    rows = _source_rows(sources, kind)
    needle = query.strip().lower()
    if needle:
        rows = [r for r in rows
                if any(needle in str(v).lower() for k, v in r.items() if not k.startswith("_"))]
    total = len(rows)
    page = max(1, page)
    return {
        "kind": kind,
        "blurb": SOURCE_BLURB[kind],
        "columns": SOURCE_COLUMNS[kind],
        "total": total,
        "page": page,
        "pages": max(1, (total + size - 1) // size),
        "rows": rows[(page - 1) * size: page * size],
    }


# ── aggregates for the overview ──────────────────────────────────────────────


def shape(sources: SourceBundle, result: ReconResult) -> dict[str, Any]:
    """The distributions the overview draws, all of them counts of the result.

    No metric here needs the answer key, which is the point: an operator can
    read the whole first screen on a book nobody has labels for.
    """
    tiers: dict[str, int] = {}
    rules: dict[str, int] = {}
    for m in result.matches:
        tiers[m.tier] = tiers.get(m.tier, 0) + 1
        rules[m.rule_id] = rules.get(m.rule_id, 0) + 1

    severities: dict[str, int] = {}
    classes: dict[str, int] = {}
    at_risk: dict[str, int] = {}
    for e in result.exceptions:
        level, _ = severity(e.residual_paise)
        severities[level] = severities.get(level, 0) + 1
        at_risk[level] = at_risk.get(level, 0) + abs(e.residual_paise or 0)
        name = e.suspected_class.value if e.suspected_class else "not classified"
        classes[name] = classes.get(name, 0) + 1

    # Matched, and still owed an explanation. This is deliberately not folded
    # into the credit bar: it is not outstanding credit, it is a declared gap
    # inside a settled pairing, and merging the two would misstate both.
    carried = [m for m in result.matches if m.variance_paise]

    matched_lines = {m.left_ids[0] for m in result.matches_for_leg(2)}
    total_credit = sum(b.amount_paise for b in sources.bank_lines)
    matched_credit = sum(b.amount_paise for b in sources.bank_lines
                         if b.bank_line_id in matched_lines)

    return {
        "tiers": [{"tier": t, "count": c} for t, c in sorted(tiers.items())],
        "rules": [{"rule_id": r, "label": RULE_LABEL.get(r, r),
                   "tier": TIER_OF_RULE.get(r, "—"), "count": c}
                  for r, c in sorted(rules.items(), key=lambda kv: -kv[1])],
        "severities": [{"level": level, "count": severities.get(level, 0),
                        "at_risk": money(at_risk.get(level, 0))}
                       for level in ("critical", "warn", "minor", "structural")],
        "classes": sorted(({"name": n, "count": c} for n, c in classes.items()),
                          key=lambda d: -d["count"]),
        "credit": {
            "total": money(total_credit),
            "matched": money(matched_credit),
            "outstanding": money(total_credit - matched_credit),
            "matched_share": matched_credit / total_credit if total_credit else 0.0,
            "lines_total": len(sources.bank_lines),
            "lines_matched": len(matched_lines),
        },
        "variance": {
            "total": money(sum(m.variance_paise for m in carried)),
            "count": len(carried),
        },
    }


# ── the published result artefacts ───────────────────────────────────────────

#: A first line worth putting in a list, per file we ship.
RESULT_BLURB = {
    "benchrec_recoagent": "RecoAgent on BenchRec, the ICAIF industry benchmark — the one claim here that is not self-generated.",
    "benchrec_baseline": "The matcher that ships with BenchRec, scored on the same file by the same code.",
    "tolerance_sweep": "What the Leg 2 tolerance buys and what it costs, swept.",
}


def results_index() -> list[dict[str, Any]]:
    """The published artefacts, newest run first within each family."""
    if not RESULTS_DIR.is_dir():
        return []
    out = []
    for path in sorted(RESULTS_DIR.glob("*.txt")):
        out.append({
            "name": path.name,
            "stem": path.stem,
            "bytes": path.stat().st_size,
            "blurb": RESULT_BLURB.get(path.stem, ""),
        })
    return out


def result_file(name: str) -> str | None:
    """One artefact, by name, from the results directory and nowhere else.

    Matched against the listing rather than joined and normalised: a path this
    process will read should be one it already offered, not one a request
    described.
    """
    if name not in {entry["name"] for entry in results_index()}:
        return None
    return (RESULTS_DIR / name).read_text()
