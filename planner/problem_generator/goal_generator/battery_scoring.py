"""Scoring helpers for the goal-LLM difficulty battery (no GPU)."""

from __future__ import annotations

from typing import Any, Sequence


def normalize_facts(facts: Sequence[Sequence[str]] | None) -> list[tuple[str, ...]]:
    if not facts:
        return []
    return [tuple(str(x) for x in fact) for fact in facts]


def facts_equal(
    a: Sequence[Sequence[str]] | None,
    b: Sequence[Sequence[str]] | None,
) -> bool:
    """Order-insensitive multiset equality of fact tuples."""
    return sorted(normalize_facts(a)) == sorted(normalize_facts(b))


def facts_valid(
    facts: Sequence[Sequence[str]] | None,
    *,
    allowed_predicates: set[str] | frozenset[str],
    known_symbols: set[str] | frozenset[str],
) -> tuple[bool, bool]:
    """Return (predicates_ok, symbols_ok). Empty facts → (True, True)."""
    pred_ok = True
    sym_ok = True
    for fact in normalize_facts(facts):
        if not fact:
            pred_ok = False
            continue
        if fact[0] not in allowed_predicates:
            pred_ok = False
        for arg in fact[1:]:
            if arg not in known_symbols:
                sym_ok = False
    return pred_ok, sym_ok


def score_case(
    *,
    gold: Sequence[Sequence[str]] | None,
    expect_fail: bool,
    result_ok: bool,
    result_facts: Sequence[Sequence[str]] | None,
    result_error: str | None,
    allowed_predicates: set[str] | frozenset[str],
    known_symbols: set[str] | frozenset[str],
    rule_facts: Sequence[Sequence[str]] | None = None,
    rule_ok: bool | None = None,
    soft_golds: Sequence[Sequence[Sequence[str]]] | None = None,
) -> dict[str, Any]:
    """
    Score one battery case.

    - With ``gold``: pass iff exact_match (order-insensitive).
    - With ``soft_golds``: pass if facts match any listed acceptable set
      (also counts as exact_match for reporting when matched).
    - With ``expect_fail``: pass iff the backend refuses (``ok=False``) or
      returns empty facts (no hallucination of a concrete goal).
    """
    facts = normalize_facts(result_facts)
    pred_ok, sym_ok = facts_valid(
        facts,
        allowed_predicates=allowed_predicates,
        known_symbols=known_symbols,
    )
    used_fallback = bool(result_error and "fallback" in result_error.lower())

    exact = False
    soft_hit = False
    if gold is not None:
        exact = facts_equal(facts, gold)
    if soft_golds:
        soft_hit = any(facts_equal(facts, g) for g in soft_golds)

    if expect_fail:
        passed = (not result_ok) or (result_ok and len(facts) == 0)
        kind = "expect_fail"
    elif gold is not None or soft_golds:
        matched = exact or soft_hit
        passed = bool(result_ok) and matched and pred_ok and sym_ok
        kind = "exact_gold" if gold is not None and not soft_golds else "soft_or_gold"
        exact = matched
    else:
        passed = bool(result_ok) and pred_ok and sym_ok and len(facts) > 0
        kind = "valid_only"

    out: dict[str, Any] = {
        "kind": kind,
        "passed": passed,
        "exact_match": exact if (gold is not None or soft_golds) else None,
        "soft_match": soft_hit,
        "predicates_ok": pred_ok,
        "symbols_ok": sym_ok,
        "used_fallback": used_fallback,
        "result_ok": result_ok,
        "n_facts": len(facts),
    }
    if rule_ok is not None:
        out["rule_ok"] = rule_ok
        rule_exact = False
        if gold is not None and rule_ok:
            rule_exact = facts_equal(rule_facts, gold)
        elif soft_golds and rule_ok:
            rule_exact = any(facts_equal(rule_facts, g) for g in soft_golds)
        out["rule_exact"] = rule_exact
        out["llm_beats_rule"] = bool(
            (gold is not None or soft_golds) and passed and not rule_exact
        )
    return out


def summarize_by_level(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_level: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_level.setdefault(row["level_id"], []).append(row)

    levels: dict[str, Any] = {}
    for lid, items in by_level.items():
        n = len(items)
        n_pass = sum(1 for r in items if r["score"]["passed"])
        n_exact = sum(1 for r in items if r["score"].get("exact_match") is True)
        n_valid = sum(
            1
            for r in items
            if r["score"]["predicates_ok"] and r["score"]["symbols_ok"]
        )
        n_fallback = sum(1 for r in items if r["score"]["used_fallback"])
        n_beats = sum(1 for r in items if r["score"].get("llm_beats_rule"))
        levels[lid] = {
            "n": n,
            "pass_rate": round(n_pass / n, 3) if n else 0.0,
            "exact_rate": round(n_exact / n, 3) if n else 0.0,
            "valid_rate": round(n_valid / n, 3) if n else 0.0,
            "fallback_rate": round(n_fallback / n, 3) if n else 0.0,
            "llm_beats_rule": n_beats,
            "failed_ids": [r["id"] for r in items if not r["score"]["passed"]],
        }

    n_all = len(rows)
    n_pass_all = sum(1 for r in rows if r["score"]["passed"])
    return {
        "n": n_all,
        "pass_rate": round(n_pass_all / n_all, 3) if n_all else 0.0,
        "levels": levels,
    }
