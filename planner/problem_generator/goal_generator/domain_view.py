"""
Compact domain view for the goal LLM user prompt.

Same shape whether the domain is a fixed template or an online-enriched file:
predicates + action signatures. Callers must not label actions as \"added\".

Kept free of ``planner.domain_llm`` imports to avoid circular imports through
``problem_generator`` ↔ ``domain_llm``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from planner.domain_store import load_base_domain
from planner.problem_generator.goal_generator.predicates import DOMAIN_PREDICATES

__all__ = [
    "action_signature_dict",
    "compact_domain_view",
    "enrichment_authored_payload",
    "enrichment_outcome_predicates",
    "extract_actions_from_domain_text",
    "facts_use_any_predicate",
    "filter_facts_to_predicates",
    "format_domain_actions_block",
]

_ACTION_HEAD = re.compile(r"\(:action\s+([a-zA-Z][\w-]*)")
_COMMENT = re.compile(r";[^\n]*")
_RAW_PRED = re.compile(r"\(\s*([a-zA-Z][\w-]*)")


def _strip_pddl_comments(text: str) -> str:
    return _COMMENT.sub("", text)


def _action_block(text: str, start: int) -> str | None:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _domain_predicate_names(domain_text: str) -> set[str]:
    body = _strip_pddl_comments(domain_text)
    start = body.find("(:predicates")
    if start == -1:
        return set()
    block = _action_block(body, start) or ""
    names = set(re.findall(r"\(([a-zA-Z][\w-]*)", block))
    names.discard("predicates")
    return names


def _section_after(block: str, keyword: str) -> str:
    """Return the sexpr (or atom) that follows ``keyword`` inside an action."""
    idx = block.find(keyword)
    if idx < 0:
        return ""
    after = block[idx + len(keyword) :].lstrip()
    if not after:
        return ""
    if after.startswith("("):
        return _action_block(after, 0) or ""
    return after.split("\n", 1)[0].strip()


def extract_actions_from_domain_text(domain_text: str) -> list[dict[str, str]]:
    """Parse ``(:action …)`` blocks into name/parameters/precondition/effect."""
    body = _strip_pddl_comments(domain_text or "")
    out: list[dict[str, str]] = []
    for match in _ACTION_HEAD.finditer(body):
        name = match.group(1)
        block = _action_block(body, match.start())
        if not block:
            continue
        out.append(
            {
                "name": name,
                "parameters": _section_after(block, ":parameters") or "",
                "precondition": _section_after(block, ":precondition") or "",
                "effect": _section_after(block, ":effect") or "",
            }
        )
    return out


def action_signature_dict(action: Mapping[str, Any]) -> dict[str, str]:
    """Normalise a domain_additions / enricher action dict for prompts & reports."""
    return {
        "name": str(action.get("name") or "").strip(),
        "parameters": str(action.get("parameters") or "").strip(),
        "precondition": str(action.get("precondition") or "").strip(),
        "effect": str(action.get("effect") or "").strip(),
    }


def _merge_actions(
    base: Sequence[Mapping[str, Any]],
    extra: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    by_name: dict[str, dict[str, str]] = {}
    for raw in [*base, *extra]:
        sig = action_signature_dict(raw)
        name = sig["name"]
        if not name:
            continue
        by_name[name.lower().replace("_", "-")] = sig
    return list(by_name.values())


def compact_domain_view(
    domain_template: str,
    *,
    domain_text: str | None = None,
    domain_additions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the domain slice shown to the goal LLM.

    Prefer ``domain_text`` (enriched or stock). Fall back to the fixed template
    file plus optional ``domain_additions.new_actions`` / ``new_predicates``.
    """
    additions = dict(domain_additions or {})
    text = domain_text
    if not text:
        try:
            text = load_base_domain(domain_template)
        except FileNotFoundError:
            text = ""

    actions = extract_actions_from_domain_text(text) if text else []
    predicates = (
        set(_domain_predicate_names(text))
        if text
        else set(DOMAIN_PREDICATES.get(domain_template, ()))
    )

    extra_actions = [
        action_signature_dict(a)
        for a in (additions.get("new_actions") or [])
        if isinstance(a, Mapping)
    ]
    actions = _merge_actions(actions, extra_actions)

    for decl in additions.get("new_predicates") or []:
        match = re.match(r"\(\s*([a-zA-Z][\w-]*)", str(decl).strip())
        if match:
            predicates.add(match.group(1))

    return {
        "predicates": sorted(predicates),
        "actions": actions,
    }


def format_domain_actions_block(actions: Sequence[Mapping[str, Any]]) -> str:
    """Multi-line block for the goal user prompt."""
    if not actions:
        return "(none)"
    lines: list[str] = []
    for act in actions:
        sig = action_signature_dict(act)
        lines.append(
            f"- {sig['name']}: parameters={sig['parameters'] or '—'}; "
            f"precondition={sig['precondition'] or '—'}; "
            f"effect={sig['effect'] or '—'}"
        )
    return "\n".join(lines)


def _normalize_pred(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def _predicate_from_fact(fact: Sequence[Any]) -> str:
    if not fact:
        return ""
    head = str(fact[0])
    if head == "_raw_fact" and len(fact) >= 2:
        match = _RAW_PRED.match(str(fact[1]).strip())
        return _normalize_pred(match.group(1) if match else "")
    return _normalize_pred(head)


def facts_use_any_predicate(
    facts: Sequence[Sequence[Any]],
    predicates: Sequence[str] | frozenset[str],
) -> bool:
    """True when at least one goal fact uses a predicate in ``predicates``."""
    targets = {_normalize_pred(p) for p in predicates if str(p).strip()}
    if not targets:
        return False
    return any(_predicate_from_fact(f) in targets for f in facts if f)


def enrichment_outcome_predicates(
    domain_additions: Mapping[str, Any] | None,
) -> frozenset[str]:
    """
    Predicates introduced by online enrichment (declarations + positive effects).

    Used to detect when a primary :goal ignored the enriched outcome and fell
    back to stock fluents such as ``holding``. Delegates to
    ``EnrichmentContext`` so negated effect atoms are not treated as outcomes.
    """
    if not domain_additions:
        return frozenset()
    from planner.enrichment_effects import EnrichmentContext

    ctx = EnrichmentContext.from_domain_additions(domain_additions)
    return frozenset(_normalize_pred(p) for p in ctx.predicates() if p)


def filter_facts_to_predicates(
    facts: Sequence[Sequence[Any]],
    allowed: Sequence[str] | frozenset[str],
) -> list[Any]:
    """
    Drop goal facts whose predicate is outside ``allowed``.

    ``_raw_fact`` tuples keep the predicate inside the sexpr string; those are
    checked the same way. Empty ``allowed`` means no filtering.
    """
    allow = {_normalize_pred(p) for p in allowed if str(p).strip()}
    if not allow:
        return [tuple(f) for f in facts]

    kept: list[Any] = []
    for fact in facts:
        if not fact:
            continue
        if _predicate_from_fact(fact) in allow:
            kept.append(tuple(fact))
    return kept


def enrichment_authored_payload(
    domain_additions: Mapping[str, Any] | None,
    *,
    skills: Sequence[str] = (),
    domain_path: str | None = None,
    reused: bool = False,
) -> dict[str, Any] | None:
    """Structured enrichment artefact for summary.json / planning_eval."""
    if not domain_additions:
        return None
    actions = [
        action_signature_dict(a)
        for a in (domain_additions.get("new_actions") or [])
        if isinstance(a, Mapping) and str(a.get("name") or "").strip()
    ]
    if not actions and not skills:
        return None
    return {
        "skills": [str(s) for s in skills],
        "actions": actions,
        "new_predicates": [
            str(p) for p in (domain_additions.get("new_predicates") or []) if str(p).strip()
        ],
        "new_types": [
            str(t) for t in (domain_additions.get("new_types") or []) if str(t).strip()
        ],
        "domain_path": domain_path,
        "reused": bool(reused),
        "init_facts": [
            list(f)
            for f in (domain_additions.get("init_facts") or [])
            if isinstance(f, (list, tuple))
        ],
    }
