"""
Shared enrichment goal derivation from VLM ``domain_additions``.

Extracts PDDL goal facts from the positive effects of VLM-generated action
definitions. Used by both the legacy ``infer_goal_state`` and the hybrid
``HybridProblemSession.ensure_goal`` paths so enrichment semantics stay
identical.

Moved from ``legacy._goal_from_enrichment_action`` in Session 19.
"""

from __future__ import annotations

import re
from typing import Sequence

from .init_generator.renderer import PddlFact

_PDDL_KW = {"and", "or", "not", "when", "forall", "exists"}


def goal_from_enrichment_action(
    action_def: dict, step_args: dict
) -> str | None:
    """
    Given a VLM-generated action definition and the concrete step arguments,
    derive the PDDL goal fact by finding the first positive (non-negated) predicate
    in the action's effect and substituting the actual argument values.

    Works with any predicate name the VLM chose — no hardcoded names.
    Returns a PDDL fact string like '(poured bottle glass)', or None if derivation fails.
    """
    effect_str = action_def.get("effect", "")
    params_str = action_def.get("parameters", "")

    param_names = re.findall(r"\?([a-zA-Z][\w-]*)", params_str)
    arg_values = list(step_args.values())
    binding = {
        pn: str(arg_values[i])
        for i, pn in enumerate(param_names)
        if i < len(arg_values)
    }

    def _top_level_exprs(text: str) -> list[str]:
        """Extract direct children expressions (respects nested parens)."""
        exprs, depth, start = [], 0, -1
        for i, c in enumerate(text):
            if c == "(":
                if depth == 0:
                    start = i
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and start != -1:
                    exprs.append(text[start : i + 1])
                    start = -1
        return exprs

    def _find_positive_pred(expr: str) -> str | None:
        """Recursively search for the first positive predicate fact."""
        expr = expr.strip()
        if not expr.startswith("("):
            return None
        inner = expr[1:-1].strip()
        head = inner.split()[0] if inner else ""

        if head in ("not",):
            return None
        if head in ("and", "or"):
            for child in _top_level_exprs(inner[len(head) :].strip()):
                result = _find_positive_pred(child)
                if result:
                    return result
            return None
        if head in _PDDL_KW:
            return None

        tokens = inner.split()
        pred_name = tokens[0]
        pred_args = []
        for tok in tokens[1:]:
            if tok.startswith("?"):
                pred_args.append(binding.get(tok[1:], tok))
            elif tok not in ("-", "item", "location", "object"):
                pred_args.append(tok)
        return (
            f"({pred_name} {' '.join(pred_args)})" if pred_args else f"({pred_name})"
        )

    return _find_positive_pred(effect_str)


def goals_from_domain_additions(
    domain_additions: dict | None,
    steps: Sequence,
) -> list[PddlFact]:
    """
    Derive ``_raw_fact`` goal tuples from ``domain_additions.new_actions``
    matched against non-standard plan steps.

    This is the shared logic used by both ``legacy.infer_goal_state`` and the
    hybrid ``ensure_goal`` enrichment path.
    """
    if not domain_additions:
        return []

    enrichment_actions: dict[str, dict] = {}
    for act in domain_additions.get("new_actions", []):
        name = act.get("name", "")
        if name:
            enrichment_actions[name] = act

    if not enrichment_actions:
        return []

    _STANDARD = {"pick", "place", "unstack", "stack", "pick-from-container",
                  "place-in-container", "look_at", "look-at",
                  "open-container", "close-container", "navigate-to",
                  "pick_from_container", "place_in_container",
                  "open_container", "close_container", "navigate_to"}

    goals: list[PddlFact] = []
    for step in steps:
        primitive = getattr(step, "primitive", None)
        if primitive is None and isinstance(step, dict):
            primitive = step.get("primitive", "")
        args = getattr(step, "args", None)
        if args is None and isinstance(step, dict):
            args = step.get("args", {})

        if not primitive or primitive in _STANDARD:
            continue

        act_def = enrichment_actions.get(primitive)
        if act_def:
            fact = goal_from_enrichment_action(act_def, args or {})
            if fact:
                goals.append(("_raw_fact", fact))

    return goals
