"""Parse + validate R1 enrichment (call 1) and assignment (call 2) payloads."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Sequence

from planner.domain_llm import EnrichmentPayloadError
from planner.problem_generator.enrichment_goal import goal_from_enrichment_action
from planner.problem_generator.goal_generator.json_facts import extract_json_object
from planner.skill_catalog import (
    normalize_catalog_skill,
    ros_primitive_for_action,
    skill_signature,
)

_TYPED_ATOM = re.compile(r"\?[\w-]+\s*-\s*[a-zA-Z]")
_PLACEHOLDER_PREDICATES = frozenset(
    {"pred", "predicate", "new-pred", "new-predicate", "fluent", "fact", "p"}
)
_LOGICAL_OPS = frozenset(
    {"and", "or", "not", "imply", "forall", "exists", "when"}
)

R1EnrichmentPayloadError = EnrichmentPayloadError


class R1AssignmentError(ValueError):
    """Call-2 init_facts payload violated the affordance contract."""


@dataclass(frozen=True)
class ParsedR1Action:
    skill: str
    action: dict[str, str]

    @property
    def action_name(self) -> str:
        return str(self.action.get("name", ""))


@dataclass(frozen=True)
class ParsedR1Enrichment:
    skills: tuple[str, ...]
    actions: tuple[ParsedR1Action, ...]
    new_predicates: tuple[str, ...] = ()
    new_types: tuple[str, ...] = ()
    reason: str = ""
    refused: bool = False


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _require_sexpr(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("(") or not text.endswith(")"):
        raise R1EnrichmentPayloadError(f"{label} must be a PDDL s-expression")
    if not _balanced(text):
        raise R1EnrichmentPayloadError(f"{label} has unbalanced parentheses")
    return text


def predicate_head(declaration: str) -> str:
    match = re.match(r"\(\s*([a-zA-Z][\w-]*)", declaration.strip())
    return match.group(1) if match else declaration.strip()


def is_affordance_predicate(name: str) -> bool:
    return str(name or "").strip().lower().startswith("can-")


def affordance_predicate_names(declarations: Sequence[str]) -> frozenset[str]:
    return frozenset(
        predicate_head(d).lower()
        for d in declarations
        if is_affordance_predicate(predicate_head(d))
    )


def _atoms_in_formula(sexpr: str) -> list[tuple[str, int]]:
    """Predicate name + arity for atoms in a formula (skips logical ops)."""
    found: list[tuple[str, int]] = []
    depth = 0
    start = -1
    for i, char in enumerate(sexpr):
        if char == "(":
            if depth == 0:
                start = i
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                inner = sexpr[start + 1 : i].strip()
                if inner:
                    head = inner.split()[0].lower()
                    if head not in _LOGICAL_OPS:
                        vars_found = re.findall(r"\?[a-zA-Z][\w-]*", inner)
                        arity = len(vars_found) if vars_found else max(
                            0, len(inner.split()) - 1
                        )
                        found.append((head, arity))
                    else:
                        found.extend(_atoms_in_formula(inner[len(head) :]))
                start = -1
    return found


def _parse_one_action(
    action_raw: dict,
    *,
    allowed: set[str],
    skill_hint: str | None,
) -> ParsedR1Action:
    name = str(action_raw.get("name", "")).strip().lower()
    skill = normalize_catalog_skill(skill_hint or name)
    if skill not in allowed:
        raise R1EnrichmentPayloadError(
            f"skill {skill!r} is not in the allowed set "
            f"{sorted(allowed) or '(empty)'}"
        )
    sig = skill_signature(skill)
    if sig is None or ros_primitive_for_action(skill) is None:
        raise R1EnrichmentPayloadError(f"skill {skill!r} has no ROS primitive")
    if normalize_catalog_skill(name) != skill:
        raise R1EnrichmentPayloadError(
            f"action name {name!r} does not match skill {skill!r} "
            f"(expected {sig.pddl_action!r})"
        )

    parameters = _require_sexpr(action_raw.get("parameters"), "parameters")
    precondition = _require_sexpr(action_raw.get("precondition"), "precondition")
    effect = _require_sexpr(action_raw.get("effect"), "effect")

    for label, formula in (("precondition", precondition), ("effect", effect)):
        if _TYPED_ATOM.search(formula):
            raise R1EnrichmentPayloadError(
                f"{label} declares parameter types inside a formula — "
                "types belong in 'parameters', not in preconditions or effects"
            )

    declared = re.findall(r"\?([a-zA-Z][\w-]*)", parameters)
    if not (sig.min_params <= len(declared) <= sig.max_params):
        raise R1EnrichmentPayloadError(
            f"action '{sig.pddl_action}' needs {sig.min_params}–"
            f"{sig.max_params} parameters ({', '.join(sig.roles)}), got "
            f"{len(declared)}"
        )
    used = set(re.findall(r"\?([a-zA-Z][\w-]*)", precondition + " " + effect))
    unbound = sorted(used - set(declared))
    if unbound:
        raise R1EnrichmentPayloadError(
            "effect/precondition use undeclared variables: "
            + ", ".join(f"?{v}" for v in unbound)
        )
    goal_fact = goal_from_enrichment_action(
        {"effect": effect, "parameters": parameters}, {}
    )
    if goal_fact is None:
        raise R1EnrichmentPayloadError(
            "effect asserts no positive predicate — no goal fact can be derived"
        )
    return ParsedR1Action(
        skill=skill,
        action={
            "name": sig.pddl_action,
            "parameters": parameters,
            "precondition": precondition,
            "effect": effect,
        },
    )


def parse_r1_enrichment(
    raw: str,
    *,
    allowed_skills: Sequence[str],
) -> ParsedR1Enrichment:
    """Validate call-1 JSON: 1–N catalog skills + affordance predicates."""
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise R1EnrichmentPayloadError(f"invalid enrichment JSON: {exc}") from exc

    if data.get("refuse"):
        return ParsedR1Enrichment(
            skills=(),
            actions=(),
            reason=str(data.get("reason", "")).strip() or "model refused",
            refused=True,
        )

    allowed = {normalize_catalog_skill(s) for s in allowed_skills}

    actions_raw = data.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        single = data.get("action")
        if isinstance(single, dict):
            actions_raw = [single]
        else:
            raise R1EnrichmentPayloadError("missing 'actions' list")

    skills_raw = data.get("skills")
    if isinstance(skills_raw, str):
        skills_raw = [skills_raw]
    if not isinstance(skills_raw, list) or not skills_raw:
        skill_one = data.get("skill")
        skills_raw = [skill_one] if skill_one else [
            a.get("name") for a in actions_raw if isinstance(a, dict)
        ]

    parsed_actions: list[ParsedR1Action] = []
    for i, action_raw in enumerate(actions_raw):
        if not isinstance(action_raw, dict):
            raise R1EnrichmentPayloadError(f"actions[{i}] is not an object")
        hint = None
        if i < len(skills_raw) and skills_raw[i]:
            hint = str(skills_raw[i])
        parsed_actions.append(
            _parse_one_action(action_raw, allowed=allowed, skill_hint=hint)
        )

    if not parsed_actions:
        raise R1EnrichmentPayloadError("no actions authored")

    new_predicates = tuple(
        _require_sexpr(p, "new predicate")
        for p in (data.get("new_predicates") or [])
    )
    declared_new = {predicate_head(p).lower() for p in new_predicates}
    affordances = affordance_predicate_names(new_predicates)
    if not affordances:
        raise R1EnrichmentPayloadError(
            "R1 enrichment must declare at least one unary can-* affordance "
            "predicate in new_predicates"
        )

    new_arities: dict[str, int] = {}
    for decl in new_predicates:
        head = predicate_head(decl).lower()
        new_arities[head] = len(re.findall(r"\?[a-zA-Z][\w-]*", decl))
        if is_affordance_predicate(head) and new_arities[head] != 1:
            raise R1EnrichmentPayloadError(
                f"affordance '{head}' must be unary, got arity {new_arities[head]}"
            )

    for parsed in parsed_actions:
        pre = parsed.action["precondition"]
        eff = parsed.action["effect"]
        pre_preds = {name for name, _ in _atoms_in_formula(pre)}
        if not (pre_preds & affordances):
            raise R1EnrichmentPayloadError(
                f"action '{parsed.action_name}' precondition does not use any "
                f"can-* affordance {sorted(affordances)}"
            )
        goal_fact = goal_from_enrichment_action(parsed.action, {})
        if goal_fact is None:
            raise R1EnrichmentPayloadError(
                f"action '{parsed.action_name}' has no positive effect"
            )
        goal_predicate = goal_fact.strip("()").split()[0].lower()
        if goal_predicate in _PLACEHOLDER_PREDICATES:
            raise R1EnrichmentPayloadError(
                f"predicate {goal_predicate!r} is a placeholder from the prompt"
            )
        if goal_predicate not in declared_new:
            raise R1EnrichmentPayloadError(
                f"effect asserts {goal_predicate!r}, which the action does not "
                f"declare in new_predicates "
                f"({', '.join(sorted(declared_new)) or 'none'})"
            )
        for label, formula in (("precondition", pre), ("effect", eff)):
            for pred, arity in _atoms_in_formula(
                formula if formula.startswith("(") else f"({formula})"
            ):
                if pred not in new_arities:
                    continue
                expected = new_arities[pred]
                if arity != expected:
                    raise R1EnrichmentPayloadError(
                        f"{label} uses '{pred}' with {arity} argument(s) but "
                        f"new_predicates declares arity {expected}"
                    )

    new_types = tuple(
        str(t).strip() for t in (data.get("new_types") or []) if str(t).strip()
    )
    skills = tuple(a.skill for a in parsed_actions)
    return ParsedR1Enrichment(
        skills=skills,
        actions=tuple(parsed_actions),
        new_predicates=new_predicates,
        new_types=new_types,
        reason=str(data.get("reason", "")).strip(),
    )


def parse_r1_assignment(
    raw: str,
    *,
    allowed_predicates: Sequence[str],
    scene_symbols: Sequence[str],
) -> list[tuple[str, str]]:
    """Validate call-2 JSON ``init_facts`` against affordances and objects."""
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise R1AssignmentError(f"invalid assignment JSON: {exc}") from exc

    allowed = {str(p).strip().lower() for p in allowed_predicates if str(p).strip()}
    symbols = {str(s).strip().lower() for s in scene_symbols if str(s).strip()}
    raw_facts = data.get("init_facts")
    if raw_facts is None:
        raise R1AssignmentError("missing 'init_facts' list")
    if not isinstance(raw_facts, list):
        raise R1AssignmentError("'init_facts' must be a list")

    facts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_facts:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise R1AssignmentError(
                f"init_facts entries must be [predicate, object], got {item!r}"
            )
        pred = str(item[0]).strip().lower()
        obj = str(item[1]).strip().lower()
        if pred not in allowed:
            raise R1AssignmentError(
                f"predicate {pred!r} is not an authored affordance "
                f"{sorted(allowed) or '(empty)'}"
            )
        if obj not in symbols:
            raise R1AssignmentError(
                f"object {obj!r} is not in the scene "
                f"{sorted(symbols) or '(empty)'}"
            )
        key = (pred, obj)
        if key in seen:
            continue
        seen.add(key)
        facts.append(key)
    return facts
