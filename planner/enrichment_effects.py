"""
Loop hooks for online-enriched catalog skills (Session 30).

An enriched action has two halves, and they come from different places:

**The manipulation half is fixed** — it is whatever the ROS body in
``ros2_ws/.../primitives/`` actually does to the gripper. ``pour`` releases the
source vessel (detach + open gripper); ``tilt`` / ``stir`` / ``cut`` never touch
the gripper. That is hard-coded here in ``CATALOG_SKILL_EFFECTS`` because it is
a property of the robot, not of any generated PDDL.

**The symbolic half is authored at runtime** — the fluent an enriched action
asserts (``poured``, ``stirred``, whatever the Session 29 text LLM chose) is
only known from that task's ``domain_additions``. ``EnrichmentContext`` reads it
back out with the Session 19 helper, so the tracker asserts exactly the fact the
planner will look for in ``:goal``, with no second source of truth.

Verification policy (design §4.5): these fluents are **not perceptually
verifiable** — no adapter reports "liquid moved". The verifier therefore scores
only the observable consequences (gripper, holding, placements) and refuses to
call an enriched step GREEN on the strength of the fluent alone; a scene that
looks right lands on YELLOW so the VLM, not the script, gets the last word.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from planner.problem_generator.enrichment_goal import goal_from_enrichment_action

__all__ = [
    "CATALOG_SKILL_EFFECTS",
    "CatalogSkillEffect",
    "EnrichmentContext",
    "catalog_skill_effect",
    "ground_enrichment_goal",
    "merge_contexts",
    "predicates_from_domain_additions",
    "step_args_for_catalog_action",
]


@dataclass(frozen=True)
class CatalogSkillEffect:
    """What a catalog skill's ROS body does to the robot, symbolically."""

    primitive: str
    releases_object: bool  # gripper opens and the held item is let go
    requires_holding: bool  # the ROS body assumes something is in the gripper
    perceptually_verifiable: bool  # can any adapter confirm the fluent?
    note: str = ""


# Underscore + hyphen spellings both resolve through ``normalize`` below.
CATALOG_SKILL_EFFECTS: Mapping[str, CatalogSkillEffect] = {
    effect.primitive: effect
    for effect in (
        CatalogSkillEffect(
            "pour",
            releases_object=True,
            requires_holding=True,
            perceptually_verifiable=False,
            note=(
                "pour.py tilts the held source over the target, then detaches, "
                "opens the gripper and retreats — the source ends un-held and "
                "its resting place is only known from the next perception pass"
            ),
        ),
        CatalogSkillEffect(
            "tilt",
            releases_object=False,
            requires_holding=True,
            perceptually_verifiable=False,
            note="tilt.py rotates the held item about the wrist; no release",
        ),
        CatalogSkillEffect(
            "stir",
            releases_object=False,
            requires_holding=False,
            perceptually_verifiable=False,
            note="stir.py traces a circle inside the container; no release",
        ),
        CatalogSkillEffect(
            "cut",
            releases_object=False,
            requires_holding=False,
            perceptually_verifiable=False,
            note="cut.py makes downward strokes on the workpiece; no release",
        ),
        CatalogSkillEffect(
            "drill",
            releases_object=False,
            requires_holding=True,
            perceptually_verifiable=False,
            note="plan-only workshop skill: bore a hole; no Gazebo body",
        ),
        CatalogSkillEffect(
            "paint",
            releases_object=True,
            requires_holding=True,
            perceptually_verifiable=False,
            note="plan-only workshop skill: coat a surface from a held source",
        ),
        CatalogSkillEffect(
            "clamp",
            releases_object=False,
            requires_holding=True,
            perceptually_verifiable=False,
            note="plan-only workshop skill: secure a workpiece; no Gazebo body",
        ),
    )
}


def _normalize(name: str) -> str:
    return str(name or "").strip().lower().replace("-", "_")


def catalog_skill_effect(primitive: str) -> CatalogSkillEffect | None:
    """Fixed ROS-body semantics for a catalog skill, or ``None``."""
    return CATALOG_SKILL_EFFECTS.get(_normalize(primitive))


@dataclass(frozen=True)
class EnrichmentContext:
    """
    The enriched actions in play for one task.

    Built from the ``domain_additions`` that Session 29 persisted alongside the
    enriched domain (or from a VLM plan's additions, for the legacy ablation).
    Empty context = fixed-domain task; every hook below then no-ops, which is
    why the default path is unchanged.
    """

    actions: Mapping[str, dict] = None  # type: ignore[assignment]
    declared_predicates: frozenset[str] = frozenset()
    catalog_only: bool = True

    def __post_init__(self) -> None:
        if self.actions is None:
            object.__setattr__(self, "actions", {})

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def empty(cls) -> "EnrichmentContext":
        return cls(actions={}, declared_predicates=frozenset())

    @classmethod
    def from_domain_additions(
        cls,
        domain_additions: Mapping[str, Any] | None,
    ) -> "EnrichmentContext":
        """
        Read enriched actions + predicate names out of a ``domain_additions`` dict.

        ``catalog_only`` records whether every action maps to a real ROS
        primitive. Legacy open-vocab VLM additions set it False so the loop can
        log them as an ablation rather than treating them as the paper path.
        """
        if not domain_additions:
            return cls.empty()

        from planner.skill_catalog import ros_primitive_for_action

        actions: dict[str, dict] = {}
        catalog_only = True
        for act in domain_additions.get("new_actions", []) or []:
            if not isinstance(act, Mapping):
                continue
            name = str(act.get("name", "")).strip()
            if not name:
                continue
            actions[_normalize(name)] = dict(act)
            if ros_primitive_for_action(name) is None:
                catalog_only = False

        predicates = {
            name
            for decl in (domain_additions.get("new_predicates", []) or [])
            if (name := _predicate_name(decl))
        }
        for act in actions.values():
            fact = goal_from_enrichment_action(act, {})
            if fact:
                predicates.add(fact.strip("()").split()[0])

        return cls(
            actions=actions,
            declared_predicates=frozenset(predicates),
            catalog_only=catalog_only if actions else True,
        )

    # ── Queries ──────────────────────────────────────────────────────────────

    def __bool__(self) -> bool:
        return bool(self.actions)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.actions))

    def knows(self, primitive: str) -> bool:
        """True when ``primitive`` is one of this task's enriched actions."""
        return _normalize(primitive) in self.actions

    def predicates(self) -> frozenset[str]:
        """Predicate names this task's enrichment may put in ``:init``."""
        return self.declared_predicates

    def effect_for(self, primitive: str) -> CatalogSkillEffect | None:
        """ROS-body semantics for an enriched action the robot can execute."""
        if not self.knows(primitive):
            return None
        return catalog_skill_effect(primitive)

    def is_executable(self, primitive: str) -> bool:
        """True when the enriched action maps to a catalog skill with a body."""
        return self.effect_for(primitive) is not None

    def fact_for(
        self,
        primitive: str,
        args: Mapping[str, Any] | None = None,
    ) -> tuple[str, ...] | None:
        """
        The grounded fluent an enriched step asserts, as a PDDL fact tuple.

        Delegates to the Session 19 helper so the tracker asserts precisely what
        ``goals_from_domain_additions`` put in ``:goal``. Returns ``None`` when
        the action is unknown or its arguments cannot be bound.
        """
        action = self.actions.get(_normalize(primitive))
        if action is None:
            return None
        rendered = goal_from_enrichment_action(action, dict(args or {}))
        if not rendered:
            return None
        tokens = rendered.strip("()").split()
        if not tokens:
            return None
        # Unbound parameters mean the step did not supply enough arguments.
        if any(tok.startswith("?") for tok in tokens[1:]):
            return None
        return tuple(tokens)


# PDDL parameter roles → the argument keys the ROS primitives and the tracker
# already speak (see planner/primitive_transitions.py ``arg`` lookups).
_ROLE_ARG_KEYS: Mapping[str, str] = {
    "item": "object",
    "location": "location",
    "container": "container",
    "source": "source",
    "target": "target",
}


def step_args_for_catalog_action(
    action: str,
    args: Sequence[str],
) -> dict[str, str]:
    """
    Name the positional arguments of a grounded enriched action.

    ``(pour bottle glass)`` → ``{"source": "bottle", "target": "glass"}``, using
    the catalog signature's roles so the ROS primitive, the tracker and the
    fluent binding all read the same keys.
    """
    from planner.skill_catalog import skill_signature

    signature = skill_signature(action)
    roles = signature.roles if signature else ()
    named: dict[str, str] = {}
    for index, value in enumerate(args):
        if index < len(roles):
            key = _ROLE_ARG_KEYS.get(roles[index], roles[index])
        else:
            key = f"arg{index}"
        named[key] = str(value)
    return named


def ground_enrichment_goal(
    context: EnrichmentContext,
    command: str,
    symbols: Sequence[str],
    *,
    binder: Any | None = None,
) -> list[tuple[str, str]]:
    """
    Bind each enriched action's fluent to concrete scene objects.

    Needed on the ``control=fd`` path: there are no VLM steps to read the
    arguments from, so the ``:goal`` has to be grounded from the request and the
    scene. ``binder`` (the single text LLM, Session 30) is asked first because a
    paraphrase like *"get me something to drink"* names no objects at all; the
    deterministic fallback fills parameters in the order the objects appear in
    the command. Actions that can be bound neither way are skipped, and the
    caller refuses rather than planning toward a half-bound goal.

    Returns ``_raw_fact`` goal tuples, the same shape
    ``goals_from_domain_additions`` produces.
    """
    if not context:
        return []

    ordered = _symbols_in_command_order(command, symbols)
    goals: list[tuple[str, str]] = []
    for name in context.action_names:
        action = context.actions[name]
        arity = len(re.findall(r"\?[a-zA-Z][\w-]*", str(action.get("parameters", ""))))
        if arity == 0:
            continue

        bound: Sequence[str] | None = None
        if binder is not None:
            bound = binder.bind(command, action, list(symbols), arity=arity)
        if not bound:
            bound = ordered[:arity] if len(ordered) >= arity else None
        if not bound:
            continue

        fact = context.fact_for(
            name,
            {f"p{i}": value for i, value in enumerate(bound)},
        )
        if fact:
            goals.append(("_raw_fact", f"({' '.join(fact)})"))
    return goals


def _symbols_in_command_order(command: str, symbols: Sequence[str]) -> list[str]:
    """Scene symbols mentioned in ``command``, in order of first appearance."""
    text = str(command or "").lower()
    hits: list[tuple[int, str]] = []
    for symbol in symbols:
        needle = str(symbol or "").lower()
        if not needle:
            continue
        position = text.find(needle)
        if position < 0:
            position = text.find(needle.replace("_", " "))
        if position >= 0:
            hits.append((position, symbol))
    seen: set[str] = set()
    ordered: list[str] = []
    for _, symbol in sorted(hits):
        if symbol not in seen:
            seen.add(symbol)
            ordered.append(symbol)
    return ordered


_PREDICATE_HEAD = re.compile(r"\(\s*([a-zA-Z][\w-]*)")


def _predicate_name(declaration: Any) -> str:
    match = _PREDICATE_HEAD.match(str(declaration or "").strip())
    return match.group(1) if match else ""


def predicates_from_domain_additions(
    domain_additions: Mapping[str, Any] | None,
) -> frozenset[str]:
    """Convenience for callers that only need the ``:init`` predicate allowlist."""
    return EnrichmentContext.from_domain_additions(domain_additions).predicates()


def merge_contexts(contexts: Iterable[EnrichmentContext]) -> EnrichmentContext:
    """Union several contexts (multi-skill enrichment)."""
    actions: dict[str, dict] = {}
    predicates: set[str] = set()
    catalog_only = True
    for ctx in contexts:
        actions.update(ctx.actions)
        predicates |= set(ctx.declared_predicates)
        catalog_only = catalog_only and ctx.catalog_only
    return EnrichmentContext(
        actions=actions,
        declared_predicates=frozenset(predicates),
        catalog_only=catalog_only,
    )
