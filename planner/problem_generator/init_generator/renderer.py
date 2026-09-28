"""
Pure SceneState → PDDL ``:init`` renderer (no LLM / VLM).

See ``docs/hybrid_problem_generator_design.md`` §3.2.
"""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

from .schema import SceneState

logger = logging.getLogger(__name__)

PddlFact = tuple[str, ...]

# v1 relation predicates mapped directly to init facts.
# Session 23: ``reachable`` dropped from the four complete domains (was always
# asserted). Schema may still carry the field; it is not emitted to PDDL.
_KNOWN_RELATION_PREDICATES: frozenset[str] = frozenset(
    {
        "on",
        "stacked-on",
        "in-container",
        "clear",
        "camera-aimed-at",
        "holding",
    }
)

# Stable emission order (matches legacy ``generate_problem`` where practical).
_PREDICATE_ORDER: dict[str, int] = {
    "on": 0,
    "stacked-on": 1,
    "in-container": 2,
    "holding": 3,
    "clear": 4,
    "gripper-empty": 5,
    "camera-aimed-at": 6,
    "open": 7,
    "closed": 8,
}


def _fact_key(fact: PddlFact) -> tuple:
    pred = fact[0]
    return (_PREDICATE_ORDER.get(pred, 99), fact[1:])


def _format_fact(fact: PddlFact) -> str:
    if len(fact) == 1:
        return f"({fact[0]})"
    return f"({fact[0]} {' '.join(fact[1:])})"


class InitRenderer:
    """Deterministic mapping ``SceneState`` → PDDL init facts."""

    def __init__(self, *, extra_predicates: frozenset[str] | None = None) -> None:
        # Session 30: predicates an online-enriched domain declared for this
        # task (e.g. ``poured``). They are emitted like the v1 relations so a
        # completed enriched action survives into the next ``:init``; anything
        # still unknown keeps warning, which is what catches typos.
        self.extra_predicates: frozenset[str] = frozenset(extra_predicates or ())

    def render_facts(self, scene: SceneState) -> list[PddlFact]:
        """
        Map SceneState fields and relations to PDDL init fact tuples.

        Emits when present:
          (on i l), (holding i), (gripper-empty),
          (clear i),
          and domain extras: (stacked-on a b), (in-container i c), (open c), (closed c).

        ``camera-aimed-at`` is never written to ``:init``. The fluent is achieved
        only by ``look-at``; seeding it would skip that action.

        Unknown relation predicates are skipped with a warning.
        ``ObjectFact.reachable`` / ``LocationFact.reachable`` are ignored: the
        four complete domains no longer declare ``(reachable …)`` (Session 23).
        """
        facts: set[PddlFact] = set()
        holding = scene.robot.holding

        for rel in scene.relations:
            pred = rel.predicate
            if pred == "camera-aimed-at":
                continue  # look-at must achieve this; never seed :init
            if pred in self.extra_predicates:
                facts.add((pred, *rel.args))
                continue
            if pred not in _KNOWN_RELATION_PREDICATES:
                msg = f"InitRenderer: ignoring unknown relation predicate {pred!r}"
                logger.warning(msg)
                warnings.warn(msg, stacklevel=2)
                continue

            if pred == "holding":
                continue  # prefer robot.holding

            if pred == "on" and holding and len(rel.args) >= 1 and rel.args[0] == holding:
                continue  # held object has no (on ...) fact

            facts.add((pred, *rel.args))

        for obj in scene.objects:
            if obj.clear is True:
                facts.add(("clear", obj.name))

        for loc in scene.locations:
            if scene.domain_template != "containers_manipulation":
                continue
            if loc.open is True:
                facts.add(("open", loc.name))
            elif loc.open is False:
                facts.add(("closed", loc.name))

        if holding:
            facts.add(("holding", holding))
        elif scene.robot.gripper_empty:
            facts.add(("gripper-empty",))

        return sorted(facts, key=_fact_key)

    def render_section(self, scene: SceneState, *, indent: str = "  ") -> str:
        """Return the ``(:init ...)`` block string (without surrounding problem)."""
        facts = self.render_facts(scene)
        inner = indent * 2
        lines = [f"{indent}(:init"]
        for fact in facts:
            lines.append(f"{inner}{_format_fact(fact)}")
        lines.append(f"{indent})")
        return "\n".join(lines)
