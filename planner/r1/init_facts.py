"""Attach R1 unary affordance facts to a SceneState for ``:init``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from planner.problem_generator.init_generator.schema import RelationFact, SceneState

PddlFact = tuple[str, ...]


def apply_init_facts(
    scene: SceneState,
    facts: Sequence[Sequence[str] | tuple[str, ...]],
    *,
    source: str = "mock",
) -> SceneState:
    """Return a copy of ``scene`` with extra unary (or n-ary) relations."""
    extra: list[RelationFact] = []
    existing = {(rel.predicate, tuple(rel.args)) for rel in scene.relations}
    for fact in facts:
        if not fact:
            continue
        pred = str(fact[0]).strip()
        args = [str(a).strip() for a in fact[1:]]
        key = (pred, tuple(args))
        if key in existing:
            continue
        extra.append(
            RelationFact(
                predicate=pred,
                args=args,
                source=source,
                confidence=1.0,
            )
        )
        existing.add(key)
    if not extra:
        return scene
    return replace(scene, relations=list(scene.relations) + extra)


def init_facts_from_additions(
    domain_additions: Mapping[str, Any] | None,
) -> list[PddlFact]:
    raw = (domain_additions or {}).get("init_facts") or []
    facts: list[PddlFact] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and item:
            facts.append(tuple(str(x) for x in item))
    return facts


def apply_init_facts_from_additions(
    scene: SceneState,
    domain_additions: Mapping[str, Any] | None,
) -> SceneState:
    facts = init_facts_from_additions(domain_additions)
    if not facts:
        return scene
    return apply_init_facts(scene, facts)
