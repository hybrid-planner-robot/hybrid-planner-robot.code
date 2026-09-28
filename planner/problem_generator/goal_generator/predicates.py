"""Allowed PDDL goal predicates per domain template."""

from __future__ import annotations

from typing import Sequence

DOMAIN_PREDICATES: dict[str, frozenset[str]] = {
    "manipulation_base": frozenset({"on", "holding", "camera-aimed-at"}),
    "manipulation_stacking": frozenset({"on", "holding", "camera-aimed-at", "stacked-on"}),
    "containers_manipulation": frozenset({"on", "holding", "camera-aimed-at", "in-container"}),
    "navigation_manipulation": frozenset({"on", "holding", "camera-aimed-at", "stacked-on"}),
}


def resolve_allowed_predicates(
    domain_template: str,
    allowed_predicates: Sequence[str] | None = None,
) -> frozenset[str]:
    if allowed_predicates is not None:
        return frozenset(allowed_predicates)
    return DOMAIN_PREDICATES.get(
        domain_template,
        DOMAIN_PREDICATES["manipulation_base"],
    )
