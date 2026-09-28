"""
Closed planner-action list and fair scene dump for the llm_pddl / llm_plan
baselines (eval session E0).

Reads ``planner.skill_catalog``; does not patch it. Extra names already
present on the stacking/container templates are listed here as short
signatures only — no PDDL ``(:action)`` bodies.
"""

from __future__ import annotations

import json
from typing import Mapping

from planner.problem_generator.goal_generator.scene_compact import compact_scene_json
from planner.problem_generator.init_generator.schema import SceneState
from planner.skill_catalog import (
    CATALOG_SKILLS,
    catalog_skills_for_world,
    catalog_summary_lines,
    catalog_to_pddl_action,
    normalize_catalog_skill,
)

__all__ = [
    "BASELINE_PLANNER_ACTIONS",
    "EXTRA_PLANNER_ACTIONS",
    "baseline_action_summary_lines",
    "baseline_planner_actions_for_world",
    "fair_compact_scene",
    "is_baseline_planner_action",
    "normalize_baseline_action",
]

# Template-only names needed on suite v1 (stack / containers). Underscore form.
EXTRA_PLANNER_ACTIONS: frozenset[str] = frozenset(
    {
        "stack",
        "unstack",
        "pick_from_container",
        "place_in_container",
    }
)

# Household default (tabletop / kitchen / office). Workshop uses
# :func:`baseline_planner_actions_for_world`.
BASELINE_PLANNER_ACTIONS: frozenset[str] = CATALOG_SKILLS | EXTRA_PLANNER_ACTIONS


def baseline_planner_actions_for_world(world: str | None = None) -> frozenset[str]:
    """Closed action names the llm_plan / llm_pddl prompts may list for ``world``."""
    return catalog_skills_for_world(world) | EXTRA_PLANNER_ACTIONS

# Short prompt signatures (roles + one-line description). Not ROS dispatch.
# Roles use the item/location vocabulary; hyphen names come from the catalog map.
_EXTRA_SIGNATURES: Mapping[str, tuple[tuple[str, ...], str]] = {
    "stack": (("item", "item"), "place the held item onto another item"),
    "unstack": (("item", "item"), "pick the top item off another item"),
    "pick_from_container": (
        ("item", "location"),
        "pick an item from an open container",
    ),
    "place_in_container": (
        ("item", "location"),
        "place the held item into an open container",
    ),
}


def normalize_baseline_action(name: str) -> str:
    """Hyphen / mixed spellings → canonical underscore form."""
    return normalize_catalog_skill(name)


def is_baseline_planner_action(name: str, world: str | None = None) -> bool:
    """True when ``name`` is in the closed list for ``world`` (household if omitted)."""
    return normalize_baseline_action(name) in baseline_planner_actions_for_world(
        world
    )


def _format_extra_line(name: str) -> str:
    roles, description = _EXTRA_SIGNATURES[name]
    args = ", ".join(roles)
    return f"{catalog_to_pddl_action(name)}({args}) — {description}"


def baseline_action_summary_lines(
    skills: frozenset[str] | None = None,
    *,
    world: str | None = None,
) -> list[str]:
    """Prompt-ready ``name(roles) — description`` lines for the baseline list."""
    wanted = (
        skills
        if skills is not None
        else baseline_planner_actions_for_world(world)
    )
    extra_part = frozenset(n for n in wanted if n in EXTRA_PLANNER_ACTIONS)
    catalog_part = wanted - extra_part
    by_name: dict[str, str] = {}
    catalog_lines = catalog_summary_lines(catalog_part)
    for name, line in zip(sorted(catalog_part), catalog_lines):
        by_name[name] = line
    for name in extra_part:
        by_name[name] = _format_extra_line(name)
    return [by_name[n] for n in sorted(by_name)]


def fair_compact_scene(scene: SceneState) -> str:
    """
    Compact scene JSON for baseline prompts: no poses, no ``domain_template``.

    Calls :func:`compact_scene_json` then drops the key even when SceneState
    carries a template (R0 dump is otherwise identical).
    """
    payload = json.loads(compact_scene_json(scene, include_poses=False))
    payload.pop("domain_template", None)
    return json.dumps(payload, indent=2)
