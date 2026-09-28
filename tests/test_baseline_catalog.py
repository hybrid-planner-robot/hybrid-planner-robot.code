"""E0 — frozen baseline action list and fair compact-scene dump."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.baseline_catalog import (
    BASELINE_PLANNER_ACTIONS,
    EXTRA_PLANNER_ACTIONS,
    baseline_action_summary_lines,
    baseline_planner_actions_for_world,
    fair_compact_scene,
    is_baseline_planner_action,
    normalize_baseline_action,
)
from planner.problem_generator.goal_generator.scene_compact import compact_scene_json
from planner.problem_generator.init_generator.schema import (
    LocationFact,
    ObjectFact,
    Orientation,
    Pose,
    Position,
    RelationFact,
    RobotFacts,
    SceneState,
)
from planner.skill_catalog import CATALOG_SKILLS, catalog_summary_lines


def _scene_with_template() -> SceneState:
    return SceneState(
        objects=[
            ObjectFact(
                name="red_cup",
                source="mock",
                confidence=1.0,
                location="table",
                clear=True,
                pose=Pose(
                    position=Position(0.4, 0.0, 0.8),
                    orientation=Orientation(0.0, 0.0, 0.0, 1.0),
                ),
            ),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="mock",
                confidence=1.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
        domain_template="manipulation_stacking",
    )


def test_constant_is_catalog_union_extras():
    assert EXTRA_PLANNER_ACTIONS == {
        "stack",
        "unstack",
        "pick_from_container",
        "place_in_container",
    }
    assert EXTRA_PLANNER_ACTIONS.isdisjoint(CATALOG_SKILLS)
    assert BASELINE_PLANNER_ACTIONS == CATALOG_SKILLS | EXTRA_PLANNER_ACTIONS


def test_normalize_hyphen_and_underscore():
    assert normalize_baseline_action("pick-from-container") == "pick_from_container"
    assert normalize_baseline_action("Look-At") == "look_at"
    assert normalize_baseline_action("stack") == "stack"
    assert is_baseline_planner_action("place-in-container")
    assert is_baseline_planner_action("pour")
    assert not is_baseline_planner_action("solder")
    assert not is_baseline_planner_action("levitate")
    assert not is_baseline_planner_action("paint")
    assert is_baseline_planner_action("paint", world="workshop")
    assert is_baseline_planner_action("drill", world="workshop")
    assert is_baseline_planner_action("clamp", world="workshop")
    assert not is_baseline_planner_action("pour", world="workshop")
    assert is_baseline_planner_action("cut", world="workshop")
    assert is_baseline_planner_action("pour", world="kitchen")


def test_summary_lines_reuse_catalog_plus_short_extras():
    catalog = catalog_summary_lines()
    combined = baseline_action_summary_lines()
    for line in catalog:
        assert line in combined
    joined = "\n".join(combined)
    assert "stack(item, item) — " in joined
    assert "unstack(item, item) — " in joined
    assert "pick-from-container(item, location) — " in joined
    assert "place-in-container(item, location) — " in joined
    assert len(combined) == len(BASELINE_PLANNER_ACTIONS)


def test_workshop_summary_lists_workshop_skills_not_household_enrichment():
    household = "\n".join(baseline_action_summary_lines())
    workshop = "\n".join(baseline_action_summary_lines(world="workshop"))
    assert "pour(" in household
    assert "paint(" not in household
    assert "paint(" in workshop
    assert "drill(" in workshop
    assert "clamp(" in workshop
    assert "pour(" not in workshop
    assert "stir(" not in workshop
    assert "tilt(" not in workshop
    assert "cut(" in workshop
    assert "pick(" in workshop
    kitchen = baseline_planner_actions_for_world("kitchen")
    tabletop = baseline_planner_actions_for_world("tabletop")
    assert kitchen == tabletop == BASELINE_PLANNER_ACTIONS
    assert "paint" in baseline_planner_actions_for_world("workshop")
    assert "pour" not in baseline_planner_actions_for_world("workshop")


def test_skill_catalog_summary_unchanged_for_extras():
    """Extras live in baseline_catalog; skill_catalog still has no signatures."""
    assert catalog_summary_lines(EXTRA_PLANNER_ACTIONS) == []
    assert catalog_summary_lines() == catalog_summary_lines(CATALOG_SKILLS)
    for name in EXTRA_PLANNER_ACTIONS:
        assert name not in CATALOG_SKILLS


def test_fair_compact_scene_strips_domain_template():
    scene = _scene_with_template()
    raw = compact_scene_json(scene)
    assert json.loads(raw)["domain_template"] == "manipulation_stacking"

    fair = fair_compact_scene(scene)
    payload = json.loads(fair)
    assert "domain_template" not in payload
    assert "domain_template" not in fair
    assert "pose" not in payload["objects"][0]
    assert payload["objects"][0]["name"] == "red_cup"
    assert payload["robot"]["gripper_empty"] is True
