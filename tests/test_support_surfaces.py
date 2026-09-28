"""Support-surface inference and oracle goal checking (geometry only, no sim)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.support_surfaces import (
    SupportSurface,
    as_xyz,
    check_goal_facts,
    infer_support_surface,
    load_world_surfaces,
    on_relations,
    surfaces_from_sdf,
)

WORLDS_DIR = (
    Path(__file__).resolve().parent.parent
    / "ros2_ws/src/vlm_robot_planner_bringup/worlds"
)

# Geometry of the shipped tabletop world, so the numbers below are not invented:
#   table   pose (0.5, 0.0, 0.0), surface link at z=0.75, box 0.90 x 0.60 x 0.04
#   shelf_b pose (0.5, -0.25, 0.78), surface box 0.15 x 0.15 x 0.02
TABLE = SupportSurface("table", 0.5, 0.0, 0.77, 0.45, 0.30)
SHELF = SupportSurface("shelf_b", 0.5, -0.25, 0.79, 0.075, 0.075)
SURFACES = [TABLE, SHELF]


# ── as_xyz ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pose",
    [(0.5, -0.25, 0.85), [0.5, -0.25, 0.85], {"x": 0.5, "y": -0.25, "z": 0.85}],
)
def test_as_xyz_accepts_tuple_and_mapping(pose):
    assert as_xyz(pose) == (0.5, -0.25, 0.85)


@pytest.mark.parametrize("pose", [None, {}, {"x": 1.0}, "nope", ()])
def test_as_xyz_rejects_unusable(pose):
    assert as_xyz(pose) is None


# ── infer_support_surface ────────────────────────────────────────────────────


def test_nested_surfaces_pick_the_highest():
    """The shelf sits inside the table footprint, so height must break the tie."""
    # Real pose of red_cup after a successful place onto shelf_b.
    assert infer_support_surface((0.5001, -0.2496, 0.850), SURFACES) == "shelf_b"


def test_item_outside_shelf_footprint_is_on_the_table():
    # Real pose of blue_box: only 0.15 m from the shelf centre, but outside it.
    assert infer_support_surface((0.5, -0.10, 0.80), SURFACES) == "table"


def test_item_held_high_above_every_surface_has_no_support():
    assert infer_support_surface((0.5, 0.0, 1.30), SURFACES) is None
    assert (
        infer_support_surface((0.5, 0.0, 1.30), SURFACES, default="table") == "table"
    )


def test_object_stuck_in_the_gripper_is_not_resting_on_the_surface_below():
    """Real pose of red_cup after a failed place: directly over shelf_b, 0.20 m up.

    A generous clearance bound reported this as `(on red_cup shelf_b)`, which
    would have turned a failed place into a satisfied goal.
    """
    assert infer_support_surface((0.499, -0.251, 0.990), SURFACES) is None
    check = check_goal_facts(
        [["on", "red_cup", "shelf_b"]],
        {"red_cup": {"x": 0.499, "y": -0.251, "z": 0.990}},
        SURFACES,
    )
    assert check.status == "violated"


def test_clearance_bound_still_accepts_a_tall_prop_resting_on_a_surface():
    # Centre 0.10 m above the table top — a 0.20 m tall bottle standing on it.
    assert infer_support_surface((0.5, 0.0, 0.87), SURFACES) == "table"


def test_item_beyond_the_table_footprint_has_no_support():
    assert infer_support_surface((2.0, 2.0, 0.80), SURFACES) is None


def test_item_below_a_surface_is_not_supported_by_it():
    """Something under the shelf must not be reported as resting on it."""
    assert infer_support_surface((0.5, -0.25, 0.60), SURFACES) is None


def test_overhang_within_margin_still_counts():
    # 1 cm past the shelf edge — an object may overhang its surface.
    assert infer_support_surface((0.5, -0.335, 0.82), SURFACES) == "shelf_b"


def test_unusable_pose_returns_default():
    assert infer_support_surface(None, SURFACES, default="table") == "table"


# ── on_relations ─────────────────────────────────────────────────────────────


def test_on_relations_maps_each_item_to_its_surface():
    poses = {
        "red_cup": {"x": 0.5001, "y": -0.2496, "z": 0.850},
        "blue_box": {"x": 0.5, "y": -0.10, "z": 0.80},
        "coke_can": {"x": 0.40, "y": 0.10, "z": 0.83},
    }
    assert on_relations(poses, SURFACES) == {
        "red_cup": "shelf_b",
        "blue_box": "table",
        "coke_can": "table",
    }


def test_on_relations_omits_unsupported_items_without_default():
    poses = {"held": {"x": 0.5, "y": 0.0, "z": 1.30}}
    assert on_relations(poses, SURFACES) == {}
    assert on_relations(poses, SURFACES, default="table") == {"held": "table"}


# ── check_goal_facts ─────────────────────────────────────────────────────────


def test_goal_satisfied_when_object_is_on_the_expected_surface():
    poses = {"red_cup": {"x": 0.5001, "y": -0.2496, "z": 0.850}}
    check = check_goal_facts([["on", "red_cup", "shelf_b"]], poses, SURFACES)
    assert check.status == "satisfied"
    assert check.satisfied and not check.violated


def test_goal_violated_when_object_never_moved():
    """The executor can report a clean run while the object is still on the table."""
    poses = {"red_cup": {"x": 0.30, "y": 0.10, "z": 0.80}}
    check = check_goal_facts([["on", "red_cup", "shelf_b"]], poses, SURFACES)
    assert check.status == "violated"
    assert check.facts[0]["observed"] == "table"


def test_goal_violated_when_object_fell_off_every_surface():
    poses = {"red_cup": {"x": 1.60, "y": 0.90, "z": 0.05}}
    check = check_goal_facts([["on", "red_cup", "shelf_b"]], poses, SURFACES)
    assert check.status == "violated"
    assert check.facts[0]["observed"] is None


def test_all_facts_must_hold():
    poses = {
        "red_cup": {"x": 0.5001, "y": -0.2496, "z": 0.850},
        "blue_box": {"x": 0.5, "y": -0.10, "z": 0.80},
    }
    facts = [["on", "red_cup", "shelf_b"], ["on", "blue_box", "shelf_b"]]
    assert check_goal_facts(facts, poses, SURFACES).status == "violated"


def test_predicates_that_poses_cannot_decide_are_unverifiable():
    for fact in (
        ["holding", "red_cup"],
        ["camera-aimed-at", "red_cup"],
        ["in-container", "pen", "drawer"],
        ["stacked-on", "a", "b"],
    ):
        check = check_goal_facts([fact], {}, SURFACES)
        assert check.status == "unverifiable", fact
        assert not check.satisfied


def test_missing_pose_is_unverifiable_not_a_failure():
    check = check_goal_facts([["on", "ghost", "shelf_b"]], {}, SURFACES)
    assert check.status == "unverifiable"


def test_a_violation_outweighs_unverifiable_facts():
    poses = {"red_cup": {"x": 0.30, "y": 0.10, "z": 0.80}}
    facts = [["holding", "red_cup"], ["on", "red_cup", "shelf_b"]]
    assert check_goal_facts(facts, poses, SURFACES).status == "violated"


def test_no_goal_facts_is_unverifiable():
    assert check_goal_facts([], {}, SURFACES).status == "unverifiable"
    assert check_goal_facts(None, {}, SURFACES).status == "unverifiable"


def test_as_dict_and_summary_line_are_serialisable():
    poses = {"red_cup": {"x": 0.5001, "y": -0.2496, "z": 0.850}}
    check = check_goal_facts([["on", "red_cup", "shelf_b"]], poses, SURFACES)
    payload = check.as_dict()
    assert payload["status"] == "satisfied"
    assert payload["facts"][0]["predicate"] == "on"
    import json

    json.dumps(payload)  # must survive going into summary.json
    assert "on red_cup shelf_b" in check.summary_line()


# ── SDF parsing ──────────────────────────────────────────────────────────────


def test_parses_the_shipped_tabletop_world():
    surfaces = load_world_surfaces(WORLDS_DIR / "tabletop.world")
    assert set(surfaces) == {"table", "shelf_b"}
    assert surfaces["table"].top_z == pytest.approx(0.77)
    assert surfaces["table"].half_x == pytest.approx(0.45)
    assert surfaces["table"].half_y == pytest.approx(0.30)
    assert surfaces["shelf_b"].top_z == pytest.approx(0.79)
    assert surfaces["shelf_b"].center_y == pytest.approx(-0.25)
    assert surfaces["shelf_b"].half_x == pytest.approx(0.075)


def test_tabletop_geometry_reproduces_the_observed_layout():
    """End-to-end: parsed SDF + poses read from Gazebo after a successful place."""
    surfaces = load_world_surfaces(WORLDS_DIR / "tabletop.world")
    poses = {
        "red_cup": {"x": 0.5001, "y": -0.2496, "z": 0.850},
        "blue_box": {"x": 0.5000, "y": -0.1000, "z": 0.800},
    }
    assert on_relations(poses, surfaces.values()) == {
        "red_cup": "shelf_b",
        "blue_box": "table",
    }


def test_models_without_a_surface_link_are_ignored():
    sdf = """<sdf version='1.6'><world name='w'>
      <model name='red_cup'>
        <pose>0.3 0.1 0.8 0 0 0</pose>
        <link name='link'>
          <collision name='c'><geometry><box><size>0.05 0.05 0.1</size></box></geometry></collision>
        </link>
      </model>
      <model name='shelf_b'>
        <pose>0.5 -0.25 0.78 0 0 0</pose>
        <link name='surface'>
          <collision name='c'><geometry><box><size>0.15 0.15 0.02</size></box></geometry></collision>
        </link>
      </model>
    </world></sdf>"""
    assert set(surfaces_from_sdf(sdf)) == {"shelf_b"}


def test_link_pose_offsets_the_surface_height():
    sdf = """<sdf version='1.6'><world name='w'>
      <model name='table'>
        <pose>0.5 0.0 0.0 0 0 0</pose>
        <link name='surface'>
          <pose>0 0 0.75 0 0 0</pose>
          <collision name='c'><geometry><box><size>0.90 0.60 0.04</size></box></geometry></collision>
        </link>
      </model>
    </world></sdf>"""
    surface = surfaces_from_sdf(sdf)["table"]
    assert surface.top_z == pytest.approx(0.77)
    assert surface.center_x == pytest.approx(0.5)


def test_non_box_surface_is_skipped():
    sdf = """<sdf version='1.6'><world name='w'>
      <model name='round_table'>
        <link name='surface'>
          <collision name='c'><geometry><cylinder><radius>0.4</radius></cylinder></geometry></collision>
        </link>
      </model>
    </world></sdf>"""
    assert surfaces_from_sdf(sdf) == {}


def test_malformed_or_missing_input_yields_no_surfaces():
    assert surfaces_from_sdf("<sdf><world>") == {}
    assert surfaces_from_sdf("") == {}
    assert load_world_surfaces(WORLDS_DIR / "does_not_exist.world") == {}
