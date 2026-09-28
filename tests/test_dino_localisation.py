"""Session 25 — DINO localisation helpers + fusion provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.dino_localisation import (
    SNAP_RADIUS_M,
    build_localisation_report,
    catalog_prop_names,
    extend_fused_from_provenance,
    fuzzy_gazebo_name_match,
    init_fed_by_label,
    nearest_gazebo_snap,
    plink0_to_world,
    pose_provenance_label,
    poses_for_init_from_estimates,
    refine_z_with_height,
    summarize_shortcuts,
    world_to_plink0,
)
from planner.hybrid_runtime import (
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    SceneSource,
    scene_from_dino_payload,
    scene_from_xyz_poses,
)
from vlm.planner import PlanStep, VLMPlan


def _place_plan() -> VLMPlan:
    return VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": "shelf"},
            ),
        ],
        raw_output="",
        domain_template="manipulation_base",
    )


def test_frame_roundtrip():
    w = plink0_to_world(0.30, -0.05, 0.025)
    assert w["x"] == pytest.approx(0.50)
    assert w["y"] == pytest.approx(-0.05)
    assert w["z"] == pytest.approx(0.795)
    p = world_to_plink0(w["x"], w["y"], w["z"])
    assert p["x"] == pytest.approx(0.30)
    assert p["z"] == pytest.approx(0.025)


def test_catalog_prefers_gazebo_items_over_locations():
    names = catalog_prop_names(
        gazebo_poses={
            "red_cup": {"x": 0.5, "y": 0.0, "z": 0.8},
            "shelf_b": {"x": 0.7, "y": 0.2, "z": 0.9},
            "table": {"x": 0.6, "y": 0.0, "z": 0.77},
        },
        world_props=["red_cup", "blue_box", "shelf_b"],
        location_models={"shelf_b", "table", "shelf"},
        infra={"floor", "sun"},
        held=None,
    )
    assert names == ["red_cup"]


def test_catalog_falls_back_to_world_props():
    names = catalog_prop_names(
        gazebo_poses={},
        world_props=["red_cup", "blue_box", "shelf_b"],
        location_models={"shelf_b"},
        held="blue_box",
    )
    assert names == ["red_cup"]


def test_fuzzy_namematch_unique_only():
    gz = {"red_cup": {}, "blue_box": {}}
    assert fuzzy_gazebo_name_match("cup", gz) == "red_cup"
    assert fuzzy_gazebo_name_match("box", gz) == "blue_box"
    assert fuzzy_gazebo_name_match("blue", gz) == "blue_box"
    assert fuzzy_gazebo_name_match("red_cup", gz) is None  # exact key → no NameMatch
    # Ambiguous substring → no match
    gz2 = {"red_cup": {}, "tea_cup": {}}
    assert fuzzy_gazebo_name_match("cup", gz2) is None


def test_nearest_snap_within_radius():
    gz = {
        "red_cup": {"x": 0.50, "y": 0.0, "z": 0.80},  # plink0 ≈ (0.30, 0.0)
        "blue_box": {"x": 0.80, "y": 0.3, "z": 0.80},
    }
    hit = nearest_gazebo_snap(0.31, 0.01, gz, radius_m=SNAP_RADIUS_M)
    assert hit is not None
    name, dist, pose = hit
    assert name == "red_cup"
    assert dist < 0.05
    assert pose["x"] == pytest.approx(0.30)
    miss = nearest_gazebo_snap(1.0, 1.0, gz, radius_m=SNAP_RADIUS_M)
    assert miss is None


def test_nearest_snap_excludes_locations():
    # Query near shelf_b; without exclude the furniture would win.
    gz = {
        "red_cup": {"x": 0.50, "y": 0.0, "z": 0.80},       # plink0 (0.30, 0)
        "shelf_b": {"x": 0.52, "y": 0.02, "z": 0.95},       # plink0 (0.32, 0.02)
    }
    hit_all = nearest_gazebo_snap(0.325, 0.02, gz)
    assert hit_all is not None and hit_all[0] == "shelf_b"
    hit = nearest_gazebo_snap(
        0.325, 0.02, gz, exclude={"shelf_b", "table"}
    )
    assert hit is not None and hit[0] == "red_cup"


def test_localisation_report_delta_cm():
    dino = {"red_cup": {"x": 0.30, "y": 0.0, "z": 0.025}}  # plink0
    oracle = {"red_cup": {"x": 0.52, "y": 0.0, "z": 0.80}}  # world
    rows = build_localisation_report(dino, oracle)
    assert len(rows) == 1
    row = rows[0]
    assert row["has_both"] is True
    # world dino = (0.50, 0, 0.795); oracle (0.52, 0, 0.80) → Δxy=2cm, Δz≈-0.5cm
    assert row["delta_xy_cm"] == pytest.approx(2.0, abs=0.05)
    assert row["delta_z_cm"] == pytest.approx(-0.5, abs=0.05)


def test_localisation_report_only_names():
    dino = {"red_cup": {"x": 0.30, "y": 0.0, "z": 0.025}}
    oracle = {
        "red_cup": {"x": 0.50, "y": 0.0, "z": 0.80},
        "blue_box": {"x": 0.40, "y": 0.1, "z": 0.80},
        "shelf_b": {"x": 0.70, "y": 0.2, "z": 0.95},
    }
    rows = build_localisation_report(
        dino, oracle, only_names=["red_cup"]
    )
    assert [r["name"] for r in rows] == ["red_cup"]
    assert rows[0]["has_both"] is True


def test_poses_for_init_real_z_and_legacy_tuple():
    modern = poses_for_init_from_estimates(
        {"red_cup": {"x": 0.3, "y": 0.0, "z": 0.04}},
        frame="plink0",
    )
    assert modern["red_cup"]["z"] == pytest.approx(0.04)
    legacy = poses_for_init_from_estimates(
        {"red_cup": (0.3, 0.0)},
        frame="plink0",
    )
    # fallback world 0.82 → plink0 z = 0.05
    assert legacy["red_cup"]["z"] == pytest.approx(0.05)


def test_refine_z_with_height_only_without_depth():
    assert refine_z_with_height(0.025, 0.10, used_depth=True) == pytest.approx(0.025)
    assert refine_z_with_height(0.025, 0.10, used_depth=False) == pytest.approx(0.075)


def test_pose_provenance_labels():
    assert (
        pose_provenance_label(
            perception_only=True, name_match=[], sim_snap=[], had_dino=True
        )
        == "dino_raw"
    )
    assert (
        pose_provenance_label(
            perception_only=False,
            name_match=["cup"],
            sim_snap=[],
            had_dino=True,
        )
        == "name_match"
    )
    assert (
        pose_provenance_label(
            perception_only=False,
            name_match=[],
            sim_snap=["red_cup"],
            had_dino=True,
        )
        == "dino_snapped"
    )


def test_fused_from_requested_vs_init_fed_by():
    oracle = scene_from_xyz_poses(
        {"red_cup": {"x": 0.5, "y": 0.0, "z": 0.82}},
        known_locations=["table", "shelf"],
        on_surface={"red_cup": "table"},
    )
    dino = scene_from_dino_payload(
        [{"name": "red_cup", "box": [1, 2, 3, 4], "score": 0.9}],
        poses={"red_cup": {"x": 0.48, "y": 0.01, "z": 0.81}},
        on_surface={"red_cup": "table"},
        known_locations=["table", "shelf"],
    )
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.DINO,
        command="place red_cup on shelf",
        known_locations=["table", "shelf"],
    )
    session.perception_provenance = {
        "perception_only": True,
        "shortcuts": summarize_shortcuts(perception_only=True),
        "pose_provenance": "dino_raw",
    }
    _, fused = session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, dino_scene=dino
    )
    ff = session.last_scene_compare["fused_from"]
    assert ff["requested"] == "dino"
    assert ff["init_fed_by"] == "dino"
    assert ff["oracle"] is False
    assert ff["dino"] is True
    assert ff["perception_only"] is True
    assert ff["pose_provenance"] == "dino_raw"
    assert "oracle" not in (fused.meta.sources_used if fused.meta else [])
    snap = session.metrics_snapshot()
    assert snap["perception_provenance"]["pose_provenance"] == "dino_raw"


def test_extend_fused_from_helper():
    base = {"oracle": True, "dino": True, "sim_like": True}
    out = extend_fused_from_provenance(
        base,
        requested="fused",
        perception_only=False,
        shortcuts=summarize_shortcuts(
            perception_only=False, sim_snap=["red_cup"]
        ),
        pose_provenance="dino_snapped",
    )
    assert out["requested"] == "fused"
    assert out["init_fed_by"] == "fused"
    assert out["shortcuts"]["sim_snap_count"] == 1
    assert init_fed_by_label(oracle=False, dino=True) == "dino"
