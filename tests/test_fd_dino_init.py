"""Session 25 — FD control builds :init from DINO estimates (offline)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.hybrid_runtime import (
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    SceneSource,
    make_domain_stub_plan,
)

_REPO = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "run_loop_host",
    _REPO / "scripts" / "run_loop_host.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_rlh = importlib.util.module_from_spec(_SPEC)
# Avoid executing docker-side main; load helpers only.
sys.modules["run_loop_host"] = _rlh
_SPEC.loader.exec_module(_rlh)


def test_fd_build_hybrid_problem_dino_only_no_oracle_in_fusion():
    """scene_source=dino + detections → solvable :init; fused_from.oracle=False."""
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.DINO,
        command="place the red cup on the shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
        perception_provenance={
            "perception_only": True,
            "pose_provenance": "dino_raw",
            "shortcuts": {
                "perception_only": True,
                "name_match": [],
                "sim_snap": [],
                "any_oracle_substitute": False,
            },
        },
    )
    stub = make_domain_stub_plan(
        "place the red cup on the shelf", "manipulation_base"
    )
    gazebo = {
        "red_cup": {"x": 0.55, "y": 0.05, "z": 0.82},
        "blue_box": {"x": 0.45, "y": -0.10, "z": 0.82},
        "shelf_b": {"x": 0.70, "y": 0.25, "z": 0.95},
    }
    # panda_link0 estimates with real z (not hardcoded 0.82 world).
    last_dino = {
        "red_cup": {"x": 0.34, "y": 0.04, "z": 0.04},
        "blue_box": {"x": 0.24, "y": -0.11, "z": 0.03},
    }
    detections = [
        {"name": "red_cup", "box": [10, 20, 40, 60], "score": 0.91},
        {"name": "blue_box", "box": [50, 30, 80, 70], "score": 0.88},
    ]
    pddl, ok = _rlh._fd_build_hybrid_problem(
        hybrid_session=session,
        stub_plan=stub,
        gazebo_poses=gazebo,
        dino_detections=detections,
        last_dino_est=last_dino,
        problem_name="test_fd_dino",
        pre_scan_ok=True,
        world_name="tabletop",
        perception_only=True,
    )
    assert ok is True
    assert pddl is not None
    assert "(on red_cup" in pddl
    ff = session.last_scene_compare["fused_from"]
    assert ff["requested"] == "dino"
    assert ff["init_fed_by"] == "dino"
    assert ff["oracle"] is False
    assert ff["dino"] is True
    assert ff["pose_provenance"] == "dino_raw"
    # Poses in SceneState should carry perceived z (world), not hardcoded absence.
    dino_scene = session.last_dino_scene
    assert dino_scene is not None
    cup = next(o for o in dino_scene.objects if o.name == "red_cup")
    assert cup.pose is not None
    # plink0 z=0.04 → world z ≈ 0.81
    assert cup.pose.position.z == pytest.approx(0.81, abs=0.02)


def test_look_at_goal_does_not_seed_camera_aimed_at():
    """Look-at-as-goal must not inherit the pick shortcut (seeded fluent)."""
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.DINO,
        command="look at the wood cube",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
        perception_provenance={
            "perception_only": True,
            "pose_provenance": "dino_raw",
            "shortcuts": {
                "perception_only": True,
                "name_match": [],
                "sim_snap": [],
                "any_oracle_substitute": False,
            },
        },
    )
    stub = make_domain_stub_plan("look at the wood cube", "manipulation_base")
    gazebo = {
        "wood_cube": {"x": 0.55, "y": 0.05, "z": 0.82},
        "red_cup": {"x": 0.45, "y": -0.10, "z": 0.82},
        "shelf_b": {"x": 0.70, "y": 0.25, "z": 0.95},
    }
    last_dino = {
        "wood_cube": {"x": 0.34, "y": 0.04, "z": 0.04},
        "red_cup": {"x": 0.24, "y": -0.11, "z": 0.03},
    }
    detections = [
        {"name": "wood_cube", "box": [10, 20, 40, 60], "score": 0.91},
        {"name": "red_cup", "box": [50, 30, 80, 70], "score": 0.88},
    ]
    pddl, ok = _rlh._fd_build_hybrid_problem(
        hybrid_session=session,
        stub_plan=stub,
        gazebo_poses=gazebo,
        dino_detections=detections,
        last_dino_est=last_dino,
        problem_name="test_fd_lookat",
        pre_scan_ok=True,
        world_name="tabletop",
        perception_only=True,
    )
    assert ok is True
    init = pddl[pddl.index("(:init") : pddl.index("(:goal")]
    assert "(camera-aimed-at wood_cube)" not in init
    assert "(camera-aimed-at wood_cube)" in pddl[pddl.index("(:goal") :]
    assert "(gripper-empty)" in init


def test_place_does_not_seed_camera_aimed_at_after_pre_scan():
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.DINO,
        command="place the wood cube on the shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
        perception_provenance={
            "perception_only": True,
            "pose_provenance": "dino_raw",
            "shortcuts": {
                "perception_only": True,
                "name_match": [],
                "sim_snap": [],
                "any_oracle_substitute": False,
            },
        },
    )
    stub = make_domain_stub_plan(
        "place the wood cube on the shelf", "manipulation_base"
    )
    gazebo = {
        "wood_cube": {"x": 0.55, "y": 0.05, "z": 0.82},
        "shelf_b": {"x": 0.70, "y": 0.25, "z": 0.95},
    }
    last_dino = {"wood_cube": {"x": 0.34, "y": 0.04, "z": 0.04}}
    detections = [{"name": "wood_cube", "box": [10, 20, 40, 60], "score": 0.91}]
    pddl, ok = _rlh._fd_build_hybrid_problem(
        hybrid_session=session,
        stub_plan=stub,
        gazebo_poses=gazebo,
        dino_detections=detections,
        last_dino_est=last_dino,
        problem_name="test_fd_place_seed",
        pre_scan_ok=True,
        world_name="tabletop",
        perception_only=True,
    )
    assert ok is True
    init = pddl[pddl.index("(:init") : pddl.index("(:goal")]
    assert "(camera-aimed-at" not in init


def test_fd_dino_aborts_without_detections():
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        scene_source=SceneSource.DINO,
        command="place red_cup on shelf",
        known_locations=["table", "shelf"],
    )
    stub = make_domain_stub_plan("place red_cup on shelf", "manipulation_base")
    pddl, ok = _rlh._fd_build_hybrid_problem(
        hybrid_session=session,
        stub_plan=stub,
        gazebo_poses={"red_cup": {"x": 0.5, "y": 0.0, "z": 0.82}},
        dino_detections=[],
        last_dino_est={},
        problem_name="empty",
        world_name="tabletop",
        perception_only=True,
    )
    assert ok is False
    assert pddl is None


def test_fd_build_ready_inventory_scene_skips_empty_catalog():
    from planner.problem_generator.init_generator.adapters.mock import OracleMockAdapter

    scene = OracleMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.INVENTORY,
        command="place the red cup on the shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
        perception_provenance={
            "perception_only": True,
            "pose_provenance": "vlm_inventory_dino",
            "shortcuts": {
                "perception_only": True,
                "name_match": [],
                "sim_snap": [],
                "any_oracle_substitute": False,
            },
        },
    )
    stub = make_domain_stub_plan(
        "place the red cup on the shelf", "manipulation_base"
    )
    pddl, ok = _rlh._fd_build_hybrid_problem(
        hybrid_session=session,
        stub_plan=stub,
        gazebo_poses={},
        dino_detections=[],
        last_dino_est={"red_cup": {"x": 0.3, "y": 0.0, "z": 0.04}},
        problem_name="inventory_ready",
        pre_scan_ok=True,
        world_name="household",
        perception_only=True,
        ready_dino_scene=scene,
    )
    assert ok is True
    assert pddl is not None
    assert "(on red_cup" in pddl
    ff = session.last_scene_compare["fused_from"]
    assert ff["requested"] == "inventory"
    assert ff["oracle"] is False
    assert ff["dino"] is True
