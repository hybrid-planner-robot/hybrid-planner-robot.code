"""Session 14 regressions: noisy / missing DINO still fuse safely into hybrid."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.hybrid_runtime import (
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    scene_compare_snapshot,
    scene_from_dino_payload,
)
from planner.problem_generator.init_generator.adapters.oracle import OracleAdapter
from planner.problem_generator.init_generator.builder import build_scene
from planner.state_verifier import StateVerifier
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


def test_empty_dino_detections_return_none_and_oracle_only_hybrid():
    assert scene_from_dino_payload([]) is None
    assert scene_from_dino_payload(None) is None

    oracle = OracleAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command="place red_cup on shelf",
        known_locations=["table", "shelf"],
    )
    pddl, fused = session.generate_hybrid_problem(
        _place_plan(),
        oracle_scene=oracle,
        dino_scene=None,
        problem_name="empty_dino",
    )
    assert "(on red_cup table)" in pddl
    assert "dino" not in (fused.meta.sources_used if fused.meta else [])
    assert session.last_scene_compare is not None
    assert session.last_scene_compare["dino"] is None
    assert session.last_scene_compare["oracle"]["objects"]


def test_noisy_dino_names_fuse_without_dropping_oracle():
    oracle = OracleAdapter.load()
    # Noisy label → blu_box; oracle has blue_box.
    dino = scene_from_dino_payload(
        [
            {"name": "red cup", "box": [1, 2, 3, 4], "score": 0.9},
            {"name": "blu box", "box": [5, 6, 7, 8], "score": 0.7},
        ],
        poses={
            "red_cup": {"x": 0.4, "y": -0.05, "z": 0.82},
            "blu_box": {"x": 0.38, "y": 0.12, "z": 0.82},
        },
        on_surface={"red_cup": "table", "blu_box": "table"},
        known_locations=["table", "shelf"],
    )
    assert dino is not None

    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command="place red_cup on shelf",
        known_locations=["table", "shelf"],
    )
    pddl, fused = session.generate_hybrid_problem(
        _place_plan(),
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="noisy_dino",
    )

    names = {o.name for o in fused.objects}
    assert "red_cup" in names
    assert "blue_box" in names  # oracle retained
    assert "blu_box" in names  # noisy DINO kept alongside
    assert "(on red_cup table)" in pddl
    notes = " ".join(fused.meta.fusion_notes)
    assert "name mismatch" in notes
    compare = scene_compare_snapshot(oracle, dino, fused)
    assert "blu_box" in (compare["dino"] or {})["objects"]
    assert "blue_box" in (compare["oracle"] or {})["objects"]
    assert session.metrics_snapshot().get("scene_compare") is not None


def test_partial_dino_keeps_oracle_only_object():
    oracle = OracleAdapter.load()
    dino = scene_from_dino_payload(
        [{"name": "red_cup", "box": [0, 0, 10, 10], "score": 0.8}],
        poses={"red_cup": {"x": 0.42, "y": -0.05, "z": 0.82}},
        on_surface={"red_cup": "table"},
        known_locations=["table", "shelf"],
    )
    fused = build_scene(oracle_scene=oracle, dino_scene=dino, sim_like=True)
    names = {o.name for o in fused.objects}
    assert "red_cup" in names
    assert "blue_box" in names  # missing from DINO, kept from oracle
    assert any("oracle-only" in n for n in fused.meta.fusion_notes)


def test_mvp_noisy_dino_does_not_force_false_green_path():
    """Session 14 stays on MVP (no verifier). Full-mode verify prefers YELLOW over false GREEN."""
    from planner.primitive_transitions import CompletedAction

    oracle = OracleAdapter.load()
    dino = scene_from_dino_payload(
        [{"name": "blu box", "box": [0, 0, 1, 1], "score": 0.5}],
        known_locations=["table", "shelf"],
        on_surface={"blu_box": "table"},
    )
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place red_cup on shelf",
    )
    _, fused = session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, dino_scene=dino
    )
    assert session.verdict_counts == {"GREEN": 0, "YELLOW": 0, "RED": 0}

    pick = CompletedAction("pick", {"object": "red_cup"}, success_flag=True)
    # Observed still matches pre-pick scene → cannot be GREEN vs pick expectation.
    result = StateVerifier().verify(fused, pick, fused)
    assert result.verdict in {"YELLOW", "RED"}
    assert result.verdict != "GREEN"
