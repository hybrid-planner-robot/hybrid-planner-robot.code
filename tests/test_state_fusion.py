"""Tests for deterministic SceneState fusion (Session 7)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.init_generator.adapters.dino import DinoAdapter
from planner.problem_generator.init_generator.adapters.mock import (
    DinoMockAdapter,
    OracleMockAdapter,
)
from planner.problem_generator.init_generator.adapters.oracle import OracleAdapter
from planner.problem_generator.init_generator.builder import InitBuilder, build_scene
from planner.problem_generator.init_generator.fusion import FusionEngine
from planner.problem_generator.init_generator.renderer import InitRenderer
from planner.problem_generator.init_generator.schema import (
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)
from planner.state_tracker import CompletedAction, StateTracker


@pytest.fixture
def oracle_scene():
    return OracleMockAdapter.load()


@pytest.fixture
def dino_scene():
    return DinoMockAdapter.load()


@pytest.fixture
def fusion() -> FusionEngine:
    return FusionEngine()


def _on_pairs(scene: SceneState) -> set[tuple[str, str]]:
    return {
        (r.args[0], r.args[1])
        for r in scene.relations
        if r.predicate == "on" and len(r.args) >= 2
    }


def test_tracker_wins_robot_facts_over_dino(oracle_scene, dino_scene, fusion):
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))

    scene = fusion.merge(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
    )

    assert scene.robot.source == "tracker"
    assert scene.robot.holding == "red_cup"
    assert scene.robot.gripper_empty is False
    assert dino_scene.robot.gripper_empty is True


def test_oracle_wins_on_relation_over_dino(oracle_scene, dino_scene, fusion):
    scene = fusion.merge(oracle_scene=oracle_scene, dino_scene=dino_scene)

    red_on = next(
        r for r in scene.relations if r.predicate == "on" and r.args[0] == "red_cup"
    )
    assert red_on.source == "oracle"
    assert red_on.args == ["red_cup", "table"]
    assert ("on", "red_cup", "table") in {
        (r.predicate, r.args[0], r.args[1]) for r in scene.relations
    }


def test_tracker_moved_objects_after_pick_place(oracle_scene, dino_scene, fusion):
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))

    scene = fusion.merge(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
    )

    assert scene.robot.gripper_empty is True
    assert scene.robot.holding is None
    assert ("red_cup", "shelf") in _on_pairs(scene)
    assert ("red_cup", "table") not in _on_pairs(scene)
    assert any("tracker override" in n for n in scene.meta.fusion_notes)


def test_name_mismatch_recorded_in_fusion_notes(oracle_scene, dino_scene, fusion):
    scene = fusion.merge(oracle_scene=oracle_scene, dino_scene=dino_scene)

    object_names = {o.name for o in scene.objects}
    assert "blue_box" in object_names
    assert "blu_box" in object_names

    notes = " ".join(scene.meta.fusion_notes)
    assert "name mismatch" in notes
    assert "blu_box" in notes or "blue_box" in notes


def test_missing_detection_keeps_oracle_only_object(fusion):
    oracle = SceneState(
        objects=[
            ObjectFact(
                name="red_cup",
                source="oracle",
                confidence=1.0,
                location="table",
            ),
            ObjectFact(
                name="green_bottle",
                source="oracle",
                confidence=1.0,
                location="table",
            ),
        ],
        locations=[],
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="oracle",
                confidence=1.0,
            ),
            RelationFact(
                predicate="on",
                args=["green_bottle", "table"],
                source="oracle",
                confidence=1.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="oracle",
            confidence=1.0,
        ),
    )
    dino = SceneState(
        objects=[
            ObjectFact(
                name="red_cup",
                source="dino",
                confidence=0.8,
                location="table",
            ),
        ],
        locations=[],
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="dino",
                confidence=0.8,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="dino",
            confidence=0.8,
        ),
    )

    scene = fusion.merge(oracle_scene=oracle, dino_scene=dino)
    assert {o.name for o in scene.objects} == {"green_bottle", "red_cup"}
    assert any("oracle-only" in n for n in scene.meta.fusion_notes)


def test_build_scene_with_tracker_moved_relations(oracle_scene, dino_scene):
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))

    scene = build_scene(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
    )

    assert "tracker" in scene.meta.sources_used
    assert "fusion" in scene.meta.sources_used
    assert ("red_cup", "shelf") in _on_pairs(scene)


def test_init_builder_end_to_end_init_facts(oracle_scene, dino_scene):
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))

    builder = InitBuilder()
    facts = builder.build_init(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
    )

    assert ("on", "red_cup", "shelf") in facts
    assert ("on", "red_cup", "table") not in facts
    assert ("gripper-empty",) in facts
    assert not any(f[0] == "holding" for f in facts)


def test_init_renderer_end_to_end_from_build_scene(oracle_scene, dino_scene):
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))

    scene = build_scene(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
    )
    facts = InitRenderer().render_facts(scene)

    assert ("on", "red_cup", "shelf") in facts
    section = InitRenderer().render_section(scene)
    assert "(on red_cup shelf)" in section


def test_merge_requires_input(fusion):
    with pytest.raises(ValueError, match="at least one"):
        fusion.merge()


def test_tracker_holding_removes_on_relation(oracle_scene, fusion):
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))

    scene = fusion.merge(oracle_scene=oracle_scene, tracker=tracker)

    assert scene.robot.holding == "red_cup"
    assert ("red_cup", "table") not in _on_pairs(scene)
    assert any("removed" in n and "red_cup" in n for n in scene.meta.fusion_notes)


def test_build_scene_production_adapters_with_tracker():
    """Real-shaped oracle/DINO fixture payloads through build_scene + StateTracker."""
    oracle_scene = OracleAdapter.load()
    dino_scene = DinoAdapter.load()

    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))

    scene = build_scene(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
    )

    assert "oracle" in scene.meta.sources_used
    assert "dino" in scene.meta.sources_used
    assert "tracker" in scene.meta.sources_used
    assert ("red_cup", "shelf") in _on_pairs(scene)
    assert ("red_cup", "table") not in _on_pairs(scene)
    assert scene.robot.gripper_empty is True

    object_names = {o.name for o in scene.objects}
    assert "blue_box" in object_names
    assert "blu_box" in object_names
    assert any("name mismatch" in n for n in scene.meta.fusion_notes)
