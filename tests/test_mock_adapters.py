"""Tests for mock adapters (oracle, DINO, tracker) — no ROS/Gazebo/GPU."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.init_generator.adapters.mock import (
    DinoMockAdapter,
    OracleMockAdapter,
    TrackerMockAdapter,
)
from planner.problem_generator.init_generator.schema import SceneStateError

_MOCK_DIR = (
    Path(__file__).resolve().parent.parent
    / "planner"
    / "problem_generator"
    / "init_generator"
    / "mock"
)


def test_oracle_mock_default_fixture():
    scene = OracleMockAdapter.load()
    assert scene.meta.sources_used == ["oracle"]
    assert scene.robot.gripper_empty is True
    assert {o.name for o in scene.objects} == {"red_cup", "blue_box"}
    assert ("on", "red_cup", "table") in {
        (r.predicate, r.args[0], r.args[1]) for r in scene.relations
    }
    assert all(o.source == "oracle" and o.confidence == 1.0 for o in scene.objects)


def test_oracle_mock_from_file():
    scene = OracleMockAdapter.load(_MOCK_DIR / "oracle_table_scene.json")
    assert len(scene.locations) == 2
    assert scene.frame_id == "panda_link0"


def test_dino_mock_normalizes_noisy_names():
    scene = DinoMockAdapter.load()
    names = {o.name for o in scene.objects}
    assert "red_cup" in names
    assert "blu_box" in names
    assert all(o.source == "dino" for o in scene.objects)
    assert all(o.confidence < 1.0 for o in scene.objects)


def test_dino_mock_skips_location_detections():
    scene = DinoMockAdapter.load()
    object_names = {o.name for o in scene.objects}
    assert "shelf" not in object_names
    assert any(loc.name == "shelf" for loc in scene.locations)


def test_dino_mock_on_relations():
    scene = DinoMockAdapter.load()
    on_pairs = {(r.args[0], r.args[1]) for r in scene.relations if r.predicate == "on"}
    assert ("red_cup", "table") in on_pairs
    assert ("blu_box", "table") in on_pairs


def test_tracker_mock_holding_scene():
    scene = TrackerMockAdapter.load()
    assert scene.robot.gripper_empty is False
    assert scene.robot.holding == "red_cup"
    assert scene.robot.camera_aimed_at == "red_cup"
    assert scene.robot.source == "tracker"
    assert scene.relations == []
    assert scene.objects[0].name == "red_cup"


def test_tracker_mock_from_file():
    scene = TrackerMockAdapter.load(_MOCK_DIR / "holding_mid_task.json")
    assert "tracker" in scene.meta.sources_used


def test_adapter_rejects_wrong_format(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "unknown", "objects": []}')
    with pytest.raises(SceneStateError, match="expected format"):
        OracleMockAdapter.load(bad)


def test_all_default_fixtures_validate():
    for adapter in (OracleMockAdapter, DinoMockAdapter, TrackerMockAdapter):
        scene = adapter.load()
        scene.validate()
