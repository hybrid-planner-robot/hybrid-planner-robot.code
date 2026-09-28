"""Tests for SceneState schema load/save and validation."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.init_generator.schema import (
    LocationFact,
    ObjectFact,
    Orientation,
    Pose,
    Position,
    RelationFact,
    RobotFacts,
    SceneState,
    SceneStateError,
)


def _minimal_scene(**overrides) -> SceneState:
    robot = overrides.pop("robot", None) or RobotFacts(
        gripper_empty=True,
        holding=None,
        camera_aimed_at=None,
        source="mock",
        confidence=1.0,
    )
    return SceneState(
        objects=overrides.pop(
            "objects",
            [
                ObjectFact(
                    name="red_cup",
                    source="mock",
                    confidence=1.0,
                    clear=True,
                )
            ],
        ),
        locations=overrides.pop(
            "locations",
            [
                LocationFact(
                    name="table",
                    source="mock",
                    confidence=1.0,
                )
            ],
        ),
        relations=overrides.pop(
            "relations",
            [
                RelationFact(
                    predicate="on",
                    args=["red_cup", "table"],
                    source="mock",
                    confidence=1.0,
                )
            ],
        ),
        robot=robot,
        **overrides,
    )


def test_from_dict_minimal_example():
    scene = SceneState.from_dict(
        {
            "schema_version": "1.0",
            "objects": [
                {
                    "name": "red_cup",
                    "type": "item",
                    "clear": True,
                    "reachable": True,
                    "source": "mock",
                    "confidence": 1.0,
                }
            ],
            "locations": [
                {
                    "name": "table",
                    "type": "location",
                    "reachable": True,
                    "source": "mock",
                    "confidence": 1.0,
                }
            ],
            "relations": [
                {
                    "predicate": "on",
                    "args": ["red_cup", "table"],
                    "source": "mock",
                    "confidence": 1.0,
                }
            ],
            "robot": {
                "gripper_empty": True,
                "holding": None,
                "camera_aimed_at": None,
                "source": "mock",
                "confidence": 1.0,
            },
        }
    )
    assert scene.objects[0].name == "red_cup"
    assert scene.relations[0].predicate == "on"
    assert scene.robot.gripper_empty is True


def test_round_trip_json():
    scene = _minimal_scene(timestamp="2026-07-20T12:00:00Z")
    restored = SceneState.from_json(scene.to_json())
    assert restored.to_dict() == scene.to_dict()


def test_save_and_load_json_file():
    scene = _minimal_scene()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.json"
        scene.save_json(path)
        loaded = SceneState.load_json(path)
        assert loaded.objects[0].name == "red_cup"
        assert json.loads(path.read_text())["schema_version"] == "1.0"


def test_holding_invariant_valid():
    scene = _minimal_scene(
        relations=[],
        robot=RobotFacts(
            gripper_empty=False,
            holding="red_cup",
            camera_aimed_at="red_cup",
            source="tracker",
            confidence=1.0,
        ),
    )
    scene.validate()


def test_holding_invariant_rejects_empty_gripper_with_holding():
    scene = _minimal_scene(
        robot=RobotFacts(
            gripper_empty=True,
            holding="red_cup",
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    with pytest.raises(SceneStateError, match="gripper_empty"):
        scene.validate()


def test_holding_invariant_rejects_full_gripper_without_holding():
    scene = _minimal_scene(
        robot=RobotFacts(
            gripper_empty=False,
            holding=None,
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    with pytest.raises(SceneStateError, match="holding is null"):
        scene.validate()


def test_rejects_invalid_confidence():
    scene = _minimal_scene(
        objects=[
            ObjectFact(name="cup", source="mock", confidence=1.5),
        ],
    )
    with pytest.raises(SceneStateError, match="confidence"):
        scene.validate()


def test_rejects_missing_schema_version():
    with pytest.raises(SceneStateError, match="schema_version"):
        SceneState.from_dict({"objects": [], "locations": [], "relations": [], "robot": {}})


def test_pose_serialization():
    scene = _minimal_scene(
        objects=[
            ObjectFact(
                name="red_cup",
                source="oracle",
                confidence=1.0,
                pose=Pose(
                    position=Position(x=0.4, y=0.0, z=0.82),
                    orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
        ],
    )
    data = scene.to_dict()
    assert data["objects"][0]["pose"]["position"]["x"] == 0.4
