"""Tests for production oracle/DINO adapters — fixtures only, no Gazebo/GPU."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.init_generator.adapters.dino import DinoAdapter
from planner.problem_generator.init_generator.adapters.oracle import OracleAdapter
from planner.problem_generator.init_generator.schema import SceneStateError
from simulation.oracle.world_state import (
    ObjectState,
    Orientation,
    Pose,
    Position,
    WorldState,
)

_FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent
    / "planner"
    / "problem_generator"
    / "init_generator"
    / "fixtures"
)


def _sample_world_state() -> WorldState:
    return WorldState(
        objects=[
            ObjectState(
                name="red_cup",
                pose=Pose(
                    position=Position(x=0.42, y=-0.05, z=0.82),
                    orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                location="table",
            ),
            ObjectState(
                name="blue_box",
                pose=Pose(
                    position=Position(x=0.38, y=0.12, z=0.82),
                    orientation=Orientation(x=0.0, y=0.0, z=0.707, w=0.707),
                ),
                location="table",
            ),
        ],
        gripper_empty=True,
    )


def test_oracle_adapter_from_world_state():
    scene = OracleAdapter.from_world_state(
        _sample_world_state(),
        known_locations=["table", "shelf"],
    )

    assert scene.meta.sources_used == ["oracle"]
    assert scene.robot.gripper_empty is True
    assert scene.robot.source == "oracle"
    assert scene.robot.confidence == 1.0
    assert {o.name for o in scene.objects} == {"red_cup", "blue_box"}
    assert all(o.pose is not None for o in scene.objects)
    assert ("on", "red_cup", "table") in {
        (r.predicate, r.args[0], r.args[1]) for r in scene.relations
    }
    assert {loc.name for loc in scene.locations} == {"shelf", "table"}


def test_oracle_adapter_load_fixture():
    scene = OracleAdapter.load(_FIXTURES_DIR / "oracle_world_state.json")

    assert scene.frame_id == "panda_link0"
    assert scene.domain_template == "manipulation_base"
    assert len(scene.objects) == 2
    scene.validate()


def test_oracle_adapter_pose_only_world_state():
    """GazeboOracle snapshot without symbolic locations still yields valid scene."""
    ws = WorldState(
        objects=[
            ObjectState(
                name="red_cup",
                pose=Pose(
                    position=Position(x=0.5, y=0.0, z=0.82),
                    orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
        ],
        gripper_empty=True,
    )
    scene = OracleAdapter.from_world_state(ws)

    assert scene.objects[0].location is None
    assert scene.relations == []
    assert scene.objects[0].clear is None
    assert scene.objects[0].pose is not None


def test_dino_adapter_from_detections():
    detections = [
        {"name": "red_cup", "box": [10, 20, 30, 40], "score": 0.87},
        {"name": "blu box", "box": [50, 60, 70, 80], "score": 0.72},
    ]
    scene = DinoAdapter.from_detections(
        detections,
        poses={
            "red_cup": {"x": 0.41, "y": -0.06, "z": 0.81},
            "blu_box": {"x": 0.39, "y": 0.11, "z": 0.83},
        },
        on_surface={"red_cup": "table", "blu_box": "table"},
        known_locations=["table"],
    )

    names = {o.name for o in scene.objects}
    assert names == {"blu_box", "red_cup"}
    assert all(o.source == "dino" for o in scene.objects)
    assert scene.objects[0].pose is not None
    assert ("red_cup", "table") in {
        (r.args[0], r.args[1]) for r in scene.relations if r.predicate == "on"
    }
    assert scene.robot.gripper_empty is True
    assert scene.robot.holding is None
    assert scene.robot.camera_aimed_at is None


def test_dino_adapter_from_detect_output():
    boxes = {
        "red_cup": [[100, 100, 200, 200], [10, 10, 20, 20]],
        "table": [[0, 0, 640, 480]],
    }
    scene = DinoAdapter.from_detect_output(
        boxes,
        scores={"red_cup": 0.9},
        known_locations=["table"],
        on_surface={"red_cup": "table"},
    )

    assert {o.name for o in scene.objects} == {"red_cup"}
    assert any(loc.name == "table" for loc in scene.locations)
    assert "table" not in {o.name for o in scene.objects}


def test_dino_adapter_load_fixture():
    scene = DinoAdapter.load(_FIXTURES_DIR / "dino_detection_payload.json")

    names = {o.name for o in scene.objects}
    assert "red_cup" in names
    assert "blu_box" in names
    assert "shelf" not in names
    assert any(loc.name == "shelf" for loc in scene.locations)
    scene.validate()


def test_dino_adapter_skips_location_detections():
    scene = DinoAdapter.from_detections(
        [{"name": "shelf", "box": [0, 0, 1, 1], "score": 0.6}],
        known_locations=["shelf", "table"],
    )
    assert scene.objects == []
    assert {loc.name for loc in scene.locations} == {"shelf", "table"}


def test_adapter_rejects_wrong_fixture_format(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"format": "unknown"}')
    with pytest.raises(SceneStateError, match="expected format"):
        OracleAdapter.load(bad)
    with pytest.raises(SceneStateError, match="expected format"):
        DinoAdapter.load(bad)


@pytest.mark.integration
def test_oracle_adapter_live_gazebo_smoke():
    """Optional smoke when Gazebo + ROS are available."""
    pytest.importorskip("rclpy")
    pytest.skip("live Gazebo oracle smoke — run manually when sim is up")
