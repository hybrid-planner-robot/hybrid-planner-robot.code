"""Unit tests for StateTracker (mock actions, no ROS)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.init_generator.schema import RobotFacts
from planner.state_tracker import CompletedAction, StateTracker
from vlm.planner import PlanStep


@pytest.fixture
def tracker() -> StateTracker:
    return StateTracker()


def test_default_snapshot_empty_gripper(tracker):
    snap = tracker.snapshot()
    assert snap.gripper_empty is True
    assert snap.holding is None
    assert snap.camera_aimed_at is None
    assert snap.source == "tracker"
    assert snap.confidence == 1.0
    assert tracker.moved_objects() == {}


def test_pick_then_place_sequence(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup", "source": "table"}))
    mid = tracker.snapshot()
    assert mid.gripper_empty is False
    assert mid.holding == "red_cup"
    assert tracker.moved_objects() == {}

    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))
    end = tracker.snapshot()
    assert end.gripper_empty is True
    assert end.holding is None
    assert tracker.moved_objects() == {"red_cup": "shelf"}
    assert tracker.moved_relations() == [("on", "red_cup", "shelf")]


def test_plan_step_pick_place_compatible(tracker):
    tracker.apply(PlanStep(primitive="pick", args={"object": "blue_box"}))
    assert tracker.snapshot().holding == "blue_box"

    tracker.apply(
        PlanStep(primitive="place", args={"object": "blue_box", "location": "table"})
    )
    assert tracker.snapshot().holding is None
    assert tracker.moved_objects()["blue_box"] == "table"


def test_look_at_sets_camera(tracker):
    tracker.apply(CompletedAction("look_at", {"target": "red_cup"}))
    assert tracker.snapshot().camera_aimed_at == "red_cup"
    assert tracker.snapshot().gripper_empty is True

    # PDDL hyphen form
    tracker.apply(CompletedAction("look-at", {"target": "blue_box"}))
    assert tracker.snapshot().camera_aimed_at == "blue_box"


def test_look_at_preserves_holding(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("look_at", {"target": "shelf_marker"}))
    snap = tracker.snapshot()
    assert snap.holding == "red_cup"
    assert snap.gripper_empty is False
    assert snap.camera_aimed_at == "shelf_marker"


def test_pick_clears_prior_moved_record(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "table"}))
    assert tracker.moved_objects() == {"red_cup": "table"}

    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    assert tracker.snapshot().holding == "red_cup"
    assert "red_cup" not in tracker.moved_objects()


def test_unstack_and_stack(tracker):
    tracker.apply(CompletedAction("unstack", {"top": "bowl", "bot": "plate", "l": "table"}))
    assert tracker.snapshot().holding == "bowl"
    assert tracker.snapshot().gripper_empty is False

    tracker.apply(CompletedAction("stack", {"object": "bowl", "bot": "plate"}))
    assert tracker.snapshot().holding is None
    assert tracker.moved_objects() == {"bowl": "plate"}
    assert tracker.moved_relations() == [("stacked-on", "bowl", "plate")]


def test_pick_from_and_place_in_container(tracker):
    tracker.apply(
        CompletedAction("pick-from-container", {"object": "red_cup", "container": "drawer"})
    )
    assert tracker.snapshot().holding == "red_cup"

    tracker.apply(
        CompletedAction(
            "place-in-container",
            {"object": "red_cup", "container": "drawer"},
        )
    )
    assert tracker.snapshot().gripper_empty is True
    assert tracker.moved_relations() == [("in-container", "red_cup", "drawer")]


def test_place_uses_held_object_when_args_omit_object(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"location": "shelf"}))
    assert tracker.moved_objects() == {"red_cup": "shelf"}
    assert tracker.snapshot().holding is None


def test_unknown_primitive_is_noop(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    before = tracker.snapshot()
    tracker.apply(CompletedAction("pour", {"source": "red_cup", "target": "bowl"}))
    after = tracker.snapshot()
    assert after == before
    assert tracker.moved_objects() == {}


def test_reset_clears_state(tracker):
    tracker.apply(CompletedAction("look_at", {"target": "red_cup"}))
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))

    tracker.reset()
    snap = tracker.snapshot()
    assert snap.gripper_empty is True
    assert snap.holding is None
    assert snap.camera_aimed_at is None
    assert tracker.moved_objects() == {}


def test_reset_with_seed():
    seed = RobotFacts(
        gripper_empty=False,
        holding="red_cup",
        camera_aimed_at="red_cup",
        source="manual",
        confidence=0.9,
    )
    tracker = StateTracker(initial=seed)
    snap = tracker.snapshot()
    assert snap.holding == "red_cup"
    assert snap.gripper_empty is False
    assert snap.source == "tracker"
    assert snap.confidence == 0.9

    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "table"}))
    tracker.reset(initial=seed)
    assert tracker.snapshot().holding == "red_cup"
    assert tracker.moved_objects() == {}


def test_as_partial_scene(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))
    partial = tracker.as_partial_scene()

    assert partial["robot"]["gripper_empty"] is True
    assert partial["robot"]["holding"] is None
    assert partial["robot"]["source"] == "tracker"
    assert partial["relations"] == [
        {
            "predicate": "on",
            "args": ["red_cup", "shelf"],
            "source": "tracker",
            "confidence": 1.0,
        }
    ]


def test_multi_object_pick_place_sequence(tracker):
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(CompletedAction("place", {"object": "red_cup", "location": "shelf"}))
    tracker.apply(CompletedAction("pick", {"object": "blue_box"}))
    tracker.apply(CompletedAction("place", {"object": "blue_box", "location": "table"}))

    assert tracker.snapshot().gripper_empty is True
    assert tracker.moved_objects() == {
        "red_cup": "shelf",
        "blue_box": "table",
    }
