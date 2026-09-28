"""Mid-task hybrid regression fixtures (Session 12)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator import generate_problem
from planner.problem_generator.init_generator.renderer import InitRenderer
from planner.problem_generator.init_generator.schema import RobotFacts, SceneState
from planner.state_tracker import CompletedAction, StateTracker
from planner.state_verifier import StateVerifier
from vlm.planner import PlanStep, VLMPlan

_MID_TASK = (
    Path(__file__).resolve().parent.parent
    / "planner"
    / "problem_generator"
    / "init_generator"
    / "mock"
    / "mid_task"
)


def _load(name: str) -> SceneState:
    return SceneState.load_json(_MID_TASK / name)


@pytest.mark.parametrize(
    "fixture,expect_holding,expect_camera,expect_rel",
    [
        ("holding_after_pick.json", "red_cup", None, None),
        ("camera_aimed_at.json", None, "red_cup", ("on", "red_cup", "table")),
        ("after_place_on_shelf.json", None, None, ("on", "red_cup", "shelf")),
        (
            "stacking_after_stack.json",
            None,
            None,
            ("stacked-on", "blue_block", "red_block"),
        ),
        (
            "container_after_place_in.json",
            None,
            None,
            ("in-container", "red_cup", "box"),
        ),
    ],
)
def test_mid_task_fixture_loads_and_renders(
    fixture, expect_holding, expect_camera, expect_rel
):
    scene = _load(fixture)
    assert scene.robot.holding == expect_holding
    assert scene.robot.camera_aimed_at == expect_camera
    if expect_holding:
        assert scene.robot.gripper_empty is False
    else:
        assert scene.robot.gripper_empty is True

    facts = InitRenderer().render_facts(scene)
    if expect_holding:
        assert ("holding", expect_holding) in facts
    if expect_camera:
        assert ("camera-aimed-at", expect_camera) not in facts
    if expect_rel:
        assert expect_rel in facts
    assert not any(f[0] == "camera-aimed-at" for f in facts)


def test_pick_place_sequence_tracker_matches_after_place_fixture():
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
    tracker.apply(
        CompletedAction("place", {"object": "red_cup", "location": "shelf"})
    )
    assert tracker.snapshot().holding is None
    assert tracker.moved_objects()["red_cup"] == "shelf"

    fixture = _load("after_place_on_shelf.json")
    assert fixture.robot.holding is None
    assert any(
        r.predicate == "on" and r.args == ["red_cup", "shelf"]
        for r in fixture.relations
    )


def test_look_at_then_hybrid_goal_not_init():
    scene = _load("camera_aimed_at.json")
    plan = VLMPlan(
        goal="look at red_cup",
        steps=[PlanStep(primitive="look_at", args={"object": "red_cup"})],
        raw_output="",
        domain_template="manipulation_base",
    )
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("camera-aimed-at", "red_cup")],
        use_hybrid=True,
        problem_name="mid_look",
    )
    init = pddl[pddl.index("(:init") : pddl.index("(:goal")]
    assert "(camera-aimed-at red_cup)" not in init
    assert "(camera-aimed-at red_cup)" in pddl[pddl.index("(:goal") :]


def test_holding_mid_task_hybrid_init():
    scene = _load("holding_after_pick.json")
    plan = VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": "shelf"},
            )
        ],
        raw_output="",
        domain_template="manipulation_base",
    )
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("on", "red_cup", "shelf")],
        use_hybrid=True,
        problem_name="mid_hold",
    )
    assert "(holding red_cup)" in pddl
    assert "(on red_cup shelf)" in pddl


def test_stacking_and_container_fixtures_domain_templates():
    stack = _load("stacking_after_stack.json")
    cont = _load("container_after_place_in.json")
    assert stack.domain_template == "manipulation_stacking"
    assert cont.domain_template == "containers_manipulation"
    assert any(r.predicate == "stacked-on" for r in stack.relations)
    assert any(r.predicate == "in-container" for r in cont.relations)


def test_verifier_expect_stack_and_place_in_container():
    pre_stack = SceneState(
        objects=[],
        locations=[],
        relations=[],
        robot=RobotFacts(
            gripper_empty=False,
            holding="blue_block",
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    stacked = StateVerifier().expect(
        pre_stack,
        CompletedAction(
            "stack", {"object": "blue_block", "bottom": "red_block"}
        ),
    )
    assert any(
        r.predicate == "stacked-on" and r.args == ["blue_block", "red_block"]
        for r in stacked.relations
    )

    pre_box = SceneState(
        objects=[],
        locations=[],
        relations=[],
        robot=RobotFacts(
            gripper_empty=False,
            holding="red_cup",
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    placed = StateVerifier().expect(
        pre_box,
        CompletedAction(
            "place-in-container", {"object": "red_cup", "container": "box"}
        ),
    )
    assert any(
        r.predicate == "in-container" and r.args == ["red_cup", "box"]
        for r in placed.relations
    )


def test_verifier_thresholds_prefer_yellow_band():
    """Defaults stay conservative: mid-band mismatch is YELLOW, not GREEN."""
    v = StateVerifier()
    assert v.green_threshold == 0.15
    assert v.red_threshold == 0.60
    # Synthetic: relation-only mismatch weight = 0.20 → YELLOW band
    assert v.green_threshold < 0.20 <= v.red_threshold
