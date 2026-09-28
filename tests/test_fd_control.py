"""Session 22 — FD control mode (VLM out of the action loop)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.fast_downward import result_from_actions
from planner.hybrid_runtime import (
    ENV_CONTROL,
    ControlMode,
    HybridMode,
    HybridProblemSession,
    IncompatibleControlHybridError,
    assert_control_hybrid_compatible,
    command_is_look_at_goal,
    control_hybrid_incompatibility,
    make_domain_stub_plan,
    plan_fd_from_problem,
    resolve_control_mode,
    scene_from_xyz_poses,
    select_domain_template,
)
from planner.pipeline import Pipeline
from planner.plan_parser import PrimitiveCall
from vlm.planner import PlanStep, VLMPlan


DOMAINS_DIR = Path(__file__).resolve().parent.parent / "pddl" / "domains"


class _StubFD:
    def __init__(self, actions: list[str] | None):
        self._actions = actions
        self.call_count = 0
        self.last_problem: str | None = None
        self.last_domain: str | None = None

    def solve_from_strings(self, domain_text: str, problem_text: str):
        self.call_count += 1
        self.last_domain = domain_text
        self.last_problem = problem_text
        return self._actions


class _SpyVLM:
    """Must never be called on the FD control path."""

    def __init__(self):
        self.plan_calls = 0

    def plan(self, command: str, images: list) -> VLMPlan:
        self.plan_calls += 1
        raise AssertionError("vision VLM must not plan actions under control=fd")


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, ControlMode.VLM_STEPS),
        ("", ControlMode.VLM_STEPS),
        ("vlm_steps", ControlMode.VLM_STEPS),
        ("vlm", ControlMode.VLM_STEPS),
        ("fd", ControlMode.FD),
        ("FD", ControlMode.FD),
        ("fast_downward", ControlMode.FD),
        ("fast-downward", ControlMode.FD),
        ("pddl", ControlMode.FD),
    ],
)
def test_resolve_control_mode(raw, expected):
    assert resolve_control_mode(raw) == expected
    if raw is None:
        assert resolve_control_mode(env={}) == ControlMode.VLM_STEPS
        assert resolve_control_mode(env={ENV_CONTROL: "fd"}) == ControlMode.FD


def test_control_fd_refuses_hybrid_full():
    """Session 23: full+fd would load an unused verifier VLM — refuse explicitly."""
    msg = control_hybrid_incompatibility(ControlMode.FD, HybridMode.FULL)
    assert msg is not None
    assert "incompatible" in msg.lower()
    assert "--hybrid mvp" in msg
    with pytest.raises(IncompatibleControlHybridError, match="incompatible"):
        assert_control_hybrid_compatible(ControlMode.FD, HybridMode.FULL)


@pytest.mark.parametrize(
    "control,hybrid",
    [
        (ControlMode.FD, HybridMode.MVP),
        (ControlMode.FD, HybridMode.OFF),
        (ControlMode.VLM_STEPS, HybridMode.FULL),
        (ControlMode.VLM_STEPS, HybridMode.MVP),
        (ControlMode.VLM_STEPS, HybridMode.OFF),
    ],
)
def test_control_hybrid_compatible_combos(control, hybrid):
    assert control_hybrid_incompatibility(control, hybrid) is None
    assert_control_hybrid_compatible(control, hybrid)


@pytest.mark.parametrize(
    "command,expected",
    [
        ("place red_cup on shelf", "manipulation_base"),
        ("pick the mug", "manipulation_base"),
        ("stack blue_block on red_block", "manipulation_stacking"),
        ("put pen into the drawer", "containers_manipulation"),
        ("place phone in the box", "containers_manipulation"),
        ("pour the can into the glass", "manipulation_base"),
        ("navigate to the kitchen", "navigation_manipulation"),
        ("go to the table", "navigation_manipulation"),
        ("", "manipulation_base"),
    ],
)
def test_select_domain_template(command, expected):
    assert select_domain_template(command) == expected


def test_infer_grasp_target_from_command():
    from planner.hybrid_runtime import infer_grasp_target

    items = ["red_cup", "blue_box", "wood_cube"]
    assert infer_grasp_target("place the red cup on the shelf", items) == "red_cup"
    assert infer_grasp_target("pick blue_box", items) == "blue_box"
    assert infer_grasp_target("do something", items) is None


def test_command_is_look_at_goal_covers_suite_phrasings():
    assert command_is_look_at_goal("look at the wood cube")
    assert command_is_look_at_goal("inspect the wooden cube more closely")
    assert command_is_look_at_goal("look_at red_cup")
    assert not command_is_look_at_goal("place the wood cube on the shelf")
    assert not command_is_look_at_goal("pick the wood cube")
    assert not command_is_look_at_goal("pour the can into the glass")


def test_empty_fd_plan_is_success_unsolvable_is_none():
    """Goal already holds → []; no plan file → None. Do not conflate them."""
    empty = result_from_actions([])
    assert empty["success"] is True
    assert empty["actions"] == []
    assert empty["primitives"] == []
    missing = result_from_actions(None)
    assert missing["success"] is False
    assert "unsolvable" in missing["error"]


def test_hybrid_fd_init_does_not_emit_camera_aimed_at():
    """Tracker/oracle may know the aim; :init still omits camera-aimed-at."""
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
    )
    session.note_completed(PlanStep(primitive="look_at", args={"target": "red_cup"}))
    scene = scene_from_xyz_poses(
        {"red_cup": {"x": 0.4, "y": 0.0, "z": 0.82}},
        known_locations=["table", "shelf"],
        on_surface={"red_cup": "table"},
        gripper_empty=True,
        camera_aimed_at="red_cup",
        domain_template="manipulation_base",
    )
    stub = make_domain_stub_plan(session.command, session.domain_template)
    pddl, fused = session.generate_hybrid_problem(stub, oracle_scene=scene)
    assert fused.robot.camera_aimed_at == "red_cup"
    init = pddl[pddl.index("(:init") : pddl.index("(:goal")]
    assert "(camera-aimed-at red_cup)" not in init


def test_hybrid_problem_without_vlm_action_sketch():
    """Honest :init/:goal from SceneState + GoalGenerator alone."""
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
    )
    scene = scene_from_xyz_poses(
        {"red_cup": {"x": 0.4, "y": 0.0, "z": 0.82}},
        known_locations=["table", "shelf"],
        on_surface={"red_cup": "table"},
        gripper_empty=True,
        domain_template="manipulation_base",
    )
    stub = make_domain_stub_plan(session.command, session.domain_template)
    pddl, fused = session.generate_hybrid_problem(
        stub,
        oracle_scene=scene,
        problem_name="fd_control_test",
    )
    assert fused is not None
    assert "(:init" in pddl
    assert "(:goal" in pddl
    assert "red_cup" in pddl
    assert "shelf" in pddl
    # No action sketch leaked into problem from empty stub steps
    assert stub.steps == []


def test_pipeline_uses_pddl_problem_override_without_calling_vlm():
    stub = make_domain_stub_plan("place red_cup on shelf", "manipulation_base")
    # Minimal well-formed problem matching manipulation_base symbols.
    problem = """(define (problem fd_override)
  (:domain manipulation-base)
  (:objects
    red_cup - item
    table shelf - location
  )
  (:init
    (on red_cup table)
    (clear red_cup)
    (gripper-empty)
    (hand-free)
  )
  (:goal (and (on red_cup shelf)))
)
"""
    fd = _StubFD(
        [
            "(pick red_cup table)",
            "(place red_cup shelf)",
        ]
    )
    spy = _SpyVLM()
    pipeline = Pipeline(
        vlm=spy,
        fd_planner=fd,
        domains_dir=DOMAINS_DIR,
        repair_retries=0,
    )
    result = pipeline.run(
        "place red_cup on shelf",
        images=[],
        vlm_plan=stub,
        pddl_problem=problem,
    )
    assert result.success
    assert spy.plan_calls == 0
    assert fd.call_count == 1
    assert fd.last_problem == problem
    assert [p.name for p in result.primitives] == ["pick", "place"]
    assert stub.steps == []


def test_plan_fd_from_problem_helper_mock_fd():
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
    )
    scene = scene_from_xyz_poses(
        {"red_cup": {"x": 0.4, "y": 0.0, "z": 0.82}},
        known_locations=["table", "shelf"],
        on_surface={"red_cup": "table"},
        gripper_empty=True,
        domain_template="manipulation_base",
    )
    stub = make_domain_stub_plan(session.command, session.domain_template)
    pddl, _ = session.generate_hybrid_problem(stub, oracle_scene=scene)
    fd = _StubFD(["(pick red_cup table)", "(place red_cup shelf)"])
    result = plan_fd_from_problem(
        session.command,
        domain_template=session.domain_template,
        pddl_problem=pddl,
        fd_planner=fd,
        domains_dir=DOMAINS_DIR,
        repair_retries=0,
    )
    assert result.success
    assert fd.call_count == 1
    assert isinstance(result.primitives[0], PrimitiveCall)
    # Problem passed to FD is the hybrid one, not legacy VLM-inferred
    assert "(:goal" in (fd.last_problem or "")
    assert result.vlm_plan is not None
    assert result.vlm_plan.steps == []


def test_pipeline_pddl_override_requires_stub_plan():
    pipeline = Pipeline(
        vlm=None,
        fd_planner=_StubFD([]),
        domains_dir=DOMAINS_DIR,
    )
    result = pipeline.run(
        "place cup on shelf",
        images=[],
        vlm_plan=None,
        pddl_problem="(define (problem x) (:domain manipulation-base))",
    )
    assert not result.success
    assert result.failure_stage == "vlm"
