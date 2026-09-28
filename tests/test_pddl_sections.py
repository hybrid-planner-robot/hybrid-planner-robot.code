"""Persist :init/:goal even when Fast Downward fails."""

from __future__ import annotations

import json
from pathlib import Path

from planner.hybrid_runtime import (
    HybridMode,
    HybridProblemSession,
    make_domain_stub_plan,
    scene_from_xyz_poses,
)
from planner.pddl_sections import (
    parse_fluents,
    split_problem,
    write_fd_problem_artifacts,
)

_PROBLEM = """(define (problem fd_fail_inspect)
  (:domain manipulation-base)
  (:objects
    red_cup - item
    table shelf - location
  )
  (:init
    (on red_cup table)
    (clear red_cup)
    (gripper-empty)
  )
  (:goal (and (on red_cup shelf)))
)
"""


def test_split_problem_separates_init_and_goal():
    split = split_problem(_PROBLEM)
    assert split["pddl_init"] is not None
    assert "(:init" in split["pddl_init"]
    assert "(gripper-empty)" in split["pddl_init"]
    assert "(:goal" not in split["pddl_init"]
    assert split["pddl_goal"] is not None
    assert "(:goal" in split["pddl_goal"]
    assert ["on", "red_cup", "table"] in split["init_facts"]
    assert ["gripper-empty"] in split["init_facts"]
    assert split["goal_facts"] == [["on", "red_cup", "shelf"]]


def test_parse_fluents_unwraps_goal_and():
    facts = parse_fluents("  (:goal (and (camera-aimed-at red_cup)))")
    assert facts == [["camera-aimed-at", "red_cup"]]


def test_fd_fail_writes_problem_init_goal(tmp_path: Path):
    """Hybrid problem + mock FD fail still leaves inspectable PDDL on disk."""
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="look at the red cup",
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
        problem_name="fd_fail_lookat",
    )
    assert fused is not None
    assert "(:init" in pddl and "(:goal" in pddl

    iter_dir = tmp_path / "iter_01"
    fail = {"success": False, "error": "unsolvable (mock FD fail)"}
    art = write_fd_problem_artifacts(iter_dir, pddl, fd_result=fail)

    assert (iter_dir / "problem.pddl").is_file()
    assert (iter_dir / "init.pddl").is_file()
    assert (iter_dir / "goal.pddl").is_file()
    assert (iter_dir / "fd_plan.json").is_file()
    assert (iter_dir / "problem.pddl").read_text(encoding="utf-8") == pddl
    dumped = json.loads((iter_dir / "fd_plan.json").read_text(encoding="utf-8"))
    assert dumped["success"] is False
    assert "unsolvable" in dumped["error"]
    assert art["init_facts"], "parsed :init fluents should be non-empty"
    assert any(f and f[0] == "gripper-empty" for f in art["init_facts"])
    assert any(f and f[0] == "on" for f in art["init_facts"])
    assert art["pddl_goal"] and "(:goal" in art["pddl_goal"]
