"""Tests for hybrid problem generation (use_hybrid=True)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator import generate_problem
from planner.problem_generator.init_generator.adapters.mock import OracleMockAdapter
from planner.problem_generator.goal_generator.backends.rule_based import RuleBasedGoalGenerator
from vlm.planner import VLMPlan, PlanStep


def _plan(
    goal: str = "place red_cup on shelf",
    *,
    template: str = "manipulation_base",
    steps: list[tuple[str, dict]] | None = None,
) -> VLMPlan:
    if steps is None:
        steps = [
            ("pick", {"object": "red_cup"}),
            ("place", {"object": "red_cup", "location": "shelf"}),
        ]
    return VLMPlan(
        goal=goal,
        steps=[PlanStep(primitive=p, args=a) for p, a in steps],
        raw_output="",
        domain_template=template,
    )


def test_hybrid_oracle_scene_place_goal():
    scene = OracleMockAdapter.load()
    plan = _plan(goal="place red_cup on shelf")
    goal = RuleBasedGoalGenerator().generate(
        plan.goal,
        [o.name for o in scene.objects],
        locations=[loc.name for loc in scene.locations],
    )
    assert goal.ok

    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=goal.facts,
        use_hybrid=True,
        problem_name="hybrid-place",
    )

    assert "(define (problem hybrid-place)" in pddl
    assert "(:domain manipulation-base)" in pddl
    objects_block = pddl.split("(:objects")[1].split(")")[0]
    assert "red_cup" in objects_block
    assert "blue_box" in objects_block
    assert "table" in objects_block
    assert "shelf" in objects_block
    assert "(on red_cup table)" in pddl
    assert "(on blue_box table)" in pddl
    assert "(gripper-empty)" in pddl
    assert "(on red_cup shelf)" in pddl
    assert "(on red_cup shelf_b)" not in pddl


def test_hybrid_auto_goal_from_plan_command():
    scene = OracleMockAdapter.load()
    plan = _plan(goal="pick up the red cup")

    pddl = generate_problem(
        plan,
        scene_state=scene,
        use_hybrid=True,
    )

    assert "(holding red_cup)" in pddl
    assert "(on red_cup table)" in pddl


def test_hybrid_requires_scene_state():
    plan = _plan()
    with pytest.raises(ValueError, match="scene_state"):
        generate_problem(plan, use_hybrid=True)


def test_hybrid_domain_from_scene_template():
    scene = OracleMockAdapter.load()
    plan = _plan(template="manipulation_stacking")
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("holding", "red_cup")],
        use_hybrid=True,
    )
    assert "(:domain manipulation-base)" in pddl


def test_hybrid_domain_name_override():
    scene = OracleMockAdapter.load()
    plan = _plan()
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("on", "red_cup", "shelf")],
        use_hybrid=True,
        domain_name="custom-domain",
    )
    assert "(:domain custom-domain)" in pddl


def test_hybrid_unions_plan_only_symbols():
    scene = OracleMockAdapter.load()
    plan = _plan(
        goal="place red_cup on shelf",
        steps=[
            ("pick", {"object": "red_cup"}),
            ("place", {"object": "red_cup", "location": "shelf_b"}),
        ],
    )
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("on", "red_cup", "shelf_b")],
        use_hybrid=True,
    )
    assert "shelf_b" in pddl
    assert "(on red_cup shelf_b)" in pddl


def test_default_path_unchanged_legacy():
    """use_hybrid=False must match legacy generate_problem output."""
    from planner.problem_generator.legacy import generate_problem as legacy_generate

    plan = _plan(
        goal="place red_cup on shelf",
        steps=[
            ("pick", {"object": "red_cup"}),
            ("place", {"object": "red_cup", "location": "shelf"}),
        ],
    )
    scene = OracleMockAdapter.load()

    legacy_pddl = legacy_generate(plan, problem_name="legacy")
    facade_pddl = generate_problem(plan, problem_name="legacy")
    hybrid_pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("on", "red_cup", "shelf")],
        use_hybrid=True,
        problem_name="legacy",
    )

    assert facade_pddl == legacy_pddl
    assert "(on red_cup source_red_cup)" in legacy_pddl
    assert "(on red_cup table)" in hybrid_pddl


def test_render_objects_declares_container_subtype():
    from planner.problem_generator.assembler import render_objects_section

    block = render_objects_section(
        {"red_grapes", "green_grapes"},
        {"blue_tablecloth", "white_bowl"},
        {"white_bowl"},
    )
    assert "green_grapes red_grapes - item" in block
    assert "blue_tablecloth - location" in block
    assert "white_bowl - container" in block
    assert "white_bowl - location" not in block


def test_hybrid_container_location_typed_in_objects():
    from planner.problem_generator.init_generator.adapters.mock import (
        OracleMockAdapter,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_hybrid"
        / "scene_bowl_as_container.json"
    )
    scene = OracleMockAdapter.load(fixture)
    scene.domain_template = "containers_manipulation"
    plan = _plan(
        goal="put the red grapes and the green grapes in the white bowl",
        template="containers_manipulation",
        steps=[],
    )
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[
            ("in-container", "red_grapes", "white_bowl"),
            ("in-container", "green_grapes", "white_bowl"),
        ],
        use_hybrid=True,
        problem_name="bowl-as-container",
    )
    objects_block = pddl.split("(:objects")[1].split("(:init")[0]
    assert "white_bowl - container" in objects_block
    assert "white_bowl - location" not in objects_block
    assert "(in-container red_grapes white_bowl)" in pddl
    assert "(open white_bowl)" in pddl


def test_hybrid_bowl_on_base_domain_is_location_not_open():
    from planner.problem_generator.init_generator.adapters.mock import (
        OracleMockAdapter,
    )

    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "image_hybrid"
        / "scene_bowl_as_container.json"
    )
    scene = OracleMockAdapter.load(fixture)
    scene.domain_template = "manipulation_base"
    plan = _plan(goal="pick the red grapes", template="manipulation_base", steps=[])
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("holding", "red_grapes")],
        use_hybrid=True,
        problem_name="bowl-on-base",
    )
    objects_block = pddl.split("(:objects")[1].split("(:init")[0]
    assert "white_bowl - location" in objects_block
    assert "white_bowl - container" not in objects_block
    assert "(open white_bowl)" not in pddl


def test_hybrid_promotes_on_destination_item_to_location():
    """Goal (on pen notebook) cannot ground if notebook stays typed as item."""
    from planner.problem_generator.init_generator.schema import (
        LocationFact,
        ObjectFact,
        RelationFact,
        RobotFacts,
        SceneState,
    )

    scene = SceneState(
        objects=[
            ObjectFact(name="black_pen", source="mock", confidence=1.0, clear=True),
            ObjectFact(name="notebook", source="mock", confidence=1.0, clear=True),
        ],
        locations=[LocationFact(name="table", source="mock", confidence=1.0)],
        relations=[
            RelationFact(
                predicate="on", args=["black_pen", "table"], source="mock", confidence=1.0
            ),
            RelationFact(
                predicate="on", args=["notebook", "table"], source="mock", confidence=1.0
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
        domain_template="manipulation_base",
    )
    plan = _plan(goal="put the black pen on the notebook", steps=[])
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=[("on", "black_pen", "notebook")],
        use_hybrid=True,
        problem_name="pen-on-notebook",
    )
    objects_block = pddl.split("(:objects")[1].split("(:init")[0]
    init = pddl[pddl.index("(:init") : pddl.index("(:goal")]
    item_line = next(l for l in objects_block.splitlines() if "- item" in l)
    loc_line = next(l for l in objects_block.splitlines() if "- location" in l)
    assert "notebook" not in item_line
    assert "notebook" in loc_line
    assert "(on notebook table)" not in init
    assert "(on black_pen notebook)" in pddl[pddl.index("(:goal") :]
