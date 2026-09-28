"""
Comparative goal-backend evaluation: rule_based vs local_llm (± cloud llm).

Default CI uses mocked local generate_fn fixtures with gold goals (no GPU).
Metrics: exact-match facts, predicate/symbol validity; optional hybrid
problem assembly smoke with fixed SceneState → :init.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator import generate_problem
from planner.problem_generator.goal_generator.backends.llm import LLMGoalGenerator
from planner.problem_generator.goal_generator.backends.local_llm import LocalLLMGoalGenerator
from planner.problem_generator.goal_generator.backends.rule_based import RuleBasedGoalGenerator
from planner.problem_generator.goal_generator.predicates import resolve_allowed_predicates
from planner.problem_generator.init_generator.schema import (
    LocationFact,
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)
from vlm.planner import PlanStep, VLMPlan


@dataclass(frozen=True)
class GoalCase:
    name: str
    command: str
    gold: tuple[tuple[str, ...], ...]
    domain_template: str = "manipulation_base"


_CASES: list[GoalCase] = [
    GoalCase("pick", "pick up the red cup", (("holding", "red_cup"),)),
    GoalCase("place", "place red_cup on shelf", (("on", "red_cup", "shelf"),)),
    GoalCase("look", "look at blue_box", (("camera-aimed-at", "blue_box"),)),
]


def _scene() -> SceneState:
    return SceneState(
        objects=[
            ObjectFact(
                name="red_cup",
                source="mock",
                confidence=1.0,
                location="table",
                clear=True,
            ),
            ObjectFact(
                name="blue_box",
                source="mock",
                confidence=1.0,
                location="table",
                clear=True,
            ),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
            LocationFact(name="shelf", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="mock",
                confidence=1.0,
            ),
            RelationFact(
                predicate="on",
                args=["blue_box", "table"],
                source="mock",
                confidence=1.0,
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


def _objects_locations(scene: SceneState) -> tuple[list[str], list[str]]:
    return [o.name for o in scene.objects], [loc.name for loc in scene.locations]


def _gold_responder(gold: tuple[tuple[str, ...], ...]):
    def generate(system: str, user: str) -> str:
        assert "compact_scene_state" in user
        assert "allowed_predicates" in user
        return json.dumps({"facts": [list(f) for f in gold]})

    return generate


def _fact_valid(
    facts: list[tuple],
    *,
    allowed: frozenset[str],
    known: frozenset[str],
) -> bool:
    if not facts:
        return False
    for fact in facts:
        if fact[0] not in allowed:
            return False
        for arg in fact[1:]:
            if arg not in known:
                return False
    return True


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_rule_based_matches_gold(case: GoalCase):
    scene = _scene()
    objs, locs = _objects_locations(scene)
    result = RuleBasedGoalGenerator().generate(
        case.command,
        objs,
        locations=locs,
        domain_template=case.domain_template,
    )
    assert result.ok is True
    assert tuple(result.facts) == case.gold


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_local_llm_mocked_matches_gold(case: GoalCase):
    scene = _scene()
    objs, locs = _objects_locations(scene)
    gen = LocalLLMGoalGenerator(
        generate_fn=_gold_responder(case.gold),
        fallback_rule_based=False,
    )
    result = gen.generate(
        case.command,
        objs,
        locations=locs,
        domain_template=case.domain_template,
        scene_state=scene,
    )
    assert result.ok is True
    assert result.backend == "local_llm"
    assert tuple(result.facts) == case.gold


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_rule_based_vs_local_llm_exact_match(case: GoalCase):
    scene = _scene()
    objs, locs = _objects_locations(scene)
    rule = RuleBasedGoalGenerator().generate(
        case.command,
        objs,
        locations=locs,
        domain_template=case.domain_template,
    )
    local = LocalLLMGoalGenerator(
        generate_fn=_gold_responder(case.gold),
        fallback_rule_based=False,
    ).generate(
        case.command,
        objs,
        locations=locs,
        domain_template=case.domain_template,
        scene_state=scene,
    )
    assert rule.ok and local.ok
    assert rule.facts == local.facts == list(case.gold)


def test_validity_metrics_across_backends():
    scene = _scene()
    objs, locs = _objects_locations(scene)
    known = frozenset(objs) | frozenset(locs)
    allowed = resolve_allowed_predicates("manipulation_base")

    rule = RuleBasedGoalGenerator()
    local = LocalLLMGoalGenerator(
        generate_fn=_gold_responder((("holding", "red_cup"),)),
        fallback_rule_based=False,
    )
    cloud = LLMGoalGenerator(
        complete_fn=lambda s, u: json.dumps({"facts": [["holding", "red_cup"]]}),
        fallback_rule_based=False,
    )

    results = {
        "rule_based": rule.generate("pick red_cup", objs, locations=locs),
        "local_llm": local.generate(
            "pick red_cup", objs, locations=locs, scene_state=scene
        ),
        "llm": cloud.generate("pick red_cup", objs, locations=locs),
    }

    for name, result in results.items():
        assert result.ok, name
        assert _fact_valid(result.facts, allowed=allowed, known=known), name
        assert result.facts == [("holding", "red_cup")], name


def test_hybrid_problem_smoke_with_local_llm_goal():
    """Assemble hybrid PDDL with fixed :init SceneState + local_llm :goal."""
    scene = _scene()
    objs, locs = _objects_locations(scene)
    gold = (("on", "red_cup", "shelf"),)
    goal = LocalLLMGoalGenerator(
        generate_fn=_gold_responder(gold),
        fallback_rule_based=False,
    ).generate(
        "place red_cup on shelf",
        objs,
        locations=locs,
        scene_state=scene,
    )
    assert goal.ok

    plan = VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(primitive="place", args={"object": "red_cup", "location": "shelf"}),
        ],
        raw_output="",
        domain_template="manipulation_base",
    )
    pddl = generate_problem(
        plan,
        scene_state=scene,
        goal_facts=goal.facts,
        use_hybrid=True,
        problem_name="compare-local-goal",
    )
    assert "(on red_cup table)" in pddl  # init from scene
    assert "(on red_cup shelf)" in pddl  # goal from local_llm
    assert "(define (problem compare-local-goal)" in pddl


def test_generate_problem_default_still_legacy_without_hybrid_flag():
    """Session 9b must not change generate_problem default behavior."""
    plan = VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(primitive="place", args={"object": "red_cup", "location": "shelf"}),
        ],
        raw_output="",
        domain_template="manipulation_base",
    )
    pddl = generate_problem(plan)
    assert "(on red_cup shelf)" in pddl
    assert "compare-local-goal" not in pddl
