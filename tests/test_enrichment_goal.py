"""
Session 19 — Enrichment goal parity tests.

Verifies that:
- The shared ``goal_from_enrichment_action`` produces identical results to the
  old legacy inline version.
- ``goals_from_domain_additions`` correctly extracts _raw_fact tuples from
  plan steps + domain_additions.
- ``plan_hybrid_compatible`` accepts enrichment plans.
- ``HybridProblemSession.ensure_goal`` uses enrichment grounding only as a
  fallback when the primary goal path is empty (no merge).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.enrichment_goal import (
    goal_from_enrichment_action,
    goals_from_domain_additions,
)


# ── Fixtures: VLM-style action defs ──────────────────────────────────────────

POUR_ACTION = {
    "name": "pour",
    "parameters": "(?source - item ?target - item)",
    "precondition": "(and (holding ?source) (reachable ?target))",
    "effect": "(and (not (holding ?source)) (poured ?source ?target))",
}

TILT_ACTION = {
    "name": "tilt",
    "parameters": "(?obj - item)",
    "precondition": "(holding ?obj)",
    "effect": "(and (tilted ?obj) (not (upright ?obj)))",
}

SHAKE_ACTION = {
    "name": "shake",
    "parameters": "(?container - item)",
    "precondition": "(holding ?container)",
    "effect": "(shaken ?container)",
}

# Action with only negated effects — should return None.
CLEAR_ACTION = {
    "name": "clear",
    "parameters": "(?x - item)",
    "precondition": "(dirty ?x)",
    "effect": "(not (dirty ?x))",
}


# ── goal_from_enrichment_action ──────────────────────────────────────────────

def test_pour_goal():
    fact = goal_from_enrichment_action(POUR_ACTION, {"source": "bottle", "target": "glass"})
    assert fact == "(poured bottle glass)"


def test_tilt_goal():
    fact = goal_from_enrichment_action(TILT_ACTION, {"obj": "can"})
    assert fact == "(tilted can)"


def test_shake_goal():
    fact = goal_from_enrichment_action(SHAKE_ACTION, {"container": "jar"})
    assert fact == "(shaken jar)"


def test_only_negated_effects_returns_none():
    fact = goal_from_enrichment_action(CLEAR_ACTION, {"x": "plate"})
    assert fact is None


def test_empty_effect_returns_none():
    fact = goal_from_enrichment_action({"effect": "", "parameters": ""}, {})
    assert fact is None


def test_legacy_delegation_matches():
    """Legacy wrapper delegates to the same shared function."""
    from planner.problem_generator.legacy import _goal_from_enrichment_action

    fact_shared = goal_from_enrichment_action(POUR_ACTION, {"source": "bottle", "target": "glass"})
    fact_legacy = _goal_from_enrichment_action(POUR_ACTION, {"source": "bottle", "target": "glass"})
    assert fact_shared == fact_legacy


# ── goals_from_domain_additions ──────────────────────────────────────────────

@dataclass
class _FakeStep:
    primitive: str
    args: dict = field(default_factory=dict)


def test_goals_from_domain_additions_pour():
    da = {"new_actions": [POUR_ACTION]}
    steps = [
        _FakeStep("pick", {"object": "bottle", "source": "table"}),
        _FakeStep("pour", {"source": "bottle", "target": "glass"}),
    ]
    goals = goals_from_domain_additions(da, steps)
    assert goals == [("_raw_fact", "(poured bottle glass)")]


def test_goals_from_domain_additions_multiple():
    da = {"new_actions": [POUR_ACTION, TILT_ACTION]}
    steps = [
        _FakeStep("pour", {"source": "bottle", "target": "glass"}),
        _FakeStep("tilt", {"obj": "can"}),
    ]
    goals = goals_from_domain_additions(da, steps)
    assert len(goals) == 2
    assert ("_raw_fact", "(poured bottle glass)") in goals
    assert ("_raw_fact", "(tilted can)") in goals


def test_goals_from_domain_additions_none():
    goals = goals_from_domain_additions(None, [])
    assert goals == []


def test_goals_from_domain_additions_standard_steps_skipped():
    da = {"new_actions": [POUR_ACTION]}
    steps = [_FakeStep("pick", {"object": "cup"})]
    goals = goals_from_domain_additions(da, steps)
    assert goals == []


def test_goals_from_domain_additions_unknown_enrichment_ignored():
    da = {"new_actions": [POUR_ACTION]}
    steps = [_FakeStep("spin", {"thing": "top"})]  # not in new_actions
    goals = goals_from_domain_additions(da, steps)
    assert goals == []


# ── plan_hybrid_compatible with enrichment ───────────────────────────────────

@dataclass
class _FakePlan:
    steps: list = field(default_factory=list)
    domain_template: str = "manipulation_base"
    domain_additions: dict | None = None
    goal: str = ""


def test_plan_hybrid_compatible_enrichment_accepted():
    from planner.hybrid_runtime import plan_hybrid_compatible

    plan = _FakePlan(
        steps=[
            _FakeStep("pick", {"object": "bottle"}),
            _FakeStep("pour", {"source": "bottle", "target": "glass"}),
        ],
        domain_additions={"new_actions": [POUR_ACTION]},
    )
    assert plan_hybrid_compatible(plan) is True


def test_plan_hybrid_compatible_unknown_prim_rejected():
    from planner.hybrid_runtime import plan_hybrid_compatible

    plan = _FakePlan(
        steps=[_FakeStep("fly", {"dest": "moon"})],
        domain_additions=None,
    )
    assert plan_hybrid_compatible(plan) is False


def test_plan_hybrid_compatible_enrichment_prim_not_in_new_actions():
    from planner.hybrid_runtime import plan_hybrid_compatible

    plan = _FakePlan(
        steps=[_FakeStep("pour", {"source": "bottle", "target": "glass"})],
        domain_additions={"new_actions": []},  # pour not declared
    )
    assert plan_hybrid_compatible(plan) is False


# ── HybridProblemSession.ensure_goal (no merge; fallback only) ───────────────

def test_ensure_goal_discards_stock_goal_when_enrichment_outcome_ignored():
    """holding(...) while pour authored → discard primary, ground enrichment."""
    from planner.enrichment_effects import EnrichmentContext
    from planner.hybrid_runtime import GoalBackend, HybridMode, HybridProblemSession
    from planner.problem_generator.init_generator.schema import (
        LocationFact, ObjectFact, RobotFacts, SceneState,
    )

    scene = SceneState(
        objects=[
            ObjectFact(name="bottle", source="mock", confidence=1.0, location="table"),
            ObjectFact(name="glass", source="mock", confidence=1.0, location="table"),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[],
        robot=RobotFacts(gripper_empty=True, holding=None,
                         camera_aimed_at=None, source="mock", confidence=1.0),
        domain_template="manipulation_base",
    )

    da = {
        "new_actions": [POUR_ACTION],
        "new_predicates": ["(poured ?s - item ?t - item)"],
    }
    plan = _FakePlan(
        steps=[
            _FakeStep("pick", {"object": "bottle"}),
            _FakeStep("pour", {"source": "bottle", "target": "glass"}),
        ],
        domain_additions=da,
        goal="get me something to drink",
    )

    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command="pick up the bottle",  # stock goal; enrichment outcome ignored
    )
    session.set_enrichment(
        EnrichmentContext.from_domain_additions(da),
        skills=["pour"],
        domain_additions=da,
    )

    facts = session.ensure_goal(scene, plan=plan)

    assert facts == [("_raw_fact", "(poured bottle glass)")]
    assert "ignored_stock_goal" in (session.goal_backend_used or "")
    assert "enrichment_fallback" in (session.goal_backend_used or "")


def test_ensure_goal_keeps_primary_when_it_uses_enrichment_outcome():
    """Primary poured(...) is kept — no mechanical merge/replace."""
    import json

    from planner.enrichment_effects import EnrichmentContext
    from planner.hybrid_runtime import GoalBackend, HybridMode, HybridProblemSession
    from planner.problem_generator.init_generator.schema import (
        LocationFact, ObjectFact, RobotFacts, SceneState,
    )

    def generate(system: str, user: str) -> str:
        return json.dumps({"facts": [["poured", "bottle", "glass"]]})

    scene = SceneState(
        objects=[
            ObjectFact(name="bottle", source="mock", confidence=1.0, location="table"),
            ObjectFact(name="glass", source="mock", confidence=1.0, location="table"),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[],
        robot=RobotFacts(gripper_empty=True, holding=None,
                         camera_aimed_at=None, source="mock", confidence=1.0),
        domain_template="manipulation_base",
    )
    da = {
        "new_actions": [POUR_ACTION],
        "new_predicates": ["(poured ?s - item ?t - item)"],
    }
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.LOCAL_LLM,
        command="pour from the bottle into the glass",
        local_generate_fn=generate,
    )
    session.set_enrichment(
        EnrichmentContext.from_domain_additions(da),
        skills=["pour"],
        domain_additions=da,
    )

    facts = session.ensure_goal(scene)
    assert facts == [("poured", "bottle", "glass")]
    assert "enrichment_fallback" not in (session.goal_backend_used or "")
    assert "ignored_stock_goal" not in (session.goal_backend_used or "")


def test_ensure_goal_enrichment_fallback_when_primary_empty():
    """Unreadable command → empty primary → bind enrichment effect alone."""
    from planner.enrichment_effects import EnrichmentContext
    from planner.hybrid_runtime import GoalBackend, HybridMode, HybridProblemSession
    from planner.problem_generator.init_generator.schema import (
        LocationFact, ObjectFact, RobotFacts, SceneState,
    )

    scene = SceneState(
        objects=[
            ObjectFact(name="bottle", source="mock", confidence=1.0, location="table"),
            ObjectFact(name="glass", source="mock", confidence=1.0, location="table"),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[],
        robot=RobotFacts(gripper_empty=True, holding=None,
                         camera_aimed_at=None, source="mock", confidence=1.0),
        domain_template="manipulation_base",
    )

    da = {
        "new_actions": [POUR_ACTION],
        "new_predicates": ["(poured ?s - item ?t - item)"],
    }
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command="get me something to drink",
    )
    session.set_enrichment(
        EnrichmentContext.from_domain_additions(da),
        skills=["pour"],
        domain_additions=da,
    )

    plan = _FakePlan(
        steps=[_FakeStep("pour", {"source": "bottle", "target": "glass"})],
        domain_additions=da,
    )
    facts = session.ensure_goal(scene, plan=plan)

    assert facts == [("_raw_fact", "(poured bottle glass)")]
    assert "enrichment_fallback" in (session.goal_backend_used or "")


def test_ensure_goal_no_enrichment_without_plan():
    from planner.hybrid_runtime import GoalBackend, HybridMode, HybridProblemSession
    from planner.problem_generator.init_generator.schema import (
        LocationFact, ObjectFact, RobotFacts, SceneState,
    )

    scene = SceneState(
        objects=[ObjectFact(name="cup", source="mock", confidence=1.0, location="table")],
        locations=[LocationFact(name="table", source="mock", confidence=1.0, reachable=True)],
        relations=[],
        robot=RobotFacts(gripper_empty=True, holding=None,
                         camera_aimed_at=None, source="mock", confidence=1.0),
        domain_template="manipulation_base",
    )

    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command="pick up the cup",
    )

    facts = session.ensure_goal(scene)
    assert all(f[0] != "_raw_fact" for f in facts)
    assert "enrichment" not in (session.goal_backend_used or "")


def test_local_llm_sees_domain_actions_and_extra_predicates():
    """Goal LLM user prompt includes effective domain actions + predicates."""
    import json

    from planner.hybrid_runtime import GoalBackend, HybridMode, HybridProblemSession
    from planner.problem_generator.init_generator.schema import (
        LocationFact, ObjectFact, RobotFacts, SceneState,
    )

    captured: dict[str, str] = {}

    def generate(system: str, user: str) -> str:
        captured["user"] = user
        return json.dumps({"facts": [["poured", "bottle", "glass"]]})

    scene = SceneState(
        objects=[
            ObjectFact(name="bottle", source="mock", confidence=1.0, location="table"),
            ObjectFact(name="glass", source="mock", confidence=1.0, location="table"),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[],
        robot=RobotFacts(gripper_empty=True, holding=None,
                         camera_aimed_at=None, source="mock", confidence=1.0),
        domain_template="manipulation_base",
    )
    da = {
        "new_actions": [POUR_ACTION],
        "new_predicates": ["(poured ?s - item ?t - item)"],
    }
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.LOCAL_LLM,
        command="pour from the bottle into the glass",
        local_generate_fn=generate,
    )
    session.set_enrichment(None, domain_additions=da)

    facts = session.ensure_goal(scene)
    assert facts == [("poured", "bottle", "glass")]
    assert "domain_actions:" in captured["user"]
    assert "pour" in captured["user"]
    assert "poured" in captured["user"]
    assert "enrichment" not in (session.goal_backend_used or "")


def test_metrics_include_enrichment_authored():
    from planner.enrichment_effects import EnrichmentContext
    from planner.hybrid_runtime import GoalBackend, HybridMode, HybridProblemSession

    da = {"new_actions": [POUR_ACTION], "new_predicates": ["(poured ?a ?b)"]}
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command="pour",
    )
    session.set_enrichment(
        EnrichmentContext.from_domain_additions(da),
        skills=["pour"],
        domain_path="/tmp/enriched.pddl",
        reused=True,
        domain_additions=da,
    )
    snap = session.metrics_snapshot()
    authored = snap["enrichment_authored"]
    assert authored is not None
    assert authored["skills"] == ["pour"]
    assert authored["actions"][0]["name"] == "pour"
    assert "poured" in authored["actions"][0]["effect"]
    assert authored["reused"] is True
    assert authored["domain_path"] == "/tmp/enriched.pddl"
