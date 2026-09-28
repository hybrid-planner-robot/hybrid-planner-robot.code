"""R1 enricher: call 1 (multi-skill + affordances) and call 2 (assignment).

Injected generate_fn only — no GPU, no Gazebo. R0 prompts/parsers are not used.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.domain_llm import (
    ENRICH_SYSTEM_PROMPT,
    EnrichmentPayloadError,
    domain_action_names,
    domain_predicate_names,
)
from planner.domain_store import load_base_domain
from planner.hybrid_runtime import (
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    SceneSource,
    make_domain_stub_plan,
)
from planner.online_enrichment import (
    DomainCompleteness,
    DomainSelectionResult,
    EnrichmentRequest,
    EnrichmentStatus,
    SelectionBackend,
)
from planner.domain_enricher import DomainAdditions, DomainEnricher
from planner.problem_generator.init_generator.adapters.mock import OracleMockAdapter
from planner.problem_generator.init_generator.schema import (
    LocationFact,
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)
from planner.r1.enricher import R1DomainEnricher
from planner.r1.init_facts import apply_init_facts
from planner.r1.mock_llm import mock_host_generate_fn
from planner.r1.parse import parse_r1_assignment, parse_r1_enrichment
from planner.r1.prompts import R1_ASSIGN_SYSTEM_PROMPT, R1_ENRICH_SYSTEM_PROMPT
from planner.r1.scenes import mock_scene_file_for_world


def _request(
    command: str = "pour the can into the glass",
    skills: tuple[str, ...] = ("pour",),
    symbols: tuple[str, ...] = ("can", "glass", "cup", "mug", "table"),
) -> EnrichmentRequest:
    selection = DomainSelectionResult(
        template="manipulation_base",
        completeness=DomainCompleteness.INCOMPLETE,
        needed_skills=skills,
        reason="gap",
        backend=SelectionBackend.LLM.value,
    )
    return EnrichmentRequest(
        template="manipulation_base",
        command=command,
        scene_symbols=symbols,
        candidate_skills=skills,
        reason="gap",
        selection=selection,
        base_domain_text=load_base_domain("manipulation_base"),
    )


def test_r1_prompt_is_not_r0_prompt():
    assert "affordance" in R1_ENRICH_SYSTEM_PROMPT.lower()
    assert "can-" in R1_ENRICH_SYSTEM_PROMPT
    assert "can-pour" not in ENRICH_SYSTEM_PROMPT


def test_r1_prompts_are_not_suite_few_shots():
    """System prompts must not contain a worked suite item (pour/can/glass)."""
    blob = (R1_ENRICH_SYSTEM_PROMPT + "\n" + R1_ASSIGN_SYSTEM_PROMPT).lower()
    assert '"skills": ["pour"]' not in blob
    assert "pour the can" not in blob
    assert "into the glass" not in blob
    # Object gold from kitchen/tabletop fixtures must not appear as examples.
    for token in ("tea_box", "wood_cube", "coke_can"):
        assert token not in blob


def test_parse_r1_pour_payload():
    raw = json.dumps(
        {
            "skills": ["pour"],
            "actions": [
                {
                    "name": "pour",
                    "parameters": "(?s - item ?t - item)",
                    "precondition": (
                        "(and (holding ?s) (can-pour ?s) (can-be-poured ?t))"
                    ),
                    "effect": (
                        "(and (poured ?s ?t) (not (holding ?s)) (gripper-empty))"
                    ),
                }
            ],
            "new_predicates": [
                "(poured ?s - item ?t - item)",
                "(can-pour ?x - item)",
                "(can-be-poured ?x - item)",
            ],
        }
    )
    parsed = parse_r1_enrichment(raw, allowed_skills=["pour"])
    assert parsed.skills == ("pour",)
    assert "can-pour" in parsed.actions[0].action["precondition"]


def test_parse_r1_rejects_missing_affordance():
    raw = json.dumps(
        {
            "skills": ["pour"],
            "actions": [
                {
                    "name": "pour",
                    "parameters": "(?s - item ?t - item)",
                    "precondition": "(and (holding ?s))",
                    "effect": "(and (poured ?s ?t))",
                }
            ],
            "new_predicates": ["(poured ?s - item ?t - item)"],
        }
    )
    with pytest.raises(EnrichmentPayloadError, match="can-"):
        parse_r1_enrichment(raw, allowed_skills=["pour"])


def test_parse_r1_multi_skill():
    raw = json.dumps(
        {
            "skills": ["pour", "stir"],
            "actions": [
                {
                    "name": "pour",
                    "parameters": "(?s - item ?t - item)",
                    "precondition": (
                        "(and (holding ?s) (can-pour ?s) (can-be-poured ?t))"
                    ),
                    "effect": "(and (poured ?s ?t) (not (holding ?s)) (gripper-empty))",
                },
                {
                    "name": "stir",
                    "parameters": "(?c - item)",
                    "precondition": "(and (holding ?c) (can-be-stirred ?c))",
                    "effect": "(and (stirred ?c))",
                },
            ],
            "new_predicates": [
                "(poured ?s - item ?t - item)",
                "(can-pour ?x - item)",
                "(can-be-poured ?x - item)",
                "(stirred ?c - item)",
                "(can-be-stirred ?x - item)",
            ],
        }
    )
    parsed = parse_r1_enrichment(raw, allowed_skills=["pour", "stir"])
    assert parsed.skills == ("pour", "stir")


def test_parse_assignment_filters_unknown_objects():
    with pytest.raises(Exception, match="wood_cube"):
        parse_r1_assignment(
            json.dumps({"init_facts": [["can-pour", "wood_cube"]]}),
            allowed_predicates=["can-pour"],
            scene_symbols=["can", "glass"],
        )


def test_r1_enricher_two_calls_and_merge():
    enricher = R1DomainEnricher(generate_fn=mock_host_generate_fn)
    outcome = enricher.enrich(_request())
    assert outcome.status == EnrichmentStatus.ENRICHED
    assert "pour" in outcome.skills_grounded
    assert outcome.domain_text is not None
    assert "pour" in domain_action_names(outcome.domain_text)
    assert "can-pour" in domain_predicate_names(outcome.domain_text)
    facts = [tuple(f) for f in (outcome.domain_additions or {}).get("init_facts") or []]
    assert ("can-pour", "can") in facts
    assert ("can-be-poured", "glass") in facts
    assert ("can-pour", "wood_cube") not in facts
    assert len(enricher.calls) >= 2


def test_r1_multi_skill_enricher():
    enricher = R1DomainEnricher(generate_fn=mock_host_generate_fn)
    outcome = enricher.enrich(
        _request(
            "pour the can into the cup and stir the cup",
            skills=("pour", "stir"),
            symbols=("can", "cup", "spoon", "table"),
        )
    )
    assert outcome.status == EnrichmentStatus.ENRICHED
    assert set(outcome.skills_grounded) >= {"pour", "stir"}
    names = domain_action_names(outcome.domain_text or "")
    assert "pour" in names and "stir" in names


def test_r1_refuse_ungenerable():
    enricher = R1DomainEnricher(generate_fn=mock_host_generate_fn)
    outcome = enricher.enrich(
        _request("solder the broken wire on the board", skills=("pour",))
    )
    # mock refuses ungenerable even if pour was listed
    assert outcome.status == EnrichmentStatus.REFUSED


def test_init_facts_reach_pddl(tmp_path: Path):
    scene = OracleMockAdapter.load(mock_scene_file_for_world("kitchen"))
    scene = apply_init_facts(scene, [("can-pour", "can"), ("can-be-poured", "glass")])
    additions = {
        "new_predicates": [
            "(poured ?s - item ?t - item)",
            "(can-pour ?x - item)",
            "(can-be-poured ?x - item)",
        ],
        "new_actions": [
            {
                "name": "pour",
                "parameters": "(?s - item ?t - item)",
                "precondition": "(and (holding ?s) (can-pour ?s) (can-be-poured ?t))",
                "effect": "(and (poured ?s ?t))",
            }
        ],
        "init_facts": [["can-pour", "can"], ["can-be-poured", "glass"]],
    }
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.ORACLE,
        command="pour the can into the glass",
        domain_template="manipulation_base",
        known_locations=["table"],
        domain_additions=additions,
    )
    from planner.enrichment_effects import EnrichmentContext

    session.set_enrichment(
        EnrichmentContext.from_domain_additions(additions),
        skills=["pour"],
        domain_additions=additions,
    )
    stub = make_domain_stub_plan(
        "pour the can into the glass",
        "manipulation_base",
        domain_additions=additions,
    )
    pddl, fused = session.generate_hybrid_problem(
        stub, oracle_scene=scene, dino_scene=scene
    )
    assert "(can-pour can)" in pddl
    assert "(can-be-poured glass)" in pddl
    assert "(can-pour wood_cube)" not in pddl
    cube_facts = [r for r in fused.relations if r.predicate == "can-pour"]
    assert all(r.args != ["wood_cube"] for r in cube_facts)


def test_mock_scene_files_exist():
    assert mock_scene_file_for_world("tabletop").is_file()
    assert mock_scene_file_for_world("kitchen").is_file()
    assert mock_scene_file_for_world("workshop").name == "workshop.json"
    assert mock_scene_file_for_world("workshop").is_file()


def _r1_pour_additions(init_facts: list[list[str]]) -> dict:
    return {
        "new_types": [],
        "new_predicates": [
            "(poured ?s - item ?t - item)",
            "(can-pour ?x - item)",
            "(can-be-poured ?x - item)",
        ],
        "new_actions": [
            {
                "name": "pour",
                "parameters": "(?s - item ?t - item)",
                "precondition": "(and (holding ?s) (can-pour ?s) (can-be-poured ?t))",
                "effect": "(and (poured ?s ?t) (not (holding ?s)) (gripper-empty))",
            }
        ],
        "modified_preconditions": {},
        "init_facts": [list(f) for f in init_facts],
    }


def _holding_scene(held: str, other: str) -> SceneState:
    return SceneState(
        objects=[
            ObjectFact(name=held, source="mock", confidence=1.0, clear=True),
            ObjectFact(
                name=other, source="mock", confidence=1.0, location="table", clear=True
            ),
        ],
        locations=[LocationFact(name="table", source="mock", confidence=1.0)],
        relations=[
            RelationFact(
                predicate="on", args=[other, "table"], source="mock", confidence=1.0
            ),
            RelationFact(predicate="clear", args=[held], source="mock", confidence=1.0),
            RelationFact(predicate="clear", args=[other], source="mock", confidence=1.0),
        ],
        robot=RobotFacts(
            gripper_empty=False,
            holding=held,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
        domain_template="manipulation_base",
    )


def _pour_domain_and_problem(held: str, other: str, init_facts: list[list[str]]):
    from planner.problem_generator.assembler import generate_problem_hybrid

    additions = _r1_pour_additions(init_facts)
    merged = DomainEnricher().enrich(
        load_base_domain("manipulation_base"),
        DomainAdditions(
            new_predicates=list(additions["new_predicates"]),
            new_actions=list(additions["new_actions"]),
        ),
    )
    assert merged.is_valid, merged.errors
    scene = apply_init_facts(_holding_scene(held, other), init_facts)
    stub = make_domain_stub_plan(
        f"pour the {held} into the {other}",
        "manipulation_base",
        domain_additions=additions,
    )
    problem = generate_problem_hybrid(
        stub, scene, goal_facts=[("poured", held, other)]
    )
    return merged.domain_text, problem


def test_cube_without_can_pour_absent_from_init():
    """R1-4: a cube must not receive can-pour, so pour is not applicable."""
    domain, problem = _pour_domain_and_problem(
        "wood_cube",
        "glass",
        [["can-be-poured", "glass"]],
    )
    assert "(can-pour wood_cube)" not in problem
    assert "(can-be-poured glass)" in problem
    assert "(holding wood_cube)" in problem
    assert "can-pour" in domain


@pytest.mark.skipif(
    shutil.which("fast-downward.py") is None
    and shutil.which("fast-downward") is None,
    reason="Fast Downward not on PATH (offline suite uses the stub planner)",
)
def test_fd_pour_only_when_source_has_can_pour():
    from planner.fast_downward import FastDownwardPlanner

    planner = FastDownwardPlanner()
    domain_ok, problem_ok = _pour_domain_and_problem(
        "can",
        "glass",
        [["can-pour", "can"], ["can-be-poured", "glass"]],
    )
    plan_ok = planner.solve_from_strings(domain_ok, problem_ok)
    assert plan_ok is not None
    assert any("pour" in action for action in plan_ok)

    domain_no, problem_no = _pour_domain_and_problem(
        "wood_cube",
        "glass",
        [["can-be-poured", "glass"]],
    )
    plan_no = planner.solve_from_strings(domain_no, problem_no)
    assert plan_no is None
