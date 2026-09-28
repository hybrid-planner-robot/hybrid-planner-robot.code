"""E2a — LLM-PDDL author domain+problem with injected generate_fn (no GPU, no FD)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.domain_llm import pddl_syntax_errors
from planner.llm_pddl_baseline import (
    EXIT_AUTHORED,
    EXIT_INVALID,
    EXIT_REFUSED,
    PROMPTS_DIR,
    author_pddl,
    build_pddl_prompts,
    load_pddl_system_prompt,
)

_WOOD_CUBE_SCENE = json.dumps(
    {
        "objects": [{"name": "wood_cube", "type": "item"}],
        "locations": [
            {"name": "table", "type": "location"},
            {"name": "shelf", "type": "location"},
        ],
        "relations": [{"predicate": "on", "args": ["wood_cube", "table"]}],
        "robot": {
            "gripper_empty": True,
            "holding": None,
            "camera_aimed_at": None,
        },
    },
    indent=2,
)

_PLACE_DOMAIN = """
(define (domain toy-place)
  (:requirements :strips :typing)
  (:types widget pad)
  (:predicates
    (resting ?w - widget ?p - pad)
    (gripped ?w - widget)
    (free-hand)
  )
  (:action pick
    :parameters (?w - widget ?p - pad)
    :precondition (and (resting ?w ?p) (free-hand))
    :effect (and (gripped ?w) (not (resting ?w ?p)) (not (free-hand)))
  )
  (:action place
    :parameters (?w - widget ?p - pad)
    :precondition (gripped ?w)
    :effect (and (resting ?w ?p) (free-hand) (not (gripped ?w)))
  )
)
""".strip()

_PLACE_PROBLEM = """
(define (problem place-cube)
  (:domain toy-place)
  (:objects
    wood_cube - widget
    table shelf - pad
  )
  (:init
    (resting wood_cube table)
    (free-hand)
  )
  (:goal (resting wood_cube shelf))
)
""".strip()

_FORBIDDEN_PROMPT = (
    "manipulation_base",
    "manipulation-base",
    "manipulation_stacking",
    "containers_manipulation",
    "navigation_manipulation",
    "pddl/domains",
    "manipulation_complete",
)


def test_toy_few_shot_domain_passes_syntax_check():
    domain = (PROMPTS_DIR / "toy_domain.pddl").read_text(encoding="utf-8")
    assert pddl_syntax_errors(domain) == []
    assert "(:action" in domain
    problem = (PROMPTS_DIR / "toy_problem.pddl").read_text(encoding="utf-8")
    assert "(:init" in problem
    assert "(:goal" in problem


def test_prompt_excludes_repo_templates():
    system, user = build_pddl_prompts(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE
    )
    blob = load_pddl_system_prompt() + "\n" + system + "\n" + user
    for token in _FORBIDDEN_PROMPT:
        assert token not in blob, token
    for path in PROMPTS_DIR.iterdir():
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_PROMPT:
            assert token not in text, f"{path.name}: {token}"
    assert "wood_cube" in user
    assert "place the wood cube on the shelf" in user
    assert "STAGE:pddl_refuse" in system
    assert "toy-table" in load_pddl_system_prompt()


def test_workshop_pddl_refuse_prompt_lists_paint_not_pour():
    system, _user = build_pddl_prompts(
        "paint the wood board", _WOOD_CUBE_SCENE, world="workshop"
    )
    assert "paint(" in system
    assert "pour(" not in system
    kitchen, _ = build_pddl_prompts(
        "pour the can", _WOOD_CUBE_SCENE, world="kitchen"
    )
    assert "pour(" in kitchen
    assert "paint(" not in kitchen


def test_place_authors_parsable_domain_and_problem_with_goal():
    def generate(system: str, user: str) -> str:
        for token in _FORBIDDEN_PROMPT:
            assert token not in system
        return json.dumps(
            {
                "domain": _PLACE_DOMAIN,
                "problem": _PLACE_PROBLEM,
                "refuse": False,
                "reason": "put the cube on the shelf",
            }
        )

    outcome = author_pddl(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE, generate
    )
    assert outcome.exit_reason == EXIT_AUTHORED
    assert outcome.refuse is False
    assert outcome.domain is not None
    assert "(:action" in outcome.domain
    assert pddl_syntax_errors(outcome.domain) == []
    assert outcome.problem is not None
    assert "(:goal" in outcome.problem
    assert "(:init" in outcome.problem


def test_solder_refuse_clears_domain():
    def generate(system: str, user: str) -> str:
        return json.dumps(
            {
                "domain": _PLACE_DOMAIN,
                "problem": _PLACE_PROBLEM,
                "refuse": True,
                "reason": "no solder skill",
            }
        )

    outcome = author_pddl("solder the pipe", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_REFUSED
    assert outcome.refuse is True
    assert outcome.domain is None
    assert outcome.problem is None


def test_garbage_text_is_invalid_pddl():
    outcome = author_pddl(
        "place the wood cube on the shelf",
        _WOOD_CUBE_SCENE,
        lambda s, u: "not json and not pddl at all",
    )
    assert outcome.exit_reason == EXIT_INVALID
    assert outcome.domain is None
    assert outcome.refuse is False


def test_problem_missing_goal_is_invalid_pddl():
    broken = _PLACE_PROBLEM.replace("(:goal (resting wood_cube shelf))", "")

    def generate(system: str, user: str) -> str:
        return json.dumps(
            {"domain": _PLACE_DOMAIN, "problem": broken, "refuse": False}
        )

    outcome = author_pddl("place the cube", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_INVALID
    assert outcome.domain == _PLACE_DOMAIN
    assert outcome.problem == broken
    assert "(:goal" in (outcome.error or "")


def test_fenced_pddl_fallback_when_json_fails():
    raw = (
        "here you go\n"
        "```pddl\n"
        f"{_PLACE_DOMAIN}\n"
        "```\n"
        "```pddl\n"
        f"{_PLACE_PROBLEM}\n"
        "```\n"
    )
    outcome = author_pddl(
        "place the wood cube on the shelf",
        _WOOD_CUBE_SCENE,
        lambda s, u: raw,
    )
    assert outcome.exit_reason == EXIT_AUTHORED
    assert "(:action" in (outcome.domain or "")
    assert "(:goal" in (outcome.problem or "")


def test_assemble_pddl_from_staged_pieces():
    from planner.llm_pddl_baseline import assemble_pddl

    domain, problem, unknown = assemble_pddl(
        {
            "types": ["item", "location"],
            "predicates": ["(on ?x - item ?y - location)", "(hand-empty)"],
        },
        [
            {
                "name": "pick",
                "parameters": "(?x - item)",
                "precondition": "(hand-empty)",
                "effect": "(not (hand-empty))",
            }
        ],
        {
            "objects": {"amber_mug": "item", "oak_stool": "location"},
            "init": ["(on amber_mug oak_stool)", "(hand-empty)"],
        },
        {"goal": ["(on amber_mug oak_stool)"]},
    )
    assert unknown == ()
    assert pddl_syntax_errors(domain) == []
    assert "(:action pick" in domain
    assert "(:init" in problem and "(:goal" in problem
    assert "amber_mug" in problem


def test_staged_pddl_five_calls_assembles_domain_and_problem():
    calls: list[str] = []

    def generate(system: str, user: str) -> str:
        calls.append(next(l for l in system.splitlines() if l.startswith("STAGE:")))
        if "STAGE:pddl_refuse" in system:
            return json.dumps({"refuse": False, "reason": "ok"})
        if "STAGE:pddl_schema" in system:
            return json.dumps(
                {
                    "types": ["widget", "pad"],
                    "predicates": [
                        "(resting ?w - widget ?p - pad)",
                        "(gripped ?w - widget)",
                        "(free-hand)",
                    ],
                }
            )
        if "STAGE:pddl_actions" in system:
            return json.dumps(
                {
                    "actions": [
                        {
                            "name": "pick",
                            "parameters": "(?w - widget ?p - pad)",
                            "precondition": "(and (resting ?w ?p) (free-hand))",
                            "effect": (
                                "(and (gripped ?w) (not (resting ?w ?p)) "
                                "(not (free-hand)))"
                            ),
                        },
                        {
                            "name": "place",
                            "parameters": "(?w - widget ?p - pad)",
                            "precondition": "(gripped ?w)",
                            "effect": (
                                "(and (resting ?w ?p) (free-hand) "
                                "(not (gripped ?w)))"
                            ),
                        },
                    ]
                }
            )
        if "STAGE:pddl_init" in system:
            return json.dumps(
                {
                    "objects": {
                        "wood_cube": "widget",
                        "table": "pad",
                        "shelf": "pad",
                    },
                    "init": ["(resting wood_cube table)", "(free-hand)"],
                }
            )
        assert "STAGE:pddl_goal" in system
        return json.dumps({"goal": ["(resting wood_cube shelf)"]})

    outcome = author_pddl(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE, generate
    )
    assert calls == [
        "STAGE:pddl_refuse",
        "STAGE:pddl_schema",
        "STAGE:pddl_actions",
        "STAGE:pddl_init",
        "STAGE:pddl_goal",
    ]
    assert outcome.exit_reason == EXIT_AUTHORED
    assert outcome.domain is not None
    assert pddl_syntax_errors(outcome.domain) == []
    assert "(:action pick" in outcome.domain
    assert "(:goal" in (outcome.problem or "")
    assert "wood_cube" in (outcome.problem or "")
    assert len(outcome.stages) == 5
