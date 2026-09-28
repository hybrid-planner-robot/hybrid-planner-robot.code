"""
Offline mock generate_fn for llm_plan / llm_pddl (eval E4).

Looks up the **verbatim** suite v1 task string (not a paraphrase, not an
LLM-decision cache keyed by NL). Ungenerable ids refuse; other ids emit
gold ``plan_actions`` / ``gold_goal`` so the battery can classify the path.
Smoke tasks outside the suite (e.g. ``solder the pipe``) still refuse on
keyword. Not a quality claim.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from planner.skill_catalog import catalog_to_pddl_action

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SUITE_PATH = (
    _REPO_ROOT / "tests" / "fixtures" / "planning_eval" / "suite_v1.json"
)
_KEYWORD_REFUSE = ("solder", "levitate", "weld")

_EXPLICIT_PLACE_PLAN = (
    {"name": "pick", "args": ["wood_cube"]},
    {"name": "place", "args": ["wood_cube", "shelf"]},
)


_CASES: tuple[dict[str, Any], ...] | None = None


def _suite_cases() -> tuple[dict[str, Any], ...]:
    """Load suite v1 once (fixture index, not an LLM-decision cache)."""
    global _CASES
    if _CASES is None:
        data = json.loads(_SUITE_PATH.read_text(encoding="utf-8"))
        _CASES = tuple(data["cases"])
    return _CASES


def suite_task_texts() -> tuple[str, ...]:
    return tuple(c["task"] for c in _suite_cases())


def case_for_task(task: str) -> dict[str, Any] | None:
    """Exact strip match on suite v1 ``task`` (verbatim)."""
    wanted = str(task or "").strip()
    for case in _suite_cases():
        if str(case.get("task") or "").strip() == wanted:
            return case
    return None


def task_from_user_prompt(user: str) -> str:
    for line in (user or "").splitlines():
        if line.startswith("task:"):
            return line.split(":", 1)[1].strip()
    return str(user or "").strip()


def _keyword_refuse(task: str) -> bool:
    blob = task.lower()
    return any(word in blob for word in _KEYWORD_REFUSE)


def _refuse_payload(*, pddl: bool) -> str:
    body: dict[str, Any] = {
        "refuse": True,
        "reason": "no catalog skill for this task",
    }
    if pddl:
        body["domain"] = ""
        body["problem"] = ""
    else:
        body["actions"] = []
    return json.dumps(body)


def _goal_atoms(facts: list[list[str]]) -> str:
    parts = ["(" + " ".join(str(x) for x in fact) + ")" for fact in facts if fact]
    return " ".join(parts) if parts else "(dummy)"


def _predicates_from_facts(facts: list[list[str]]) -> str:
    seen: dict[str, int] = {"dummy": 0}
    for fact in facts:
        if not fact:
            continue
        name = str(fact[0])
        arity = max(0, len(fact) - 1)
        seen.setdefault(name, arity)
    chunks: list[str] = []
    for name, arity in seen.items():
        if arity <= 0:
            chunks.append(f"({name})")
        else:
            vars_ = " ".join(f"?x{i}" for i in range(arity))
            chunks.append(f"({name} {vars_})")
    return "\n    ".join(chunks)


def _actions_pddl(names: list[str]) -> str:
    blocks = []
    for raw in names or ["pick"]:
        name = catalog_to_pddl_action(raw)
        blocks.append(
            "  (:action "
            + name
            + "\n    :parameters ()\n    :precondition (dummy)\n"
            "    :effect (dummy)\n  )"
        )
    return "\n".join(blocks)


def domain_and_problem_for_case(case: dict[str, Any]) -> tuple[str, str]:
    expect = case.get("expect") or {}
    plan_actions = list(expect.get("plan_actions") or ["pick"])
    gold = list(expect.get("gold_goal") or [])
    domain = (
        "(define (domain mock-baseline)\n"
        "  (:requirements :strips)\n"
        "  (:predicates\n    "
        + _predicates_from_facts(gold)
        + "\n  )\n"
        + _actions_pddl(plan_actions)
        + "\n)\n"
    )
    problem = (
        "(define (problem mock-case)\n"
        "  (:domain mock-baseline)\n"
        "  (:objects)\n"
        "  (:init (dummy))\n"
        "  (:goal (and "
        + _goal_atoms(gold)
        + "))\n"
        ")\n"
    )
    return domain.strip(), problem.strip()


def plan_actions_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    if case.get("id") == "explicit_place":
        return [dict(a) for a in _EXPLICIT_PLACE_PLAN]
    expect = case.get("expect") or {}
    names = list(expect.get("plan_actions") or [])
    return [{"name": n, "args": []} for n in names]


_ACTION_HEAD = re.compile(r"\(:action\s+([\w-]+)", re.IGNORECASE)


def mock_fast_downward_actions(
    domain_text: str,
    problem_text: str,
) -> list[str] | None:
    """Same method surface as FastDownwardPlanner.solve_from_strings (no binary)."""
    if not str(domain_text or "").strip() or not str(problem_text or "").strip():
        return None
    names = _ACTION_HEAD.findall(domain_text)
    if not names:
        return None
    lowered = {n.lower() for n in names}
    if "pick" in lowered and "place" in lowered and "wood_cube" in problem_text:
        return ["(pick wood_cube table)", "(place wood_cube shelf)"]
    return [f"({n})" for n in names]


class MockFastDownward:
    """CI/smoke FD client: same method as the host, no binary / no container."""

    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        return mock_fast_downward_actions(domain_text, problem_text)


def _stage_from_system(system: str) -> str:
    for line in (system or "").splitlines()[:8]:
        if line.startswith("STAGE:"):
            return line.split(":", 1)[1].strip()
    return ""


def generate_plan_mock(system: str, user: str) -> str:
    task = task_from_user_prompt(user)
    case = case_for_task(task)
    refuse = (case and case.get("family") == "ungenerable") or _keyword_refuse(task)
    stage = _stage_from_system(system)
    if refuse:
        return _refuse_payload(pddl=False)
    if stage == "plan_refuse":
        return json.dumps({"refuse": False, "reason": "catalog covers this"})
    if stage == "plan_intent":
        names = []
        if case is not None:
            names = list((case.get("expect") or {}).get("plan_actions") or [])
        if not names:
            names = ["pick", "place"]
        return json.dumps({"skills": names, "reason": "mock intent from suite gold"})
    if case is None:
        return json.dumps(
            {
                "actions": [dict(a) for a in _EXPLICIT_PLACE_PLAN],
                "reason": "place the cube on the shelf",
                "refuse": False,
            }
        )
    return json.dumps(
        {
            "actions": plan_actions_for_case(case),
            "reason": "mock plan from suite gold",
            "refuse": False,
        }
    )


def generate_pddl_mock(system: str, user: str) -> str:
    task = task_from_user_prompt(user)
    case = case_for_task(task)
    refuse = (case and case.get("family") == "ungenerable") or _keyword_refuse(task)
    stage = _stage_from_system(system)
    if refuse:
        return _refuse_payload(pddl=True)
    if case is None:
        case = case_for_task("place the wood cube on the shelf")
        assert case is not None
    if stage in ("",) or stage not in {
        "pddl_refuse",
        "pddl_schema",
        "pddl_actions",
        "pddl_init",
        "pddl_goal",
    }:
        domain, problem = domain_and_problem_for_case(case)
        return json.dumps(
            {
                "domain": domain,
                "problem": problem,
                "refuse": False,
                "reason": "mock PDDL from suite gold",
            }
        )
    expect = case.get("expect") or {}
    gold = list(expect.get("gold_goal") or [])
    plan_actions = list(expect.get("plan_actions") or ["pick"])
    if stage == "pddl_refuse":
        return json.dumps({"refuse": False, "reason": "catalog covers this"})
    if stage == "pddl_schema":
        return json.dumps(
            {
                "types": ["object"],
                "predicates": [
                    line.strip()
                    for line in _predicates_from_facts(gold).splitlines()
                    if line.strip()
                ],
                "reason": "mock schema",
            }
        )
    if stage == "pddl_actions":
        actions = []
        for raw in plan_actions:
            name = catalog_to_pddl_action(raw)
            actions.append(
                {
                    "name": name,
                    "parameters": "()",
                    "precondition": "(dummy)",
                    "effect": "(dummy)",
                }
            )
        return json.dumps({"actions": actions, "reason": "mock actions"})
    if stage == "pddl_init":
        return json.dumps(
            {
                "objects": {"dummy": "object"},
                "init": ["(dummy)"],
                "reason": "mock init",
            }
        )
    return json.dumps(
        {
            "goal": ["(" + " ".join(str(x) for x in fact) + ")" for fact in gold]
            or ["(dummy)"],
            "reason": "mock goal",
        }
    )
