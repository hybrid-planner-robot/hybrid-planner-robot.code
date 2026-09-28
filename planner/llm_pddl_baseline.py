"""
LLM-PDDL baseline: staged authoring of a domain + problem (no Fast Downward).

Stages (no repo templates; toy few-shot only):

1. refuse — catalog can model the task?
2. schema — types + predicates
3. actions — catalog-named ``:action`` bodies
4. init — ``:objects`` + ``:init`` from the scene
5. goal — ``:goal`` facts

A one-shot payload with ``domain`` + ``problem`` (or two PDDL fences) is
still accepted (CI injects). ``refuse=true`` yields no domain text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from planner.baseline_catalog import (
    baseline_action_summary_lines,
    baseline_planner_actions_for_world,
    normalize_baseline_action,
)
from planner.domain_llm import pddl_syntax_errors, strip_pddl_comments
from planner.problem_generator.goal_generator.json_facts import extract_json_object
from planner.skill_catalog import catalog_to_pddl_action
from planner.call_timings import invoke_llm, last_llm_s
from planner.text_llm import GenerateFn
from prompts import prompt_path

__all__ = [
    "EXIT_AUTHORED",
    "EXIT_INVALID",
    "EXIT_REFUSED",
    "LlmPddlOutcome",
    "PROMPTS_DIR",
    "assemble_pddl",
    "author_pddl",
    "build_pddl_prompts",
    "load_pddl_system_prompt",
    "parse_pddl_payload",
]

EXIT_AUTHORED = "authored"
EXIT_REFUSED = "refused"
EXIT_INVALID = "invalid_pddl"

PROMPTS_DIR = prompt_path("llm_pddl")

_FENCE = re.compile(r"```(?:pddl)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_DOMAIN_HEAD = re.compile(r"\(define\s+\(domain\b", re.IGNORECASE)
_PROBLEM_HEAD = re.compile(r"\(define\s+\(problem\b", re.IGNORECASE)

_STAGE_FILES = (
    "refuse.md",
    "schema.md",
    "actions.md",
    "init.md",
    "goal.md",
)


@dataclass(frozen=True)
class LlmPddlOutcome:
    domain: str | None
    problem: str | None
    reason: str
    refuse: bool
    exit_reason: str
    raw: str | None = None
    error: str | None = None
    syntax_errors: tuple[str, ...] = ()
    stages: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "problem": self.problem,
            "reason": self.reason,
            "refuse": self.refuse,
            "exit_reason": self.exit_reason,
            "error": self.error,
            "syntax_errors": list(self.syntax_errors),
            "stages": [dict(s) for s in self.stages],
        }


def _catalog_block(world: str | None = None) -> str:
    lines = "\n".join(
        f"- {line}" for line in baseline_action_summary_lines(world=world)
    )
    return (
        "robot_actions (closed — action names only, no template files):\n"
        + lines
        + "\n"
    )


def _load_stage(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip() + "\n"


def _toy_block() -> str:
    toy_domain = (PROMPTS_DIR / "toy_domain.pddl").read_text(
        encoding="utf-8"
    )
    toy_problem = (PROMPTS_DIR / "toy_problem.pddl").read_text(
        encoding="utf-8"
    )
    return (
        "\n## Few-shot toy (invented; not a repository template)\n\n"
        "Example domain:\n```pddl\n"
        + toy_domain.strip()
        + "\n```\n\nExample problem:\n```pddl\n"
        + toy_problem.strip()
        + "\n```\n"
    )


def load_pddl_system_prompt() -> str:
    """All staged system texts + toy (for template-exclusion greps)."""
    parts = [_load_stage(name) for name in _STAGE_FILES]
    return "".join(parts) + _toy_block()


def _user_scene_task(command: str, scene_json: str, extra: str = "") -> str:
    body = (
        f"scene:\n{scene_json}\n\n"
        f"task: {str(command or '').strip()}\n"
    )
    if extra:
        body += extra
    return body


def build_pddl_refuse_prompts(
    command: str, scene_json: str, *, world: str | None = None
) -> tuple[str, str]:
    system = _load_stage("refuse.md") + "\n" + _catalog_block(world)
    return system, _user_scene_task(command, scene_json)


def build_pddl_schema_prompts(
    command: str, scene_json: str, *, world: str | None = None
) -> tuple[str, str]:
    del world  # schema stage has no catalog block; signature matches the others
    system = _load_stage("schema.md") + _toy_block()
    return system, _user_scene_task(command, scene_json)


def build_pddl_actions_prompts(
    command: str,
    scene_json: str,
    *,
    schema: dict[str, Any],
    world: str | None = None,
) -> tuple[str, str]:
    system = (
        _load_stage("actions.md")
        + "\n"
        + _catalog_block(world)
        + _toy_block()
    )
    extra = "\nschema:\n" + json.dumps(schema, indent=2) + "\n"
    return system, _user_scene_task(command, scene_json, extra)


def build_pddl_init_prompts(
    command: str,
    scene_json: str,
    *,
    schema: dict[str, Any],
    world: str | None = None,
) -> tuple[str, str]:
    del world
    system = _load_stage("init.md") + _toy_block()
    extra = "\nschema:\n" + json.dumps(schema, indent=2) + "\n"
    return system, _user_scene_task(command, scene_json, extra)


def build_pddl_goal_prompts(
    command: str,
    scene_json: str,
    *,
    schema: dict[str, Any],
    world: str | None = None,
) -> tuple[str, str]:
    del world
    system = _load_stage("goal.md") + _toy_block()
    extra = "\nschema:\n" + json.dumps(schema, indent=2) + "\n"
    return system, _user_scene_task(command, scene_json, extra)


def build_pddl_prompts(
    command: str, scene_json: str, *, world: str | None = None
) -> tuple[str, str]:
    """First-stage (refuse) prompts."""
    return build_pddl_refuse_prompts(command, scene_json, world=world)


def _as_refuse(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def _extract_json(raw: str) -> dict[str, Any] | None:
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _outcome(
    *,
    domain: str | None = None,
    problem: str | None = None,
    reason: str = "",
    refuse: bool = False,
    exit_reason: str,
    raw: str | None = None,
    error: str | None = None,
    syntax_errors: tuple[str, ...] = (),
    stages: tuple[dict[str, Any], ...] = (),
) -> LlmPddlOutcome:
    return LlmPddlOutcome(
        domain=domain,
        problem=problem,
        reason=reason,
        refuse=refuse,
        exit_reason=exit_reason,
        raw=raw,
        error=error,
        syntax_errors=syntax_errors,
        stages=stages,
    )


def _extract_fenced_pddl(raw: str) -> tuple[str, str] | None:
    blocks = [m.group(1).strip() for m in _FENCE.finditer(raw or "")]
    domain = next((b for b in blocks if _DOMAIN_HEAD.search(b)), None)
    problem = next((b for b in blocks if _PROBLEM_HEAD.search(b)), None)
    if domain and problem:
        return domain, problem
    return None


def _problem_errors(problem_text: str) -> list[str]:
    body = strip_pddl_comments(problem_text).lower()
    if not body.strip():
        return ["empty problem text"]
    errors: list[str] = []
    if "(:init" not in body:
        errors.append("problem missing (:init")
    if "(:goal" not in body:
        errors.append("problem missing (:goal")
    return errors


def parse_pddl_payload(
    raw: str,
    *,
    stages: tuple[dict[str, Any], ...] = (),
) -> LlmPddlOutcome:
    """JSON first; fenced PDDL only if JSON fails and two labeled defines exist."""
    domain: str | None = None
    problem: str | None = None
    reason = ""
    json_ok = False
    try:
        data = extract_json_object(raw)
        json_ok = True
    except (json.JSONDecodeError, ValueError, TypeError):
        data = None

    if json_ok and isinstance(data, dict):
        reason = str(data.get("reason") or "").strip()
        if _as_refuse(data.get("refuse")):
            return _outcome(
                reason=reason or "model refused",
                refuse=True,
                exit_reason=EXIT_REFUSED,
                raw=raw,
                stages=stages,
            )
        domain_raw = data.get("domain")
        problem_raw = data.get("problem")
        domain = domain_raw.strip() if isinstance(domain_raw, str) else None
        problem = problem_raw.strip() if isinstance(problem_raw, str) else None
    else:
        fenced = _extract_fenced_pddl(raw)
        if fenced is None:
            return _outcome(
                exit_reason=EXIT_INVALID,
                raw=raw,
                error="invalid PDDL JSON (no clear domain/problem fences)",
                stages=stages,
            )
        domain, problem = fenced

    if not domain or not problem:
        return _outcome(
            domain=domain,
            problem=problem,
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="missing domain or problem text",
            stages=stages,
        )

    syntax = tuple(pddl_syntax_errors(domain))
    if syntax:
        return _outcome(
            domain=domain,
            problem=problem,
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="; ".join(syntax),
            syntax_errors=syntax,
            stages=stages,
        )

    problem_errs = _problem_errors(problem)
    if problem_errs:
        return _outcome(
            domain=domain,
            problem=problem,
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="; ".join(problem_errs),
            syntax_errors=tuple(problem_errs),
            stages=stages,
        )

    return _outcome(
        domain=domain,
        problem=problem,
        reason=reason or "LLM PDDL",
        exit_reason=EXIT_AUTHORED,
        raw=raw,
        stages=stages,
    )


def _paren(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    if not s.startswith("("):
        return f"({s})"
    return s


def _atom(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "(" + " ".join(parts) + ")" if parts else ""
    return _paren(str(value))


def _objects_line(raw: Any) -> str:
    if isinstance(raw, dict):
        chunks = []
        for name, typ in raw.items():
            n = str(name).strip()
            t = str(typ).strip() or "object"
            if n:
                chunks.append(f"{n} - {t}")
        return "\n    ".join(chunks)
    if isinstance(raw, (list, tuple)):
        return "\n    ".join(str(x).strip() for x in raw if str(x).strip())
    return str(raw or "").strip()


def assemble_pddl(
    schema: dict[str, Any],
    actions: Sequence[dict[str, Any]],
    init_payload: dict[str, Any],
    goal_payload: dict[str, Any],
    *,
    world: str | None = None,
) -> tuple[str, str, tuple[str, ...]]:
    """Build domain+problem. Returns (domain, problem, unknown_action_names)."""
    types = schema.get("types") or ["object"]
    if isinstance(types, str):
        types = [types]
    types_s = " ".join(str(t).strip() for t in types if str(t).strip()) or "object"

    preds = schema.get("predicates") or []
    if isinstance(preds, str):
        preds = [preds]
    pred_lines = [_paren(p) for p in preds if str(p).strip()]
    pred_block = "\n    ".join(pred_lines) or "(dummy)"

    unknown: list[str] = []
    act_blocks: list[str] = []
    allowed = baseline_planner_actions_for_world(world)
    for raw_act in actions:
        if not isinstance(raw_act, dict):
            continue
        raw_name = str(raw_act.get("name") or "").strip()
        canon = normalize_baseline_action(raw_name)
        if canon not in allowed:
            if raw_name:
                unknown.append(raw_name)
        pddl_name = catalog_to_pddl_action(canon or raw_name or "act")
        params = str(raw_act.get("parameters") or "()").strip()
        if not params.startswith("("):
            params = f"({params})"
        pre = str(raw_act.get("precondition") or "(and)").strip()
        eff = str(raw_act.get("effect") or "(and)").strip()
        act_blocks.append(
            "  (:action "
            + pddl_name
            + "\n    :parameters "
            + params
            + "\n    :precondition "
            + pre
            + "\n    :effect "
            + eff
            + "\n  )"
        )
    if not act_blocks:
        act_blocks.append(
            "  (:action noop\n    :parameters ()\n"
            "    :precondition (and)\n    :effect (and)\n  )"
        )

    domain = (
        "(define (domain llm-authored)\n"
        "  (:requirements :strips :typing)\n"
        f"  (:types {types_s})\n"
        "  (:predicates\n    "
        + pred_block
        + "\n  )\n"
        + "\n".join(act_blocks)
        + "\n)\n"
    )

    objects_line = _objects_line(init_payload.get("objects")) or "dummy - object"
    init_atoms = init_payload.get("init") or []
    if isinstance(init_atoms, str):
        init_atoms = [init_atoms]
    init_block = "\n    ".join(
        a for a in (_atom(x) for x in init_atoms) if a
    ) or "(dummy)"

    goal_atoms = goal_payload.get("goal") or goal_payload.get("facts") or []
    if isinstance(goal_atoms, str):
        goal_atoms = [goal_atoms]
    goal_block = " ".join(a for a in (_atom(x) for x in goal_atoms) if a) or "(dummy)"

    problem = (
        "(define (problem llm-task)\n"
        "  (:domain llm-authored)\n"
        "  (:objects\n    "
        + objects_line
        + "\n  )\n"
        "  (:init\n    "
        + init_block
        + "\n  )\n"
        "  (:goal (and "
        + goal_block
        + "))\n"
        ")\n"
    )
    return domain.strip(), problem.strip(), tuple(unknown)


def _looks_like_oneshot(data: dict[str, Any] | None, raw: str) -> bool:
    if data is not None:
        domain = data.get("domain")
        problem = data.get("problem")
        if isinstance(domain, str) and domain.strip():
            return True
        if isinstance(problem, str) and problem.strip():
            return True
    return _extract_fenced_pddl(raw) is not None


def _generate(
    generate_fn: GenerateFn,
    system: str,
    user: str,
    *,
    stages: list[dict[str, Any]],
    name: str,
) -> str | LlmPddlOutcome:
    try:
        raw = invoke_llm(name, lambda: generate_fn(system, user))
    except Exception as exc:  # noqa: BLE001
        return _outcome(
            exit_reason=EXIT_INVALID,
            error=f"text LLM PDDL call failed ({name}): {exc}",
            stages=tuple(stages),
        )
    stages.append({"stage": name, "raw": raw, "s": last_llm_s()})
    return raw


def _schema_from_json(data: dict[str, Any]) -> dict[str, Any]:
    types = data.get("types") or ["object"]
    preds = data.get("predicates") or []
    if isinstance(types, str):
        types = [types]
    if isinstance(preds, str):
        preds = [preds]
    return {"types": list(types), "predicates": list(preds)}


def _actions_from_json(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("actions") or []
    if isinstance(raw, dict):
        raw = [raw]
    return [item for item in raw if isinstance(item, dict)]


def author_pddl(
    command: str,
    scene_json: str,
    generate_fn: GenerateFn,
    *,
    world: str | None = None,
) -> LlmPddlOutcome:
    """Refuse → schema → actions → init → goal, or one-shot domain+problem."""
    stages: list[dict[str, Any]] = []

    system, user = build_pddl_refuse_prompts(command, scene_json, world=world)
    raw = _generate(generate_fn, system, user, stages=stages, name="refuse")
    if isinstance(raw, LlmPddlOutcome):
        return raw

    data = _extract_json(raw)
    if _looks_like_oneshot(data, raw):
        return parse_pddl_payload(raw, stages=tuple(stages))
    if data is None:
        return parse_pddl_payload(raw, stages=tuple(stages))
    if _as_refuse(data.get("refuse")):
        return _outcome(
            reason=str(data.get("reason") or "").strip() or "model refused",
            refuse=True,
            exit_reason=EXIT_REFUSED,
            raw=raw,
            stages=tuple(stages),
        )

    system, user = build_pddl_schema_prompts(command, scene_json, world=world)
    raw = _generate(generate_fn, system, user, stages=stages, name="schema")
    if isinstance(raw, LlmPddlOutcome):
        return raw
    data = _extract_json(raw)
    if _looks_like_oneshot(data, raw):
        return parse_pddl_payload(raw, stages=tuple(stages))
    if data is None:
        return _outcome(
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="invalid schema JSON",
            stages=tuple(stages),
        )
    if _as_refuse(data.get("refuse")):
        return _outcome(
            reason=str(data.get("reason") or "").strip() or "model refused",
            refuse=True,
            exit_reason=EXIT_REFUSED,
            raw=raw,
            stages=tuple(stages),
        )
    schema = _schema_from_json(data)

    system, user = build_pddl_actions_prompts(
        command, scene_json, schema=schema, world=world
    )
    raw = _generate(generate_fn, system, user, stages=stages, name="actions")
    if isinstance(raw, LlmPddlOutcome):
        return raw
    data = _extract_json(raw)
    if _looks_like_oneshot(data, raw):
        return parse_pddl_payload(raw, stages=tuple(stages))
    if data is None:
        return _outcome(
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="invalid actions JSON",
            stages=tuple(stages),
        )
    actions = _actions_from_json(data)

    system, user = build_pddl_init_prompts(
        command, scene_json, schema=schema, world=world
    )
    raw = _generate(generate_fn, system, user, stages=stages, name="init")
    if isinstance(raw, LlmPddlOutcome):
        return raw
    data = _extract_json(raw)
    if _looks_like_oneshot(data, raw):
        return parse_pddl_payload(raw, stages=tuple(stages))
    if data is None:
        return _outcome(
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="invalid init JSON",
            stages=tuple(stages),
        )
    init_payload = data

    system, user = build_pddl_goal_prompts(
        command, scene_json, schema=schema, world=world
    )
    raw = _generate(generate_fn, system, user, stages=stages, name="goal")
    if isinstance(raw, LlmPddlOutcome):
        return raw
    data = _extract_json(raw)
    if _looks_like_oneshot(data, raw):
        return parse_pddl_payload(raw, stages=tuple(stages))
    if data is None:
        return _outcome(
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="invalid goal JSON",
            stages=tuple(stages),
        )
    if _as_refuse(data.get("refuse")):
        return _outcome(
            reason=str(data.get("reason") or "").strip() or "model refused",
            refuse=True,
            exit_reason=EXIT_REFUSED,
            raw=raw,
            stages=tuple(stages),
        )

    domain, problem, unknown = assemble_pddl(
        schema, actions, init_payload, data, world=world
    )
    reason = str(data.get("reason") or "").strip() or "LLM PDDL staged"
    if unknown:
        return _outcome(
            domain=domain,
            problem=problem,
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="actions not in catalog: " + ", ".join(unknown),
            stages=tuple(stages),
        )
    return parse_pddl_payload(
        json.dumps({"domain": domain, "problem": problem, "reason": reason}),
        stages=tuple(stages),
    )
