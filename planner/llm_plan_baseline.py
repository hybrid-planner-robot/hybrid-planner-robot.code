"""
LLM-plan baseline: staged text-LLM calls, then a grounded action list.

Stages (no Fast Downward, no repo templates):

1. refuse — catalog can achieve the task?
2. intent — which catalog skills, no arguments
3. ground — JSON actions with scene symbols

Closed names come from ``BASELINE_PLANNER_ACTIONS`` (E0). A one-shot
payload that already contains ``actions`` is still accepted (CI injects).
Unknown action names invalidate the whole plan by default (D8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

from planner.baseline_catalog import (
    baseline_action_summary_lines,
    baseline_planner_actions_for_world,
    normalize_baseline_action,
)
from planner.plan_parser import PrimitiveCall, normalize_to_primitives
from planner.problem_generator.goal_generator.json_facts import extract_json_object
from planner.skill_catalog import catalog_to_pddl_action
from planner.call_timings import invoke_llm, last_llm_s
from planner.text_llm import GenerateFn
from prompts import load_prompt, prompt_path

__all__ = [
    "EXIT_INVALID",
    "EXIT_PLANNED",
    "EXIT_REFUSED",
    "LlmPlanAction",
    "LlmPlanOutcome",
    "PLAN_SYSTEM_PREAMBLE",
    "PROMPTS_DIR",
    "UnknownActionPolicy",
    "build_plan_prompts",
    "parse_plan_payload",
    "plan_to_host_fields",
    "plan_with_llm",
    "scene_symbols_from_json",
]

EXIT_PLANNED = "planned"
EXIT_REFUSED = "refused"
EXIT_INVALID = "invalid_plan"

UnknownActionPolicy = Literal["invalidate", "drop"]

PROMPTS_DIR = prompt_path("llm_plan")

PLAN_SYSTEM_PREAMBLE = load_prompt("llm_plan", "system.md")


@dataclass(frozen=True)
class LlmPlanAction:
    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class LlmPlanOutcome:
    actions: tuple[LlmPlanAction, ...]
    reason: str
    refuse: bool
    exit_reason: str
    raw: str | None = None
    error: str | None = None
    stages: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [
                {"name": a.name, "args": list(a.args)} for a in self.actions
            ],
            "reason": self.reason,
            "refuse": self.refuse,
            "exit_reason": self.exit_reason,
            "stages": [dict(s) for s in self.stages],
        }


def scene_symbols_from_json(scene_json: str) -> frozenset[str]:
    """Object and location names from a compact-scene JSON dump."""
    try:
        data = json.loads(scene_json)
    except (json.JSONDecodeError, TypeError):
        return frozenset()
    if not isinstance(data, dict):
        return frozenset()
    names: set[str] = set()
    for key in ("objects", "locations"):
        for item in data.get(key) or []:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
            else:
                name = str(item or "").strip()
            if name:
                names.add(name)
    return frozenset(names)


def _catalog_block(world: str | None = None) -> str:
    lines = "\n".join(
        f"- {line}" for line in baseline_action_summary_lines(world=world)
    )
    return "robot_actions (closed — nothing else is executable):\n" + lines + "\n"


def _load_stage(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip() + "\n"


def _user_scene_task(command: str, scene_json: str, extra: str = "") -> str:
    body = (
        f"scene:\n{scene_json}\n\n"
        f"task: {str(command or '').strip()}\n"
    )
    if extra:
        body += extra
    return body


def build_plan_refuse_prompts(
    command: str, scene_json: str, *, world: str | None = None
) -> tuple[str, str]:
    system = _load_stage("refuse.md") + "\n" + _catalog_block(world)
    return system, _user_scene_task(command, scene_json)


def build_plan_intent_prompts(
    command: str, scene_json: str, *, world: str | None = None
) -> tuple[str, str]:
    system = _load_stage("intent.md") + "\n" + _catalog_block(world)
    return system, _user_scene_task(command, scene_json)


def build_plan_ground_prompts(
    command: str,
    scene_json: str,
    *,
    intent_skills: Sequence[str] | None = None,
    world: str | None = None,
) -> tuple[str, str]:
    system = _load_stage("ground.md") + "\n" + _catalog_block(world)
    extra = ""
    if intent_skills:
        extra = "\nintent_skills: " + ", ".join(intent_skills) + "\n"
    return system, _user_scene_task(command, scene_json, extra)


def build_plan_prompts(
    command: str, scene_json: str, *, world: str | None = None
) -> tuple[str, str]:
    """First-stage (refuse) prompts; catalog + scene + NL."""
    return build_plan_refuse_prompts(command, scene_json, world=world)


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
    actions: tuple[LlmPlanAction, ...] = (),
    reason: str = "",
    refuse: bool = False,
    exit_reason: str,
    raw: str | None = None,
    error: str | None = None,
    stages: tuple[dict[str, Any], ...] = (),
) -> LlmPlanOutcome:
    return LlmPlanOutcome(
        actions=actions,
        reason=reason,
        refuse=refuse,
        exit_reason=exit_reason,
        raw=raw,
        error=error,
        stages=stages,
    )


def parse_plan_payload(
    raw: str,
    *,
    scene_symbols: frozenset[str],
    on_unknown_action: UnknownActionPolicy = "invalidate",
    stages: tuple[dict[str, Any], ...] = (),
    world: str | None = None,
) -> LlmPlanOutcome:
    """
    Validate LLM JSON against the closed action list and scene symbols.

    Default (D8): a name outside the world catalog invalidates the
    whole plan. ``on_unknown_action="drop"`` keeps only in-list actions.
    Unknown args always invalidate. Out-of-list names never appear in
    ``actions``.
    """
    allowed = baseline_planner_actions_for_world(world)
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return _outcome(
            exit_reason=EXIT_INVALID,
            raw=raw,
            error=f"invalid plan JSON: {exc}",
            stages=stages,
        )

    reason = str(data.get("reason") or "").strip()
    if _as_refuse(data.get("refuse")):
        return _outcome(
            reason=reason or "model refused",
            refuse=True,
            exit_reason=EXIT_REFUSED,
            raw=raw,
            stages=stages,
        )

    actions_raw = data.get("actions")
    if not isinstance(actions_raw, list):
        return _outcome(
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="plan JSON missing 'actions' list",
            stages=stages,
        )

    kept: list[LlmPlanAction] = []
    dropped: list[str] = []
    for item in actions_raw:
        if not isinstance(item, dict):
            return _outcome(
                reason=reason,
                exit_reason=EXIT_INVALID,
                raw=raw,
                error=f"invalid action entry: {item!r}",
                stages=stages,
            )
        name = normalize_baseline_action(str(item.get("name") or ""))
        raw_args = item.get("args") or []
        if isinstance(raw_args, str):
            raw_args = [raw_args]
        if not isinstance(raw_args, (list, tuple)):
            return _outcome(
                reason=reason,
                exit_reason=EXIT_INVALID,
                raw=raw,
                error=f"invalid args for {name!r}: {item.get('args')!r}",
                stages=stages,
            )
        args = tuple(str(a).strip() for a in raw_args if str(a).strip())

        if name not in allowed:
            if on_unknown_action == "drop":
                if name:
                    dropped.append(name)
                continue
            return _outcome(
                reason=reason,
                exit_reason=EXIT_INVALID,
                raw=raw,
                error=f"action {name!r} is not in the closed planner action list",
                stages=stages,
            )

        unknown = [a for a in args if a not in scene_symbols]
        if unknown:
            return _outcome(
                reason=reason,
                exit_reason=EXIT_INVALID,
                raw=raw,
                error=f"unknown scene symbols in args: {unknown!r}",
                stages=stages,
            )
        kept.append(LlmPlanAction(name=name, args=args))

    if not kept:
        extra = (
            f" (dropped non-catalog actions: {', '.join(dropped)})"
            if dropped
            else ""
        )
        return _outcome(
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error=f"empty action list{extra}" if extra else "empty action list",
            stages=stages,
        )

    if dropped:
        reason = (
            f"{reason} (dropped non-catalog actions: {', '.join(dropped)})"
        ).strip()
    return _outcome(
        actions=tuple(kept),
        reason=reason or "LLM plan",
        exit_reason=EXIT_PLANNED,
        raw=raw,
        stages=stages,
    )


def _generate(
    generate_fn: GenerateFn,
    system: str,
    user: str,
    *,
    stages: list[dict[str, Any]],
    name: str,
) -> str | LlmPlanOutcome:
    try:
        raw = invoke_llm(name, lambda: generate_fn(system, user))
    except Exception as exc:  # noqa: BLE001
        return _outcome(
            exit_reason=EXIT_INVALID,
            error=f"text LLM plan call failed ({name}): {exc}",
            stages=tuple(stages),
        )
    stages.append({"stage": name, "raw": raw, "s": last_llm_s()})
    return raw


def _parse_intent_skills(
    raw: str, *, world: str | None = None
) -> tuple[list[str], str, bool, str | None]:
    data = _extract_json(raw)
    if data is None:
        return [], "", False, "invalid intent JSON"
    reason = str(data.get("reason") or "").strip()
    if _as_refuse(data.get("refuse")):
        return [], reason or "model refused", True, None
    if isinstance(data.get("actions"), list):
        return [], reason, False, None
    raw_skills = data.get("skills") or []
    if isinstance(raw_skills, str):
        raw_skills = [raw_skills]
    if not isinstance(raw_skills, (list, tuple)):
        return [], reason, False, "intent JSON missing 'skills' list"
    allowed = baseline_planner_actions_for_world(world)
    skills: list[str] = []
    for item in raw_skills:
        name = normalize_baseline_action(str(item))
        if name in allowed and name not in skills:
            skills.append(name)
    return skills, reason, False, None


def plan_with_llm(
    command: str,
    scene_json: str,
    generate_fn: GenerateFn,
    *,
    on_unknown_action: UnknownActionPolicy = "invalidate",
    world: str | None = None,
) -> LlmPlanOutcome:
    """Refuse → intent → ground. One-shot ``actions`` payloads still parse."""
    symbols = scene_symbols_from_json(scene_json)
    stages: list[dict[str, Any]] = []

    system, user = build_plan_refuse_prompts(command, scene_json, world=world)
    raw = _generate(generate_fn, system, user, stages=stages, name="refuse")
    if isinstance(raw, LlmPlanOutcome):
        return raw

    data = _extract_json(raw)
    if data is None:
        return _outcome(
            exit_reason=EXIT_INVALID,
            raw=raw,
            error="invalid plan JSON: refuse stage",
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
    if isinstance(data.get("actions"), list):
        return parse_plan_payload(
            raw,
            scene_symbols=symbols,
            on_unknown_action=on_unknown_action,
            stages=tuple(stages),
            world=world,
        )

    system, user = build_plan_intent_prompts(command, scene_json, world=world)
    raw = _generate(generate_fn, system, user, stages=stages, name="intent")
    if isinstance(raw, LlmPlanOutcome):
        return raw
    intent = _extract_json(raw)
    if intent is not None and isinstance(intent.get("actions"), list):
        return parse_plan_payload(
            raw,
            scene_symbols=symbols,
            on_unknown_action=on_unknown_action,
            stages=tuple(stages),
            world=world,
        )
    skills, reason, refused, err = _parse_intent_skills(raw, world=world)
    if refused:
        return _outcome(
            reason=reason or "model refused",
            refuse=True,
            exit_reason=EXIT_REFUSED,
            raw=raw,
            stages=tuple(stages),
        )
    if err:
        return _outcome(
            reason=reason,
            exit_reason=EXIT_INVALID,
            raw=raw,
            error=err,
            stages=tuple(stages),
        )

    system, user = build_plan_ground_prompts(
        command, scene_json, intent_skills=skills, world=world
    )
    raw = _generate(generate_fn, system, user, stages=stages, name="ground")
    if isinstance(raw, LlmPlanOutcome):
        return raw
    return parse_plan_payload(
        raw,
        scene_symbols=symbols,
        on_unknown_action=on_unknown_action,
        stages=tuple(stages),
        world=world,
    )


def plan_to_host_fields(outcome: LlmPlanOutcome) -> dict[str, Any]:
    """
    Same ``fd_actions`` / ``fd_primitives`` / ``n_plan_actions`` shape the
    host writes for the battery. No Fast Downward — names come from the LLM
    plan, PDDL spelling via ``catalog_to_pddl_action``, ROS bodies via
    ``normalize_to_primitives``.
    """
    fd_actions: list[str] = []
    calls: list[PrimitiveCall] = []
    for action in outcome.actions:
        pddl_name = catalog_to_pddl_action(action.name)
        if action.args:
            fd_actions.append(f"({pddl_name} {' '.join(action.args)})")
        else:
            fd_actions.append(f"({pddl_name})")
        calls.append(PrimitiveCall(name=pddl_name, args=list(action.args)))
    prims = normalize_to_primitives(calls)
    return {
        "fd_actions": fd_actions,
        "fd_primitives": [{"name": p.name, "args": list(p.args)} for p in prims],
        "n_plan_actions": len(fd_actions),
    }
