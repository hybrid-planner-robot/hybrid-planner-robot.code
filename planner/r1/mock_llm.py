"""Canned generate_fn for R1 / host ``--mock-llm`` (no GPU).

Dispatches on the system prompt: domain select, R1 enrich, R1 assignment,
R0 enrich, or goal facts. Task text is matched verbatim against suite_v1/v2.
"""

from __future__ import annotations

import json
import re
from typing import Any

from planner.r1.prompts import R1_ASSIGN_SYSTEM_PROMPT, R1_ENRICH_SYSTEM_PROMPT

_TASK_RE = re.compile(r"(?:task|command):\s*(.+)", re.IGNORECASE)


def _task_from_user(user: str) -> str:
    for line in user.splitlines():
        match = _TASK_RE.match(line.strip())
        if match:
            return match.group(1).strip()
    return user.strip().splitlines()[0] if user.strip() else ""


def _pour_r1() -> dict[str, Any]:
    return {
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
        "reason": "mock pour with affordances",
    }


def _stir_r1() -> dict[str, Any]:
    return {
        "skills": ["stir"],
        "actions": [
            {
                "name": "stir",
                "parameters": "(?c - item)",
                "precondition": "(and (holding ?c) (can-be-stirred ?c))",
                "effect": "(and (stirred ?c) (not (holding ?c)) (gripper-empty))",
            }
        ],
        "new_predicates": [
            "(stirred ?c - item)",
            "(can-be-stirred ?x - item)",
        ],
        "reason": "mock stir with affordance",
    }


def _cut_r1() -> dict[str, Any]:
    return {
        "skills": ["cut"],
        "actions": [
            {
                "name": "cut",
                "parameters": "(?i - item)",
                "precondition": "(and (holding ?i) (can-be-cut ?i))",
                "effect": "(and (cut-open ?i) (not (holding ?i)) (gripper-empty))",
            }
        ],
        "new_predicates": [
            "(cut-open ?i - item)",
            "(can-be-cut ?x - item)",
        ],
        "reason": "mock cut with affordance",
    }


def _tilt_r1() -> dict[str, Any]:
    return {
        "skills": ["tilt"],
        "actions": [
            {
                "name": "tilt",
                "parameters": "(?i - item)",
                "precondition": "(and (holding ?i) (can-tilt ?i))",
                "effect": "(and (tilted ?i))",
            }
        ],
        "new_predicates": [
            "(tilted ?i - item)",
            "(can-tilt ?x - item)",
        ],
        "reason": "mock tilt with affordance",
    }


def _drill_r1() -> dict[str, Any]:
    return {
        "skills": ["drill"],
        "actions": [
            {
                "name": "drill",
                "parameters": "(?i - item)",
                "precondition": "(and (holding ?i) (can-be-drilled ?i))",
                "effect": (
                    "(and (drilled ?i) (not (holding ?i)) (gripper-empty))"
                ),
            }
        ],
        "new_predicates": [
            "(drilled ?i - item)",
            "(can-be-drilled ?x - item)",
        ],
        "reason": "mock drill with affordance",
    }


def _paint_r1() -> dict[str, Any]:
    return {
        "skills": ["paint"],
        "actions": [
            {
                "name": "paint",
                "parameters": "(?s - item ?t - item)",
                "precondition": (
                    "(and (holding ?s) (can-paint ?s) (can-be-painted ?t))"
                ),
                "effect": (
                    "(and (painted ?s ?t) (not (holding ?s)) (gripper-empty))"
                ),
            }
        ],
        "new_predicates": [
            "(painted ?s - item ?t - item)",
            "(can-paint ?x - item)",
            "(can-be-painted ?x - item)",
        ],
        "reason": "mock paint with affordances",
    }


def _clamp_r1() -> dict[str, Any]:
    return {
        "skills": ["clamp"],
        "actions": [
            {
                "name": "clamp",
                "parameters": "(?i - item)",
                "precondition": "(and (holding ?i) (can-be-clamped ?i))",
                "effect": (
                    "(and (clamped ?i) (not (holding ?i)) (gripper-empty))"
                ),
            }
        ],
        "new_predicates": [
            "(clamped ?i - item)",
            "(can-be-clamped ?x - item)",
        ],
        "reason": "mock clamp with affordance",
    }


def _pour_then_stir_r1() -> dict[str, Any]:
    pour = _pour_r1()
    stir = _stir_r1()
    return {
        "skills": ["pour", "stir"],
        "actions": list(pour["actions"]) + list(stir["actions"]),
        "new_predicates": list(pour["new_predicates"]) + list(stir["new_predicates"]),
        "reason": "mock pour then stir",
    }


def _r0_action(skill: str) -> dict[str, Any]:
    bodies = {
        "pour": {
            "name": "pour",
            "parameters": "(?s - item ?t - item)",
            "precondition": "(and (holding ?s))",
            "effect": "(and (poured ?s ?t) (not (holding ?s)) (gripper-empty))",
            "new_predicates": ["(poured ?s - item ?t - item)"],
        },
        "stir": {
            "name": "stir",
            "parameters": "(?c - item)",
            "precondition": "(and (holding ?c))",
            "effect": "(and (stirred ?c) (not (holding ?c)) (gripper-empty))",
            "new_predicates": ["(stirred ?c - item)"],
        },
        "cut": {
            "name": "cut",
            "parameters": "(?i - item)",
            "precondition": "(and (holding ?i))",
            "effect": "(and (cut-open ?i) (not (holding ?i)) (gripper-empty))",
            "new_predicates": ["(cut-open ?i - item)"],
        },
        "tilt": {
            "name": "tilt",
            "parameters": "(?i - item)",
            "precondition": "(and (holding ?i))",
            "effect": "(and (tilted ?i))",
            "new_predicates": ["(tilted ?i - item)"],
        },
        "drill": {
            "name": "drill",
            "parameters": "(?i - item)",
            "precondition": "(and (holding ?i))",
            "effect": "(and (drilled ?i) (not (holding ?i)) (gripper-empty))",
            "new_predicates": ["(drilled ?i - item)"],
        },
        "paint": {
            "name": "paint",
            "parameters": "(?s - item ?t - item)",
            "precondition": "(and (holding ?s))",
            "effect": "(and (painted ?s ?t) (not (holding ?s)) (gripper-empty))",
            "new_predicates": ["(painted ?s - item ?t - item)"],
        },
        "clamp": {
            "name": "clamp",
            "parameters": "(?i - item)",
            "precondition": "(and (holding ?i))",
            "effect": "(and (clamped ?i) (not (holding ?i)) (gripper-empty))",
            "new_predicates": ["(clamped ?i - item)"],
        },
    }
    body = bodies[skill]
    return {
        "skill": skill,
        "action": {
            "name": body["name"],
            "parameters": body["parameters"],
            "precondition": body["precondition"],
            "effect": body["effect"],
        },
        "new_predicates": body["new_predicates"],
        "new_types": [],
        "reason": f"mock R0 {skill}",
    }


_UNGENERABLE = (
    "solder",
    "broken wire",
    "circuit",
    "join the wires",
    "float in the air",
    "stay in the air",
    "levitate",
    "hover",
    "weld",
    "welding",
)

_COMPLETE_CUES = (
    "place the wood cube",
    "wooden cube belongs on the shelf",
    "place the red cup",
    "red cup belongs on the shelf",
    "pick the wood cube",
    "hand me the wooden cube",
    "look at the wood cube",
    "inspect the wooden cube",
    "stack the wood cube",
    "small tower",
    "place the wood cube on the shelf then stack",
    "wooden cube belongs on the shelf, then put the blue box",
    "place the hammer",
    "hammer belongs on the metal tray",
    "the hammer belongs",
    "pick the screwdriver",
    "look at the power drill",
    "inspect the power drill",
    "stack the wood_block",
    "stack the wood block",
    "wood_block on top of wood_block2",
    "wood block on top of wood_block2",
)


def _is_ungenerable(task: str) -> bool:
    low = task.lower()
    return any(cue in low for cue in _UNGENERABLE)


def _is_complete(task: str) -> bool:
    low = task.lower()
    return any(cue in low for cue in _COMPLETE_CUES)


def _needed_skill(task: str) -> str | None:
    low = task.lower()
    if any(
        w in low
        for w in (
            "solder",
            "float",
            "air by itself",
            "hover",
            "circuit",
            "weld",
        )
    ):
        return None
    if ("drill" in low or "hole in" in low) and "look" not in low and "inspect" not in low:
        return "drill"
    if "paint" in low:
        return "paint"
    if "clamp" in low or "secure the" in low or "secured in the clamp" in low:
        return "clamp"
    if "pour" in low or "drink" in low or "thirsty" in low or "fill the cup" in low:
        if "stir" in low or "mix" in low:
            return "pour+stir"
        return "pour"
    if "stir" in low or "mix" in low or "sugar" in low:
        return "stir"
    if "cut" in low or "sliced" in low:
        return "cut"
    if "tilt" in low or "tip the can" in low:
        return "tilt"
    return None


def _select_payload(task: str) -> dict[str, Any]:
    if _is_ungenerable(task):
        return {
            "template": "manipulation_base",
            "completeness": "incomplete",
            "needed_skills": [],
            "reason": "mock ungenerable",
        }
    if "stack" in task.lower() or "tower" in task.lower() or "blue box on top" in task.lower():
        return {
            "template": "manipulation_stacking",
            "completeness": "complete",
            "needed_skills": [],
            "reason": "mock stacking complete",
        }
    if _is_complete(task):
        return {
            "template": "manipulation_base",
            "completeness": "complete",
            "needed_skills": [],
            "reason": "mock template complete",
        }
    skill = _needed_skill(task)
    if skill == "pour+stir":
        return {
            "template": "manipulation_base",
            "completeness": "incomplete",
            "needed_skills": ["pour", "stir"],
            "reason": "mock multi-skill gap",
        }
    if skill:
        return {
            "template": "manipulation_base",
            "completeness": "incomplete",
            "needed_skills": [skill],
            "reason": f"mock needs {skill}",
        }
    return {
        "template": "manipulation_base",
        "completeness": "complete",
        "needed_skills": [],
        "reason": "mock default complete",
    }


def _r1_enrich_payload(task: str) -> dict[str, Any]:
    if _is_ungenerable(task):
        return {"refuse": True, "reason": "mock refuse ungenerable"}
    skill = _needed_skill(task)
    if skill == "pour+stir":
        return _pour_then_stir_r1()
    if skill == "pour":
        return _pour_r1()
    if skill == "stir":
        return _stir_r1()
    if skill == "cut":
        return _cut_r1()
    if skill == "tilt":
        return _tilt_r1()
    if skill == "drill":
        return _drill_r1()
    if skill == "paint":
        return _paint_r1()
    if skill == "clamp":
        return _clamp_r1()
    return {"refuse": True, "reason": "mock refuse no catalog skill"}


def _r0_enrich_payload(task: str) -> dict[str, Any]:
    if _is_ungenerable(task):
        return {"refuse": True, "reason": "mock refuse ungenerable"}
    skill = _needed_skill(task)
    if skill == "pour+stir":
        return _r0_action("pour")
    if skill in {"pour", "stir", "cut", "tilt", "drill", "paint", "clamp"}:
        return _r0_action(skill)
    return {"refuse": True, "reason": "mock refuse no catalog skill"}


def _assign_payload(user: str) -> dict[str, Any]:
    facts: list[list[str]] = []
    low = user.lower()
    objects: list[str] = []
    for line in user.splitlines():
        if line.lower().startswith("scene_objects:"):
            objects = [o.strip() for o in line.split(":", 1)[1].split(",") if o.strip()]
    objs = {o.lower() for o in objects} or set()

    def has(*names: str) -> list[str]:
        return [n for n in names if n in objs]

    if "can-pour" in low:
        for name in has("can", "mug"):
            facts.append(["can-pour", name])
    if "can-be-poured" in low:
        for name in has("glass", "cup", "mug", "glass2"):
            facts.append(["can-be-poured", name])
    if "can-be-stirred" in low:
        for name in has("cup", "mug", "glass"):
            facts.append(["can-be-stirred", name])
    if "can-be-cut" in low:
        for name in has("tea_box", "wood_board"):
            facts.append(["can-be-cut", name])
    if "can-tilt" in low:
        for name in has("can"):
            facts.append(["can-tilt", name])
    if "can-stir" in low:
        for name in has("spoon"):
            facts.append(["can-stir", name])
    if "can-cut" in low:
        for name in has("knife"):
            facts.append(["can-cut", name])
    if "can-be-drilled" in low:
        for name in has("metal_plate", "wood_board"):
            facts.append(["can-be-drilled", name])
    if "can-paint" in low:
        for name in has("paint_can", "paint_brush"):
            facts.append(["can-paint", name])
    if "can-be-painted" in low:
        for name in has("wood_board", "metal_plate"):
            facts.append(["can-be-painted", name])
    if "can-be-clamped" in low:
        for name in has("wood_board", "wood_block", "wood_block2"):
            facts.append(["can-be-clamped", name])
    return {"init_facts": facts, "reason": "mock assignment"}


def _goal_payload(task: str) -> dict[str, Any]:
    low = task.lower()
    if "look" in low or "inspect" in low:
        item = "power_drill" if "drill" in low else "wood_cube"
        return {"facts": [["camera-aimed-at", item]]}
    if "paint" in low:
        return {"facts": [["painted", "paint_can", "wood_board"]]}
    if "drill" in low or "hole in" in low:
        target = "wood_board" if "wood_board" in low or "wood board" in low else "metal_plate"
        return {"facts": [["drilled", target]]}
    if "clamp" in low or "secure" in low:
        return {"facts": [["clamped", "wood_board"]]}
    if "pour" in low and "stir" in low:
        return {"facts": [["poured", "can", "cup"], ["stirred", "cup"]]}
    if "drink" in low or "thirsty" in low:
        return {"facts": [["poured", "can", "glass"]]}
    if "pour" in low or "fill the cup" in low:
        target = "cup" if "cup" in low else "glass"
        return {"facts": [["poured", "can", target]]}
    if "stir" in low or "mix" in low or "sugar" in low:
        return {"facts": [["stirred", "cup"]]}
    if "cut" in low or "sliced" in low:
        item = "wood_board" if "wood" in low or "board" in low else "tea_box"
        return {"facts": [["cut-open", item]]}
    if "tilt" in low or "tip the can" in low:
        return {"facts": [["tilted", "can"]]}
    if "pick" in low or "hand me" in low:
        item = "screwdriver" if "screwdriver" in low else "wood_cube"
        return {"facts": [["holding", item]]}
    if "stack" in low or "tower" in low or "blue box on top" in low or "on top of wood_block" in low:
        if "wood_block" in low or "wood block" in low:
            return {"facts": [["stacked-on", "wood_block", "wood_block2"]]}
        if "shelf" in low:
            return {"facts": [["stacked-on", "blue_box", "wood_cube"]]}
        return {"facts": [["stacked-on", "wood_cube", "blue_box"]]}
    if "red cup" in low:
        return {"facts": [["on", "red_cup", "shelf"]]}
    if "hammer" in low or "metal tray" in low:
        return {"facts": [["on", "hammer", "metal_tray"]]}
    if "shelf" in low:
        return {"facts": [["on", "wood_cube", "shelf"]]}
    return {"facts": [["on", "wood_cube", "shelf"]]}


def mock_fd_result(task: str, domain_text: str | None = None) -> dict:
    """Canned Fast Downward outcome for host ``--mock-fd`` (CI, no binary)."""
    from planner.domain_llm import domain_action_names
    from planner.fast_downward import result_from_actions

    if _is_ungenerable(task):
        return result_from_actions(None)
    names = domain_action_names(domain_text or "")
    skill = _needed_skill(task)
    low = task.lower()

    def have(*acts: str) -> bool:
        return all(a in names for a in acts)

    if skill == "pour+stir" and have("pour", "stir"):
        return result_from_actions(
            ["(pick can table)", "(pour can cup)", "(stir cup)"]
        )
    if skill == "pour" and have("pour"):
        target = "cup" if "cup" in low else "glass"
        return result_from_actions(
            [f"(pick can table)", f"(pour can {target})"]
        )
    if skill == "stir" and have("stir"):
        return result_from_actions(["(pick cup table)", "(stir cup)"])
    if skill == "cut" and have("cut"):
        item = "wood_board" if "wood" in low or "board" in low else "tea_box"
        return result_from_actions([f"(pick {item} table)", f"(cut {item})"])
    if skill == "tilt" and have("tilt"):
        return result_from_actions(["(pick can table)", "(tilt can)"])
    if skill == "drill" and have("drill"):
        item = "wood_board" if "wood_board" in low or "wood board" in low else "metal_plate"
        return result_from_actions([f"(pick {item} table)", f"(drill {item})"])
    if skill == "paint" and have("paint"):
        return result_from_actions(
            ["(pick paint_can table)", "(paint paint_can wood_board)"]
        )
    if skill == "clamp" and have("clamp"):
        return result_from_actions(
            ["(pick wood_board table)", "(clamp wood_board)"]
        )
    if "look" in low or "inspect" in low:
        item = "power_drill" if "drill" in low else "wood_cube"
        return result_from_actions([f"(look-at {item})"])
    if "pick" in low or "hand me" in low:
        item = "screwdriver" if "screwdriver" in low else "wood_cube"
        return result_from_actions([f"(pick {item} table)"])
    if "wood_block" in low or "wood block" in low:
        if "stack" in low or "on top" in low:
            return result_from_actions(
                ["(pick wood_block table)", "(stack wood_block wood_block2)"]
            )
    if ("stack" in low or "tower" in low or "blue box on top" in low) and (
        "shelf" in low
    ):
        if have("stack", "place"):
            return result_from_actions(
                [
                    "(pick wood_cube table)",
                    "(place wood_cube shelf)",
                    "(pick blue_box table)",
                    "(stack blue_box wood_cube)",
                ]
            )
    if "stack" in low or "tower" in low:
        return result_from_actions(
            ["(pick wood_cube table)", "(stack wood_cube blue_box)"]
        )
    if "red cup" in low:
        return result_from_actions(
            ["(pick red_cup table)", "(place red_cup shelf)"]
        )
    if "hammer" in low or "metal tray" in low:
        return result_from_actions(
            ["(pick hammer table)", "(place hammer metal_tray)"]
        )
    if "shelf" in low or "place" in low or "belongs" in low:
        return result_from_actions(
            ["(pick wood_cube table)", "(place wood_cube shelf)"]
        )
    return result_from_actions(["(pick wood_cube table)"])


def mock_host_generate_fn(system: str, user: str) -> str:
    """Single mock for select / R0 enrich / R1 enrich / assignment / goal."""
    task = _task_from_user(user)
    sys_l = (system or "").lower()
    if "init_facts" in sys_l or system.strip() == R1_ASSIGN_SYSTEM_PROMPT.strip():
        return json.dumps(_assign_payload(user))
    if "can-pour" in sys_l or "affordance" in sys_l or system.strip() == R1_ENRICH_SYSTEM_PROMPT.strip():
        return json.dumps(_r1_enrich_payload(task))
    if "current_domain_pddl:" in user and "not inventing a new robot capability" in sys_l:
        return json.dumps(_r0_enrich_payload(task))
    if "completeness" in sys_l or "needed_skills" in sys_l:
        return json.dumps(_select_payload(task))
    if "allowed_predicates" in user.lower() or "compact_scene_state" in user:
        return json.dumps(_goal_payload(task))
    if "current_domain_pddl:" in user:
        return json.dumps(_r1_enrich_payload(task))
    return json.dumps(_select_payload(task))
