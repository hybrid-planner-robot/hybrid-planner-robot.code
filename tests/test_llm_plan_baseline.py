"""E1a — LLM-plan JSON path with injected generate_fn (no GPU, no FD)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.baseline_catalog import BASELINE_PLANNER_ACTIONS, baseline_action_summary_lines
from planner.call_timings import reset, snapshot
from planner.llm_plan_baseline import (
    EXIT_INVALID,
    EXIT_PLANNED,
    EXIT_REFUSED,
    LlmPlanAction,
    LlmPlanOutcome,
    build_plan_prompts,
    plan_to_host_fields,
    plan_with_llm,
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


def _json_plan(actions, *, reason="ok", refuse=False) -> str:
    return json.dumps({"actions": actions, "reason": reason, "refuse": refuse})


def test_place_wood_cube_pick_then_place():
    def generate(system: str, user: str) -> str:
        assert "wood_cube" in user
        assert "place the wood cube on the shelf" in user
        for line in baseline_action_summary_lines():
            assert line in system
        return _json_plan(
            [
                {"name": "pick", "args": ["wood_cube"]},
                {"name": "place", "args": ["wood_cube", "shelf"]},
            ],
            reason="place the cube on the shelf",
        )

    outcome = plan_with_llm(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE, generate
    )
    assert outcome.exit_reason == EXIT_PLANNED
    assert outcome.refuse is False
    assert [(a.name, list(a.args)) for a in outcome.actions] == [
        ("pick", ["wood_cube"]),
        ("place", ["wood_cube", "shelf"]),
    ]


def test_solder_refuse_is_distinct_from_invalid():
    def generate(system: str, user: str) -> str:
        return _json_plan([], reason="no solder skill", refuse=True)

    outcome = plan_with_llm("solder the pipe", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_REFUSED
    assert outcome.refuse is True
    assert outcome.actions == ()


def test_solder_action_name_invalidates_whole_plan():
    def generate(system: str, user: str) -> str:
        return _json_plan([{"name": "solder", "args": ["wood_cube"]}])

    outcome = plan_with_llm("solder the pipe", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_INVALID
    assert outcome.refuse is False
    assert outcome.actions == ()
    assert all(a.name != "solder" for a in outcome.actions)
    assert "solder" not in {a.name for a in outcome.actions}


def test_levitate_action_name_invalidates_whole_plan():
    def generate(system: str, user: str) -> str:
        return _json_plan(
            [
                {"name": "pick", "args": ["wood_cube"]},
                {"name": "levitate", "args": ["wood_cube"]},
            ]
        )

    outcome = plan_with_llm("levitate the cube", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_INVALID
    names = [a.name for a in outcome.actions]
    assert names == []
    assert "levitate" not in names
    assert "pick" not in names  # D8: whole plan dropped, not silent skip


def test_name_outside_list_is_invalid():
    def generate(system: str, user: str) -> str:
        return _json_plan([{"name": "fly", "args": ["wood_cube"]}])

    outcome = plan_with_llm("fly the cube", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_INVALID
    assert outcome.actions == ()
    assert not is_in_plan(outcome, "fly")


def test_drop_branch_keeps_in_list_actions_only():
    def generate(system: str, user: str) -> str:
        return _json_plan(
            [
                {"name": "pick", "args": ["wood_cube"]},
                {"name": "levitate", "args": ["wood_cube"]},
                {"name": "place", "args": ["wood_cube", "shelf"]},
            ]
        )

    outcome = plan_with_llm(
        "place the wood cube on the shelf",
        _WOOD_CUBE_SCENE,
        generate,
        on_unknown_action="drop",
    )
    assert outcome.exit_reason == EXIT_PLANNED
    names = [a.name for a in outcome.actions]
    assert names == ["pick", "place"]
    assert "levitate" not in names


def test_unreadable_json_is_invalid_plan():
    outcome = plan_with_llm(
        "place the wood cube on the shelf",
        _WOOD_CUBE_SCENE,
        lambda s, u: "not json at all",
    )
    assert outcome.exit_reason == EXIT_INVALID
    assert outcome.actions == ()
    assert outcome.refuse is False


def test_unknown_arg_invalidates_plan():
    def generate(system: str, user: str) -> str:
        return _json_plan([{"name": "pick", "args": ["moon_rock"]}])

    outcome = plan_with_llm("pick the moon rock", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_INVALID
    assert outcome.actions == ()


def test_hyphen_names_normalize_into_the_frozen_list():
    def generate(system: str, user: str) -> str:
        return _json_plan(
            [{"name": "pick-from-container", "args": ["wood_cube", "shelf"]}]
        )

    outcome = plan_with_llm("pick the cube from the shelf", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_PLANNED
    assert outcome.actions[0].name == "pick_from_container"
    assert outcome.actions[0].name in BASELINE_PLANNER_ACTIONS


def test_refuse_true_clears_actions_even_if_model_emitted_some():
    def generate(system: str, user: str) -> str:
        return _json_plan(
            [{"name": "pick", "args": ["wood_cube"]}],
            reason="cannot solder",
            refuse=True,
        )

    outcome = plan_with_llm("solder the pipe", _WOOD_CUBE_SCENE, generate)
    assert outcome.exit_reason == EXIT_REFUSED
    assert outcome.actions == ()


def test_plan_to_host_fields_matches_battery_shape():
    outcome = LlmPlanOutcome(
        actions=(
            LlmPlanAction("pick", ("wood_cube",)),
            LlmPlanAction("place", ("wood_cube", "shelf")),
        ),
        reason="ok",
        refuse=False,
        exit_reason=EXIT_PLANNED,
    )
    fields = plan_to_host_fields(outcome)
    assert fields["fd_actions"] == [
        "(pick wood_cube)",
        "(place wood_cube shelf)",
    ]
    assert fields["n_plan_actions"] == 2
    assert fields["fd_primitives"][0]["name"] == "pick"
    assert fields["fd_primitives"][1]["name"] == "place"


def test_prompt_has_closed_list_and_scene_not_repo_templates():
    system, user = build_plan_prompts(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE
    )
    assert "wood_cube" in user
    assert "place the wood cube on the shelf" in user
    assert "STAGE:plan_refuse" in system
    assert "pick(item)" in system
    assert "stack(item, item)" in system
    assert "manipulation_base" not in system
    assert "manipulation_stacking" not in system
    assert "(:action" not in system
    assert "Fast Downward" not in system
    assert "vlm_steps" not in system


def test_workshop_plan_prompt_lists_paint_not_pour():
    system, _user = build_plan_prompts(
        "paint the wood board", _WOOD_CUBE_SCENE, world="workshop"
    )
    assert "paint(" in system
    assert "drill(" in system
    assert "clamp(" in system
    assert "pour(" not in system
    household, _ = build_plan_prompts(
        "paint the wood board", _WOOD_CUBE_SCENE, world="kitchen"
    )
    assert "pour(" in household
    assert "paint(" not in household


def test_workshop_accepts_paint_and_rejects_pour():
    scene = json.dumps(
        {
            "objects": [
                {"name": "paint_can"},
                {"name": "wood_board"},
            ],
            "locations": [{"name": "table"}],
        }
    )

    def paint_ok(system: str, user: str) -> str:
        assert "paint(" in system
        return _json_plan(
            [
                {"name": "pick", "args": ["paint_can"]},
                {"name": "paint", "args": ["paint_can", "wood_board"]},
            ]
        )

    ok = plan_with_llm(
        "paint the wood board", scene, paint_ok, world="workshop"
    )
    assert ok.exit_reason == EXIT_PLANNED
    assert [a.name for a in ok.actions] == ["pick", "paint"]

    def pour_not_workshop(system: str, user: str) -> str:
        del system, user
        return _json_plan(
            [{"name": "pour", "args": ["paint_can", "wood_board"]}]
        )

    bad = plan_with_llm(
        "paint the wood board", scene, pour_not_workshop, world="workshop"
    )
    assert bad.exit_reason == EXIT_INVALID
    assert bad.actions == ()


def test_staged_plan_three_calls_when_refuse_is_just_a_gate():
    calls: list[str] = []

    def generate(system: str, user: str) -> str:
        calls.append(system.splitlines()[0])
        if "STAGE:plan_refuse" in system:
            return json.dumps({"refuse": False, "reason": "ok"})
        if "STAGE:plan_intent" in system:
            return json.dumps({"skills": ["pick", "place"], "reason": "move"})
        assert "STAGE:plan_ground" in system
        assert "pick, place" in user or "intent_skills:" in user
        return _json_plan(
            [
                {"name": "pick", "args": ["wood_cube"]},
                {"name": "place", "args": ["wood_cube", "shelf"]},
            ]
        )

    outcome = plan_with_llm(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE, generate
    )
    assert [c.replace("STAGE:", "") for c in calls] == [
        "plan_refuse",
        "plan_intent",
        "plan_ground",
    ]
    assert outcome.exit_reason == EXIT_PLANNED
    assert len(outcome.stages) == 3
    assert [(a.name, list(a.args)) for a in outcome.actions] == [
        ("pick", ["wood_cube"]),
        ("place", ["wood_cube", "shelf"]),
    ]


def test_staged_plan_records_split_and_total_llm_s():
    reset()

    def generate(system: str, user: str) -> str:
        if "STAGE:plan_refuse" in system:
            return json.dumps({"refuse": False, "reason": "ok"})
        if "STAGE:plan_intent" in system:
            return json.dumps({"skills": ["pick", "place"], "reason": "move"})
        return _json_plan(
            [
                {"name": "pick", "args": ["wood_cube"]},
                {"name": "place", "args": ["wood_cube", "shelf"]},
            ]
        )

    outcome = plan_with_llm(
        "place the wood cube on the shelf", _WOOD_CUBE_SCENE, generate
    )
    assert outcome.exit_reason == EXIT_PLANNED
    snap = snapshot()
    assert [c["name"] for c in snap["llm_calls"]] == ["refuse", "intent", "ground"]
    assert snap["n_llm_calls"] == 3
    assert snap["llm_s"] == round(sum(c["s"] for c in snap["llm_calls"]), 3)
    assert snap["fd_s"] is None
    assert [s["stage"] for s in outcome.stages] == ["refuse", "intent", "ground"]
    assert all(isinstance(s.get("s"), float) for s in outcome.stages)


def is_in_plan(outcome, name: str) -> bool:
    return any(a.name == name for a in outcome.actions)
