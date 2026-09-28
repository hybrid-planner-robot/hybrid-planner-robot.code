"""Offline tests for planning_eval suite v4. Does not replace v1–v3."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "eval_planning_battery.py"
_TABLETOP = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v4_tabletop_mock.json"
_KITCHEN = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v4_kitchen_mock.json"
_WORKSHOP = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v4_workshop_mock.json"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SCRIPT.parent))

import eval_planning_battery as battery  # noqa: E402

_ARMS = ("llm_plan", "llm_pddl", "enrich", "r1")
_SHARED_BASE = {
    "template_complete",
    "ungenerable",
    "long_sequence",
    "underspecified",
}


def _family_ids(suite: dict, family: str) -> list[str]:
    return [c["id"] for c in suite["cases"] if c["family"] == family]


def _phrasing_split(cases: list[dict]) -> tuple[int, int]:
    e = sum(1 for c in cases if c["phrasing"] == "explicit")
    i = sum(1 for c in cases if c["phrasing"] == "implicit")
    return e, i


def test_v4_tabletop_has_no_enrichment_family():
    suite = battery.load_suite(_TABLETOP)
    assert suite["defaults"]["report_style"] == "families"
    assert suite["defaults"]["world"] == "tabletop"
    assert suite["defaults"]["arms"] == list(_ARMS)
    assert all(c["world"] == "tabletop" for c in suite["cases"])
    by_fam = {fam: _family_ids(suite, fam) for fam in _SHARED_BASE}
    assert "needs_enrichment" not in {c["family"] for c in suite["cases"]}
    assert len(by_fam["template_complete"]) == 10
    assert len(by_fam["ungenerable"]) == 4
    assert len(by_fam["long_sequence"]) == 2
    assert len(by_fam["underspecified"]) == 4
    assert len(suite["cases"]) == 20
    e, i = _phrasing_split(suite["cases"])
    assert (e, i) == (10, 10)
    ungen_tasks = {
        c["task"] for c in suite["cases"] if c["family"] == "ungenerable"
    }
    assert "solder the broken wire" in ungen_tasks
    assert "the circuit is broken, join the wires" in ungen_tasks
    assert "make the cup float in the air" in ungen_tasks
    assert all("on the board" not in t for t in ungen_tasks)


def test_v4_kitchen_workshop_are_symmetric():
    kitchen = battery.load_suite(_KITCHEN)
    workshop = battery.load_suite(_WORKSHOP)
    assert kitchen["defaults"]["world"] == "kitchen"
    assert workshop["defaults"]["world"] == "workshop"
    k_fam = {
        fam: len(_family_ids(kitchen, fam))
        for fam in list(_SHARED_BASE) + ["needs_enrichment"]
    }
    w_fam = {
        fam: len(_family_ids(workshop, fam))
        for fam in list(_SHARED_BASE) + ["needs_enrichment"]
    }
    assert k_fam == w_fam
    assert k_fam["template_complete"] == 10
    assert k_fam["needs_enrichment"] == 10
    assert k_fam["ungenerable"] == 4
    assert k_fam["long_sequence"] == 2
    assert k_fam["underspecified"] == 4
    assert len(kitchen["cases"]) == len(workshop["cases"]) == 30
    for suite in (kitchen, workshop):
        e, i = _phrasing_split(suite["cases"])
        assert (e, i) == (15, 15)
        gap_e, gap_i = _phrasing_split(
            [c for c in suite["cases"] if c["family"] == "needs_enrichment"]
        )
        assert (gap_e, gap_i) == (5, 5)
    assert _family_ids(kitchen, "template_complete") == _family_ids(
        workshop, "template_complete"
    )
    assert _family_ids(kitchen, "ungenerable") == _family_ids(
        workshop, "ungenerable"
    )
    k_skills = {
        c["expect"]["skill"]
        for c in kitchen["cases"]
        if c["family"] == "needs_enrichment"
    }
    w_skills = {
        c["expect"]["skill"]
        for c in workshop["cases"]
        if c["family"] == "needs_enrichment"
    }
    assert k_skills == {"pour", "stir", "cut", "tilt"}
    assert w_skills == {"cut", "drill", "paint", "clamp"}
    k_ungen = {
        c["task"]
        for c in kitchen["cases"]
        if c["family"] == "ungenerable"
    }
    w_ungen = {
        c["task"]
        for c in workshop["cases"]
        if c["family"] == "ungenerable"
    }
    t_ungen = {
        c["task"]
        for c in battery.load_suite(_TABLETOP)["cases"]
        if c["family"] == "ungenerable"
    }
    assert k_ungen == w_ungen == t_ungen
    assert "multi_skill" not in {c["family"] for c in kitchen["cases"]}
    assert "multi_skill" not in {c["family"] for c in workshop["cases"]}


def test_v4_kitchen_oracle_has_place_target():
    scene = json.loads(
        (_REPO / "tests" / "fixtures" / "llm_plan" / "kitchen.json").read_text()
    )
    locs = {loc["name"] for loc in scene["locations"]}
    assert locs == {"table", "tray"}
    objs = {o["name"] for o in scene["objects"]}
    assert {"mug", "cup", "tea_box", "plate", "glass"} <= objs


def _dry_suite(suite_path: Path, tmp_path: Path, n_cases: int) -> dict:
    out = tmp_path / suite_path.stem
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry",
            "--suite",
            str(suite_path),
            "--out-dir",
            str(out),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] == report["total"] == n_cases * 4
    assert report["arms"] == list(_ARMS)
    assert "claims" not in report
    families = report["families"]["by_family"]
    assert "template_complete" in families
    assert "ungenerable" in families
    md = (out / "report.md").read_text()
    assert "## Families" in md
    assert "## Timings" in md
    assert "llm_s" in md
    assert "Claim 1" not in md
    assert "Claim 2" not in md
    assert "Claim 3" not in md
    timings = report["timings"]
    assert timings["all"]["n"] == n_cases * 4
    assert "llm_plan" in timings["by_arm"]
    assert timings["by_arm"]["llm_plan"]["llm_s"]["n"] == n_cases
    assert timings["by_arm"]["llm_plan"]["fd_s"]["n"] == 0
    case = report["cases"][0]
    assert "timings" in case
    assert "llm_calls" in case
    return report


def test_v4_tabletop_dry_writes_family_report(tmp_path: Path):
    report = _dry_suite(_TABLETOP, tmp_path, 20)
    assert "needs_enrichment" not in report["families"]["by_family"]
    worlds = {row["world"] for row in report["cases"]}
    assert worlds == {"tabletop"}
    enrich_complete = [
        row
        for row in report["cases"]
        if row["arm"] == "enrich" and row["family"] == "template_complete"
    ]
    assert len(enrich_complete) == 10
    assert all(row["path"] == "complete" for row in enrich_complete)
    assert all(row["ok"] for row in enrich_complete)


def test_v4_kitchen_and_workshop_dry(tmp_path: Path):
    k = _dry_suite(_KITCHEN, tmp_path, 30)
    w = _dry_suite(_WORKSHOP, tmp_path, 30)
    assert k["families"]["by_family"]["needs_enrichment"]["enrich"]["plan_correct"][
        "all"
    ]["n"] == 10
    assert w["families"]["by_family"]["needs_enrichment"]["enrich"]["plan_correct"][
        "all"
    ]["n"] == 10
    assert {row["world"] for row in k["cases"]} == {"kitchen"}
    assert {row["world"] for row in w["cases"]} == {"workshop"}


def test_v4_kitchen_place_nl_uses_tray():
    kitchen = battery.load_suite(_KITCHEN)
    place = next(c for c in kitchen["cases"] if c["id"] == "explicit_place")
    assert place["task"] == "place the mug on the tray"
    assert place["expect"]["gold_goal"] == [["on", "mug", "tray"]]
    assert "target tray" not in {c["task"] for c in kitchen["cases"]}


def test_v4_stack_nl_avoids_tower_on_kitchen_and_workshop():
    for path in (_KITCHEN, _WORKSHOP):
        suite = battery.load_suite(path)
        stack = next(c for c in suite["cases"] if c["id"] == "implicit_stack")
        assert "tower" not in stack["task"].lower()
        assert "on top of" in stack["task"]
    tabletop = battery.load_suite(_TABLETOP)
    assert "tower" in next(
        c["task"] for c in tabletop["cases"] if c["id"] == "implicit_stack"
    )


def test_v4_handover_allowlist_includes_place():
    for path in (_TABLETOP, _KITCHEN, _WORKSHOP):
        suite = battery.load_suite(path)
        for case_id in ("explicit_handover", "implicit_handover"):
            case = next(c for c in suite["cases"] if c["id"] == case_id)
            assert "place" in case["expect"]["accept"]["allowed_actions"]
            assert "pick" in case["expect"]["accept"]["allowed_actions"]


def test_underspecified_look_at_is_not_spurious():
    expect = {
        "gold_goal": [["holding", "wood_cube"]],
        "accept": {
            "refuse": True,
            "allowed_actions": ["pick", "place"],
            "allowed_objects": ["wood_cube"],
        },
    }
    row = {
        "refused": False,
        "plan_found": True,
        "invalid_pddl": False,
        "invalid_plan": False,
        "goal_facts": [],
        "fd_primitives": [
            {"name": "look-at", "args": ["wood_cube"]},
            {"name": "pick", "args": ["wood_cube"]},
            {"name": "place", "args": ["wood_cube", "table"]},
            {"name": "navigate-to", "args": ["table"]},
        ],
    }
    assert battery.underspecified_case_ok(row, expect) is True


def test_underspecified_gap_skill_still_spurious():
    expect = {
        "accept": {
            "refuse": True,
            "allowed_actions": ["pick", "place"],
            "allowed_objects": ["mug"],
        },
    }
    row = {
        "refused": False,
        "plan_found": True,
        "invalid_pddl": False,
        "invalid_plan": False,
        "goal_facts": [],
        "fd_primitives": [
            {"name": "pour", "args": ["tea_box", "mug"]},
        ],
    }
    assert battery.underspecified_case_ok(row, expect) is False
