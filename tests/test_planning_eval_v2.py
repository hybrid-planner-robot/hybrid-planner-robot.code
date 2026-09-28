"""Offline tests for planning_eval suite v2 (mock-init). Does not replace v1."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "eval_planning_battery.py"
_SUITE = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v2_mock.json"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import eval_planning_battery as battery  # noqa: E402


def test_suite_v2_keeps_v1_ids_and_adds_long_cases():
    suite = battery.load_suite(_SUITE)
    ids = [c["id"] for c in suite["cases"]]
    assert "explicit_place" in ids
    assert "explicit_pour" in ids
    assert "explicit_shelf_then_stack" in ids
    assert "explicit_pour_then_stir" not in ids
    assert "implicit_pour_then_stir" not in ids
    assert len(suite["cases"]) == 26
    families = {c["family"] for c in suite["cases"]}
    assert "long_sequence" in families
    assert "multi_skill" not in families
    pour = next(c for c in suite["cases"] if c["id"] == "explicit_pour")
    assert pour["expect"]["plan_actions"] == ["pick", "pour"]
    assert "r1" in pour["expect"]
    assert "can-pour" in pour["expect"]["affordances"]


def test_v2_dry_default_arms_pass(tmp_path: Path):
    import subprocess

    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry",
            "--suite",
            str(_SUITE),
            "--out-dir",
            str(out),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    import json

    report = json.loads((out / "report.json").read_text())
    assert report["arms"] == ["llm_plan", "llm_pddl", "enrich", "r1"]
    assert report["passed"] == report["total"]
    assert any(row["arm"] == "r1" for row in report["cases"])
    r1_pour = next(
        row
        for row in report["cases"]
        if row["arm"] == "r1" and row["id"] == "explicit_pour"
    )
    assert r1_pour["affordances_declared"] is True
    assert r1_pour["init_assignment_ok"] is True
    md = (out / "report.md").read_text()
    assert "## R1 — affordance enrichment" in md
    assert "Claim 1" in md
    assert "Multi-skill" not in md
    assert all(row.get("family") != "multi_skill" for row in report["cases"])
    cases_md = (out / "report_cases.md").read_text()
    assert "| id | Task | arm | actions | new_predicates | Fd_action | verdict |" in cases_md


def test_build_loop_cmd_r1_and_mock_init():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "explicit_pour")
    cmd = battery.build_loop_cmd(
        case,
        suite["defaults"],
        arm="r1",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source="oracle",
        perception_only=False,
        mock_init=True,
        mock_llm=True,
    )
    assert "--enrichment-profile" in cmd
    assert "r1" in cmd
    assert "--mock-scene" in cmd
    assert "--mock-llm" in cmd
    assert "--online-enrichment" in cmd
    enrich = battery.build_loop_cmd(
        case,
        suite["defaults"],
        arm="enrich",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source="oracle",
        perception_only=False,
        mock_init=True,
        mock_llm=True,
    )
    assert "--enrichment-profile" not in enrich


def test_v1_dry_unchanged():
    import subprocess

    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--dry"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_mock_init_mock_llm_subset(tmp_path: Path):
    import json
    import subprocess

    out = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--mock-init",
            "--mock-llm",
            "--suite",
            str(_SUITE),
            "--arms",
            "enrich,r1",
            "--cases",
            "explicit_place,explicit_pour",
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
    assert report["passed"] == report["total"] == 4
    r1 = next(
        row
        for row in report["cases"]
        if row["arm"] == "r1" and row["id"] == "explicit_pour"
    )
    assert r1["affordances_declared"] is True
    assert r1.get("init_assignment_ok") is True


def test_baselines_hold_parent_llm_host_arms_do_not():
    assert battery.arm_loads_llm_in_parent("llm_plan") is True
    assert battery.arm_loads_llm_in_parent("llm_pddl") is True
    assert battery.arm_loads_llm_in_parent("llm_plan", mock_llm=True) is False
    assert battery.arm_loads_llm_in_parent("enrich") is False
    assert battery.arm_loads_llm_in_parent("r1") is False
