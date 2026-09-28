"""Offline tests for planning_eval suite v3. Does not replace v1 or v2."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "eval_planning_battery.py"
_SUITE = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v3_mock.json"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SCRIPT.parent))

import eval_planning_battery as battery  # noqa: E402


def _case(suite: dict, case_id: str) -> dict:
    return next(c for c in suite["cases"] if c["id"] == case_id)


def _rescore(case: dict, arm: str, **updates):
    row = battery._dry_row(case, arm)
    row.update(updates)
    return battery.attach_score(row)


def test_suite_v3_replaces_open_intent_and_keeps_claim_shape():
    suite = battery.load_suite(_SUITE)
    ids = [c["id"] for c in suite["cases"]]
    assert len(suite["cases"]) == 31
    assert len(ids) == len(set(ids))
    by_fam = {}
    for case in suite["cases"]:
        by_fam.setdefault(case["family"], []).append(case["id"])
    assert len(by_fam["template_complete"]) == 10
    assert len(by_fam["needs_enrichment"]) == 10
    assert len(by_fam["ungenerable"]) == 4
    assert len(by_fam["long_sequence"]) == 2
    assert by_fam["underspecified"] == [
        "implicit_pick",
        "implicit_drink",
        "implicit_thirsty",
        "implicit_stir",
        "implicit_tilt",
    ]
    tasks = {c["id"]: c["task"] for c in suite["cases"]}
    assert tasks["implicit_holding"] == "the wooden cube should be in the gripper"
    assert tasks["implicit_fill_glass"] == "the glass should be filled from the can"
    assert tasks["implicit_fill_mug"] == "the mug should be filled from the can"
    assert tasks["implicit_stirred"] == "the contents of the cup should be mixed"
    assert tasks["implicit_can_tilted"] == "the can should be tilted"
    assert "hand me" in tasks["implicit_pick"]
    assert "thirsty" in tasks["implicit_thirsty"]
    assert "something to drink" in tasks["implicit_drink"]
    mug = _case(suite, "implicit_fill_mug")
    assert mug["family"] == "needs_enrichment"
    assert mug["expect"]["gold_goal"] == [["poured", "can", "mug"]]
    assert "soft_golds" not in mug["expect"]
    thirsty = _case(suite, "implicit_thirsty")
    assert thirsty["family"] == "underspecified"
    assert thirsty["expect"]["accept"]["refuse"] is True


def test_v3_dry_passes_and_writes_intermediate_report(tmp_path: Path):
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
    report = json.loads((out / "report.json").read_text())
    assert report["passed"] == report["total"] == 124  # 31 × 4 default arms
    claims = report["claims"]
    assert claims["2_enrichment"]["plan_correct"]["all"]["n"] == 10
    assert claims["2_enrichment"]["plan_correct"]["all"]["ok"] == 10
    claim2_ids = {
        row["id"]
        for row in report["cases"]
        if row["arm"] == "enrich" and row["family"] == "needs_enrichment"
    }
    assert "implicit_thirsty" not in claim2_ids
    assert "implicit_fill_mug" in claim2_ids
    assert claims["underspecified"]["ok"]["all"]["n"] == 20  # 5 × 4
    assert claims["underspecified"]["ok"]["all"]["ok"] == 20
    md = (out / "report.md").read_text()
    assert "## Underspecified (not Claim 1–3)" in md
    cases_md = (out / "report_cases.md").read_text()
    header = "| id | Task | arm | actions | new_predicates | Fd_action | verdict |"
    assert header in cases_md
    assert "implicit_thirsty" in cases_md
    assert "implicit_fill_mug" in cases_md


def test_underspecified_accepts_refuse_and_vessel_offer_not_cut():
    suite = battery.load_suite(_SUITE)
    thirsty = _case(suite, "implicit_thirsty")
    dry = battery._dry_row(thirsty, "enrich")
    assert dry["ok"] is True
    assert "plausible" in dry["verdict"]

    refused = _rescore(
        thirsty,
        "enrich",
        exit_reason="refused",
        exit_code=3,
        fd_actions=[],
        fd_primitives=[],
        n_plan_actions=0,
        enrichment_used=False,
        enrichment_skills=[],
        goal_facts=[],
    )
    assert refused["ok"] is True
    assert refused["refused"] is True

    offer = _rescore(
        thirsty,
        "enrich",
        exit_reason="planned",
        fd_actions=["(pick glass)"],
        fd_primitives=[{"name": "pick", "args": ["glass"]}],
        n_plan_actions=1,
        goal_facts=[["holding", "glass"]],
        enrichment_used=False,
        enrichment_skills=[],
    )
    assert offer["ok"] is True
    assert offer["plan_correct"] is False

    cut = _rescore(
        thirsty,
        "enrich",
        exit_reason="planned",
        fd_actions=["(pick tea_box)", "(cut tea_box)"],
        fd_primitives=[
            {"name": "pick", "args": ["tea_box"]},
            {"name": "cut", "args": ["tea_box"]},
        ],
        n_plan_actions=2,
        goal_facts=[["cut-open", "tea_box"]],
        enrichment_skills=["cut"],
        enrichment_used=True,
    )
    assert cut["ok"] is False
    assert "spurious" in cut["verdict"]


def test_unique_fill_still_requires_pour():
    suite = battery.load_suite(_SUITE)
    fill = _case(suite, "implicit_fill_glass")
    holding_only = _rescore(
        fill,
        "enrich",
        exit_reason="planned",
        fd_actions=["(pick glass)"],
        fd_primitives=[{"name": "pick", "args": ["glass"]}],
        n_plan_actions=1,
        goal_facts=[["holding", "glass"]],
        enrichment_used=False,
        enrichment_skills=[],
    )
    assert holding_only["ok"] is False


def test_intermediate_report_columns_only():
    suite = battery.load_suite(_SUITE)
    row = battery._dry_row(_case(suite, "explicit_pour"), "r1")
    md = battery.format_intermediate_report([row])
    lines = md.splitlines()
    assert lines[0] == (
        "| id | Task | arm | actions | new_predicates | Fd_action | verdict |"
    )
    body = lines[2]
    assert "explicit_pour" in body
    assert "pour the can into the glass" in body
    assert "| r1 |" in body
