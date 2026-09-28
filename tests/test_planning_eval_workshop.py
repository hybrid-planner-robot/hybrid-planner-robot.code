"""Offline tests for the workshop planning-eval suite. Does not change v3 claims."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "eval_planning_battery.py"
_SUITE = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v3_workshop_mock.json"
_V3 = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v3_mock.json"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_SCRIPT.parent))

import eval_planning_battery as battery  # noqa: E402


def test_workshop_suite_is_parallel_to_v3_not_mixed_into_claim2():
    workshop = battery.load_suite(_SUITE)
    v3 = battery.load_suite(_V3)
    assert all(c["world"] == "workshop" for c in workshop["cases"])
    assert all(c.get("world") != "workshop" for c in v3["cases"])
    v3_gap = [c for c in v3["cases"] if c["family"] == "needs_enrichment"]
    assert len(v3_gap) == 10
    by_fam: dict[str, list[str]] = {}
    for case in workshop["cases"]:
        by_fam.setdefault(case["family"], []).append(case["id"])
    assert len(by_fam["template_complete"]) == 6
    assert len(by_fam["needs_enrichment"]) == 8
    assert len(by_fam["ungenerable"]) == 2
    skills = {
        c["expect"]["skill"]
        for c in workshop["cases"]
        if c["family"] == "needs_enrichment"
    }
    assert skills == {"cut", "drill", "paint", "clamp"}
    assert "pour" not in skills


def test_workshop_suite_dry_passes(tmp_path: Path):
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
    assert report["passed"] == report["total"] == 64  # 16 × 4 arms
    claims = report["claims"]
    assert claims["2_enrichment"]["plan_correct"]["all"]["n"] == 8
    assert claims["2_enrichment"]["plan_correct"]["all"]["ok"] == 8
    worlds = {row["world"] for row in report["cases"]}
    assert worlds == {"workshop"}
