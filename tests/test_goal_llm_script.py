"""Smoke tests for scripts/test_goal_llm.py (rule-only; no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "test_goal_llm.py"


def test_list_scenarios():
    r = subprocess.run(
        [sys.executable, str(_SCRIPT), "--list-scenarios"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0
    assert "table" in r.stdout
    assert "holding" in r.stdout


def test_rule_only_single_command(tmp_path):
    out = tmp_path / "out.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--scenario",
            "table",
            "--command",
            "place red_cup on shelf",
            "--rule-only",
            "--json-out",
            str(out),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "on red_cup shelf" in r.stdout.replace("(", " ").replace(")", " ")
    data = json.loads(out.read_text())
    assert data["results"][0]["rule_based"]["facts"] == [["on", "red_cup", "shelf"]]


def test_rule_only_batch():
    batch = _REPO / "tests" / "fixtures" / "goal_llm_eval" / "commands_example.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "-s",
            "table",
            "--batch",
            str(batch),
            "--rule-only",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "pick up the red cup" in r.stdout
