"""Unit tests for goal-LLM battery scoring (no GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from planner.problem_generator.goal_generator.battery_scoring import (
    facts_equal,
    score_case,
    summarize_by_level,
)

_REPO = Path(__file__).resolve().parent.parent


def test_facts_equal_order_insensitive():
    assert facts_equal(
        [["on", "a", "b"], ["holding", "a"]],
        [["holding", "a"], ["on", "a", "b"]],
    )


def test_score_exact_gold():
    s = score_case(
        gold=[["holding", "red_cup"]],
        expect_fail=False,
        result_ok=True,
        result_facts=[["holding", "red_cup"]],
        result_error=None,
        allowed_predicates={"holding", "on"},
        known_symbols={"red_cup", "table"},
        rule_ok=True,
        rule_facts=[["holding", "red_cup"]],
    )
    assert s["passed"] and s["exact_match"]


def test_score_expect_fail_refuses():
    s = score_case(
        gold=None,
        expect_fail=True,
        result_ok=False,
        result_facts=[],
        result_error="unsupported",
        allowed_predicates={"holding"},
        known_symbols={"red_cup"},
    )
    assert s["passed"]


def test_score_expect_fail_hallucination():
    s = score_case(
        gold=None,
        expect_fail=True,
        result_ok=True,
        result_facts=[["holding", "red_cup"]],
        result_error=None,
        allowed_predicates={"holding"},
        known_symbols={"red_cup"},
    )
    assert not s["passed"]


def test_score_soft_gold():
    s = score_case(
        gold=None,
        expect_fail=False,
        result_ok=True,
        result_facts=[["on", "red_cup", "shelf"]],
        result_error=None,
        allowed_predicates={"on", "holding"},
        known_symbols={"red_cup", "shelf"},
        soft_golds=[[["on", "red_cup", "shelf"]]],
        rule_ok=False,
        rule_facts=[],
    )
    assert s["passed"] and s["soft_match"] and s["llm_beats_rule"]


def test_summarize_by_level():
    rows = [
        {
            "id": "a",
            "level_id": "L0",
            "score": {
                "passed": True,
                "exact_match": True,
                "predicates_ok": True,
                "symbols_ok": True,
                "used_fallback": False,
            },
        },
        {
            "id": "b",
            "level_id": "L0",
            "score": {
                "passed": False,
                "exact_match": False,
                "predicates_ok": True,
                "symbols_ok": True,
                "used_fallback": True,
            },
        },
    ]
    summary = summarize_by_level(rows)
    assert summary["n"] == 2
    assert summary["levels"]["L0"]["pass_rate"] == 0.5
    assert summary["levels"]["L0"]["failed_ids"] == ["b"]


def test_battery_file_well_formed():
    path = _REPO / "tests" / "fixtures" / "goal_llm_eval" / "battery_v1.json"
    data = json.loads(path.read_text())
    assert data["levels"]
    ids = []
    for level in data["levels"]:
        for case in level["cases"]:
            ids.append(case["id"])
            assert "command" in case
            assert "scenario" in case
            assert case.get("gold") or case.get("expect_fail") or case.get("soft_golds")
    assert len(ids) == len(set(ids))


def test_eval_script_mock_llm():
    r = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "eval_goal_llm_battery.py"),
            "--mock-llm",
            "--levels",
            "L0,L1",
            "-q",
            "--json-out",
            str(_REPO / "data" / "goal_llm_eval" / "runs" / "_ci_mock.json"),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert "pass_rate" in r.stdout or "OVERALL" in r.stdout


def test_eval_script_mock_models_comparison_flag():
    """--models with --mock-llm still exercises the sweep CLI path (no GPU)."""
    r = subprocess.run(
        [
            sys.executable,
            str(_REPO / "scripts" / "eval_goal_llm_battery.py"),
            "--mock-llm",
            "--models",
            "mock-a,mock-b",
            "--levels",
            "L0",
            "-q",
            "--comparison-out",
            str(_REPO / "data" / "goal_llm_eval" / "runs" / "_ci_comparison.json"),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    cmp_path = _REPO / "data" / "goal_llm_eval" / "runs" / "_ci_comparison.json"
    assert cmp_path.is_file()
    data = json.loads(cmp_path.read_text())
    assert len(data["runs"]) == 2
    assert "COMPARISON" in r.stdout
