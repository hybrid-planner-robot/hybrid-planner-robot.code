"""Session 17: offline tests for hybrid mini-eval harness (no Gazebo / GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "eval_hybrid_mini.py"
_SUITE = _REPO / "tests" / "fixtures" / "hybrid_mini_eval" / "suite_v1.json"
_DRY = _REPO / "tests" / "fixtures" / "hybrid_mini_eval" / "comparison_dry.json"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import eval_hybrid_mini as mini  # noqa: E402


def test_suite_is_small_mvp_tabletop():
    suite = mini.load_suite(_SUITE)
    assert suite["world"] == "tabletop"
    assert 1 <= len(suite["tasks"]) <= 10
    mode_ids = {m["id"] for m in suite["modes"]}
    assert {"legacy", "hybrid_mvp", "hybrid_mvp_llm", "hybrid_full"} <= mode_ids
    for t in suite["tasks"]:
        for p in t.get("primitives") or []:
            assert p in {"pick", "place", "look_at", "stack", "unstack"}


def test_select_modes_skips_optional_by_default():
    suite = mini.load_suite(_SUITE)
    modes = mini.select_modes(suite, None)
    assert [m["id"] for m in modes] == ["legacy", "hybrid_mvp"]
    all_modes = mini.select_modes(suite, None, include_optional=True)
    assert len(all_modes) == 4


def test_row_from_summary_and_comparison():
    rows = mini.load_dry_rows(_DRY)
    assert len(rows) >= 8
    comparison = mini.build_comparison(rows)
    assert comparison["n_runs"] == len(rows)
    assert "legacy" in comparison["modes"]
    assert "hybrid_mvp" in comparison["modes"]
    legacy = comparison["modes"]["legacy"]
    assert 0.0 <= legacy["success_rate"] <= 1.0
    assert legacy["n"] == 4
    table = mini.format_comparison_table(comparison)
    assert "legacy" in table
    assert "hybrid_full" in table
    assert "success" in table.lower() or "%" in table


def test_aggregate_summary_json(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_a.mkdir()
    (run_a / "summary.json").write_text(
        json.dumps(
            {
                "task": "pick the red cup",
                "world": "tabletop",
                "hybrid": "off",
                "goal_backend": None,
                "success": True,
                "exit_reason": "success",
                "n_steps": 1,
                "replan_count": 0,
                "wall_time_s": 12.5,
                "vlm_call_count": 0,
                "verdict_counts": {"GREEN": 0, "YELLOW": 0, "RED": 0},
                "goal_fallback_count": 0,
            }
        ),
        encoding="utf-8",
    )
    run_b = tmp_path / "run_b"
    run_b.mkdir()
    (run_b / "summary.json").write_text(
        json.dumps(
            {
                "task": "place the red cup on the shelf",
                "world": "tabletop",
                "hybrid": "mvp",
                "goal_backend": "rule_based",
                "success": True,
                "exit_reason": "success",
                "n_steps": 2,
                "replan_count": 1,
                "wall_time_s": 40.0,
                "hybrid_problem_gen": {
                    "vlm_call_count": 0,
                    "verdict_counts": {"GREEN": 0, "YELLOW": 0, "RED": 0},
                    "goal_backend_used": "rule_based",
                    "goal_fallback_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = mini.aggregate_run_dirs(tmp_path)
    assert len(rows) == 2
    comparison = mini.build_comparison(rows)
    assert comparison["modes"]["legacy"]["success_rate"] == 1.0
    assert comparison["modes"]["hybrid_mvp"]["mean_replan_count"] == 1.0


def test_eval_script_dry_cli(tmp_path: Path):
    out = tmp_path / "comparison.json"
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry",
            "--out",
            str(out),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["session"] == 17
    assert "legacy" in payload["comparison"]["modes"]
    assert payload["known_gaps"]
    assert "legacy" in payload["table_markdown"]


def test_loop_cmd_flags():
    cmd_legacy = mini._loop_cmd(
        "pick the red cup",
        {"id": "legacy", "hybrid": "off", "goal_backend": None},
        world="tabletop",
        max_steps=10,
        container="vlm_ros2",
        sudo_docker=False,
        python_bin="python",
    )
    assert "--hybrid" not in cmd_legacy
    cmd_mvp = mini._loop_cmd(
        "pick the red cup",
        {"id": "hybrid_mvp", "hybrid": "mvp", "goal_backend": "rule_based"},
        world="tabletop",
        max_steps=10,
        container="vlm_ros2",
        sudo_docker=False,
        python_bin="python",
    )
    assert cmd_mvp[cmd_mvp.index("--hybrid") + 1] == "mvp"
    assert cmd_mvp[cmd_mvp.index("--goal-backend") + 1] == "rule_based"
