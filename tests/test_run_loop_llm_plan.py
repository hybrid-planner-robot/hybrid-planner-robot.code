"""E1b/E1c — llm_plan entry point: XOR, mock-scene wiring, summary artefacts."""

from __future__ import annotations

import ast
import json
import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "run_loop_llm_plan.py"
_HOST = _REPO / "scripts" / "run_loop_host.py"
_FIXTURE = _REPO / "tests" / "fixtures" / "llm_plan" / "wood_cube_table.json"

_SPEC = importlib.util.spec_from_file_location("run_loop_llm_plan", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["run_loop_llm_plan"] = _mod
_SPEC.loader.exec_module(_mod)


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_help_green():
    r = _run(["--help"])
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "--task" in out
    assert "--world" in out
    assert "--scene-source" in out
    assert "--perception-only" in out
    assert "--plan-only" in out
    assert "--container" in out
    assert "--mock-scene" in out
    assert "XOR" in out
    assert "scene_from_dino_payload" in out or "acquire_live_scene" in out


def test_rejects_online_enrichment():
    r = _run(["--task", "place the wood cube on the shelf", "--online-enrichment", "1"])
    assert r.returncode != 0
    err = r.stderr + r.stdout
    assert "XOR" in err
    assert "online-enrichment" in err


def test_rejects_control_fd():
    r = _run(["--task", "place the wood cube on the shelf", "--control", "fd"])
    assert r.returncode != 0
    err = r.stderr + r.stdout
    assert "XOR" in err
    assert "--control" in err


def test_without_mock_scene_stops_honestly():
    r = _run(
        [
            "--task",
            "place the wood cube on the shelf",
            "--world",
            "tabletop",
            "--container",
            "vlm_ros2_no_such_eval_container",
        ]
    )
    assert r.returncode != 0
    assert "mock-scene" in (r.stdout + r.stderr)


def test_script_does_not_import_enricher_fd_or_pruner():
    imported = _imported_modules(_SCRIPT)
    assert "planner.online_enrichment" not in imported
    assert "planner.fast_downward" not in imported
    assert "planner.domain_pruner" not in imported
    assert "planner.llm_plan_baseline" in imported
    assert any("hybrid_runtime" in m for m in imported)
    assert "vlm.perception" not in imported
    assert any("live_scene" in m for m in imported)
    src = ast.parse(_SCRIPT.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(src):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    assert "resolve_domain_for_task" not in names
    assert "select_domain_template" not in names


def test_host_has_no_baseline_flag():
    r = subprocess.run(
        [sys.executable, str(_HOST), "--help"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert "--baseline" not in r.stdout
    assert "--online-enrichment" in r.stdout
    assert "--control" in r.stdout
    help_text = r.stdout
    # Host --plan-only gate is unchanged (still documents control=fd).
    assert "--plan-only" in help_text


def test_parser_mirrors_minimum_flags():
    dests = {action.dest for action in _mod.build_parser()._actions}
    assert dests >= {
        "task",
        "world",
        "scene_source",
        "perception_only",
        "plan_only",
        "container",
        "sudo_docker",
        "online_enrichment",
        "control",
        "mock_scene",
        "mock_llm",
        "scene_file",
    }


def test_mock_place_writes_summary_with_actions(tmp_path):
    run_dir = tmp_path / "run"
    r = _run(
        [
            "--task",
            "place the wood cube on the shelf",
            "--world",
            "tabletop",
            "--scene-source",
            "dino",
            "--perception-only",
            "--plan-only",
            "--mock-scene",
            "--scene-file",
            str(_FIXTURE),
            "--mock-llm",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert r.returncode == 0, r.stderr + r.stdout
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    plan = json.loads((run_dir / "llm_plan.json").read_text(encoding="utf-8"))
    scene = json.loads((run_dir / "scene_compact.json").read_text(encoding="utf-8"))
    assert summary["baseline"] == "llm_plan"
    assert summary["exit_reason"] == "planned"
    assert summary["plan_only"] is True
    assert summary["enrichment_used"] is False
    assert summary["pddl_domain"] is None
    assert summary["pddl_problem"] is None
    assert summary["pddl_init"] is None
    assert summary["pddl_goal"] is None
    assert summary["init_facts"] is None
    assert summary["domain_template"] is None
    assert summary["n_plan_actions"] == 2
    assert "(pick wood_cube)" in summary["fd_actions"]
    assert "(place wood_cube shelf)" in summary["fd_actions"]
    assert plan["actions"][0]["name"] == "pick"
    assert "domain_template" not in scene
    assert "wood_cube" in {o["name"] for o in scene["objects"]}


def test_mock_solder_refuses(tmp_path):
    run_dir = tmp_path / "run"
    r = _run(
        [
            "--task",
            "solder the pipe",
            "--world",
            "tabletop",
            "--mock-scene",
            "--mock-llm",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert r.returncode == 3, r.stderr + r.stdout
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_reason"] == "refused"
    assert summary["fd_actions"] == []
    assert summary["n_plan_actions"] == 0
    assert summary["refuse_reason"]
    assert summary["pddl_domain"] is None
    plan = json.loads((run_dir / "llm_plan.json").read_text(encoding="utf-8"))
    assert plan["exit_reason"] == "refused"
    assert plan["actions"] == []
    assert plan.get("raw")


def test_invalid_plan_still_writes_llm_plan_json(tmp_path):
    run_dir = tmp_path / "run"
    code = _mod.main(
        [
            "--task",
            "place the wood cube on the shelf",
            "--world",
            "tabletop",
            "--mock-scene",
            "--scene-file",
            str(_FIXTURE),
            "--run-dir",
            str(run_dir),
        ],
        generate_fn=lambda s, u: "this is not plan json",
    )
    assert code == 2
    plan = json.loads((run_dir / "llm_plan.json").read_text(encoding="utf-8"))
    assert plan["exit_reason"] == "invalid_plan"
    assert plan["actions"] == []
    assert "this is not plan json" in (plan.get("raw") or "")
