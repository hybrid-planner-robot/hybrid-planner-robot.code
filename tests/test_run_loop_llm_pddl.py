"""E2b/E2c — llm_pddl entry point: XOR, author→host FD wiring, summary artefacts."""

from __future__ import annotations

import ast
import json
import importlib.util
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "run_loop_llm_pddl.py"
_HOST = _REPO / "scripts" / "run_loop_host.py"
_FD_INNER = _REPO / "scripts" / "_fd_solve_problem.py"
_FIXTURE_SCENE = _REPO / "tests" / "fixtures" / "llm_plan" / "wood_cube_table.json"
_FIXTURE_DOMAIN = _REPO / "tests" / "fixtures" / "llm_pddl" / "min_place_domain.pddl"
_FIXTURE_PROBLEM = _REPO / "tests" / "fixtures" / "llm_pddl" / "min_place_problem.pddl"

_SPEC = importlib.util.spec_from_file_location("run_loop_llm_pddl", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["run_loop_llm_pddl"] = _mod
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


def _walk_names(path: Path) -> set[str]:
    src = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(src):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def _place_argv(run_dir: Path, extra: list[str] | None = None) -> list[str]:
    argv = [
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
        str(_FIXTURE_SCENE),
        "--run-dir",
        str(run_dir),
    ]
    if extra:
        argv.extend(extra)
    return argv


class _RecordingFD:
    def __init__(self, actions: list[str] | None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._actions = actions

    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        self.calls.append((domain_text, problem_text))
        return self._actions


class _BoomFD:
    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        raise AssertionError("Fast Downward must not be called")


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
    assert "FastDownwardPlanner" in out


def test_rejects_online_enrichment():
    r = _run(["--task", "place the wood cube on the shelf", "--online-enrichment", "1"])
    assert r.returncode != 0
    err = r.stderr + r.stdout
    assert "XOR" in err
    assert "online-enrichment" in err


def test_rejects_domain_select():
    r = _run(["--task", "place the wood cube on the shelf", "--domain-select", "llm"])
    assert r.returncode != 0
    err = r.stderr + r.stdout
    assert "XOR" in err
    assert "domain-select" in err


def test_rejects_prune_flag():
    r = _run(["--task", "place the wood cube on the shelf", "--prune"])
    assert r.returncode != 0
    err = r.stderr + r.stdout
    assert "XOR" in err
    assert "prune" in err.lower()


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


def test_script_does_not_import_enricher_or_pruner():
    imported = _imported_modules(_SCRIPT)
    assert "planner.online_enrichment" not in imported
    assert "planner.domain_pruner" not in imported
    assert "planner.domain_llm" not in imported
    assert "planner.llm_pddl_baseline" in imported
    assert "planner.fast_downward" in imported
    assert any("hybrid_runtime" in m for m in imported)
    assert any("live_scene" in m for m in imported)
    assert "vlm.perception" not in imported
    names = _walk_names(_SCRIPT)
    assert "resolve_domain_for_task" not in names
    assert "select_domain_template" not in names
    assert "InitRenderer" not in names
    assert "solve_from_strings" in names
    assert "result_from_actions" in names


def test_host_fd_inner_is_solve_from_strings_then_result_from_actions():
    """After author, llm_pddl uses the same two FD calls as the host inner script."""
    host_src = _FD_INNER.read_text(encoding="utf-8")
    script_src = _SCRIPT.read_text(encoding="utf-8")
    assert "FastDownwardPlanner().solve_from_strings" in host_src
    assert "result_from_actions(actions)" in host_src
    assert "solve_from_strings(domain_text, problem_text)" in script_src
    assert "return result_from_actions(actions)" in script_src
    assert "_fd_solve_in_container" not in script_src
    called = set()
    for node in ast.walk(ast.parse(script_src)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert "resolve_domain_for_task" not in called
    assert "select_domain_template" not in called
    assert "solve_from_strings" in called
    assert "result_from_actions" in called


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
    assert "--domain-select" in r.stdout


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
        "domain_select",
        "prune",
        "pruning",
        "mock_scene",
        "mock_llm",
        "mock_fd",
        "scene_file",
    }


def test_mock_place_writes_domain_problem_and_fd_actions(tmp_path):
    run_dir = tmp_path / "run"
    domain = _FIXTURE_DOMAIN.read_text(encoding="utf-8").strip()
    problem = _FIXTURE_PROBLEM.read_text(encoding="utf-8").strip()
    recorder = _RecordingFD(
        ["(pick wood_cube table)", "(place wood_cube shelf)"]
    )

    def generate(system: str, user: str) -> str:
        assert "STAGE:pddl_refuse" in system or "toy-table" in system
        return json.dumps(
            {
                "domain": domain,
                "problem": problem,
                "refuse": False,
                "reason": "put the cube on the shelf",
            }
        )

    code = _mod.main(
        _place_argv(run_dir),
        generate_fn=generate,
        fd_planner=recorder,
    )
    assert code == 0
    assert len(recorder.calls) == 1
    called_domain, called_problem = recorder.calls[0]
    assert called_domain == domain
    assert called_problem == problem

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    fd_plan = json.loads((run_dir / "fd_plan.json").read_text(encoding="utf-8"))
    scene = json.loads((run_dir / "scene_compact.json").read_text(encoding="utf-8"))
    on_disk_domain = (run_dir / "domain.pddl").read_text(encoding="utf-8")
    on_disk_problem = (run_dir / "problem.pddl").read_text(encoding="utf-8")

    assert summary["baseline"] == "llm_pddl"
    assert summary["exit_reason"] == "planned"
    assert summary["plan_only"] is True
    assert summary["enrichment_used"] is False
    assert summary["pruner"] is False
    assert summary["domain_template"] is None
    assert summary["n_plan_actions"] == 2
    assert summary["fd_actions"] == [
        "(pick wood_cube table)",
        "(place wood_cube shelf)",
    ]
    assert summary["fd_primitives"]
    assert summary["pddl_domain"] == domain
    assert summary["pddl_problem"] == problem
    assert summary["pddl_init"]
    assert "(:init" in summary["pddl_init"]
    assert summary["pddl_goal"]
    assert "(:goal" in summary["pddl_goal"]
    assert on_disk_domain == domain
    assert on_disk_problem == problem
    assert "manipulation_base" not in on_disk_domain
    assert fd_plan["success"] is True
    assert fd_plan["actions"] == summary["fd_actions"]
    assert "domain_template" not in scene
    assert "wood_cube" in {o["name"] for o in scene["objects"]}


def test_cli_mock_fd_populates_fd_actions(tmp_path):
    run_dir = tmp_path / "run"
    r = _run(_place_argv(run_dir, ["--mock-llm", "--mock-fd"]))
    assert r.returncode == 0, r.stderr + r.stdout
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["baseline"] == "llm_pddl"
    assert summary["exit_reason"] == "planned"
    assert summary["n_plan_actions"] == 2
    assert (run_dir / "domain.pddl").is_file()
    assert (run_dir / "problem.pddl").is_file()
    assert (run_dir / "fd_plan.json").is_file()


def test_unsolvable_fd_writes_fd_plan_and_honest_no_plan(tmp_path):
    run_dir = tmp_path / "run"
    domain = _FIXTURE_DOMAIN.read_text(encoding="utf-8").strip()
    problem = _FIXTURE_PROBLEM.read_text(encoding="utf-8").strip()
    recorder = _RecordingFD(None)

    def generate(system: str, user: str) -> str:
        del system, user
        return json.dumps(
            {"domain": domain, "problem": problem, "refuse": False}
        )

    code = _mod.main(
        _place_argv(run_dir),
        generate_fn=generate,
        fd_planner=recorder,
    )
    assert code == 1
    assert recorder.calls
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    fd_plan = json.loads((run_dir / "fd_plan.json").read_text(encoding="utf-8"))
    assert summary["exit_reason"] == "fail"
    assert summary["fd_actions"] == []
    assert summary["n_plan_actions"] == 0
    assert summary["domain_template"] is None
    assert fd_plan["success"] is False
    assert (run_dir / "domain.pddl").read_text(encoding="utf-8") == domain


def test_refuse_does_not_call_fd(tmp_path):
    run_dir = tmp_path / "run"
    boom = _BoomFD()

    def generate(system: str, user: str) -> str:
        del system, user
        return json.dumps(
            {"domain": "", "problem": "", "refuse": True, "reason": "no solder"}
        )

    code = _mod.main(
        [
            "--task",
            "solder the pipe",
            "--world",
            "tabletop",
            "--mock-scene",
            "--scene-file",
            str(_FIXTURE_SCENE),
            "--run-dir",
            str(run_dir),
        ],
        generate_fn=generate,
        fd_planner=boom,
    )
    assert code == 3
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_reason"] == "refused"
    assert summary["fd_actions"] == []
    assert summary["n_plan_actions"] == 0
    assert summary["refuse_reason"]
    assert not (run_dir / "domain.pddl").exists()
    assert not (run_dir / "fd_plan.json").exists()


def test_cli_mock_solder_refuses(tmp_path):
    run_dir = tmp_path / "run"
    r = _run(
        [
            "--task",
            "solder the pipe",
            "--world",
            "tabletop",
            "--mock-scene",
            "--mock-llm",
            "--mock-fd",
            "--run-dir",
            str(run_dir),
        ]
    )
    assert r.returncode == 3, r.stderr + r.stdout
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_reason"] == "refused"
    assert summary["fd_actions"] == []
    assert not (run_dir / "fd_plan.json").exists()


def test_invalid_pddl_does_not_call_fd(tmp_path):
    run_dir = tmp_path / "run"
    boom = _BoomFD()
    code = _mod.main(
        _place_argv(run_dir),
        generate_fn=lambda s, u: "not json and not pddl at all",
        fd_planner=boom,
    )
    assert code == 2
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["exit_reason"] == "invalid_pddl"
    assert summary["fd_actions"] == []
    assert summary["n_plan_actions"] == 0
    assert not (run_dir / "domain.pddl").exists()
    assert not (run_dir / "fd_plan.json").exists()
    dumped = json.loads((run_dir / "llm_pddl.json").read_text(encoding="utf-8"))
    assert dumped["exit_reason"] == "invalid_pddl"
    assert dumped.get("raw")


def test_invalid_pddl_still_writes_domain_and_problem(tmp_path):
    run_dir = tmp_path / "run"
    broken = _FIXTURE_PROBLEM.read_text(encoding="utf-8").replace("(:goal", "(:g")
    domain = _FIXTURE_DOMAIN.read_text(encoding="utf-8").strip()
    boom = _BoomFD()

    def generate(system: str, user: str) -> str:
        del system, user
        return json.dumps(
            {"domain": domain, "problem": broken, "refuse": False}
        )

    code = _mod.main(
        _place_argv(run_dir),
        generate_fn=generate,
        fd_planner=boom,
    )
    assert code == 2
    assert (run_dir / "domain.pddl").read_text(encoding="utf-8").strip() == domain
    assert "(define (problem" in (run_dir / "problem.pddl").read_text(encoding="utf-8")
    assert not (run_dir / "fd_plan.json").exists()


def test_docker_fd_refuses_empty_domain():
    """Empty domain must not reach the container wrapper (template fallback)."""
    client = _mod.DockerFastDownward(container="unused")
    try:
        client.solve_from_strings(
            "",
            "(define (problem x) (:domain x) (:init) (:goal (and)))",
        )
    except RuntimeError as exc:
        assert "empty" in str(exc).lower()
        assert "template" in str(exc).lower()
    else:
        raise AssertionError("empty domain must raise")
