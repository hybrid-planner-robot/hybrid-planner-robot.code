"""Host --mock-scene / --enrichment-profile r1 (no Gazebo, no GPU)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SCRIPT = _REPO / "scripts" / "run_loop_host.py"
_KITCHEN = _REPO / "tests" / "fixtures" / "llm_plan" / "kitchen.json"
_TABLE = _REPO / "tests" / "fixtures" / "llm_plan" / "tabletop.json"
_WORKSHOP = _REPO / "tests" / "fixtures" / "llm_plan" / "workshop.json"


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *argv],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )


def _common(task: str, world: str, scene: Path) -> list[str]:
    return [
        "--task",
        task,
        "--world",
        world,
        "--hybrid",
        "mvp",
        "--control",
        "fd",
        "--goal-backend",
        "local_llm",
        "--domain-select",
        "llm",
        "--plan-only",
        "--mock-scene",
        "--scene-file",
        str(scene),
        "--mock-llm",
        "--mock-fd",
    ]


def test_mock_scene_help_lists_flags():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--mock-scene" in r.stdout
    assert "--enrichment-profile" in r.stdout
    assert "inventory" in r.stdout
    assert "--inventory-json" in r.stdout


def test_mock_scene_place_plan_only():
    r = _run(_common("place the wood cube on the shelf", "tabletop", _TABLE))
    assert r.returncode == 0, r.stdout + r.stderr
    combined = (r.stdout + r.stderr).lower()
    assert "mock-scene" in combined


def test_mock_scene_workshop_place_plan_only():
    r = _run(_common("place the hammer on the metal tray", "workshop", _WORKSHOP))
    assert r.returncode == 0, r.stdout + r.stderr
    combined = (r.stdout + r.stderr).lower()
    assert "mock-scene" in combined
    assert "hammer" in combined


def test_r1_workshop_drill_writes_affordances():
    r = _run(
        _common("drill a hole in the metal plate", "workshop", _WORKSHOP)
        + ["--online-enrichment", "1", "--enrichment-profile", "r1"]
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "can-be-drilled" in r.stdout
    assert "can-pour" not in r.stdout


def test_r1_pour_writes_affordances():
    r = _run(
        _common("pour the can into the glass", "kitchen", _KITCHEN)
        + ["--online-enrichment", "1", "--enrichment-profile", "r1"]
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "can-pour" in r.stdout
    runs = sorted((_REPO / "data" / "runs").glob("*kitchen*"))
    matching = [p for p in runs if "pour" in p.name]
    assert matching, f"expected a run dir, have {runs[-3:]}"
    domain = matching[-1] / "iter_01" / "domain.pddl"
    problem = matching[-1] / "iter_01" / "problem.pddl"
    assert domain.is_file(), domain
    text = domain.read_text()
    assert "can-pour" in text
    init = problem.read_text()
    assert "(can-pour can)" in init
    assert "(can-be-poured glass)" in init


def test_r0_profile_default_skips_r1_banner():
    r = _run(
        _common("pour the can into the glass", "kitchen", _KITCHEN)
        + ["--online-enrichment", "1"]
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "enrichment-profile=r1" not in r.stdout


def test_inventory_source_requires_fd():
    r = _run(["--task", "place grapes in the bowl", "--scene-source", "inventory"])
    assert r.returncode == 2
    combined = (r.stdout + r.stderr).lower()
    assert "control fd" in combined


def test_inventory_source_requires_hybrid():
    r = _run(
        [
            "--task",
            "place grapes in the bowl",
            "--scene-source",
            "inventory",
            "--control",
            "fd",
        ]
    )
    assert r.returncode == 2
    combined = (r.stdout + r.stderr).lower()
    assert "hybrid mvp" in combined


def test_mock_scene_plan_only_does_not_inject():
    r = _run(_common("place the wood cube on the shelf", "tabletop", _TABLE))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "skipping injection" in r.stdout


def test_solve_fd_mock_scene_falls_back_to_docker(monkeypatch):
    """Host has no fast-downward: --mock-scene uses docker exec, not --mock-fd."""
    import argparse
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("run_loop_host_fd", _SCRIPT)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    monkeypatch.setattr("shutil.which", lambda _name: None)
    called: dict = {}

    def fake_docker(args, *, domain_template, problem, domain_text=None):
        called["container"] = args.container
        called["template"] = domain_template
        called["problem"] = problem
        called["domain_text"] = domain_text
        return {
            "success": True,
            "actions": ["(pick wood_cube table)"],
            "primitives": [{"name": "pick", "args": ["wood_cube"]}],
        }

    monkeypatch.setattr(mod, "_fd_solve_in_container", fake_docker)
    args = argparse.Namespace(
        mock_fd=False,
        mock_scene=True,
        container="vlm_ros2",
        task="place the wood cube on the shelf",
    )
    out = mod._solve_fd(
        args,
        domain_template="manipulation_base",
        problem="(define (problem p) (:domain manipulation-base))",
        domain_text="(define (domain manipulation-base))",
    )
    assert called["container"] == "vlm_ros2"
    assert out["success"] is True
    assert out["actions"] == ["(pick wood_cube table)"]


def test_solve_fd_mock_fd_does_not_call_docker(monkeypatch):
    import argparse
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("run_loop_host_fd2", _SCRIPT)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    monkeypatch.setattr(
        mod,
        "_fd_solve_in_container",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("docker must not run")),
    )
    args = argparse.Namespace(
        mock_fd=True,
        mock_scene=True,
        container="vlm_ros2",
        task="place the wood cube on the shelf",
    )
    out = mod._solve_fd(
        args,
        domain_template="manipulation_base",
        problem="(define (problem p) (:domain manipulation-base))",
        domain_text=None,
    )
    assert out.get("success") is True
    assert out.get("actions")
