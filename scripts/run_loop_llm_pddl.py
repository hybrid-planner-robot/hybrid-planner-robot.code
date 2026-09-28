#!/usr/bin/env python3
"""
run_loop_llm_pddl.py — llm_pddl baseline (always plan-only).

XOR vs ``run_loop_host.py``: this script does not call the enricher, the
pruner, ``select_domain_template`` / ``resolve_domain_for_task``, or
``--domain-select``. The official host does not grow a ``--baseline`` flag.
No retry on a repo template; the LLM texts are the only domain.

FD (no reimplementation, no D10 extract)::

    FD = import planner.fast_downward.FastDownwardPlanner
         (same class as scripts/_fd_solve_problem.py / host control=fd).
    After author_pddl: FastDownwardPlanner.solve_from_strings(llm_domain,
    llm_problem) then planner.fast_downward.result_from_actions — the same
    two calls as _fd_solve_problem.py. --container is recorded (host mirror)
    but this script does not docker-exec: that wrapper defaults to a
    template file when domain text is empty, which would violate D6/D9.

SceneState (D10 extract, same spine as llm_plan; no PerceptionModule fork)::

    scena = import planner.live_scene.acquire_live_scene
            (same _pre_scan / _capture / _run_dino_scene_sweep as the host,
            then scene_from_dino_payload / build_scene_state).
    Offline fixture: OracleMockAdapter.load (--mock-scene).

Flow: SceneState → fair_compact_scene → author_pddl
(refuse → schema → actions → init → goal, or one-shot domain+problem)
→ (if authored) write domain.pddl + problem.pddl → host FD → fd_plan.json even on fail.
Refused / invalid_pddl stop before FD. Always plan-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from planner.call_timings import attach_snapshot, fd_span, reset as reset_timings  # noqa: E402
from planner.baseline_catalog import fair_compact_scene  # noqa: E402
from planner.fast_downward import (  # noqa: E402
    FastDownwardPlanner,
    FastDownwardTimeout,
    result_from_actions,
    result_from_timeout,
)
from planner.hybrid_runtime import (  # noqa: E402, F401
    HybridProblemSession,
    scene_from_dino_payload,
    scene_from_xyz_poses,
)
from planner.baselines.mock_llm import (  # noqa: E402
    generate_pddl_mock,
    mock_fast_downward_actions,
)
from planner.llm_pddl_baseline import (  # noqa: E402
    EXIT_AUTHORED,
    EXIT_INVALID,
    EXIT_REFUSED,
    LlmPddlOutcome,
    author_pddl,
)
from planner.pddl_sections import write_fd_problem_artifacts, write_fd_result  # noqa: E402
from planner.problem_generator.init_generator.adapters import (  # noqa: E402, F401
    DinoAdapter,
    OracleAdapter,
)
from planner.problem_generator.init_generator.adapters.mock import (  # noqa: E402
    OracleMockAdapter,
)
from planner.problem_generator.init_generator.schema import SceneState  # noqa: E402
from planner.text_llm import GenerateFn  # noqa: E402

# Live decode budget (E5): a full domain+problem JSON needs more room than
# selection/enrichment (512). Eval may inject a generate_fn with the same cap
# so both arms share one loaded 7B.
LLM_PDDL_MAX_NEW_TOKENS = 1536

FD_SPINE_NOTE = (
    "FD = import planner.fast_downward.FastDownwardPlanner "
    "(same class as scripts/_fd_solve_problem.py / host control=fd). "
    "After author: solve_from_strings(llm_domain, llm_problem) then "
    "result_from_actions. No extract; no resolve_domain_for_task."
)

SCENE_SPINE_NOTE = (
    "scena = import planner.live_scene.acquire_live_scene "
    "(D10 extract: same _pre_scan/_capture/_run_dino_scene_sweep as "
    "run_loop_host, then scene_from_dino_payload / build_scene_state). "
    "Offline: OracleMockAdapter.load (--mock-scene). "
    "No PerceptionModule fork."
)

DEFAULT_MOCK_SCENE = (
    _REPO_ROOT / "tests" / "fixtures" / "llm_plan" / "wood_cube_table.json"
)
DEFAULT_MOCK_DOMAIN = (
    _REPO_ROOT / "tests" / "fixtures" / "llm_pddl" / "min_place_domain.pddl"
)
DEFAULT_MOCK_PROBLEM = (
    _REPO_ROOT / "tests" / "fixtures" / "llm_pddl" / "min_place_problem.pddl"
)

EXIT_PLANNED = "planned"
EXIT_FAIL = "fail"

_EXIT_CODES = {
    EXIT_PLANNED: 0,
    EXIT_REFUSED: 3,
    EXIT_INVALID: 2,
    EXIT_FAIL: 1,
}


class FdPlanner(Protocol):
    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "llm_pddl baseline: the text LLM authors domain+problem, then Fast "
            "Downward (same FastDownwardPlanner as the host). Always plan-only. "
            "Does not call the enricher or pruner."
        ),
        epilog=(
            "XOR: --online-enrichment, --domain-select, and prune flags are "
            "refused (use scripts/run_loop_host.py for hybrid+FD). "
            + FD_SPINE_NOTE
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", required=True, help="Natural-language task.")
    parser.add_argument(
        "--world",
        default="office",
        help="Active Gazebo world (mirror of run_loop_host; default: office).",
    )
    parser.add_argument(
        "--scene-source",
        default="dino",
        choices=["fused", "dino", "oracle"],
        help="Same adapters as R0 (default: dino). --mock-scene skips live DINO.",
    )
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Disable SIM NameMatch / oracle snap (mirror of run_loop_host).",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Always on for this baseline (accepted for flag-mirror compatibility). "
            "No orchestrator inject, no arm motion."
        ),
    )
    parser.add_argument(
        "--container",
        default="vlm_ros2",
        help=(
            "Docker container for pre-scan / capture and optional FD. "
            "FD uses FastDownwardPlanner on the LLM texts (or docker exec of "
            "_fd_solve_problem.py with domain override). Never an empty-domain "
            "template fallback."
        ),
    )
    parser.add_argument(
        "--sudo-docker",
        action="store_true",
        help="Prefix docker exec with sudo (mirror of run_loop_host).",
    )
    parser.add_argument(
        "--mock-scene",
        action="store_true",
        help=(
            "Load SceneState from an oracle_mock_v1 fixture (no Gazebo). "
            "Default file: tests/fixtures/llm_plan/wood_cube_table.json "
            "(override with --scene-file). Without this flag, live DINO "
            "from the running sim is used (same spine as run_loop_host)."
        ),
    )
    parser.add_argument(
        "--scene-file",
        default=None,
        metavar="PATH",
        help="oracle_mock_v1 JSON for --mock-scene (default: wood_cube_table fixture).",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help=(
            "Canned generate_fn for CI/smoke: suite v1 task texts (verbatim) "
            "or keyword refuse. Not the live text LLM."
        ),
    )
    parser.add_argument(
        "--mock-fd",
        action="store_true",
        help=(
            "Canned FastDownwardPlanner.solve_from_strings for CI/smoke "
            "(does not require the fast-downward binary or a container)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Parent of a timestamped run dir (default: data/runs/llm_pddl).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Exact artefact directory (overrides --out-dir).",
    )
    xor = parser.add_argument_group("rejected host / prune flags (XOR)")
    xor.add_argument(
        "--online-enrichment",
        default=None,
        metavar="MODE",
        help="Rejected: llm_pddl does not call the enricher.",
    )
    xor.add_argument(
        "--domain-select",
        default=None,
        metavar="MODE",
        help="Rejected: llm_pddl does not select a repo template.",
    )
    xor.add_argument(
        "--prune",
        action="store_true",
        help="Rejected: llm_pddl is not the pruning pipeline.",
    )
    xor.add_argument(
        "--pruning",
        action="store_true",
        help="Rejected: llm_pddl is not the pruning pipeline.",
    )
    return parser


def _reject_host_xor(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.online_enrichment is not None:
        parser.error(
            "XOR: llm_pddl refuses --online-enrichment "
            "(does not call the enricher; use scripts/run_loop_host.py)"
        )
    if args.domain_select is not None:
        parser.error(
            "XOR: llm_pddl refuses --domain-select "
            "(does not call select_domain_template / resolve_domain_for_task)"
        )
    if args.prune or args.pruning:
        parser.error(
            "XOR: llm_pddl refuses prune flags "
            "(use scripts/run_loop_pruning.py for the pruning pipeline)"
        )


def smoke_mock_generate_fn(system: str, user: str) -> str:
    """CI/smoke: suite-verbatim mock (see planner.baselines.mock_llm)."""
    return generate_pddl_mock(system, user)


class SmokeMockFastDownward:
    """CI/smoke: same method as the host client, no binary / no container."""

    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        return mock_fast_downward_actions(domain_text, problem_text)


def _docker_fd_timeout_s() -> int:
    from planner.fast_downward import resolve_search_time_limit_s

    return resolve_search_time_limit_s() + 20


class DockerFastDownward:
    """
    Same inner FD as the host: ``python3 /workspace/scripts/_fd_solve_problem.py``
    with **LLM domain+problem as ``domain`` override** (never an empty domain,
    so the wrapper cannot fall back to a repo template).

    Used only when ``fast-downward`` is not on the host PATH (it lives in the
    ``vlm_ros2`` image). Does not call Gazebo.
    """

    def __init__(self, container: str = "vlm_ros2") -> None:
        self.container = container

    def solve_from_strings(self, domain_text: str, problem_text: str) -> list[str] | None:
        domain = (domain_text or "").strip()
        problem = (problem_text or "").strip()
        if not domain or not problem:
            raise RuntimeError(
                "DockerFastDownward refused empty domain/problem "
                "(would hit the template-file fallback in _fd_solve_problem.py)"
            )
        payload = json.dumps(
            {
                "domain_template": "unused",
                "problem": problem,
                "domain": domain,
                "domains_dir": "/workspace/pddl/domains",
            }
        )
        bash_cmd = "python3 /workspace/scripts/_fd_solve_problem.py"
        try:
            result = subprocess.run(
                ["docker", "exec", "-i", self.container, "bash", "-c", bash_cmd],
                input=payload,
                capture_output=True,
                text=True,
                timeout=_docker_fd_timeout_s(),
            )
        except subprocess.TimeoutExpired:
            return None
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if not out:
            raise RuntimeError(
                f"FD docker exec empty stdout (rc={result.returncode}): {err}"
            )
        try:
            data = json.loads(out)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"FD docker exec non-JSON: {out[:500]}") from exc
        if not data.get("success"):
            error = str(data.get("error") or "unsolvable — no plan")
            if "unsolvable" in error or "time limit" in error.lower():
                return None
            raise RuntimeError(f"Fast Downward error: {error}")
        actions = data.get("actions")
        if actions is None:
            return None
        return [str(a) for a in actions]


def call_host_fast_downward(
    domain_text: str,
    problem_text: str,
    planner: FdPlanner | None = None,
) -> dict[str, Any]:
    """
    Same Fast Downward invocation as ``scripts/_fd_solve_problem.py``
    (host ``control=fd`` inner): ``solve_from_strings`` then
    ``result_from_actions``. LLM texts only — no template path.
    """
    client = planner if planner is not None else FastDownwardPlanner()
    with fd_span():
        try:
            actions = client.solve_from_strings(domain_text, problem_text)
        except FastDownwardTimeout as exc:
            return result_from_timeout(exc.limit_s)
        except Exception as exc:  # noqa: BLE001 — same surface as _fd_solve_problem.py
            return {
                "success": False,
                "error": f"FD error: {exc}",
                "actions": [],
                "primitives": [],
            }
        return result_from_actions(actions)


def load_scene(args: argparse.Namespace) -> SceneState:
    path_raw = args.scene_file
    if args.mock_scene or path_raw:
        path = Path(path_raw) if path_raw else DEFAULT_MOCK_SCENE
        if not path.is_file():
            raise FileNotFoundError(f"mock scene not found: {path}")
        return OracleMockAdapter.load(path)
    from planner.live_scene import acquire_live_scene

    print(f"[llm_pddl] live scene via acquire_live_scene (world={args.world})")
    result = acquire_live_scene(
        world=str(args.world),
        container=str(args.container),
        scene_source=str(args.scene_source),
        perception_only=bool(args.perception_only),
        sudo_docker=bool(getattr(args, "sudo_docker", False)),
    )
    return result.scene


def resolve_generate_fn(
    args: argparse.Namespace,
    generate_fn: GenerateFn | None,
) -> GenerateFn:
    if generate_fn is not None:
        return generate_fn
    if args.mock_llm:
        return smoke_mock_generate_fn
    from planner.text_llm import get_shared_text_client

    client = get_shared_text_client(max_new_tokens=LLM_PDDL_MAX_NEW_TOKENS)
    return client.complete


def resolve_fd_planner(
    args: argparse.Namespace,
    fd_planner: FdPlanner | None,
) -> FdPlanner:
    if fd_planner is not None:
        return fd_planner
    if args.mock_fd:
        return SmokeMockFastDownward()
    if shutil.which("fast-downward"):
        return FastDownwardPlanner()
    print(
        f"[llm_pddl] fast-downward not on PATH — "
        f"docker exec {args.container} _fd_solve_problem.py "
        "(LLM domain override, no template file)"
    )
    return DockerFastDownward(container=str(args.container))


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    parent = Path(args.out_dir) if args.out_dir else (
        _REPO_ROOT / "data" / "runs" / "llm_pddl"
    )
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_tag = args.task[:30].replace(" ", "_").replace("/", "-")
    path = parent / f"{ts}_{args.world}_{task_tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_summary(
    args: argparse.Namespace,
    *,
    outcome: LlmPddlOutcome,
    run_dir: Path,
    fd_result: dict[str, Any] | None,
    pddl_art: dict[str, Any] | None,
    exit_reason: str,
) -> dict[str, Any]:
    refused = outcome.exit_reason == EXIT_REFUSED
    fd_actions = list((fd_result or {}).get("actions") or [])
    fd_prims = list((fd_result or {}).get("primitives") or [])
    sections = pddl_art or {}
    summary = {
        "baseline": "llm_pddl",
        "task": args.task,
        "world": args.world,
        "control": "fd",
        "domain_template": None,
        "scene_source": args.scene_source,
        "perception_only": bool(args.perception_only),
        "success": False,
        "exit_reason": exit_reason,
        "n_steps": 0,
        "completed_steps": [],
        "plan_only": True,
        "enrichment_used": False,
        "enrichment_skills": [],
        "pruner": False,
        "refuse_reason": outcome.reason if refused else None,
        "reason": outcome.reason,
        "error": outcome.error or (fd_result or {}).get("error"),
        "fd_actions": fd_actions,
        "fd_primitives": fd_prims,
        "n_plan_actions": len(fd_actions),
        "pddl_problem": outcome.problem,
        "pddl_init": sections.get("pddl_init"),
        "pddl_goal": sections.get("pddl_goal"),
        "pddl_domain": outcome.domain,
        "init_facts": list(sections.get("init_facts") or []) or None,
        "goal_facts": list(sections.get("goal_facts") or []) or None,
        "container": args.container,
        "run_dir": str(run_dir.relative_to(_REPO_ROOT))
        if run_dir.is_relative_to(_REPO_ROOT)
        else str(run_dir),
    }
    attach_snapshot(summary)
    return summary


def write_author_artifacts(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    outcome: LlmPddlOutcome,
    scene_json: str,
) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    payload = outcome.to_dict()
    if outcome.raw is not None:
        payload["raw"] = outcome.raw
    (run_dir / "llm_pddl.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "scene_compact.json").write_text(scene_json, encoding="utf-8")


def write_llm_pddl_files(
    run_dir: Path,
    *,
    domain_text: str | None,
    problem_text: str | None,
) -> dict[str, Any]:
    """Persist whatever the LLM wrote (valid or not) so the run is inspectable."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if domain_text:
        (run_dir / "domain.pddl").write_text(domain_text, encoding="utf-8")
    if not problem_text:
        return {}
    return write_fd_problem_artifacts(run_dir, problem_text)


def main(
    argv: list[str] | None = None,
    *,
    generate_fn: GenerateFn | None = None,
    fd_planner: FdPlanner | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _reject_host_xor(args, parser)
    reset_timings()

    print("[llm_pddl] always plan-only")
    print(f"[llm_pddl] {FD_SPINE_NOTE}")
    print(f"[llm_pddl] {SCENE_SPINE_NOTE}")

    try:
        scene = load_scene(args)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"[FAIL] {exc}")
        return 2

    scene_json = fair_compact_scene(scene)
    try:
        gen = resolve_generate_fn(args, generate_fn)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] no generate_fn ({exc}). Pass --mock-llm for smoke.")
        return 2

    outcome = author_pddl(args.task, scene_json, gen, world=args.world)
    run_dir = make_run_dir(args)

    fd_result: dict[str, Any] | None = None
    pddl_art: dict[str, Any] | None = None
    exit_reason = outcome.exit_reason

    if outcome.exit_reason == EXIT_REFUSED:
        print("[llm_pddl] refused — Fast Downward not called")
    elif outcome.exit_reason == EXIT_INVALID:
        print("[llm_pddl] invalid_pddl — Fast Downward not called")
        if outcome.domain or outcome.problem:
            pddl_art = write_llm_pddl_files(
                run_dir,
                domain_text=outcome.domain,
                problem_text=outcome.problem,
            )
    elif (
        outcome.exit_reason == EXIT_AUTHORED
        and outcome.domain
        and outcome.problem
    ):
        pddl_art = write_llm_pddl_files(
            run_dir,
            domain_text=outcome.domain,
            problem_text=outcome.problem,
        )
        planner = resolve_fd_planner(args, fd_planner)
        fd_result = call_host_fast_downward(
            outcome.domain, outcome.problem, planner
        )
        write_fd_result(run_dir, fd_result)
        if fd_result.get("success"):
            exit_reason = EXIT_PLANNED
        else:
            exit_reason = EXIT_FAIL
            print(f"[FAIL] Fast Downward: {fd_result.get('error', 'unknown')}")
    else:
        exit_reason = EXIT_INVALID
        print("[llm_pddl] authored without domain+problem — treating as invalid_pddl")

    summary = build_summary(
        args,
        outcome=outcome,
        run_dir=run_dir,
        fd_result=fd_result,
        pddl_art=pddl_art,
        exit_reason=exit_reason,
    )
    write_author_artifacts(
        run_dir, summary=summary, outcome=outcome, scene_json=scene_json
    )

    rel = summary["run_dir"]
    print(
        f"[llm_pddl] exit_reason={exit_reason} "
        f"n_plan_actions={summary['n_plan_actions']}"
    )
    print(f"[llm_pddl] summary.json → {rel}/summary.json")
    for i, act in enumerate(summary["fd_actions"], 1):
        print(f"         {i}. {act}")
    return _EXIT_CODES.get(exit_reason, 2)


if __name__ == "__main__":
    raise SystemExit(main())
