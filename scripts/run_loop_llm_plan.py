#!/usr/bin/env python3
"""
run_loop_llm_plan.py — llm_plan baseline (always plan-only).

XOR vs ``run_loop_host.py``: does not call the enricher, the pruner,
``select_domain_template`` / ``resolve_domain_for_task``, or Fast Downward.
The official host does not grow a ``--baseline`` flag. ``--plan-only`` on
the host still requires ``control=fd`` — this script never touches that gate.

SceneState (D10 extract, no PerceptionModule fork)::

    scena = import planner.live_scene.acquire_live_scene
            (same _pre_scan / _capture / _run_dino_scene_sweep as the host,
            then scene_from_dino_payload / build_scene_state).
    Offline fixture: OracleMockAdapter.load (--mock-scene).

E1c smoke uses ``--mock-scene`` + ``--mock-llm`` (or an injected generate_fn).

Flow: SceneState → fair_compact_scene → plan_with_llm
(refuse → intent → ground) → summary.json
(``baseline=llm_plan``). Stops after the JSON; no orchestrator inject.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from planner.call_timings import attach_snapshot, reset as reset_timings  # noqa: E402
from planner.baseline_catalog import fair_compact_scene  # noqa: E402
from planner.hybrid_runtime import (  # noqa: E402, F401
    HybridProblemSession,
    scene_from_dino_payload,
    scene_from_xyz_poses,
)
from planner.baselines.mock_llm import generate_plan_mock  # noqa: E402
from planner.llm_plan_baseline import (  # noqa: E402
    EXIT_INVALID,
    EXIT_PLANNED,
    EXIT_REFUSED,
    LlmPlanOutcome,
    plan_to_host_fields,
    plan_with_llm,
)
from planner.problem_generator.init_generator.adapters import (  # noqa: E402, F401
    DinoAdapter,
    OracleAdapter,
)
from planner.problem_generator.init_generator.adapters.mock import (  # noqa: E402
    OracleMockAdapter,
)
from planner.problem_generator.init_generator.schema import SceneState  # noqa: E402
from planner.text_llm import GenerateFn  # noqa: E402

# Live decode budget (E5). Plan JSON is short; 512 is enough and shares
# the process-wide client when eval injects generate_fn with a larger cap.
LLM_PLAN_MAX_NEW_TOKENS = 512

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

_EXIT_CODES = {
    EXIT_PLANNED: 0,
    EXIT_REFUSED: 3,
    EXIT_INVALID: 2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "llm_plan baseline: the text LLM emits a grounded action list. "
            "Always plan-only. Does not call Fast Downward or the enricher."
        ),
        epilog=(
            "XOR: --online-enrichment and --control are refused "
            "(use scripts/run_loop_host.py for hybrid+FD). "
            + SCENE_SPINE_NOTE
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
        "--container",
        default="vlm_ros2",
        help="Docker container for pre-scan / capture (mirror of run_loop_host).",
    )
    parser.add_argument(
        "--sudo-docker",
        action="store_true",
        help="Prefix docker exec with sudo (mirror of run_loop_host).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Parent of a timestamped run dir (default: data/runs/llm_plan).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Exact artefact directory (overrides --out-dir).",
    )
    xor = parser.add_argument_group("rejected host flags (XOR)")
    xor.add_argument(
        "--online-enrichment",
        default=None,
        metavar="MODE",
        help="Rejected: llm_plan does not call the enricher.",
    )
    xor.add_argument(
        "--control",
        default=None,
        metavar="MODE",
        help="Rejected: llm_plan does not use Fast Downward or vlm_steps.",
    )
    return parser


def _reject_host_xor(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.online_enrichment is not None:
        parser.error(
            "XOR: llm_plan refuses --online-enrichment "
            "(does not call the enricher; use scripts/run_loop_host.py)"
        )
    if args.control is not None:
        parser.error(
            "XOR: llm_plan refuses --control "
            f"{args.control!r} (no Fast Downward, not vlm_steps; "
            "this script is always plan-only)"
        )


def smoke_mock_generate_fn(system: str, user: str) -> str:
    """CI/smoke: suite-verbatim mock (see planner.baselines.mock_llm)."""
    return generate_plan_mock(system, user)


def load_scene(args: argparse.Namespace) -> SceneState:
    path_raw = args.scene_file
    if args.mock_scene or path_raw:
        path = Path(path_raw) if path_raw else DEFAULT_MOCK_SCENE
        if not path.is_file():
            raise FileNotFoundError(f"mock scene not found: {path}")
        return OracleMockAdapter.load(path)
    from planner.live_scene import acquire_live_scene

    print(f"[llm_plan] live scene via acquire_live_scene (world={args.world})")
    result = acquire_live_scene(
        world=str(args.world),
        container=str(getattr(args, "container", "vlm_ros2")),
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

    client = get_shared_text_client(max_new_tokens=LLM_PLAN_MAX_NEW_TOKENS)
    return client.complete


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
    parent = Path(args.out_dir) if args.out_dir else (
        _REPO_ROOT / "data" / "runs" / "llm_plan"
    )
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_tag = args.task[:30].replace(" ", "_").replace("/", "-")
    path = parent / f"{ts}_{args.world}_{task_tag}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_summary(
    args: argparse.Namespace,
    *,
    outcome: LlmPlanOutcome,
    run_dir: Path,
) -> dict[str, Any]:
    fields = plan_to_host_fields(outcome)
    refused = outcome.exit_reason == EXIT_REFUSED
    summary = {
        "baseline": "llm_plan",
        "task": args.task,
        "world": args.world,
        "control": None,
        "domain_template": None,
        "scene_source": args.scene_source,
        "perception_only": bool(args.perception_only),
        "success": False,
        "exit_reason": outcome.exit_reason,
        "n_steps": 0,
        "completed_steps": [],
        "plan_only": True,
        "enrichment_used": False,
        "enrichment_skills": [],
        "refuse_reason": outcome.reason if refused else None,
        "reason": outcome.reason,
        "error": outcome.error,
        "fd_actions": fields["fd_actions"],
        "fd_primitives": fields["fd_primitives"],
        "n_plan_actions": fields["n_plan_actions"],
        "pddl_problem": None,
        "pddl_init": None,
        "pddl_goal": None,
        "pddl_domain": None,
        "init_facts": None,
        "run_dir": str(run_dir.relative_to(_REPO_ROOT))
        if run_dir.is_relative_to(_REPO_ROOT)
        else str(run_dir),
    }
    attach_snapshot(summary)
    return summary


def write_artifacts(
    run_dir: Path,
    *,
    summary: dict[str, Any],
    outcome: LlmPlanOutcome,
    scene_json: str,
) -> None:
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    payload = outcome.to_dict()
    if outcome.raw is not None:
        payload["raw"] = outcome.raw
    if outcome.error is not None:
        payload["error"] = outcome.error
    (run_dir / "llm_plan.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "scene_compact.json").write_text(scene_json, encoding="utf-8")


def main(
    argv: list[str] | None = None,
    *,
    generate_fn: GenerateFn | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _reject_host_xor(args, parser)
    reset_timings()

    print("[llm_plan] always plan-only; Fast Downward not called")
    print(f"[llm_plan] {SCENE_SPINE_NOTE}")

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

    outcome = plan_with_llm(args.task, scene_json, gen, world=args.world)
    run_dir = make_run_dir(args)
    summary = build_summary(args, outcome=outcome, run_dir=run_dir)
    write_artifacts(run_dir, summary=summary, outcome=outcome, scene_json=scene_json)

    rel = summary["run_dir"]
    print(f"[llm_plan] exit_reason={outcome.exit_reason} "
          f"n_plan_actions={summary['n_plan_actions']}")
    print(f"[llm_plan] summary.json → {rel}/summary.json")
    for i, act in enumerate(summary["fd_actions"], 1):
        print(f"         {i}. {act}")
    return _EXIT_CODES.get(outcome.exit_reason, 2)


if __name__ == "__main__":
    raise SystemExit(main())
