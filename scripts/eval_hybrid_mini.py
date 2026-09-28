#!/usr/bin/env python3
"""
Session 17 — closed-loop mini eval (legacy vs hybrid).

Thin wrapper over ``run_loop_host.py``: runs a small fixed MVP suite under a
flag matrix, aggregates ``summary.json`` / ``metrics_snapshot`` fields, and
writes a comparison table. Not a new planner.

Modes (from suite fixture)::

    legacy | hybrid_mvp | hybrid_mvp_llm (opt) | hybrid_full (opt)

Examples
--------
CI / offline dry (fixture metrics, no Gazebo)::

    python scripts/eval_hybrid_mini.py --dry

Aggregate existing ``data/runs/*/summary.json`` into a table::

    python scripts/eval_hybrid_mini.py --aggregate data/runs

Live matrix (needs Docker + Gazebo + VLM)::

    python scripts/eval_hybrid_mini.py --live --modes legacy,hybrid_mvp
    python scripts/eval_hybrid_mini.py --live --modes legacy,hybrid_mvp,hybrid_full \\
        --world tabletop --tasks place_cup_shelf,pick_cup
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_SUITE = (
    _REPO_ROOT / "tests" / "fixtures" / "hybrid_mini_eval" / "suite_v1.json"
)
_DEFAULT_DRY = (
    _REPO_ROOT / "tests" / "fixtures" / "hybrid_mini_eval" / "comparison_dry.json"
)
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "hybrid_mini_eval"


def load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_modes(
    suite: dict[str, Any],
    mode_ids: list[str] | None,
    *,
    include_optional: bool = False,
) -> list[dict[str, Any]]:
    modes = list(suite.get("modes") or [])
    if mode_ids:
        wanted = {m.strip() for m in mode_ids if m.strip()}
        selected = [m for m in modes if m["id"] in wanted]
        missing = wanted - {m["id"] for m in selected}
        if missing:
            raise SystemExit(f"Unknown mode id(s): {sorted(missing)}")
        return selected
    if include_optional:
        return modes
    return [m for m in modes if not m.get("optional")]


def select_tasks(
    suite: dict[str, Any],
    task_ids: list[str] | None,
) -> list[dict[str, Any]]:
    tasks = list(suite.get("tasks") or [])
    if not task_ids:
        return tasks
    wanted = {t.strip() for t in task_ids if t.strip()}
    selected = [t for t in tasks if t["id"] in wanted]
    missing = wanted - {t["id"] for t in selected}
    if missing:
        raise SystemExit(f"Unknown task id(s): {sorted(missing)}")
    return selected


def _empty_verdicts() -> dict[str, int]:
    return {"GREEN": 0, "YELLOW": 0, "RED": 0}


def row_from_summary(
    summary: dict[str, Any],
    *,
    mode_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Normalize a run_loop_host summary.json (or dry fixture cell) into a row."""
    hybrid_metrics = summary.get("hybrid_problem_gen") or {}
    verdicts = summary.get("verdict_counts") or hybrid_metrics.get("verdict_counts")
    if not isinstance(verdicts, dict):
        verdicts = _empty_verdicts()
    return {
        "mode_id": mode_id or summary.get("mode_id") or _infer_mode_id(summary),
        "task_id": task_id or summary.get("task_id") or "",
        "task": summary.get("task") or "",
        "world": summary.get("world") or "",
        "control": summary.get("control") or "vlm_steps",
        "hybrid": summary.get("hybrid") or "off",
        "goal_backend": summary.get("goal_backend"),
        "success": bool(summary.get("success")),
        "exit_reason": summary.get("exit_reason") or "",
        "n_steps": int(summary.get("n_steps") or 0),
        "replan_count": int(summary.get("replan_count") or 0),
        "wall_time_s": float(summary.get("wall_time_s") or 0.0),
        "vlm_call_count": int(
            summary.get("vlm_call_count")
            if summary.get("vlm_call_count") is not None
            else (hybrid_metrics.get("vlm_call_count") or 0)
        ),
        "verdict_counts": {
            "GREEN": int(verdicts.get("GREEN", 0)),
            "YELLOW": int(verdicts.get("YELLOW", 0)),
            "RED": int(verdicts.get("RED", 0)),
        },
        "goal_backend_used": summary.get("goal_backend_used")
        if "goal_backend_used" in summary
        else hybrid_metrics.get("goal_backend_used"),
        "goal_fallback_count": int(
            summary.get("goal_fallback_count")
            if summary.get("goal_fallback_count") is not None
            else (hybrid_metrics.get("goal_fallback_count") or 0)
        ),
        "run_dir": summary.get("run_dir"),
        "notes": summary.get("notes") or "",
    }


def _infer_mode_id(summary: dict[str, Any]) -> str:
    hybrid = str(summary.get("hybrid") or "off").lower()
    gb = summary.get("goal_backend")
    if hybrid in {"off", "0", "false", "none", ""}:
        return "legacy"
    if hybrid == "full":
        return "hybrid_full"
    if hybrid in {"mvp", "1", "true"} and gb == "local_llm":
        return "hybrid_mvp_llm"
    if hybrid in {"mvp", "1", "true"}:
        return "hybrid_mvp"
    return f"hybrid_{hybrid}"


def summarize_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "success_rate": 0.0,
            "mean_wall_time_s": 0.0,
            "mean_replan_count": 0.0,
            "mean_vlm_call_count": 0.0,
            "total_goal_fallback": 0,
            "verdict_totals": _empty_verdicts(),
        }
    succ = sum(1 for r in rows if r["success"])
    vt = _empty_verdicts()
    for r in rows:
        for k in vt:
            vt[k] += int(r["verdict_counts"].get(k, 0))
    return {
        "n": n,
        "success_count": succ,
        "success_rate": succ / n,
        "mean_wall_time_s": round(sum(r["wall_time_s"] for r in rows) / n, 2),
        "mean_replan_count": round(sum(r["replan_count"] for r in rows) / n, 2),
        "mean_n_steps": round(sum(r["n_steps"] for r in rows) / n, 2),
        "mean_vlm_call_count": round(sum(r["vlm_call_count"] for r in rows) / n, 2),
        "total_goal_fallback": sum(r["goal_fallback_count"] for r in rows),
        "verdict_totals": vt,
    }


def build_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_mode.setdefault(r["mode_id"], []).append(r)
    modes = {
        mid: summarize_mode(mode_rows)
        for mid, mode_rows in sorted(by_mode.items())
    }
    return {
        "n_runs": len(rows),
        "modes": modes,
        "rows": rows,
    }


def format_comparison_table(comparison: dict[str, Any]) -> str:
    lines = [
        "| mode | n | success | mean wall_s | mean replan | mean VLM calls | "
        "goal_fb | GREEN/YELLOW/RED |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mid, s in (comparison.get("modes") or {}).items():
        vt = s.get("verdict_totals") or _empty_verdicts()
        lines.append(
            f"| {mid} | {s['n']} | {s['success_rate']:.0%} "
            f"({s.get('success_count', 0)}/{s['n']}) | "
            f"{s['mean_wall_time_s']:.1f} | {s['mean_replan_count']:.1f} | "
            f"{s['mean_vlm_call_count']:.1f} | {s['total_goal_fallback']} | "
            f"{vt.get('GREEN', 0)}/{vt.get('YELLOW', 0)}/{vt.get('RED', 0)} |"
        )
    return "\n".join(lines)


def load_dry_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows_raw = payload.get("rows") or payload.get("runs") or []
    return [row_from_summary(r) for r in rows_raw]


def aggregate_run_dirs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for summary_path in sorted(root.glob("**/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "run_dir" not in summary:
            try:
                summary["run_dir"] = str(summary_path.parent.relative_to(_REPO_ROOT))
            except ValueError:
                summary["run_dir"] = str(summary_path.parent)
        rows.append(row_from_summary(summary))
    return rows


def _loop_cmd(
    task: str,
    mode: dict[str, Any],
    *,
    world: str,
    max_steps: int,
    container: str,
    sudo_docker: bool,
    python_bin: str,
) -> list[str]:
    cmd = [
        python_bin,
        str(_REPO_ROOT / "scripts" / "run_loop_host.py"),
        "--task",
        task,
        "--world",
        world,
        "--max-steps",
        str(max_steps),
        "--container",
        container,
    ]
    hybrid = (mode.get("hybrid") or "off").lower()
    if hybrid not in {"off", "0", "false", "none"}:
        cmd.extend(["--hybrid", hybrid])
    gb = mode.get("goal_backend")
    if gb:
        cmd.extend(["--goal-backend", str(gb)])
    control = mode.get("control")
    if control:
        cmd.extend(["--control", str(control)])
    if sudo_docker:
        cmd.append("--sudo-docker")
    return cmd


def find_newest_summary(since_mtime: float) -> Path | None:
    runs = _REPO_ROOT / "data" / "runs"
    if not runs.exists():
        return None
    newest: Path | None = None
    newest_m = since_mtime
    for p in runs.glob("*/summary.json"):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m >= newest_m and (newest is None or m > newest_m):
            newest = p
            newest_m = m
    return newest


def run_live_matrix(
    suite: dict[str, Any],
    modes: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    *,
    world: str,
    max_steps: int,
    container: str,
    sudo_docker: bool,
    python_bin: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for task in tasks:
            cmd = _loop_cmd(
                task["command"],
                mode,
                world=world,
                max_steps=max_steps,
                container=container,
                sudo_docker=sudo_docker,
                python_bin=python_bin,
            )
            print()
            print("═" * 64)
            print(f"[LIVE] mode={mode['id']}  task={task['id']}")
            print(f"[LIVE] → {' '.join(shlex.quote(c) for c in cmd)}")
            print("─" * 64)
            t0 = time.time()
            result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
            wall = round(time.time() - t0, 2)
            summary_path = find_newest_summary(t0 - 1.0)
            if summary_path is not None and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["task_id"] = task["id"]
                summary["mode_id"] = mode["id"]
                row = row_from_summary(summary, mode_id=mode["id"], task_id=task["id"])
            else:
                row = row_from_summary(
                    {
                        "mode_id": mode["id"],
                        "task_id": task["id"],
                        "task": task["command"],
                        "world": world,
                        "hybrid": mode.get("hybrid") or "off",
                        "goal_backend": mode.get("goal_backend"),
                        "success": False,
                        "exit_reason": "no_summary",
                        "n_steps": 0,
                        "replan_count": 0,
                        "wall_time_s": wall,
                        "notes": f"loop exit={result.returncode}; summary.json missing",
                    },
                    mode_id=mode["id"],
                    task_id=task["id"],
                )
            rows.append(row)
            print(
                f"[LIVE] done success={row['success']} "
                f"replans={row['replan_count']} wall={row['wall_time_s']}s"
            )
    return rows


def known_gaps() -> list[str]:
    return [
        "Enrichment primitives (pour/tilt/…) stay legacy-only until tracker/verifier hooks (§10).",
        "VLM-as-task-complete is soft success — no oracle goal-check in this harness.",
        "local_llm + full in one matrix cell doubles VRAM pressure; run optionally.",
        "DINO name mismatch / missing detections still degrade fused :init (Session 14).",
        "Gibberish NL may hallucinate MVP goals under local_llm (Session 15).",
        "YELLOW patch may leave verdict YELLOW after 1 VLM call (Session 16).",
        "Prompt few-shot leakage not measured here — Session 23.",
        "Session 22: opt-in --control fd (default remains vlm_steps); measure FD in Session 24.",
        "Hybrid remains opt-in; legacy flag-OFF is still the default path.",
    ]


def write_report(
    comparison: dict[str, Any],
    *,
    out_path: Path,
    suite: dict[str, Any],
    source: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "hybrid_mini_eval_comparison",
        "session": 17,
        "source": source,
        "suite": suite.get("name"),
        "world": suite.get("world"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "table_markdown": format_comparison_table(comparison),
        "comparison": comparison,
        "known_gaps": known_gaps(),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Session 17: legacy vs hybrid closed-loop mini eval"
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help="Suite JSON (tasks + modes)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dry",
        action="store_true",
        help="Build table from checked-in dry fixture (CI-safe, no Gazebo)",
    )
    group.add_argument(
        "--aggregate",
        type=Path,
        nargs="?",
        const=_REPO_ROOT / "data" / "runs",
        help="Aggregate summary.json under path (default: data/runs)",
    )
    group.add_argument(
        "--live",
        action="store_true",
        help="Invoke run_loop_host for each (mode × task) cell",
    )
    parser.add_argument(
        "--dry-fixture",
        type=Path,
        default=_DEFAULT_DRY,
        help="Dry comparison fixture path",
    )
    parser.add_argument(
        "--modes",
        default=None,
        help="Comma-separated mode ids (default: non-optional modes)",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional modes (local_llm, full) when --modes omitted",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        help="Comma-separated task ids (default: all suite tasks)",
    )
    parser.add_argument("--world", default=None, help="Override suite world")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--container", default="vlm_ros2")
    parser.add_argument("--sudo-docker", action="store_true")
    parser.add_argument(
        "--python",
        default=str(_REPO_ROOT / ".venv" / "bin" / "python"),
        help="Python used to launch run_loop_host",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write comparison JSON (default under data/hybrid_mini_eval/)",
    )
    parser.add_argument(
        "--write-fixture",
        type=Path,
        default=None,
        help="Also write/update a checked-in fixture path",
    )
    args = parser.parse_args()

    suite = load_suite(args.suite)
    mode_ids = args.modes.split(",") if args.modes else None
    task_ids = args.tasks.split(",") if args.tasks else None
    modes = select_modes(suite, mode_ids, include_optional=args.include_optional)
    tasks = select_tasks(suite, task_ids)
    world = args.world or suite.get("world") or "tabletop"
    max_steps = args.max_steps or int(suite.get("max_steps") or 10)

    if args.dry:
        rows = load_dry_rows(args.dry_fixture)
        source = f"dry:{args.dry_fixture.relative_to(_REPO_ROOT)}"
    elif args.aggregate is not None:
        rows = aggregate_run_dirs(args.aggregate)
        try:
            rel = args.aggregate.relative_to(_REPO_ROOT)
        except ValueError:
            rel = args.aggregate
        source = f"aggregate:{rel}"
        if not rows:
            print(f"[WARN] No summary.json under {args.aggregate}")
    else:
        py = args.python
        if not Path(py).exists():
            py = sys.executable
        rows = run_live_matrix(
            suite,
            modes,
            tasks,
            world=world,
            max_steps=max_steps,
            container=args.container,
            sudo_docker=args.sudo_docker,
            python_bin=py,
        )
        source = "live:run_loop_host"

    comparison = build_comparison(rows)
    table = format_comparison_table(comparison)
    print()
    print("═" * 64)
    print("Session 17 — legacy vs hybrid mini eval")
    print(f"source={source}  world={world}  n_runs={comparison['n_runs']}")
    print("─" * 64)
    print(table)
    print("─" * 64)
    print("Known gaps:")
    for g in known_gaps():
        print(f"  - {g}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out
    if out is None:
        if args.dry:
            out = _DEFAULT_OUT_DIR / "comparison_dry_last.json"
        else:
            out = _DEFAULT_OUT_DIR / f"comparison_{ts}.json"
    write_report(comparison, out_path=out, suite=suite, source=source)
    print(f"[wrote] {out}")

    if args.write_fixture is not None:
        write_report(
            comparison,
            out_path=args.write_fixture,
            suite=suite,
            source=source,
        )
        print(f"[wrote fixture] {args.write_fixture}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
