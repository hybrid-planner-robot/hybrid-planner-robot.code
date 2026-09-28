#!/usr/bin/env python3
"""
Live Gazebo demo of the closed-catalog enrichment loop (Sessions 28–30).

For each case in the suite the script:

1. Resets the world poses (``/reset_world``) so the previous run does not
   contaminate the next.
2. Prints a clear banner for the audience.
3. Runs ``run_loop_host.py`` with enrichment on — stdout/stderr stay in this
   terminal, so selection, authoring, FD, and motion are all visible while
   Gazebo shows the arm.
4. Reads ``summary.json``, scores the neurosymbolic path
   (complete / enriched / refused), and continues.

Motion success is reported but does **not** decide the case score: Session 24
owns arm reliability. This harness measures whether the text LLM selected,
enriched or refused correctly, and whether the loop consumed the result.

Requires an already-running simulation (orchestrator + cameras). Prefer::

    bin/eval_enrichment_live.sh

which resets the stack, waits for readiness, then calls this script.

Examples
--------
Full suite (sim already up)::

    source .venv/bin/activate
    export VLMRP_TEXT_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
    python scripts/eval_enrichment_live.py

Subset + pause so people can watch each run::

    python scripts/eval_enrichment_live.py --cases paraphrase_drink,refuse_solder \\
        --pause 8 --model Qwen/Qwen2.5-7B-Instruct

Wait for Enter between cases (guided demo)::

    python scripts/eval_enrichment_live.py --interactive

Dry self-check of scoring (no Docker / Gazebo)::

    python scripts/eval_enrichment_live.py --dry
"""

from __future__ import annotations

import argparse
import json
import os
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
    _REPO_ROOT / "tests" / "fixtures" / "enrichment_live" / "suite_v1.json"
)
_DEFAULT_OUT = _REPO_ROOT / "data" / "enrichment_live"
_TEXT_LLM_ENV = "VLMRP_TEXT_LLM_MODEL"
_RUNS_DIR = _REPO_ROOT / "data" / "runs"


# ── Suite loading ────────────────────────────────────────────────────────────


def load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_cases(
    suite: dict[str, Any],
    case_ids: list[str] | None,
) -> list[dict[str, Any]]:
    cases = list(suite.get("cases") or [])
    if not case_ids:
        return cases
    wanted = {c.strip() for c in case_ids if c.strip()}
    selected = [c for c in cases if c["id"] in wanted]
    missing = wanted - {c["id"] for c in selected}
    if missing:
        raise SystemExit(f"Unknown case id(s): {sorted(missing)}")
    return selected


# ── Sim helpers ──────────────────────────────────────────────────────────────


def ensure_world(world: str, container: str, current: str | None) -> str:
    """Relaunch the sim when the next case needs a different Gazebo world."""
    if current == world:
        return current
    print(f"[DEMO] World change {current!r} → {world!r}: relaunching sim…")
    cmd = [
        str(_REPO_ROOT / "bin" / "reset_and_test_fd.sh"),
        "--world",
        world,
    ]
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        raise SystemExit(
            f"[FAIL] could not relaunch sim for world={world} "
            f"(exit {result.returncode})"
        )
    if not orchestrator_ready(container):
        raise SystemExit(f"[FAIL] orchestrator not ready after world={world}")
    return world


def reset_world(container: str) -> None:
    """Return objects to spawn poses between cases (soft reset)."""
    cmd = [
        "docker",
        "exec",
        container,
        "bash",
        "-lc",
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "ros2 service call /reset_world std_srvs/srv/Empty",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[WARN] /reset_world failed — continuing with the current poses")
        if result.stderr.strip():
            print(f"       {result.stderr.strip()[:200]}")
    else:
        print("[OK]   World reset.")
    time.sleep(2.0)


def orchestrator_ready(container: str) -> bool:
    cmd = [
        "docker",
        "exec",
        container,
        "bash",
        "-lc",
        "source /opt/ros/humble/setup.bash && "
        "timeout 3 ros2 topic list 2>/dev/null | grep -q '/vlm_planner/inject_plan'",
    ]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# ── Loop invocation ──────────────────────────────────────────────────────────


def build_loop_cmd(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    python_bin: str,
    container: str,
    max_steps: int | None,
    model: str | None,
) -> list[str]:
    world = case.get("world") or defaults.get("world") or "kitchen"
    hybrid = case.get("hybrid") or defaults.get("hybrid") or "mvp"
    control = case.get("control") or defaults.get("control") or "fd"
    goal_backend = (
        case.get("goal_backend") or defaults.get("goal_backend") or "local_llm"
    )
    domain_select = (
        case.get("domain_select") or defaults.get("domain_select") or "llm"
    )
    steps = max_steps or int(case.get("max_steps") or defaults.get("max_steps") or 10)
    enrich = case.get("online_enrichment")
    if enrich is None:
        enrich = bool(defaults.get("online_enrichment", True))

    cmd = [
        python_bin,
        str(_REPO_ROOT / "scripts" / "run_loop_host.py"),
        "--task",
        case["task"],
        "--world",
        world,
        "--hybrid",
        hybrid,
        "--control",
        control,
        "--goal-backend",
        goal_backend,
        "--max-steps",
        str(steps),
        "--container",
        container,
    ]
    if enrich:
        cmd.extend(["--online-enrichment", "1"])
    if domain_select:
        cmd.extend(["--domain-select", domain_select])
    return cmd


def newest_summary_after(t0: float) -> Path | None:
    """Pick the most recently written summary.json newer than ``t0``."""
    if not _RUNS_DIR.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for path in _RUNS_DIR.glob("*/summary.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime + 0.05 >= t0:
            candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def run_case(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    python_bin: str,
    container: str,
    max_steps: int | None,
    model: str | None,
    env: dict[str, str],
) -> dict[str, Any]:
    cmd = build_loop_cmd(
        case,
        defaults,
        python_bin=python_bin,
        container=container,
        max_steps=max_steps,
        model=model,
    )
    print()
    print(f"[RUN]  {' '.join(shlex.quote(c) for c in cmd)}")
    print("─" * 72)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    wall = round(time.time() - t0, 1)
    print("─" * 72)

    summary_path = newest_summary_after(t0)
    summary: dict[str, Any] = {}
    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] could not read {summary_path}: {exc}")

    row = {
        "id": case["id"],
        "task": case["task"],
        "note": case.get("note", ""),
        "expect": case.get("expect") or {},
        "exit_code": int(result.returncode),
        "wall_s": wall,
        "run_dir": (
            str(summary_path.parent.relative_to(_REPO_ROOT))
            if summary_path is not None
            else None
        ),
        "exit_reason": summary.get("exit_reason"),
        "success": bool(summary.get("success")),
        "enrichment_used": bool(summary.get("enrichment_used")),
        "enrichment_skills": list(summary.get("enrichment_skills") or []),
        "domain_persisted": summary.get("domain_persisted"),
        "domain_reused": summary.get("domain_reused"),
        "domain_select_backend": summary.get("domain_select_backend"),
        "refuse_reason": summary.get("refuse_reason"),
        "domain_template": summary.get("domain_template"),
        "n_steps": int(summary.get("n_steps") or 0),
        "replan_count": int(summary.get("replan_count") or 0),
    }
    ok, verdict = score_case(row)
    row["ok"] = ok
    row["verdict"] = verdict
    return row


# ── Scoring ──────────────────────────────────────────────────────────────────


def inferred_path(row: dict[str, Any]) -> str:
    """Map a run onto complete | enriched | refused."""
    if row["exit_code"] == 3 or row.get("exit_reason") == "refused":
        return "refused"
    if row.get("enrichment_used") or row.get("enrichment_skills"):
        return "enriched"
    return "complete"


def score_case(row: dict[str, Any]) -> tuple[bool, str]:
    """
    Score the neurosymbolic path only.

    Motion ``success`` is surfaced in the report but never flips the case:
    a correct refuse that exits 3 is a pass even with no arm motion, and an
    enriched pour that MoveIt drops is still a path pass.
    """
    expect = row.get("expect") or {}
    wanted = expect.get("path")
    got = inferred_path(row)

    if wanted and got != wanted:
        return False, f"expected path={wanted}, got {got}"

    skill = expect.get("skill")
    if skill and skill not in (row.get("enrichment_skills") or []):
        return False, (
            f"expected skill {skill!r}, got "
            f"{row.get('enrichment_skills') or 'none'}"
        )

    if got == "refused":
        return True, f"refused as expected ({(row.get('refuse_reason') or '')[:80]})"
    if got == "enriched":
        skills = ",".join(row.get("enrichment_skills") or []) or "?"
        motion = "motion OK" if row.get("success") else "motion not confirmed"
        return True, f"enriched [{skills}] — {motion}"
    motion = "motion OK" if row.get("success") else "motion not confirmed"
    return True, f"complete (no enrichment) — {motion}"


# ── Dry fixtures (harness self-check) ────────────────────────────────────────


def _dry_row(case: dict[str, Any]) -> dict[str, Any]:
    """Ideal summary for a case — proves scoring without Gazebo."""
    expect = case.get("expect") or {}
    path = expect.get("path", "complete")
    skills = [expect["skill"]] if expect.get("skill") else []
    row = {
        "id": case["id"],
        "task": case["task"],
        "note": case.get("note", ""),
        "expect": expect,
        "exit_code": 3 if path == "refused" else 0,
        "wall_s": 0.0,
        "run_dir": None,
        "exit_reason": "refused" if path == "refused" else "success",
        "success": path != "refused",
        "enrichment_used": path == "enriched",
        "enrichment_skills": skills,
        "domain_persisted": (
            "pddl/domains/enriched/demo.pddl" if path == "enriched" else None
        ),
        "domain_reused": False,
        "domain_select_backend": "llm",
        "refuse_reason": "dry refuse" if path == "refused" else None,
        "domain_template": "manipulation_base",
        "n_steps": 0 if path == "refused" else 3,
        "replan_count": 0,
    }
    ok, verdict = score_case(row)
    row["ok"] = ok
    row["verdict"] = verdict
    return row


# ── Presentation ─────────────────────────────────────────────────────────────


def banner(case: dict[str, Any], index: int, total: int) -> None:
    expect = case.get("expect") or {}
    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  CASE {index}/{total}: {case['id']:<54} ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  task   : {case['task'][:58]:<58} ║")
    print(
        f"║  expect : path={expect.get('path', '?'):<12} "
        f"skill={str(expect.get('skill') or '—'):<20} ║"
    )
    note = (case.get("note") or "")[:58]
    if note:
        print(f"║  note   : {note:<58} ║")
    print("╚" + "═" * 70 + "╝")


def pause_between(args: argparse.Namespace, remaining: int) -> None:
    if remaining <= 0:
        return
    if args.interactive:
        try:
            input(f"\n[DEMO] Press Enter for the next case ({remaining} left)… ")
        except (EOFError, KeyboardInterrupt):
            print("\n[DEMO] interrupted")
            raise SystemExit(130)
        return
    if args.pause > 0:
        print(f"\n[DEMO] Pausing {args.pause:.0f}s before the next case…")
        time.sleep(args.pause)


def format_report_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| case | path ok | verdict | exit | enrich | skills | motion | wall_s |",
        "|---|:---:|---|---:|:---:|---|:---:|---:|",
    ]
    for r in rows:
        mark = "OK" if r["ok"] else "FAIL"
        lines.append(
            f"| {r['id']} | {mark} | {r['verdict']} | {r['exit_code']} | "
            f"{'yes' if r.get('enrichment_used') else 'no'} | "
            f"{','.join(r.get('enrichment_skills') or []) or '—'} | "
            f"{'yes' if r.get('success') else 'no'} | {r['wall_s']} |"
        )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live Gazebo demo of closed-catalog enrichment (per-case loop)"
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"Suite JSON (default {_DEFAULT_SUITE.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids (default: all)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Text LLM id (sets {_TEXT_LLM_ENV}). Prefer 7B for authoring.",
    )
    parser.add_argument("--container", default="vlm_ros2")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--pause",
        type=float,
        default=5.0,
        help="Seconds to wait between cases so the audience can watch (0=off)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Wait for Enter between cases instead of --pause",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip /reset_world between cases",
    )
    parser.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Do not require the orchestrator topic before starting",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Score ideal summaries only — no Docker / Gazebo",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Report directory (default {_DEFAULT_OUT.relative_to(_REPO_ROOT)}/<ts>)",
    )
    args = parser.parse_args()

    suite = load_suite(args.suite)
    defaults = dict(suite.get("defaults") or {})
    case_ids = (
        [c.strip() for c in args.cases.split(",") if c.strip()]
        if args.cases
        else None
    )
    cases = select_cases(suite, case_ids)
    if not cases:
        print("[FAIL] no cases selected")
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out_dir or (_DEFAULT_OUT / ts)
    out.mkdir(parents=True, exist_ok=True)

    model = args.model or os.environ.get(_TEXT_LLM_ENV)
    python_bin = str(_REPO_ROOT / ".venv" / "bin" / "python")
    if not Path(python_bin).exists():
        python_bin = sys.executable

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Enrichment live demo — Gazebo + closed-catalog loop         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  suite   : {args.suite.relative_to(_REPO_ROOT)}")
    print(f"  cases   : {', '.join(c['id'] for c in cases)}")
    print(f"  model   : {model or 'default (1.5B — too weak to author PDDL)'}")
    print(f"  world   : {defaults.get('world', 'kitchen')}")
    print(f"  out     : {out.relative_to(_REPO_ROOT)}")
    print(
        "  scoring : neurosymbolic path only "
        "(motion success is reported, not required)"
    )
    print()

    if args.dry:
        rows = [_dry_row(c) for c in cases]
        for i, (case, row) in enumerate(zip(cases, rows), start=1):
            banner(case, i, len(cases))
            mark = "OK  " if row["ok"] else "FAIL"
            print(f"[{mark}] {row['verdict']}")
        passed = sum(1 for r in rows if r["ok"])
        report = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "dry": True,
            "passed": passed,
            "total": len(rows),
            "cases": rows,
        }
        (out / "report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        print()
        print(format_report_table(rows))
        print(f"\n[DEMO] {passed}/{len(rows)} path checks passed (dry)")
        print(f"[DEMO] report → {out.relative_to(_REPO_ROOT)}/report.json")
        return 0 if passed == len(rows) else 1

    if not args.skip_ready_check and not orchestrator_ready(args.container):
        print(
            f"[FAIL] Orchestrator not ready in container {args.container!r}.\n"
            "       Start the sim first, e.g.:\n"
            "         bin/eval_enrichment_live.sh\n"
            "       or:\n"
            "         bin/reset_and_test_fd.sh --world kitchen\n"
            "         python scripts/eval_enrichment_live.py …"
        )
        return 2

    env = dict(os.environ)
    if model:
        env[_TEXT_LLM_ENV] = model

    rows: list[dict[str, Any]] = []
    current_world: str | None = defaults.get("world")
    try:
        for i, case in enumerate(cases, start=1):
            banner(case, i, len(cases))
            wanted_world = case.get("world") or defaults.get("world") or "tabletop"
            if not args.dry:
                current_world = ensure_world(
                    wanted_world, args.container, current_world
                )
            if not args.no_reset:
                reset_world(args.container)
            row = run_case(
                case,
                defaults,
                python_bin=python_bin,
                container=args.container,
                max_steps=args.max_steps,
                model=model,
                env=env,
            )
            rows.append(row)
            mark = "OK  " if row["ok"] else "FAIL"
            print(f"\n[{mark}] {row['id']}: {row['verdict']}")
            if row.get("run_dir"):
                print(f"       run → {row['run_dir']}/summary.json")
            pause_between(args, remaining=len(cases) - i)
    except KeyboardInterrupt:
        print("\n[DEMO] interrupted — writing partial report")

    passed = sum(1 for r in rows if r["ok"])
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dry": False,
        "model": model,
        "suite": str(args.suite.relative_to(_REPO_ROOT)),
        "defaults": defaults,
        "passed": passed,
        "total": len(rows),
        "cases": rows,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(format_report_table(rows))
    print(f"\n[DEMO] {passed}/{len(rows)} path checks passed")
    print(f"[DEMO] report → {report_path.relative_to(_REPO_ROOT)}")
    return 0 if rows and passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
