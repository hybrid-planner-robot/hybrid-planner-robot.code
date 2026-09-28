#!/usr/bin/env python3
"""
Interactive NL command REPL for manual hybrid / loop experiments.

Loads VLM + Perception once, then repeatedly asks for a natural-language
task and runs the same closed loop as ``run_loop_host.py``.

  python scripts/lab_manual_repl.py --world tabletop --hybrid mvp

Every knob the loop takes is settable per task with a slash command, so one
session can walk through the interesting cases — fixed domain, FD control,
online enrichment, refusal — without restarting the sim:

  python scripts/lab_manual_repl.py --world kitchen --hybrid mvp \\
    --control fd --online-enrichment --domain-select llm \\
    --text-llm-model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TEXT_LLM_ENV = "VLMRP_TEXT_LLM_MODEL"

_ON = {"on", "1", "true", "yes"}
_OFF = {"off", "0", "false", "no"}


def _print_banner(args: argparse.Namespace) -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Manual lab REPL — natural-language tasks                    ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  world={args.world}  hybrid={args.hybrid or 'off'}  "
          f"control={args.control or 'vlm_steps'}  "
          f"goal_backend={args.goal_backend or 'rule_based'}  "
          f"scene_source={args.scene_source or 'fused'}  "
          f"perception_only={getattr(args, 'perception_only', False)}")
    print(f"  enrichment={'on' if args.online_enrichment else 'off'}  "
          f"domain_select={args.domain_select or 'rule_based'}  "
          f"text_llm={args.text_llm_model or os.environ.get(_TEXT_LLM_ENV) or 'default (1.5B)'}")
    print("  Commands:")
    print("    <task text>     run closed-loop for that NL task")
    print("    /help           show this help")
    print("    /world NAME     set Gazebo world tag (overview calib)")
    print("    /hybrid MODE    off | mvp | full | 1")
    print("    /control MODE   vlm_steps | fd   (fd needs hybrid=mvp)")
    print("    /goal BACKEND   rule_based | local_llm")
    print("    /scene SRC      fused | dino | oracle")
    print("    /enrich on|off  closed-catalog online enrichment")
    print("    /select BACKEND rule_based | llm   (who judges the domain)")
    print("    /model ID       text LLM id, or 'default' to unset")
    print("    /perception-only on|off   disable SIM NameMatch+snap")
    print("    /show           print the current configuration")
    print("    /quit /exit     leave REPL")
    print()


def _warn_incompatible(args: argparse.Namespace) -> None:
    """Catch the combinations the loop rejects, before the sim is disturbed."""
    hybrid = (args.hybrid or "off").lower()
    if args.control == "fd" and hybrid != "mvp":
        print(f"[LAB] control=fd requires hybrid=mvp (currently {hybrid}) — "
              "the loop will refuse. Use /hybrid mvp.")
    if args.domain_select == "llm" and not args.online_enrichment:
        print("[LAB] domain_select=llm without /enrich on: the LLM will pick "
              "the template but nothing will author PDDL for an incomplete one.")
    if args.online_enrichment and args.domain_select != "llm":
        print("[LAB] enrichment on with the keyword selector: paraphrases that "
              "never say 'pour' will read as complete. Use /select llm.")


def _run_task(task: str, args: argparse.Namespace) -> int:
    cmd = [
        str(_REPO_ROOT / ".venv" / "bin" / "python"),
        str(_REPO_ROOT / "scripts" / "run_loop_host.py"),
        "--task",
        task,
        "--world",
        args.world,
        "--max-steps",
        str(args.max_steps),
        "--container",
        args.container,
    ]
    if args.hybrid and args.hybrid.lower() not in {"off", "0", "false", "none"}:
        cmd.extend(["--hybrid", args.hybrid])
    if args.goal_backend:
        cmd.extend(["--goal-backend", args.goal_backend])
    if args.scene_source:
        cmd.extend(["--scene-source", args.scene_source])
    if args.control:
        cmd.extend(["--control", args.control])
    if args.online_enrichment:
        cmd.extend(["--online-enrichment", "1"])
    if args.domain_select:
        cmd.extend(["--domain-select", args.domain_select])
    if getattr(args, "perception_only", False):
        cmd.append("--perception-only")
    if args.sudo_docker:
        cmd.append("--sudo-docker")

    env = dict(os.environ)
    if args.text_llm_model:
        env[_TEXT_LLM_ENV] = args.text_llm_model

    _warn_incompatible(args)
    print()
    print(f"[LAB] → {' '.join(shlex.quote(c) for c in cmd)}")
    print("─" * 60)
    # Inherit stdout/stderr so this terminal is the host-loop channel.
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    print("─" * 60)
    # 3 is the loop's refusal exit: the catalog could not close the goal gap.
    verdict = " (refused — no catalog skill for the gap)" if result.returncode == 3 else ""
    print(f"[LAB] loop exit={result.returncode}{verdict}")
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual lab NL task REPL")
    parser.add_argument("--world", default="tabletop")
    parser.add_argument("--hybrid", default="mvp")
    parser.add_argument("--goal-backend", default="rule_based")
    parser.add_argument("--scene-source", default="fused",
                        choices=["fused", "dino", "oracle"])
    parser.add_argument(
        "--control",
        default=None,
        choices=["vlm_steps", "fd"],
        help="Action policy (Session 22); fd requires --hybrid mvp",
    )
    parser.add_argument(
        "--online-enrichment",
        action="store_true",
        help="Closed-catalog online enrichment (Sessions 28–30)",
    )
    parser.add_argument(
        "--domain-select",
        default=None,
        choices=["rule_based", "llm"],
        help="Who judges template + completeness (Session 29)",
    )
    parser.add_argument(
        "--text-llm-model",
        default=None,
        help=(
            f"Text LLM id for the child loop (sets {_TEXT_LLM_ENV}). The 1.5B "
            "default cannot author enrichment PDDL; use a 7B for that."
        ),
    )
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Disable SIM NameMatch + oracle snap (Session 25)",
    )
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--container", default="vlm_ros2")
    parser.add_argument("--sudo-docker", action="store_true")
    args = parser.parse_args()

    _print_banner(args)

    while True:
        try:
            line = input("lab> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line in {"/quit", "/exit", "quit", "exit"}:
            break
        if line in {"/help", "help", "?"}:
            _print_banner(args)
            continue
        if line.startswith("/world "):
            args.world = line.split(maxsplit=1)[1].strip()
            print(f"[LAB] world={args.world}")
            continue
        if line.startswith("/hybrid "):
            args.hybrid = line.split(maxsplit=1)[1].strip()
            print(f"[LAB] hybrid={args.hybrid}")
            continue
        if line.startswith("/goal "):
            args.goal_backend = line.split(maxsplit=1)[1].strip()
            print(f"[LAB] goal_backend={args.goal_backend}")
            continue
        if line.startswith("/scene "):
            args.scene_source = line.split(maxsplit=1)[1].strip()
            print(f"[LAB] scene_source={args.scene_source}")
            continue
        if line.startswith("/control "):
            value = line.split(maxsplit=1)[1].strip()
            if value not in {"vlm_steps", "fd"}:
                print("[LAB] usage: /control vlm_steps|fd")
                continue
            args.control = value
            print(f"[LAB] control={args.control}")
            _warn_incompatible(args)
            continue
        if line.startswith("/enrich"):
            parts = line.split(maxsplit=1)
            value = parts[1].strip().lower() if len(parts) > 1 else "on"
            if value in _ON:
                args.online_enrichment = True
            elif value in _OFF:
                args.online_enrichment = False
            else:
                print("[LAB] usage: /enrich on|off")
                continue
            print(f"[LAB] enrichment={'on' if args.online_enrichment else 'off'}")
            _warn_incompatible(args)
            continue
        if line.startswith("/select "):
            value = line.split(maxsplit=1)[1].strip()
            if value not in {"rule_based", "llm"}:
                print("[LAB] usage: /select rule_based|llm")
                continue
            args.domain_select = value
            print(f"[LAB] domain_select={args.domain_select}")
            _warn_incompatible(args)
            continue
        if line.startswith("/model "):
            value = line.split(maxsplit=1)[1].strip()
            args.text_llm_model = None if value == "default" else value
            print(f"[LAB] text_llm={args.text_llm_model or 'default (1.5B)'} "
                  "— loaded on the next task, first run is slow")
            continue
        if line in {"/show", "/config"}:
            _print_banner(args)
            continue
        if line.startswith("/perception-only"):
            parts = line.split(maxsplit=1)
            if len(parts) == 1 or parts[1].strip().lower() in {"on", "1", "true"}:
                args.perception_only = True
            elif parts[1].strip().lower() in {"off", "0", "false"}:
                args.perception_only = False
            else:
                print("[LAB] usage: /perception-only on|off")
                continue
            print(f"[LAB] perception_only={args.perception_only}")
            continue
        if line.startswith("/"):
            print(f"[LAB] unknown command: {line}  (try /help)")
            continue

        _run_task(line, args)

    print("[LAB] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
