#!/usr/bin/env python3
"""
PoC: RGB image → VLM object names → DINO poses → hybrid FD plan.

The photo is an arbitrary real scene. Object/location symbols come from the
VLM, not from a Gazebo world (tabletop/kitchen/workshop). ``--world`` only
selects the robot skill catalog (household vs workshop).

Downstream (default ``--planner fd``) is the hybrid path:
``run_loop_host --mock-scene --control fd --plan-only``.

``--planner llm_plan`` keeps the same image inventory + DINO scene, then
calls ``run_loop_llm_plan.py`` (text-LLM action list; no FD, no enricher).

Usage:
    python scripts/run_image_hybrid.py \\
        --image photo.png \\
        --task "put the mug next to the bottle"

    python scripts/run_image_hybrid.py \\
        --image photo.png \\
        --task "put the mug next to the bottle" \\
        --planner llm_plan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


def _write_run_dir(task: str) -> Path:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in task)[:40]
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = _REPO_ROOT / "data" / "runs" / f"{stamp}_image_{slug or 'task'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PoC live path: VLM names objects in a PNG, DINO localizes them, "
            "then the hybrid FD planner runs on that SceneState."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--world",
        default="household",
        help=(
            "Robot skill catalog only, not a Gazebo world. "
            "household (default): pour/tilt/stir/cut. "
            "workshop: cut/drill/paint/clamp. "
            "Scene objects always come from the image."
        ),
    )
    parser.add_argument(
        "--camera-dir",
        type=Path,
        default=None,
        help="Directory with camera_info.json + camera_pose.json (default: data/).",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=None,
        help="Optional depth .npy (mm, RealSense) for get_pose.",
    )
    parser.add_argument(
        "--inventory-json",
        type=Path,
        default=None,
        help="Skip the VLM: load inventory JSON from this file.",
    )
    parser.add_argument(
        "--inventory-prompt",
        type=Path,
        default=None,
        help=(
            "Override the default task-typed inventory prompt "
            "(prompts/inventory/task_typed.txt). "
            "Pass prompts/inventory/keyword.txt for the older keyword buckets."
        ),
    )
    parser.add_argument(
        "--skip-dino",
        action="store_true",
        help="Use VLM names only (no GroundingDINO). Items sit on the VLM surface.",
    )
    parser.add_argument(
        "--skip-plan",
        action="store_true",
        help="Stop after writing the scene JSON (no Fast Downward).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Inject the FD plan into the running orchestrator (real robot / sim). "
            "Requires bin/start_real.sh or bin/start_sim.sh. Default is plan-only."
        ),
    )
    parser.add_argument(
        "--planner",
        default="fd",
        choices=["fd", "llm_plan"],
        help=(
            "fd (default): hybrid Fast Downward via run_loop_host "
            "(online enrichment R0/R1). "
            "llm_plan: text-LLM grounded action list via run_loop_llm_plan.py; "
            "no Fast Downward, no enricher."
        ),
    )
    parser.add_argument(
        "--online-enrichment",
        default="1",
        metavar="MODE",
        help="Passed to run_loop_host when --planner fd (default 1).",
    )
    parser.add_argument(
        "--enrichment-profile",
        default="r0",
        choices=["r0", "r1"],
        help="Ignored when --planner llm_plan.",
    )
    parser.add_argument("--domain-select", default="llm", choices=["rule_based", "llm"])
    parser.add_argument("--mock-llm", action="store_true")
    parser.add_argument("--mock-fd", action="store_true")
    args = parser.parse_args(argv)

    image_path = args.image if args.image.is_absolute() else _REPO_ROOT / args.image
    if not image_path.is_file():
        print(f"[FAIL] image not found: {image_path}", file=sys.stderr)
        return 2

    from PIL import Image as PilImage

    from planner.image_scene import (
        execution_poses_plink0,
        perceive_image,
        release_cuda,
        scene_to_oracle_mock,
    )
    from vlm.inventory import parse_inventory

    image = PilImage.open(image_path).convert("RGB")
    run_dir = _write_run_dir(args.task)
    image.save(run_dir / "input.png")
    print(f"[image-hybrid] run_dir={run_dir.relative_to(_REPO_ROOT)}")

    inventory = None
    if args.inventory_json is not None:
        inv_path = (
            args.inventory_json
            if args.inventory_json.is_absolute()
            else _REPO_ROOT / args.inventory_json
        )
        inventory = parse_inventory(inv_path.read_text(encoding="utf-8"))
        print(
            f"[image-hybrid] inventory from file: "
            f"objects={list(inventory.object_ids())} "
            f"locations={list(inventory.location_ids())}"
        )
    else:
        from vlm.inventory import list_objects_from_image
        from vlm.planner import VLMPlanner

        print("[image-hybrid] loading Qwen-VL for object names…")
        planner = VLMPlanner()
        planner.load()
        print("[image-hybrid] VLM ready.")
        prompt_path = args.inventory_prompt
        if prompt_path is not None and not prompt_path.is_absolute():
            prompt_path = _REPO_ROOT / prompt_path
        if prompt_path is not None:
            if not prompt_path.is_file():
                print(f"[FAIL] inventory prompt not found: {prompt_path}", file=sys.stderr)
                return 2
            print(f"[image-hybrid] inventory prompt={prompt_path}")
        inventory = list_objects_from_image(
            image, planner, task=args.task, prompt_path=prompt_path
        )
        print(
            f"[image-hybrid] VLM objects={list(inventory.object_ids())} "
            f"locations={list(inventory.location_ids())}"
        )
        del planner
        release_cuda()

    perception = None
    depth = None
    if not args.skip_dino:
        from vlm.perception import PerceptionModule

        print("[image-hybrid] loading GroundingDINO…")
        perception = PerceptionModule()
        perception.load()
        print("[image-hybrid] DINO ready.")
        if args.depth is not None:
            import numpy as np

            depth_path = (
                args.depth if args.depth.is_absolute() else _REPO_ROOT / args.depth
            )
            depth = np.load(str(depth_path))

    scene, inventory, detections = perceive_image(
        image,
        task=args.task,
        perception=perception,
        inventory=inventory,
        camera_dir=args.camera_dir,
        depth_image=depth,
        skip_dino=args.skip_dino,
        prefer_overview=True,
    )
    print(
        f"[image-hybrid] scene objects={ [o.name for o in scene.objects] } "
        f"locations={ [loc.name for loc in scene.locations] }"
    )

    inv_dump = {
        "objects": [{"id": e.id, "query": e.query} for e in inventory.objects],
        "locations": [{"id": e.id, "query": e.query} for e in inventory.locations],
        "containers": [
            {"id": e.id, "query": e.query} for e in inventory.containers
        ],
    }
    (run_dir / "inventory.json").write_text(
        json.dumps(inv_dump, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    mock = scene_to_oracle_mock(
        scene, execution_poses=execution_poses_plink0(scene, detections)
    )
    scene_path = run_dir / "scene_oracle_mock.json"
    scene_path.write_text(json.dumps(mock, indent=2, ensure_ascii=False) + "\n")
    print(f"[image-hybrid] wrote {scene_path.relative_to(_REPO_ROOT)}")

    if detections and perception is not None:
        try:
            ann = perception.draw_detections(image, detections)
            ann.save(run_dir / "dino.png")
            print("[image-hybrid] wrote dino.png")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] DINO annotation failed: {exc}")

    if perception is not None:
        del perception
        release_cuda()

    if args.skip_plan:
        print("[image-hybrid] --skip-plan: not calling the planner")
        return 0

    if args.planner == "llm_plan":
        if args.execute:
            print(
                "[WARN] --execute is ignored with --planner llm_plan "
                "(baseline is always plan-only)",
                file=sys.stderr,
            )
        host = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_loop_llm_plan.py"),
            "--task",
            args.task,
            "--world",
            args.world,
            "--mock-scene",
            "--scene-file",
            str(scene_path),
            "--plan-only",
        ]
        if args.mock_llm:
            host.append("--mock-llm")
        print("[image-hybrid] llm_plan downstream:", " ".join(host[2:]))
    else:
        host = [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "run_loop_host.py"),
            "--task",
            args.task,
            "--world",
            args.world,
            "--hybrid",
            "mvp",
            "--control",
            "fd",
            "--goal-backend",
            "local_llm",
            "--domain-select",
            args.domain_select,
            "--online-enrichment",
            str(args.online_enrichment),
            "--enrichment-profile",
            args.enrichment_profile,
            "--mock-scene",
            "--scene-file",
            str(scene_path),
            "--scene-source",
            "inventory",
        ]
        if args.execute:
            print("[image-hybrid] --execute: will inject the FD plan")
        else:
            host.append("--plan-only")
        if args.mock_llm:
            host.append("--mock-llm")
        if args.mock_fd:
            host.append("--mock-fd")
        print("[image-hybrid] golden downstream:", " ".join(host[2:]))
    env = os.environ.copy()
    result = subprocess.run(host, cwd=str(_REPO_ROOT), env=env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
