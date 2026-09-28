#!/usr/bin/env python3
"""
run_loop_host.py — Closed-loop task execution (HOST side).

Implements the closed-loop architecture.

Default (``--control vlm_steps`` / unset):
  [scan] -> [capture] -> [VLM remaining plan] -> [inject] -> [wait complete] -> repeat

Opt-in Session 22 (``--control fd`` / ``VLMRP_CONTROL=fd``), requires ``--hybrid mvp``:
  [scan] -> [capture] -> [hybrid :init/:goal] -> [inject + FD plan] -> [execute]
  On failure → refresh SceneState → FD again (no VLM ``plan_next_step``).
  Session 23: ``--hybrid full --control fd`` is refused (verifier unused on FD path).
  ``--plan-only`` stops after the FD plan (scan + DINO + problem + planner;
  no inject, no arm motion).

Each vlm_steps iteration:
  1. Pre-scan: move arm to scan pose via _pre_scan.py (wrist camera view)
  2. Capture: take image from wrist camera via _capture_scene.py
  3. VLM: plan_remaining(task, image, completed_steps) -> action policy
  4. Ground: GroundingDINO -> correct object names and 3D poses
  5. Inject: send single-step plan to orchestrator
  6. Wait: _wait_step_complete.py -> get completion signal
  7. If complete: break; else: add step to completed_steps, repeat

Sim-to-real note: the same loop works on the real robot — the only difference
is that Gazebo oracle is replaced by RealSense depth in the PerceptionModule.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


from planner.live_scene import (  # noqa: E402
    _capture,
    _docker,
    _estimate_object_height,
    _get_gazebo_models,
    _get_overview_cam_data,
    _get_scene_objects,
    _infer_on_surface,
    _load_depth_array,
    _pre_scan,
    _print_localisation_report,
    _read_overview_pose_from_world,
    _run_dino_scene_sweep,
    _run_in_container,
    _world_surfaces,
)


def _check_goal_in_world(args, *, goal_facts, world_name: str) -> dict | None:
    """Check the PDDL goal facts against freshly-read world poses.

    Returns ``None`` when there is nothing to check against — no goal facts (the
    legacy path never records them), no readable surface geometry, or no poses —
    so the caller can tell "not satisfied" apart from "not measured".
    """
    if not goal_facts:
        return None
    surfaces = _world_surfaces(world_name)
    if not surfaces:
        print(f"[WARN] No surface geometry for world {world_name!r} — cannot verify goal")
        return None
    poses = _get_gazebo_models(args)
    if not poses:
        print("[WARN] No world poses available — cannot verify goal")
        return None

    from planner.support_surfaces import check_goal_facts

    check = check_goal_facts(goal_facts, poses, surfaces.values())
    out = check.as_dict()
    out["summary"] = check.summary_line()
    return out




def _annotate_handled_objects(
    image,
    placed_at: dict,
    data_dir: str,
    info_file: str = "camera_info.json",
    pose_file: str = "camera_pose.json",
) -> "PIL.Image.Image":
    """
    Annotate the image with already-handled objects using two non-obstructive elements:
    1. A small cross (+) at the projected 3D position of each placed object
    2. A text legend box in the top-left corner listing all handled objects

    The small cross minimally occludes the scene; the text box is fully readable
    by the VLM. This approach avoids covering nearby unhandled objects.
    """
    if not placed_at:
        return image

    import json
    import numpy as np
    from PIL import ImageDraw
    from pathlib import Path

    ci_path = Path(data_dir) / info_file
    cp_path = Path(data_dir) / pose_file

    K, cam_to_base, R, t = None, None, None, None
    if ci_path.exists() and cp_path.exists():
        try:
            with open(ci_path) as f:
                K = np.array(json.load(f)["K"])
            with open(cp_path) as f:
                cam_to_base = np.array(json.load(f)["cam_to_base"])
            base_to_cam = np.linalg.inv(cam_to_base)
            R = base_to_cam[:3, :3]
            t = base_to_cam[:3, 3]
        except Exception:
            pass

    dbg  = image.copy()
    draw = ImageDraw.Draw(dbg)
    W, H = dbg.width, dbg.height
    CS   = max(5, min(W, H) // 80)   # cross arm length (tiny)

    # ── 1. Small cross at each projected object position ─────────────────────
    if R is not None:
        for i, (name, (px, py)) in enumerate(placed_at.items(), 1):
            p_cam = R @ np.array([px, py, 0.025]) + t
            if p_cam[2] <= 0.05:
                continue
            u = int(K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2])
            v = int(K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2])
            if not (CS <= u < W - CS and CS <= v < H - CS):
                continue
            draw.line([u - CS, v, u + CS, v], fill=(0, 220, 0), width=2)
            draw.line([u, v - CS, u, v + CS], fill=(0, 220, 0), width=2)
            draw.text((u + CS + 1, v - CS), str(i), fill=(0, 220, 0))

    # ── 2. Text legend box in top-left corner ────────────────────────────────
    PAD   = 6
    LH    = 14   # line height
    lines = ["DONE:"] + [f" {i}. {n}" for i, n in enumerate(placed_at, 1)]
    box_w = max(len(l) for l in lines) * 7 + PAD * 2
    box_h = len(lines) * LH + PAD * 2
    draw.rectangle([2, 2, box_w, box_h], fill=(0, 60, 0))
    draw.rectangle([2, 2, box_w, box_h], outline=(0, 200, 0), width=1)
    for i, line in enumerate(lines):
        color = (180, 255, 180) if i == 0 else (220, 255, 220)
        draw.text((PAD + 2, PAD + i * LH), line, fill=color)

    return dbg


def _publish_perception_pose(
    args, object_name: str, x: float, y: float, z: float,
    height_m: float | None = None,
) -> bool:
    """Publish a perception-estimated pose to /perception/object_pose.

    height_m: estimated object height in metres (from _estimate_object_height).
              Encoded in orientation.z; None → 0.0 → orchestrator uses fallback.
    """
    height_arg = f" --height_m {height_m:.4f}" if height_m is not None else ""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_publish_perception_pose.py "
        f"--object {object_name} --x {x:.6f} --y {y:.6f} --z {z:.6f}{height_arg}"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    for line in r.stdout.decode().strip().splitlines():
        print(f"       {line}")
    return r.returncode == 0






def _wait_step_complete(args, timeout: int = 60, min_seq: int = 0) -> dict:
    """Wait for step completion signal from orchestrator.

    min_seq: ignore step_complete messages with seq < this value, preventing
    stale TRANSIENT_LOCAL (latched) messages from previous steps being accepted
    as the result of the current step.
    """
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        f"python3 /workspace/scripts/_wait_step_complete.py --timeout {timeout} --min-seq {min_seq}"
    )
    r = _run_in_container(args, bash_cmd, timeout=timeout + 5)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip())
        except Exception:
            pass
    return {"success": False, "task_complete": False}


def _sync_step_seq(args, last_seq: int) -> int:
    """Advance last_seq past any latched step_complete (avoids stale fails)."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_peek_step_seq.py --timeout 1.5"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    if r.returncode != 0:
        return last_seq
    try:
        data = json.loads(r.stdout.decode().strip())
        seq = int(data.get("seq", -1))
    except Exception:
        return last_seq
    if seq > last_seq:
        print(f"[LOOP] Synced step seq latch → {seq} (ignore stale completions)")
        return seq
    return last_seq


def _write_refusal_summary(
    args,
    *,
    refuse_reason: str,
    domain_template: str,
    select_backend: str,
    selection_reason: str,
    domain_completeness: str | None = None,
    needed_skills: tuple[str, ...] | list[str] = (),
) -> None:
    """
    Record a catalog refusal as a normal run (Session 30).

    The loop exits before the usual run directory exists, so without this a
    refusal would leave nothing behind but console output — and the mini-eval
    harness would not be able to tell "refused honestly" from "crashed".
    """
    import datetime

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    world_tag = getattr(args, "world", "unknown")
    task_tag = args.task[:30].replace(" ", "_").replace("/", "-")
    run_dir = _REPO_ROOT / "data" / "runs" / f"{ts}_{world_tag}_{task_tag}"
    summary = {
        "task": args.task,
        "world": world_tag,
        "control": getattr(args, "control", None),
        "domain_template": domain_template,
        "success": False,
        "exit_reason": "refused",
        "n_steps": 0,
        "completed_steps": [],
        "enrichment_used": False,
        "domain_persisted": None,
        "domain_select_backend": select_backend,
        "refuse_reason": refuse_reason,
        "selection_reason": selection_reason,
        "plan_only": bool(getattr(args, "plan_only", False)),
        "fd_actions": [],
        "n_plan_actions": 0,
        "domain_completeness": domain_completeness,
        "needed_skills": list(needed_skills),
    }
    from planner.call_timings import attach_snapshot

    attach_snapshot(summary)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[LOOP] summary.json → {run_dir.relative_to(_REPO_ROOT)}/summary.json")
    except Exception as exc:
        print(f"[WARN] refusal summary.json write failed: {exc}")


def _fd_docker_timeout_s() -> int:
    from planner.fast_downward import resolve_search_time_limit_s

    return resolve_search_time_limit_s() + 20


def _fd_solve_in_container(
    args,
    *,
    domain_template: str,
    problem: str,
    domain_text: str | None = None,
) -> dict:
    """
    Run Fast Downward inside Docker; return {success, actions, primitives}.

    ``domain_text`` carries an online-enriched domain (Session 30); without it
    the container reads the fixed template from ``/workspace/pddl/domains``.
    """
    from planner.call_timings import fd_span

    payload = json.dumps(
        {
            "domain_template": domain_template,
            "problem": problem,
            "domains_dir": "/workspace/pddl/domains",
            "domain": domain_text or "",
        }
    )
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_fd_solve_problem.py"
    )
    with fd_span():
        try:
            r = subprocess.run(
                _docker(args.container, args.sudo_docker) + ["bash", "-c", bash_cmd],
                input=payload.encode(),
                capture_output=True,
                timeout=_fd_docker_timeout_s(),
            )
        except subprocess.TimeoutExpired:
            from planner.fast_downward import (
                resolve_search_time_limit_s,
                result_from_timeout,
            )

            return result_from_timeout(resolve_search_time_limit_s())
        out = (r.stdout or b"").decode().strip()
        err = (r.stderr or b"").decode().strip()
        if not out:
            return {"success": False, "error": err or f"empty stdout (rc={r.returncode})"}
        try:
            return json.loads(out.splitlines()[-1])
        except Exception as exc:
            return {
                "success": False,
                "error": f"bad FD JSON: {exc}; stdout={out[:500]!r}; stderr={err[:300]!r}",
            }


def _solve_fd(
    args,
    *,
    domain_template: str,
    problem: str,
    domain_text: str | None = None,
) -> dict:
    """Fast Downward: mock canned, host binary, or Docker (``vlm_ros2``).

    ``--mock-scene`` prefers a host ``fast-downward`` on PATH. If it is missing
    (typical laptop: FD is compiled only in the ROS image), fall back to
    ``docker exec`` like ``run_loop_llm_pddl.DockerFastDownward``. Gazebo is
    not required. ``--mock-fd`` stays the CI canned path.
    """
    if getattr(args, "mock_fd", False):
        from planner.call_timings import fd_span
        from planner.r1.mock_llm import mock_fd_result

        with fd_span():
            return mock_fd_result(args.task, domain_text)
    if getattr(args, "mock_scene", False):
        import shutil

        from planner.domain_store import load_base_domain
        from planner.fast_downward import (
            FastDownwardPlanner,
            FastDownwardTimeout,
            result_from_actions,
            result_from_timeout,
        )

        if shutil.which("fast-downward"):
            text = (domain_text or "").strip() or load_base_domain(domain_template)
            planner = FastDownwardPlanner()
            try:
                actions = planner.solve_from_strings(text, problem)
                return result_from_actions(actions)
            except FileNotFoundError:
                pass
            except FastDownwardTimeout as exc:
                return result_from_timeout(exc.limit_s)
            except Exception as exc:  # noqa: BLE001
                return {
                    "success": False,
                    "error": str(exc),
                    "actions": [],
                    "primitives": [],
                }
        container = getattr(args, "container", None) or "vlm_ros2"
        print(
            f"[LOOP] fast-downward not on host PATH — "
            f"FD via docker exec {container} (no Gazebo)"
        )
        return _fd_solve_in_container(
            args,
            domain_template=domain_template,
            problem=problem,
            domain_text=domain_text,
        )
    return _fd_solve_in_container(
        args,
        domain_template=domain_template,
        problem=problem,
        domain_text=domain_text,
    )


def _fd_primitives_to_vlm_plan(
    command: str,
    domain_template: str,
    primitives: list[dict],
    *,
    domain_additions: dict | None = None,
):
    """
    Convert FD primitives → VLMPlan steps for direct orchestrator execution.

    ``domain_additions`` (Session 30) travels with the plan so ``:init``,
    ``:goal`` and the tracker/verifier hooks all see the same enriched actions.
    """
    from planner.enrichment_effects import (
        EnrichmentContext,
        step_args_for_catalog_action,
    )
    from vlm.planner import PlanStep, VLMPlan

    enrichment = EnrichmentContext.from_domain_additions(domain_additions)
    steps: list = []
    for prim in primitives:
        name = str(prim.get("name") or "")
        args = list(prim.get("args") or [])
        if name == "pick":
            steps.append(
                PlanStep(
                    "pick",
                    {
                        "object": args[0] if args else "",
                        "grasp_mode": "top_down",
                    },
                )
            )
        elif name == "place":
            steps.append(
                PlanStep(
                    "place",
                    {
                        "object": args[0] if args else "",
                        "location": args[1] if len(args) > 1 else "",
                    },
                )
            )
        elif name == "look_at":
            steps.append(
                PlanStep("look_at", {"target": args[0] if args else ""})
            )
        elif name == "navigate_to":
            steps.append(
                PlanStep("navigate_to", {"target": args[0] if args else ""})
            )
        elif enrichment.knows(name):
            steps.append(PlanStep(name, step_args_for_catalog_action(name, args)))
        else:
            # Best-effort: first arg as object/target
            payload = {}
            if args:
                payload["object"] = args[0]
            if len(args) > 1:
                payload["location"] = args[1]
            steps.append(PlanStep(name, payload))
    additions = dict(domain_additions or {})
    return VLMPlan(
        goal=command,
        steps=steps,
        raw_output="[control=fd] Fast Downward plan",
        domain_template=domain_template,
        domain_additions={
            "new_types": list(additions.get("new_types", []) or []),
            "new_predicates": list(additions.get("new_predicates", []) or []),
            "new_actions": list(additions.get("new_actions", []) or []),
            "modified_preconditions": dict(
                additions.get("modified_preconditions", {}) or {}
            ),
        },
    )


def _fd_publish_gazebo_poses(args, gazebo_poses: dict, names: set[str]) -> None:
    """Publish oracle poses (panda_link0) so pick/place have perception cache hits."""
    rbase_x, rbase_y = 0.20, 0.0
    for name in sorted(names):
        gp = gazebo_poses.get(name)
        if not gp:
            continue
        x = float(gp["x"]) - rbase_x
        y = float(gp["y"]) - rbase_y
        z = 0.025
        ok = _publish_perception_pose(args, name, x, y, z)
        if ok:
            print(f"[LOOP] FD pose publish: {name} → ({x:.3f},{y:.3f},{z:.3f})")


def _fd_publish_dino_poses(args, dino_poses: dict, names: set[str]) -> None:
    """Publish perception poses (panda_link0) for FD execution under perception-only."""
    for name in sorted(names):
        pose = dino_poses.get(name)
        if not pose:
            continue
        try:
            x, y, z = float(pose["x"]), float(pose["y"]), float(pose["z"])
        except (KeyError, TypeError, ValueError):
            continue
        ok = _publish_perception_pose(args, name, x, y, z)
        if ok:
            print(f"[LOOP] FD DINO pose publish: {name} → ({x:.3f},{y:.3f},{z:.3f})")


def _overview_image_path() -> Path:
    return _REPO_ROOT / "data" / "scene_overview.png"


def _load_overview_depth():
    path = _REPO_ROOT / "data" / "depth_overview.npy"
    if not path.is_file():
        return None
    try:
        import numpy as np

        return np.load(str(path))
    except Exception:
        return None


def _acquire_inventory_sweep(args, perception, image, *, task: str):
    """VLM names (or --inventory-json) + DINO poses on the overview frame."""
    from planner.image_scene import (
        perceive_image,
        release_cuda,
        sweep_from_perceive,
    )
    from vlm.inventory import parse_inventory

    inventory = None
    inv_path = getattr(args, "inventory_json", None)
    if inv_path:
        path = Path(inv_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        inventory = parse_inventory(path.read_text(encoding="utf-8"))
        print(
            f"[LOOP] inventory from file: objects={list(inventory.object_ids())} "
            f"locations={list(inventory.location_ids())}"
        )
    else:
        from vlm.planner import VLMPlanner

        print("[LOOP] Loading Qwen-VL for inventory names…")
        planner = VLMPlanner()
        planner.load()
        print("[OK]   inventory VLM ready.")
        from vlm.inventory import list_objects_from_image

        inventory = list_objects_from_image(image, planner, task=task)
        print(
            f"[LOOP] VLM inventory objects={list(inventory.object_ids())} "
            f"locations={list(inventory.location_ids())}"
        )
        del planner
        release_cuda()

    scene, inventory, detections = perceive_image(
        image,
        task=task,
        perception=perception,
        inventory=inventory,
        camera_dir=_REPO_ROOT / "data",
        depth_image=_load_overview_depth(),
        skip_dino=perception is None,
        prefer_overview=True,
    )
    sweep = sweep_from_perceive(scene, inventory, detections)
    print(
        f"[LOOP] inventory scene objects={[o.name for o in scene.objects]} "
        f"locations={[loc.name for loc in scene.locations]} "
        f"posed={sorted(sweep.poses_plink0)}"
    )
    return sweep


def _fd_build_hybrid_problem(
    *,
    hybrid_session,
    stub_plan,
    gazebo_poses: dict,
    dino_detections: list,
    last_dino_est: dict,
    problem_name: str,
    pre_scan_ok: bool = False,
    world_name: str = "",
    perception_only: bool = False,
    ready_dino_scene=None,
):
    """Build hybrid PDDL from SceneState + GoalGenerator (no VLM action sketch)."""
    from planner.dino_localisation import (
        poses_for_init_from_estimates,
        plink0_to_world,
    )
    from planner.hybrid_runtime import (
        SceneSource,
        ground_plan_locations,
        partition_gazebo_for_hybrid,
        rewrite_command_locations,
        scene_from_dino_payload,
        scene_from_xyz_poses,
        scene_source_skips_oracle,
    )
    from planner.hybrid_runtime import GAZEBO_LOCATION_MODELS

    _base_locs = list(hybrid_session.known_locations or ["table", "shelf"])
    if gazebo_poses:
        _, _alias_locs = partition_gazebo_for_hybrid(
            gazebo_poses, base_locations=_base_locs
        )
        ground_plan_locations(stub_plan, _alias_locs)
        hybrid_session.known_locations = list(_alias_locs)
        if not hybrid_session._goal_locked:
            _cmd = rewrite_command_locations(
                hybrid_session.command or stub_plan.goal, _alias_locs
            )
            if _cmd != hybrid_session.command:
                print(f"[LOOP] Goal command rewrite: {_cmd!r}")
                hybrid_session.command = _cmd

    _item_poses, _locs = partition_gazebo_for_hybrid(
        gazebo_poses,
        base_locations=hybrid_session.known_locations,
    )
    hybrid_session.known_locations = list(_locs)
    _on_surface_oracle = _infer_on_surface(_item_poses, world_name)
    oracle_scene = None
    if _item_poses:
        oracle_scene = scene_from_xyz_poses(
            _item_poses,
            known_locations=hybrid_session.known_locations,
            on_surface=_on_surface_oracle,
            gripper_empty=not hybrid_session.is_holding,
            holding=hybrid_session.holding_object(),
            domain_template=stub_plan.domain_template,
        )

    # Session 25: real z from depth / height estimate (no hardcoded 0.82).
    dino_poses_plink = poses_for_init_from_estimates(
        last_dino_est,
        exclude=list(hybrid_session.known_locations)
        + list(GAZEBO_LOCATION_MODELS),
        frame="plink0",
    )
    # Support inference needs Gazebo-world frame (same as oracle / SDF surfaces).
    dino_poses_world = {
        n: plink0_to_world(p["x"], p["y"], p["z"])
        for n, p in dino_poses_plink.items()
    }
    # Honest dino / inventory / perception-only: (on …) from perceived z, not oracle map.
    use_dino_surfaces = (
        perception_only
        or scene_source_skips_oracle(hybrid_session.scene_source)
    )
    if use_dino_surfaces and dino_poses_world:
        _on_surface_dino = _infer_on_surface(dino_poses_world, world_name)
    else:
        _on_surface_dino = _on_surface_oracle

    if ready_dino_scene is not None:
        dino_scene = ready_dino_scene
    else:
        dino_scene = scene_from_dino_payload(
            [
                d
                for d in dino_detections
                if (d.get("name") or d.get("label") or "")
                not in hybrid_session.known_locations
                and (d.get("name") or d.get("label") or "")
                not in GAZEBO_LOCATION_MODELS
            ],
            poses=dino_poses_world or None,
            on_surface=_on_surface_dino,
            known_locations=hybrid_session.known_locations,
            domain_template=stub_plan.domain_template,
        )
    if oracle_scene is None and dino_scene is None:
        print("[WARN] FD control: no oracle/DINO scene — cannot build hybrid problem")
        return None, False
    if scene_source_skips_oracle(hybrid_session.scene_source) and dino_scene is None:
        print(
            f"[WARN] FD control: scene_source={hybrid_session.scene_source.value} "
            "but no DINO/inventory scene"
        )
        return None, False
    if hybrid_session.scene_source == SceneSource.ORACLE and oracle_scene is None:
        print("[WARN] FD control: scene_source=oracle but no oracle scene")
        return None, False

    pddl_str, _fused = hybrid_session.generate_hybrid_problem(
        stub_plan,
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        problem_name=problem_name,
    )
    return pddl_str, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Closed-loop task execution")
    parser.add_argument("--task",       required=True)
    parser.add_argument("--max-steps",  type=int, default=10)
    parser.add_argument(
        "--max-replans", type=int, default=3,
        help=(
            "Abort the task once this many replans have been spent. Shared by "
            "control=fd and control=vlm_steps so the two are comparable "
            "(default: 3)."
        ),
    )
    parser.add_argument("--container",  default="vlm_ros2")
    parser.add_argument("--sudo-docker", action="store_true")
    parser.add_argument("--world",      default="office",
                        help="Active Gazebo world (reads overview cam pose from world file)")
    parser.add_argument(
        "--hybrid",
        default=None,
        metavar="MODE",
        help=(
            "Hybrid problem gen: off|mvp|full|1 (overrides VLMRP_HYBRID_PROBLEM_GEN). "
            "Default OFF = legacy infer_init/infer_goal."
        ),
    )
    parser.add_argument(
        "--goal-backend",
        default=None,
        choices=["rule_based", "local_llm"],
        help="Goal backend when hybrid ON (overrides VLMRP_GOAL_BACKEND; default rule_based).",
    )
    parser.add_argument(
        "--scene-source",
        default=None,
        choices=["fused", "dino", "oracle", "inventory"],
        help=(
            "Hybrid :init perception: fused=oracle+DINO (default), "
            "dino=DINO catalog sweep (real-world-like), "
            "inventory=VLM names + DINO localize (open-vocab real robot), "
            "oracle=Gazebo GT only. Overrides VLMRP_SCENE_SOURCE."
        ),
    )
    parser.add_argument(
        "--inventory-json",
        default=None,
        metavar="PATH",
        help=(
            "Skip the inventory VLM: load names from this JSON "
            "(same schema as run_image_hybrid). Used with --scene-source inventory."
        ),
    )
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help=(
            "Session 25: disable SIM NameMatch and oracle snap so DINO poses "
            "stay perception-driven. Default OFF (Session 22 fused behaviour)."
        ),
    )
    parser.add_argument(
        "--control",
        default=None,
        choices=["vlm_steps", "fd"],
        help=(
            "Action policy: vlm_steps (default) = vision VLM proposes steps; "
            "fd = Fast Downward from hybrid :init/:goal (requires --hybrid mvp; "
            "incompatible with --hybrid full). Overrides VLMRP_CONTROL."
        ),
    )
    parser.add_argument(
        "--online-enrichment",
        default=None,
        metavar="MODE",
        help=(
            "Session 28: closed-catalog online domain enrichment "
            "(1|true|on to enable). Default OFF = fixed domains only. "
            "Overrides VLMRP_ONLINE_ENRICHMENT. Incomplete domains hand off "
            "to the Domain Enricher."
        ),
    )
    parser.add_argument(
        "--domain-select",
        default=None,
        choices=["rule_based", "llm"],
        help=(
            "Session 29: who judges domain template + complete/incomplete. "
            "rule_based (default) = keyword heuristic, no GPU. llm = the "
            "single text LLM (semantic; also authors enrichment PDDL). "
            "Overrides VLMRP_DOMAIN_SELECT."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Stop after active perception + hybrid :init/:goal + Fast Downward. "
            "No orchestrator inject and no arm motion. For planning batteries."
        ),
    )
    parser.add_argument(
        "--enrichment-profile",
        default="r0",
        choices=["r0", "r1"],
        help=(
            "Which closed-catalog enricher to use when online enrichment is ON. "
            "r0 (default) = one catalog skill + outcome fluent. "
            "r1 = affordance predicates + second LLM call that assigns them "
            "to objects. Does not change R0 prompts."
        ),
    )
    parser.add_argument(
        "--mock-scene",
        action="store_true",
        help=(
            "Load SceneState from an oracle_mock_v1 fixture (no Gazebo, no DINO). "
            "Default file from --world (tabletop/kitchen) under "
            "tests/fixtures/llm_plan/; override with --scene-file."
        ),
    )
    parser.add_argument(
        "--scene-file",
        default=None,
        metavar="PATH",
        help="oracle_mock_v1 JSON for --mock-scene.",
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help=(
            "Canned generate_fn for CI (select + enrich + goal). "
            "Not the live text LLM."
        ),
    )
    parser.add_argument(
        "--mock-fd",
        action="store_true",
        help=(
            "Canned Fast Downward result for CI (no fast-downward binary, "
            "no Docker). Without this flag, --mock-scene uses the host "
            "binary if on PATH, else docker exec in --container."
        ),
    )
    args = parser.parse_args()
    from planner.call_timings import reset as _reset_timings

    _reset_timings()

    from planner.hybrid_runtime import (
        ControlMode,
        HybridMode,
        IncompatibleControlHybridError,
        SceneSource,
        assert_control_hybrid_compatible,
        completed_action_from_step,
        create_session_from_env,
        ground_plan_locations,
        make_domain_stub_plan,
        partition_gazebo_for_hybrid,
        resolve_control_mode,
        resolve_hybrid_mode,
        resolve_scene_source,
        rewrite_command_locations,
        scene_from_dino_payload,
        scene_from_xyz_poses,
        scene_source_skips_oracle,
        select_domain_template,
    )
    from planner.online_enrichment import (
        DomainCompleteness,
        EnrichmentStatus,
        make_domain_selector,
        make_goal_binder,
        make_online_enricher,
        guard_domain_selection,
        resolve_domain_for_task,
        resolve_online_enrichment,
        resolve_selection_backend,
        template_covers_command,
    )
    from planner.state_verifier import StateVerifier as _LoopStateVerifier

    control_mode = resolve_control_mode(args.control)
    hybrid_mode = resolve_hybrid_mode(args.hybrid)
    try:
        assert_control_hybrid_compatible(control_mode, hybrid_mode)
    except IncompatibleControlHybridError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(2)
    if args.plan_only and control_mode != ControlMode.FD:
        print("[FAIL] --plan-only requires --control fd (hybrid :init/:goal + FD).")
        sys.exit(2)
    if args.mock_scene and control_mode != ControlMode.FD:
        print("[FAIL] --mock-scene requires --control fd.")
        sys.exit(2)
    if args.scene_file and not args.mock_scene:
        print("[FAIL] --scene-file requires --mock-scene.")
        sys.exit(2)
    if resolve_scene_source(args.scene_source) == SceneSource.INVENTORY:
        if control_mode != ControlMode.FD:
            print("[FAIL] --scene-source inventory requires --control fd.")
            sys.exit(2)
        if hybrid_mode == HybridMode.OFF:
            print("[FAIL] --scene-source inventory requires --hybrid mvp.")
            sys.exit(2)

    mock_generate_fn = None
    if args.mock_llm:
        from planner.r1.mock_llm import mock_host_generate_fn

        mock_generate_fn = mock_host_generate_fn
        print("[LOOP] --mock-llm ON — canned generate_fn (no GPU)")

    mock_scene_state = None
    mock_scene_path = None
    if args.mock_scene:
        from planner.problem_generator.init_generator.adapters.mock import (
            OracleMockAdapter,
        )
        from planner.r1.scenes import mock_scene_file_for_world

        scene_path = (
            Path(args.scene_file) if args.scene_file else mock_scene_file_for_world(args.world)
        )
        if not scene_path.is_file():
            print(f"[FAIL] mock scene not found: {scene_path}")
            sys.exit(2)
        mock_scene_path = scene_path
        mock_scene_state = OracleMockAdapter.load(scene_path)
        print(f"[LOOP] --mock-scene {scene_path} ({len(mock_scene_state.objects)} objects)")

    inventory_sweep = None
    perception = None
    inventory_live = (
        resolve_scene_source(args.scene_source) == SceneSource.INVENTORY
        and mock_scene_state is None
    )
    if inventory_live:
        args.perception_only = True

    enrichment_on = resolve_online_enrichment(args.online_enrichment)
    select_backend = resolve_selection_backend(args.domain_select)
    # Session 30 loop state: what the enriched domain (if any) adds, so tracker,
    # verifier, :init, :goal and FD all work from the same actions.
    enrichment_ctx = None
    enriched_domain_text: str | None = None
    enrichment_additions: dict | None = None
    enrichment_skills: tuple[str, ...] = ()
    enrichment_domain_path: str | None = None
    enrichment_reused = False
    refuse_reason: str | None = None
    domain_select_backend = select_backend.value
    domain_completeness: str | None = None
    needed_skills: tuple[str, ...] = ()
    selection_reason = ""
    if mock_scene_state is not None:
        scene_symbols = tuple(o.name for o in mock_scene_state.objects) + tuple(
            loc.name for loc in mock_scene_state.locations
        )
    elif inventory_live:
        print("[LOOP] Loading PerceptionModule (inventory VLM names + DINO)…")
        from vlm.perception import PerceptionModule
        from PIL import Image as PilImage

        perception = PerceptionModule()
        perception.load()
        print("[OK]   PerceptionModule loaded.")
        print("[LOOP] scene_source=inventory — capture overview then VLM names")
        _pre_scan(args)
        time.sleep(1.0)
        cap_path = _capture(args)
        ov_path = _overview_image_path()
        img_path = ov_path if ov_path.is_file() else cap_path
        if img_path is None or not Path(img_path).is_file():
            print("[FAIL] --scene-source inventory needs an overview/wrist capture.")
            sys.exit(2)
        image = PilImage.open(str(img_path)).convert("RGB")
        try:
            inventory_sweep = _acquire_inventory_sweep(
                args, perception, image, task=args.task
            )
        except Exception as exc:
            print(f"[FAIL] inventory perception failed: {exc}")
            sys.exit(2)
        scene_symbols = tuple(o.name for o in inventory_sweep.scene.objects) + tuple(
            loc.name for loc in inventory_sweep.scene.locations
        )
        print(f"[LOOP] inventory symbols: {list(scene_symbols)}")
    else:
        # World models + furniture names the goal helper needs. _get_scene_objects
        # skips shelf_b as furniture, so locations are appended explicitly.
        scene_symbols = tuple(_get_scene_objects(args.world)) + (
            "table",
            "shelf",
            "shelf_b",
            "counter",
            "drawer",
            "tray",
            "target_tray",
            "metal_tray",
        )
    selector = make_domain_selector(select_backend, generate_fn=mock_generate_fn)
    scene_goal_probe = None
    _probe_scene = mock_scene_state or (
        inventory_sweep.scene if inventory_sweep is not None else None
    )
    if _probe_scene is not None:
        _probe_objects = tuple(o.name for o in _probe_scene.objects)
        _probe_locations = tuple(loc.name for loc in _probe_scene.locations)

        def scene_goal_probe(command, template, symbols):
            return template_covers_command(
                command,
                template,
                symbols,
                objects=_probe_objects,
                locations=_probe_locations,
            )

    if args.enrichment_profile == "r1":
        from planner.r1.enricher import R1DomainEnricher

        enricher = R1DomainEnricher(
            generate_fn=mock_generate_fn, model_id=None
        )
        print("[LOOP] enrichment-profile=r1 (affordance enricher)")
    else:
        enricher = make_online_enricher(select_backend, generate_fn=mock_generate_fn)
    if enrichment_on:
        domain_resolution = resolve_domain_for_task(
            args.task,
            enrichment_enabled=True,
            selector=selector,
            enricher=enricher,
            scene_symbols=scene_symbols,
            world=args.world,
            goal_probe=scene_goal_probe,
        )
        domain_template = domain_resolution.template
        sel = domain_resolution.selection
        domain_select_backend = sel.backend
        domain_completeness = sel.completeness.value
        needed_skills = tuple(sel.needed_skills)
        selection_reason = sel.reason
        print(
            f"[LOOP] domain selection: template={sel.template} "
            f"completeness={sel.completeness.value} "
            f"backend={sel.backend} reason={sel.reason}"
        )
        if sel.error:
            print(f"[WARN] domain selection fell back: {sel.error}")
        if domain_resolution.used_enricher and domain_resolution.enrichment is not None:
            enr = domain_resolution.enrichment
            print(
                f"[LOOP] online enrichment: status={enr.status.value} "
                f"notes={enr.notes or '(none)'}"
            )
            if enr.status == EnrichmentStatus.REFUSED:
                refuse_reason = enr.refuse_message or "enrichment refused"
                print(f"[FAIL] {refuse_reason}")
                print(
                    "[FAIL] no catalog skill can close the gap between the "
                    "request and the domain — the robot will not attempt it"
                )
                _write_refusal_summary(
                    args,
                    refuse_reason=refuse_reason,
                    domain_template=domain_template,
                    select_backend=sel.backend,
                    selection_reason=sel.reason,
                    domain_completeness=domain_completeness,
                    needed_skills=needed_skills,
                )
                sys.exit(3)
            if enr.status == EnrichmentStatus.ENRICHED:
                from planner.enrichment_effects import EnrichmentContext

                enriched_domain_text = enr.domain_text
                enrichment_additions = dict(enr.domain_additions or {})
                enrichment_ctx = EnrichmentContext.from_domain_additions(
                    enrichment_additions
                )
                enrichment_skills = tuple(enr.skills_grounded)
                enrichment_domain_path = enr.domain_path
                enrichment_reused = bool(enr.reused)
                print(
                    f"[LOOP] enriched domain persisted: {enr.domain_path} "
                    f"(skills={', '.join(enr.skills_grounded) or 'none'}, "
                    f"reused={enr.reused})"
                )
                print(
                    f"[LOOP] enriched actions live in the loop: "
                    f"{', '.join(enrichment_ctx.action_names) or 'none'} → "
                    f"ROS {', '.join(enr.ros_primitives) or 'none'}"
                )
                if not enrichment_ctx.catalog_only:
                    print(
                        "[WARN] enrichment contains actions with no ROS body — "
                        "ablation only, not the closed-catalog path"
                    )
            if enr.status == EnrichmentStatus.STUB:
                print(
                    "[LOOP] enrichment stub (rule_based backend) — use "
                    "--domain-select llm to author PDDL"
                )
        elif sel.completeness == DomainCompleteness.COMPLETE:
            print("[LOOP] domain complete — enricher not called")
    elif select_backend.value == "llm":
        domain_selector = selector
        selection = domain_selector.select(
            args.task, scene_symbols=scene_symbols, world=args.world
        )
        selection = guard_domain_selection(
            selection,
            args.task,
            scene_symbols,
            selector=domain_selector,
            world=args.world,
            goal_probe=scene_goal_probe,
        )
        domain_template = selection.template
        domain_select_backend = selection.backend
        domain_completeness = selection.completeness.value
        needed_skills = tuple(selection.needed_skills)
        selection_reason = selection.reason
        print(
            f"[LOOP] domain selection: template={selection.template} "
            f"completeness={selection.completeness.value} "
            f"backend={selection.backend} reason={selection.reason}"
        )
    else:
        domain_template = select_domain_template(args.task)

    if mock_scene_state is None and perception is None:
        print("[LOOP] Loading PerceptionModule…")
        from vlm.perception import PerceptionModule
        from PIL import Image as PilImage

        perception = PerceptionModule()
        perception.load()

        vlm = None
        need_action_vlm = control_mode == ControlMode.VLM_STEPS
        # Verifier VLM attached after hybrid session is known (vlm_steps + full only).
        if need_action_vlm:
            print("[LOOP] Loading VLM (Qwen3-VL-8B-Instruct) for step planning…")
            from vlm.planner import VLMPlanner

            vlm = VLMPlanner()
            vlm.load()
            print("[OK]   VLM + PerceptionModule loaded.\n")
        else:
            print(
                "[OK]   PerceptionModule loaded "
                "(control=fd — no VLM action policy; use --hybrid mvp).\n"
            )
    elif mock_scene_state is None:
        vlm = None
        print(
            "[OK]   PerceptionModule already loaded "
            "(scene_source=inventory — VLM names released before text LLM).\n"
        )
    else:
        vlm = None
        print("[OK]   mock-scene — PerceptionModule not loaded\n")

    _known_locs = ["table", "shelf"]
    if mock_scene_state is not None:
        _known_locs = [loc.name for loc in mock_scene_state.locations] or _known_locs
    elif inventory_sweep is not None:
        _known_locs = [
            loc.name for loc in inventory_sweep.scene.locations
        ] or _known_locs
    hybrid_session = create_session_from_env(
        command=args.task,
        domain_template=domain_template,
        hybrid_flag=args.hybrid,
        goal_backend=args.goal_backend,
        scene_source=args.scene_source,
        known_locations=_known_locs,
        local_generate_fn=mock_generate_fn,
    )
    if hybrid_session is not None:
        # Teaches tracker + verifier what the enriched actions do; a run with no
        # enrichment passes None and behaves exactly as before.
        hybrid_session.set_enrichment(
            enrichment_ctx,
            skills=enrichment_skills,
            domain_path=enrichment_domain_path,
            reused=enrichment_reused,
            backend=domain_select_backend,
            goal_binder=(
                make_goal_binder(select_backend, generate_fn=mock_generate_fn)
                if enrichment_ctx
                else None
            ),
            domain_additions=enrichment_additions,
        )
        hybrid_session.refuse_reason = refuse_reason

    if control_mode == ControlMode.FD and hybrid_session is None:
        print(
            "[FAIL] --control fd requires --hybrid mvp "
            "(honest :init/:goal without a VLM action sketch; "
            "--hybrid full is refused with control=fd)."
        )
        sys.exit(2)

    print(
        f"[LOOP] control={control_mode.value}  "
        f"domain_template={domain_template} (selected once)"
    )
    if hybrid_session is not None:
        print(
            f"[LOOP] Hybrid problem gen ON "
            f"(mode={hybrid_session.mode.value}, "
            f"scene_source={hybrid_session.scene_source.value}, "
            f"goal_backend={hybrid_session.goal_backend.value}); "
            "templates: base/stacking/containers/navigation — enrichment domain_additions fall back to legacy"
        )
        if hybrid_session.scene_source == SceneSource.DINO:
            print(
                "[LOOP] scene_source=dino — :init from DINO (+ tracker) only "
                "(no Gazebo oracle in fusion; real-world-like)"
            )
        elif hybrid_session.scene_source == SceneSource.INVENTORY:
            print(
                "[LOOP] scene_source=inventory — :init from VLM names + DINO "
                "(no Gazebo catalog sweep)"
            )
        elif hybrid_session.scene_source == SceneSource.ORACLE:
            print(
                "[LOOP] scene_source=oracle — :init from Gazebo oracle (+ tracker) only"
            )
        if args.perception_only:
            print(
                "[LOOP] --perception-only ON — NameMatch + SIM snap disabled "
                "(genuine DINO metric poses)"
            )
        if args.plan_only:
            print(
                "[LOOP] --plan-only ON — perception + FD plan only "
                "(no inject, no arm motion)"
            )
        if hybrid_session.verifier_enabled:
            from planner.problem_generator.init_generator.vlm_fusion import (
                LiveVlmFusionClient,
            )
            from vlm.planner import VLMPlanner

            # YELLOW patches only (max 1 call per verification). GREEN/RED never
            # invoke it. Under control=fd the action VLM may be unloaded — load
            # a planner instance solely for LiveVlmFusionClient.
            if vlm is None:
                print("[LOOP] Loading VLM for YELLOW verifier patches only…")
                vlm = VLMPlanner()
                vlm.load()
            hybrid_session.vlm_fusion_client = LiveVlmFusionClient.from_planner(vlm)
            print(
                "[LOOP] Phase B full mode: StateVerifier after each step "
                "(YELLOW→live VLM patch ≤1, GREEN/RED→0 VLM calls)"
            )
    else:
        print("[LOOP] Hybrid problem gen OFF (legacy generate_problem)")

    completed_steps: list[str] = []
    docker_cmd = _docker(args.container, args.sudo_docker)

    # Replanning on failure state
    _current_plan       = None   # cached full VLMPlan (remaining steps)
    _last_failed_step   = None   # step that caused last replan
    _replan_count       = 0      # how many times we've replanned
    # Session 17: exit reason for summary.json / mini-eval harness
    _exit_reason        = "max_steps"  # success | abort | fail | max_steps | planned
    _fd_plan_actions: list = []
    _fd_plan_prims: list = []
    _fd_pddl_problem: str | None = None
    _fd_pddl_init: str | None = None
    _fd_pddl_goal: str | None = None
    _fd_pddl_domain: str | None = None
    _fd_init_facts: list = []
    _loop_t0            = time.monotonic()

    import datetime
    _ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _world_tag = getattr(args, "world", "unknown")
    _task_tag  = args.task[:30].replace(" ", "_").replace("/", "-")
    _RUN_DIR   = _REPO_ROOT / "data" / "runs" / f"{_ts}_{_world_tag}_{_task_tag}"
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    _hybrid_flag = (
        None
        if hybrid_session is None
        else hybrid_session.mode.value
    )
    _goal_backend_flag = (
        None
        if hybrid_session is None
        else hybrid_session.goal_backend.value
    )
    with open(str(_RUN_DIR / "run_info.txt"), "w") as _rf:
        _rf.write(f"timestamp: {_ts}\n")
        _rf.write(f"world:     {_world_tag}\n")
        _rf.write(f"task:      {args.task}\n")
        _rf.write(f"control:   {control_mode.value}\n")
        _rf.write(f"domain_template: {domain_template}\n")
        _rf.write(f"hybrid:    {_hybrid_flag or 'off'}\n")
        _rf.write(f"goal_backend: {_goal_backend_flag or 'n/a'}\n")
    print(f"[LOOP] Run dir: {_RUN_DIR.relative_to(_REPO_ROOT)}")

    # Overview camera calibration — computed once from world file (camera is static)
    _OV_K, _OV_CTB = _get_overview_cam_data(args.world)
    if _OV_K is not None:
        print("[OK]   Overview camera calibration: ready (static from SDF)")
    else:
        print("[WARN] Overview camera calibration failed — using wrist cam for VLM")

    # Tracks destinations of placed objects; new DINO detections within
    # EXCL_RADIUS of a recorded position are skipped to prevent re-picking.
    _placed_at: dict[str, tuple[float, float]] = {}
    # last DINO estimate per name: {name: {"x","y","z"}} in panda_link0
    # (Session 25: real z; legacy callers may still see 2-tuples briefly).
    _last_dino_est: dict[str, dict[str, float]] = {}
    _last_dino_detections: list = []
    _last_localisation_report: list = []
    _EXCL_RADIUS = 0.10  # 10cm — objects within this radius are treated as identical
    # Tracks the last dispatch_seq; _wait_step_complete uses min_seq=_last_seq+1
    # to ignore stale TRANSIENT_LOCAL (latched) messages from previous steps.
    _last_seq: int = -1
    if control_mode == ControlMode.FD and (
        mock_scene_state is None or not args.plan_only
    ):
        _last_seq = _sync_step_seq(args, _last_seq)
    # Persists domain enrichments across iterations. The VLM enriches the domain
    # only when it first encounters a novel action; subsequent iterations omit it.
    # generate_problem needs the accumulated definitions to infer the PDDL goal.
    _accumulated_da: dict = {}

    for iteration in range(args.max_steps):
        print(f"\n{'─'*60}")
        print(f"  ITERAZIONE {iteration+1} / {args.max_steps}")
        print(f"  Completati: {completed_steps or ['(nessuno)']}")
        print(f"{'─'*60}")

        if mock_scene_state is not None:
            assert hybrid_session is not None
            from planner.pddl_sections import (
                write_fd_problem_artifacts,
                write_fd_result,
            )

            stub_plan = make_domain_stub_plan(
                args.task,
                domain_template,
                domain_additions=enrichment_additions,
            )
            mock_scene_state.domain_template = domain_template
            pddl_str, _fused = hybrid_session.generate_hybrid_problem(
                stub_plan,
                oracle_scene=mock_scene_state,
                dino_scene=mock_scene_state,
                problem_name=f"loop_fd_iter_{iteration+1}",
            )
            if not pddl_str:
                print("[FAIL] FD control: hybrid problem build failed (mock-scene)")
                _exit_reason = "fail"
                break
            _iter_n = iteration + 1
            _iter_dir = _RUN_DIR / f"iter_{_iter_n:02d}"
            try:
                _pddl_art = write_fd_problem_artifacts(_iter_dir, pddl_str)
            except Exception as _save_err:
                print(f"[WARN] FD problem save failed: {_save_err}")
                _pddl_art = {
                    "pddl_problem": pddl_str,
                    "pddl_init": None,
                    "pddl_goal": None,
                    "init_facts": [],
                }
            _fd_pddl_problem = _pddl_art.get("pddl_problem")
            _fd_pddl_init = _pddl_art.get("pddl_init")
            _fd_pddl_goal = _pddl_art.get("pddl_goal")
            _fd_init_facts = list(_pddl_art.get("init_facts") or [])
            print("\n  PDDL PROBLEM (hybrid, mock-scene):")
            for line in pddl_str.splitlines():
                print(f"    {line}")
            print()
            print("[LOOP] Solving with Fast Downward (mock-scene, no Gazebo)…")
            fd_result = _solve_fd(
                args,
                domain_template=domain_template,
                problem=pddl_str,
                domain_text=enriched_domain_text,
            )
            fd_ok = bool(fd_result.get("success"))
            fd_actions = list(fd_result.get("actions") or [])
            fd_prims = list(fd_result.get("primitives") or [])
            if fd_ok:
                _fd_plan_actions = list(fd_actions)
                _fd_plan_prims = list(fd_prims)
                if fd_actions:
                    print(f"[LOOP] FD plan ({len(fd_actions)} actions):")
                    for i, act in enumerate(fd_actions, 1):
                        print(f"         {i}. {act}")
                else:
                    print("[LOOP] FD plan is empty (goal already holds in :init).")
                print()
            else:
                print(
                    f"[FAIL] Fast Downward: {fd_result.get('error', 'unknown')}"
                )
            try:
                write_fd_result(_iter_dir, fd_result)
                (_iter_dir / "plan_stub.json").write_text(stub_plan.to_json())
                _domain_path = (
                    _REPO_ROOT / "pddl" / "domains" / f"{domain_template}.pddl"
                )
                _fd_domain_text = enriched_domain_text
                if not _fd_domain_text and _domain_path.exists():
                    _fd_domain_text = _domain_path.read_text()
                if _fd_domain_text:
                    _fd_pddl_domain = _fd_domain_text
                    (_iter_dir / "domain.pddl").write_text(_fd_domain_text)
                _dbg = {
                    "iteration": _iter_n,
                    "task": args.task,
                    "control": control_mode.value,
                    "domain_template": domain_template,
                    "mock_scene": True,
                    "pddl_problem": pddl_str,
                    "pddl_init": _fd_pddl_init,
                    "pddl_goal": _fd_pddl_goal,
                    "init_facts": list(_fd_init_facts),
                    "fd_plan": fd_result,
                    "hybrid_problem_gen": hybrid_session.metrics_snapshot(),
                }
                (_iter_dir / "debug.json").write_text(
                    json.dumps(_dbg, indent=2, ensure_ascii=False)
                )
            except Exception as _save_err:
                print(f"[WARN] FD debug save failed: {_save_err}")
            if not fd_ok:
                _exit_reason = "fail"
                break
            if args.plan_only:
                print(
                    f"[LOOP] --mock-scene: skipping injection / execution "
                    f"({len(fd_actions)} FD actions recorded)."
                )
                _exit_reason = "planned"
                break
            from planner.image_scene import load_execution_poses

            _exec_poses = load_execution_poses(
                mock_scene_path or Path("."), mock_scene_state
            )
            _pose_names = set(_exec_poses)
            for prim in fd_prims:
                for a in prim.get("args") or []:
                    if a and a not in {"top_down", "side"}:
                        _pose_names.add(str(a))
            print(
                f"[LOOP] --mock-scene: injecting FD plan "
                f"({len(fd_actions)} actions, {len(_exec_poses)} poses)."
            )
            _fd_publish_dino_poses(args, _exec_poses, _pose_names)
            exec_plan = _fd_primitives_to_vlm_plan(
                args.task,
                domain_template,
                fd_prims,
                domain_additions=enrichment_additions,
            )
            payload = json.dumps({
                "command": args.task,
                "vlm_plan": json.loads(exec_plan.to_json()),
                "pddl_problem": pddl_str,
                "control": "fd",
                "direct": True,
            })
            bash_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros2_ws/install/setup.bash && "
                "python3 /workspace/scripts/_publish_plan.py"
            )
            inject_result = subprocess.run(
                docker_cmd + ["bash", "-c", bash_cmd],
                input=payload.encode(),
                capture_output=True,
            )
            if inject_result.returncode != 0:
                print(
                    f"[FAIL] FD injection failed: "
                    f"{inject_result.stderr.decode().strip()}"
                )
                _exit_reason = "fail"
                break
            print(
                f"[OK]   FD plan injected "
                f"({len(exec_plan.steps)} steps, direct execute)."
            )
            while True:
                print("[LOOP] Attendo completamento step (FD plan)…")
                result = _wait_step_complete(
                    args, timeout=90, min_seq=_last_seq + 1
                )
                if "seq" in result:
                    _last_seq = result["seq"]
                prim_name = result.get("primitive", "unknown")
                step_desc = f"{prim_name}()"
                if result.get("success"):
                    if prim_name != "noop":
                        completed_steps.append(step_desc)
                        print(f"[OK]   FD step completato: {prim_name}")
                    if result.get("task_complete"):
                        print("\n[LOOP] ✅  Task completato (FD plan exhausted).")
                        _exit_reason = "success"
                        break
                else:
                    print(f"[FAIL] FD step failed: {result}")
                    _exit_reason = "fail"
                    break
            break

        # 1. Pre-scan — only when gripper is empty.
        # If holding an object, scan pose movement prevents place from succeeding
        # (MoveIt2 can't plan from scan+held_object to pre-place position).
        if hybrid_session is not None:
            holding = hybrid_session.is_holding
        else:
            last_pick  = max((i for i,s in enumerate(completed_steps) if s.startswith("pick")),  default=-1)
            last_place = max((i for i,s in enumerate(completed_steps) if s.startswith("place") or s.startswith("stack")), default=-1)
            holding = last_pick > last_place

        pre_scan_ok = False
        if not holding:
            pre_scan_ok = _pre_scan(args)
            time.sleep(1.0)
        else:
            print("[LOOP] Holding object — skip scan pose, capture from current arm position")

        # 2. Capture
        image_path = _capture(args)
        if image_path is None:
            print("[FAIL] No image — aborting loop")
            _exit_reason = "fail"
            break
        image = PilImage.open(image_path).convert("RGB")

        # Wrist snapshot saved here temporarily; moved into the iter subfolder at debug-save time
        iter_path = _RUN_DIR / f"iter_{iteration+1:02d}_wrist.png"
        image.save(str(iter_path))
        print(f"[LOOP] Snapshot: {iter_path.name}")

        # Load overview camera image for VLM (fixed reference, better perspective)
        _ov_path = _REPO_ROOT / "data" / "scene_overview.png"
        if _ov_path.exists() and _OV_K is not None:
            image_vlm = PilImage.open(str(_ov_path)).convert("RGB")
        else:
            image_vlm = image   # fallback to wrist cam
        _using_overview = (_ov_path.exists() and _OV_K is not None)

        # Persist last scan-pose image + calibration for place location detection.
        # When arm is holding an object, the camera view is distorted by the arm.
        # Using the last FREE scan gives better geometry for location detection.
        if not holding:
            import shutil
            _data = _REPO_ROOT / "data"
            for fname in ("scene.png", "camera_info.json", "camera_pose.json"):
                src = _data / fname
                if src.exists():
                    shutil.copy2(str(src), str(_data / f"last_scan_{fname}"))
            print("[LOOP] Last scan saved (arm free → used for place location detection)")

        # 3. Get Gazebo models — filter scene infrastructure (never pick/place targets)
        _INFRA = frozenset({
            'floor', 'room', 'ground_plane', 'sun', 'robot_pedestal',
            'overview_camera', 'table', 'workbench',
        })
        gazebo_poses = {k: v for k, v in _get_gazebo_models(args).items()
                        if k not in _INFRA}
        gazebo_models = list(gazebo_poses.keys())
        print(f"[LOOP] Scene objects: {gazebo_models}")

        # ── Session 22: FD control (no VLM action policy) ───────────────────
        if control_mode == ControlMode.FD:
            assert hybrid_session is not None
            stub_plan = make_domain_stub_plan(
                args.task,
                domain_template,
                domain_additions=enrichment_additions,
            )
            print(
                f"[LOOP] FD PLAN (control=fd) domain={domain_template}"
                + (" +enriched" if enriched_domain_text else "")
                + f" task={args.task!r}"
                + (f" after fail={_last_failed_step!r}" if _last_failed_step else "")
            )
            _last_failed_step = None

            # Session 25: scene-level DINO sweep (catalog props, not step args).
            # Inventory: VLM names + DINO (no Gazebo catalog).
            _fd_sweep = None
            _ready_inventory_scene = None
            _need_dino = hybrid_session.scene_source in (
                SceneSource.DINO,
                SceneSource.FUSED,
            )
            if hybrid_session.scene_source == SceneSource.INVENTORY:
                refresh = inventory_sweep is None or (iteration > 0 and not holding)
                if refresh:
                    if image_vlm is None:
                        print("[FAIL] scene_source=inventory but no captured image")
                        _exit_reason = "fail"
                        break
                    print("[LOOP] inventory refresh (VLM names + DINO)")
                    try:
                        inventory_sweep = _acquire_inventory_sweep(
                            args, perception, image_vlm, task=args.task
                        )
                    except Exception as _inv_err:
                        print(f"[FAIL] inventory perception failed: {_inv_err}")
                        _exit_reason = "fail"
                        break
                if inventory_sweep is None:
                    print("[FAIL] scene_source=inventory but no inventory scene")
                    _exit_reason = "fail"
                    break
                _last_dino_detections = list(inventory_sweep.detections)
                _last_dino_est = dict(inventory_sweep.poses_plink0)
                _ready_inventory_scene = inventory_sweep.scene
                hybrid_session.known_locations = [
                    loc.name for loc in inventory_sweep.scene.locations
                ] or hybrid_session.known_locations
                hybrid_session.perception_provenance = {
                    "perception_only": True,
                    "pose_provenance": "vlm_inventory_dino",
                    "shortcuts": {
                        "perception_only": True,
                        "name_match": [],
                        "sim_snap": [],
                        "any_oracle_substitute": False,
                    },
                    "src_label": "overview",
                }
                if _last_dino_detections:
                    try:
                        from vlm.perception import PerceptionModule as _PM

                        _ann = _PM.draw_detections(image_vlm, _last_dino_detections)
                        _dino_path = _RUN_DIR / f"iter_{iteration+1:02d}_dino.png"
                        _ann.save(str(_dino_path))
                        print(f"[LOOP] DINO annotation saved: {_dino_path.name}")
                    except Exception as _ae:
                        print(f"[WARN] DINO annotation failed: {_ae}")
            elif _need_dino and _using_overview and _OV_K is not None and _OV_CTB is not None:
                from planner.dino_localisation import catalog_prop_names
                from planner.hybrid_runtime import GAZEBO_LOCATION_MODELS as _GZ_LOCS

                _held = hybrid_session.holding_object() if holding else None
                _catalog = catalog_prop_names(
                    gazebo_poses=gazebo_poses,
                    world_props=_get_scene_objects(args.world),
                    location_models=_GZ_LOCS,
                    infra=_INFRA,
                    held=_held,
                )
                print(
                    f"[LOOP] FD DINO scene sweep ({len(_catalog)} props, "
                    f"src=overview, perception_only={args.perception_only})"
                )
                try:
                    _fd_sweep = _run_dino_scene_sweep(
                        perception=perception,
                        image=image_vlm,
                        K=_OV_K,
                        ctb=_OV_CTB,
                        src_label="overview",
                        catalog_names=_catalog,
                        gazebo_poses=gazebo_poses,
                        perception_only=args.perception_only,
                        snap_exclude=_GZ_LOCS,
                    )
                    _last_dino_detections = list(_fd_sweep["detections"])
                    _last_dino_est = dict(_fd_sweep["poses_for_init"])
                    _last_localisation_report = list(
                        _fd_sweep["localisation_errors"]
                    )
                    _print_localisation_report(_last_localisation_report)
                    hybrid_session.perception_provenance = {
                        "perception_only": args.perception_only,
                        "shortcuts": _fd_sweep["shortcuts"],
                        "pose_provenance": _fd_sweep["pose_provenance"],
                        "depth_available": _fd_sweep["depth_available"],
                        "src_label": _fd_sweep["src_label"],
                    }
                    hybrid_session.last_localisation_report = (
                        _last_localisation_report
                    )
                    if _last_dino_detections:
                        try:
                            from vlm.perception import PerceptionModule as _PM

                            _ann = _PM.draw_detections(
                                image_vlm, _last_dino_detections
                            )
                            _dino_path = (
                                _RUN_DIR / f"iter_{iteration+1:02d}_dino.png"
                            )
                            _ann.save(str(_dino_path))
                            print(
                                f"[LOOP] DINO annotation saved: {_dino_path.name}"
                            )
                        except Exception as _ae:
                            print(f"[WARN] DINO annotation failed: {_ae}")
                except Exception as _sweep_err:
                    print(f"[WARN] FD DINO sweep failed: {_sweep_err}")
                    if hybrid_session.scene_source == SceneSource.DINO:
                        print(
                            "[FAIL] scene_source=dino requires a successful "
                            "DINO sweep"
                        )
                        _exit_reason = "fail"
                        break
            elif _need_dino:
                print(
                    "[WARN] FD DINO sweep skipped — overview camera unavailable"
                )
                if hybrid_session.scene_source == SceneSource.DINO:
                    print(
                        "[FAIL] scene_source=dino but no overview camera / image"
                    )
                    _exit_reason = "fail"
                    break

            pddl_str, ok_pddl = _fd_build_hybrid_problem(
                hybrid_session=hybrid_session,
                stub_plan=stub_plan,
                gazebo_poses=(
                    {}
                    if hybrid_session.scene_source == SceneSource.INVENTORY
                    else gazebo_poses
                ),
                dino_detections=_last_dino_detections,
                last_dino_est=_last_dino_est,
                problem_name=f"loop_fd_iter_{iteration+1}",
                pre_scan_ok=pre_scan_ok,
                world_name=args.world,
                perception_only=args.perception_only,
                ready_dino_scene=_ready_inventory_scene,
            )
            if not ok_pddl or not pddl_str:
                print("[FAIL] FD control: hybrid problem build failed")
                _exit_reason = "fail"
                break

            from planner.pddl_sections import (
                write_fd_problem_artifacts,
                write_fd_result,
            )

            _iter_n = iteration + 1
            _iter_dir = _RUN_DIR / f"iter_{_iter_n:02d}"
            # Persist :init/:goal before FD so a planner fail still leaves
            # an inspectable problem (stdout dump is not an artefact).
            try:
                _pddl_art = write_fd_problem_artifacts(_iter_dir, pddl_str)
            except Exception as _save_err:
                print(f"[WARN] FD problem save failed: {_save_err}")
                _pddl_art = {
                    "pddl_problem": pddl_str,
                    "pddl_init": None,
                    "pddl_goal": None,
                    "init_facts": [],
                }
            _fd_pddl_problem = _pddl_art.get("pddl_problem")
            _fd_pddl_init = _pddl_art.get("pddl_init")
            _fd_pddl_goal = _pddl_art.get("pddl_goal")
            _fd_init_facts = list(_pddl_art.get("init_facts") or [])

            print("\n  PDDL PROBLEM (hybrid, control=fd):")
            for line in pddl_str.splitlines():
                print(f"    {line}")
            print()

            # Solve with FD inside Docker and print the plan BEFORE execution.
            print("[LOOP] Solving with Fast Downward (preview, no motion yet)…")
            fd_result = _solve_fd(
                args,
                domain_template=domain_template,
                problem=pddl_str,
                domain_text=enriched_domain_text,
            )
            fd_ok = bool(fd_result.get("success"))
            fd_actions = list(fd_result.get("actions") or [])
            fd_prims = list(fd_result.get("primitives") or [])
            if fd_ok:
                _fd_plan_actions = list(fd_actions)
                _fd_plan_prims = list(fd_prims)
                if fd_actions:
                    print(f"[LOOP] FD plan ({len(fd_actions)} actions):")
                    for i, act in enumerate(fd_actions, 1):
                        print(f"         {i}. {act}")
                else:
                    print(
                        "[LOOP] FD plan is empty "
                        "(goal already holds in :init)."
                    )
                print()
            else:
                print(
                    f"[FAIL] Fast Downward: {fd_result.get('error', 'unknown')}"
                )

            try:
                write_fd_result(_iter_dir, fd_result)
                (_iter_dir / "plan_stub.json").write_text(stub_plan.to_json())
                if _last_localisation_report:
                    (_iter_dir / "localisation_errors.json").write_text(
                        json.dumps(
                            _last_localisation_report,
                            indent=2,
                            ensure_ascii=False,
                        )
                    )
                _domain_path = (
                    _REPO_ROOT / "pddl" / "domains" / f"{domain_template}.pddl"
                )
                _fd_domain_text = enriched_domain_text
                if not _fd_domain_text and _domain_path.exists():
                    _fd_domain_text = _domain_path.read_text()
                if _fd_domain_text:
                    _fd_pddl_domain = _fd_domain_text
                    (_iter_dir / "domain.pddl").write_text(_fd_domain_text)
                if _domain_path.exists() and not (
                    _iter_dir / f"domain_{domain_template}.pddl"
                ).exists():
                    (_iter_dir / f"domain_{domain_template}.pddl").write_text(
                        _domain_path.read_text()
                    )
                _dbg = {
                    "iteration": _iter_n,
                    "task": args.task,
                    "control": control_mode.value,
                    "domain_template": domain_template,
                    "perception_only": args.perception_only,
                    "pddl_problem": pddl_str,
                    "pddl_init": _fd_pddl_init,
                    "pddl_goal": _fd_pddl_goal,
                    "init_facts": list(_fd_init_facts),
                    "fd_plan": fd_result,
                    "vlm_action_policy": False,
                    "hybrid_problem_gen": hybrid_session.metrics_snapshot(),
                    "scene_compare": hybrid_session.last_scene_compare,
                    "localisation_errors": _last_localisation_report,
                    "dino_sweep": (
                        None
                        if _fd_sweep is None
                        else {
                            "pose_provenance": _fd_sweep.get("pose_provenance"),
                            "shortcuts": _fd_sweep.get("shortcuts"),
                            "depth_available": _fd_sweep.get("depth_available"),
                            "n_detections": len(_fd_sweep.get("detections") or []),
                            "poses_for_init": _fd_sweep.get("poses_for_init"),
                        }
                    ),
                }
                (_iter_dir / "debug.json").write_text(
                    json.dumps(_dbg, indent=2, ensure_ascii=False)
                )
            except Exception as _save_err:
                print(f"[WARN] FD debug save failed: {_save_err}")

            if not fd_ok:
                _exit_reason = "fail"
                break

            if args.plan_only:
                print(
                    "[LOOP] --plan-only: skipping injection / execution "
                    f"({len(fd_actions)} FD actions recorded)."
                )
                _exit_reason = "planned"
                break

            # Publish poses for symbols in the plan (pick/place need them).
            _pose_names: set[str] = set()
            for prim in fd_prims:
                for a in prim.get("args") or []:
                    if a and a not in {"top_down", "side"}:
                        _pose_names.add(str(a))
            if hybrid_session.scene_source == SceneSource.INVENTORY:
                _pose_names |= set(_last_dino_est)
            if (
                args.perception_only or hybrid_session.scene_source == SceneSource.INVENTORY
            ) and _last_dino_est:
                _fd_publish_dino_poses(args, _last_dino_est, _pose_names)
            else:
                _fd_publish_gazebo_poses(args, gazebo_poses, _pose_names)

            # Ignore latched step_complete from previous runs/iters.
            _last_seq = _sync_step_seq(args, _last_seq)

            # Execute the printed plan directly (no second opaque FD inside
            # orchestrator) — pddl_problem kept for logs / future non-direct path.
            exec_plan = _fd_primitives_to_vlm_plan(
                args.task,
                domain_template,
                fd_prims,
                domain_additions=enrichment_additions,
            )
            payload = json.dumps({
                "command": args.task,
                "vlm_plan": json.loads(exec_plan.to_json()),
                "pddl_problem": pddl_str,
                "control": "fd",
                "direct": True,
            })
            bash_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "source /workspace/ros2_ws/install/setup.bash && "
                "python3 /workspace/scripts/_publish_plan.py"
            )
            inject_result = subprocess.run(
                docker_cmd + ["bash", "-c", bash_cmd],
                input=payload.encode(),
                capture_output=True,
            )
            if inject_result.returncode != 0:
                print(
                    f"[FAIL] FD injection failed: "
                    f"{inject_result.stderr.decode().strip()}"
                )
                _exit_reason = "fail"
                break
            print(
                f"[OK]   FD plan injected "
                f"({len(exec_plan.steps)} steps, direct execute)."
            )

            # Wait for every primitive until task_complete or failure.
            fd_failed = False
            while True:
                print("[LOOP] Attendo completamento step (FD plan)…")
                result = _wait_step_complete(
                    args, timeout=90, min_seq=_last_seq + 1
                )
                if "seq" in result:
                    _last_seq = result["seq"]
                prim_name = result.get("primitive", "unknown")
                step_desc = f"{prim_name}()"
                if result.get("success"):
                    if prim_name != "noop":
                        completed_steps.append(step_desc)
                        print(f"[OK]   FD step completato: {prim_name}")
                    if result.get("task_complete"):
                        print("\n[LOOP] ✅  Task completato (FD plan exhausted).")
                        _exit_reason = "success"
                        break
                else:
                    print(f"[FAIL] FD step failed: {result}")
                    fd_failed = True
                    _replan_count += 1
                    _last_failed_step = step_desc
                    # Fresh SceneState on replan — drop possibly-stale holding.
                    hybrid_session.tracker.reset()
                    hybrid_session.invalidate_goal()
                    if _replan_count >= args.max_replans:
                        print(
                            f"[LOOP] ❌ Troppi replan ({_replan_count}) — "
                            "task abortito"
                        )
                        _exit_reason = "abort"
                    break

            if _exit_reason in {"success", "abort", "fail"}:
                break
            if fd_failed:
                print(
                    f"[LOOP] ⚠️  FD replan #{_replan_count} — "
                    "refresh SceneState e ripianifica"
                )
                continue
            break  # success already handled

        # 4. VLM: plan next single step (measure inference time)
        # Strip arrow notation from completed_steps before passing to VLM
        # so it doesn't echo back "cube->red_cup" and cause double-arrows.
        vlm_context = [s.split("->")[-1].rstrip(")") + ")" if "->" in s else s
                       for s in completed_steps
                       if not s.startswith("skip_")]

        # Annotate image for VLM with already-handled objects.
        # Use overview camera image (fixed reference) for stable annotations.
        # Wrist cam (image) continues to be used for DINO localization.
        _data_dir_annot = str(_REPO_ROOT / "data")
        if _using_overview and _OV_CTB is not None:
            # Annotate on overview image using overview cam_to_base (fixed reference)
            import json as _jjson
            _ov_info = {"K": _OV_K.tolist()}
            _ov_pose = {"cam_to_base": _OV_CTB.tolist()}
            _tmp_info = _REPO_ROOT / "data" / "_tmp_ov_info.json"
            _tmp_pose = _REPO_ROOT / "data" / "_tmp_ov_pose.json"
            with open(str(_tmp_info), "w") as _f: _jjson.dump(_ov_info, _f)
            with open(str(_tmp_pose), "w") as _f: _jjson.dump(_ov_pose, _f)
            image_for_vlm = _annotate_handled_objects(
                image_vlm, _placed_at, str(_REPO_ROOT / "data"),
                info_file="_tmp_ov_info.json", pose_file="_tmp_ov_pose.json")
        else:
            image_for_vlm = _annotate_handled_objects(image, _placed_at, _data_dir_annot)

        if _placed_at:
            annot_path = _RUN_DIR / f"iter_{iteration+1:02d}_annotated.png"
            image_for_vlm.save(str(annot_path))
            src = "overview" if _using_overview else "wrist"
            print(f"[LOOP] Annotated image [{src}]: {annot_path.name} "
                  f"({len(_placed_at)} marker(s): {list(_placed_at.keys())})")

        # Each iteration: VLM receives the current image and generates the full remaining plan.
        # If the scene state changed unexpectedly, the plan will differ from the previous iteration.
        t_vlm = time.time()
        action_label = "REPLAN" if _last_failed_step else ("PLAN" if not vlm_context else "VERIFY+PLAN")
        print(f"[LOOP] VLM {action_label} (piano completo rimanente) per: '{args.task}'")

        _prev_plan_steps = [f"{s.primitive}({s.args})" for s in (_current_plan.steps if _current_plan else [])]
        _prev_current_plan = _current_plan   # saved to inherit grasp_mode if VLM drops it

        # Pass both overview (annotated) + wrist camera to the VLM.
        # overview → global scene state with handled-object markers
        # wrist    → close-up of current arm position / grip
        _vlm_images = [image_for_vlm]
        if image is not None and image is not image_for_vlm:
            _vlm_images.append(image)

        assert vlm is not None  # control=vlm_steps always loads the action VLM
        _current_plan = vlm.plan_remaining(
            args.task, _vlm_images, vlm_context,
            failed_step=_last_failed_step,
            prior_enrichment=_accumulated_da if _accumulated_da else None,
        )
        _last_failed_step = None
        vlm_time = time.time() - t_vlm

        # Preserve grasp_mode from previous plan if VLM omitted it during replanning.
        # The VLM sometimes drops grasp_mode=side when updating the plan because it
        # doesn't remember it specified it earlier. We inherit it so the correct
        # physical grasp is preserved across iterations.
        if _prev_current_plan and _current_plan.steps:
            _prev_picks_by_obj = {
                s.args.get("object", ""): s
                for s in _prev_current_plan.steps
                if s.primitive == "pick" and s.args.get("object")
            }
            for _s in _current_plan.steps:
                if _s.primitive == "pick" and "grasp_mode" not in _s.args:
                    _obj = _s.args.get("object", "")
                    _prev_pick = _prev_picks_by_obj.get(_obj)
                    if _prev_pick and "grasp_mode" in _prev_pick.args:
                        _s.args = dict(_s.args)
                        _s.args["grasp_mode"] = _prev_pick.args["grasp_mode"]
                        print(f"[LOOP] Inherited grasp_mode='{_s.args['grasp_mode']}' "
                              f"for pick('{_obj}') from previous plan")
        print(f"[LOOP] VLM inference    : {vlm_time:.1f}s")

        if _current_plan.steps:
            _new_steps = [f"{s.primitive}({s.args})" for s in _current_plan.steps]
            # Detect if VLM changed the plan (state verification detected a change)
            if _prev_plan_steps and _new_steps != _prev_plan_steps:
                print(f"[LOOP] ⚡ Piano AGGIORNATO dalla VLM (stato cambiato):")
            else:
                print(f"[LOOP] Piano confermato ({len(_current_plan.steps)} passi rimanenti):")
            for _i, _s in enumerate(_current_plan.steps, 1):
                _args_str = ", ".join(f"{k}={v}" for k, v in _s.args.items())
                print(f"         {_i}. {_s.primitive}({_args_str})")

        # Extract only the NEXT step for execution this iteration
        from copy import deepcopy as _dc
        if _current_plan.steps:
            plan = _dc(_current_plan)
            plan.steps = [_current_plan.steps[0]]
        else:
            plan = _current_plan   # complete=True

        # ── VLM plan summary ──────────────────────────────────────────────
        print(f"[LOOP] Domain template  : {plan.domain_template}")

        # Show domain enrichment if the VLM added anything beyond the base template
        da = plan.domain_additions
        enriched = (da.get("new_predicates") or da.get("new_actions") or
                    da.get("new_types") or da.get("modified_preconditions"))
        if enriched:
            # Persist enrichment: merge new_actions/predicates into accumulator so
            # subsequent iterations can use them even when VLM says "no enrichment".
            for key in ("new_types", "new_predicates", "new_actions", "modified_preconditions"):
                if da.get(key):
                    existing = _accumulated_da.get(key, [])
                    existing_names = {
                        a.get("name") for a in existing
                        if isinstance(a, dict) and "name" in a
                    }
                    for item in da[key]:
                        name = item.get("name") if isinstance(item, dict) else None
                        if name not in existing_names:
                            existing.append(item)
                    _accumulated_da[key] = existing
            print(f"[LOOP] ⚡ DOMAIN ENRICHMENT:")
            if da.get("new_types"):
                print(f"         new_types      : {da['new_types']}")
            if da.get("new_predicates"):
                print(f"         new_predicates : {da['new_predicates']}")
            if da.get("new_actions"):
                for a in da["new_actions"]:
                    print(f"         new_action     : {a.get('name')} "
                          f"({a.get('parameters','')}) "
                          f"pre={a.get('precondition','')} "
                          f"eff={a.get('effect','')}")
            if da.get("modified_preconditions"):
                print(f"         mod_precond    : {da['modified_preconditions']}")
        else:
            print(f"[LOOP] Domain enrichment: none (base template sufficient)")

        if not plan.steps:
            # Save final state image (no bboxes — task is complete)
            try:
                final_path = _RUN_DIR / f"loop_iter_{iteration+1:02d}.png"
                image.save(str(final_path))
                print(f"[LOOP] Snapshot finale: {final_path.name}")
            except Exception:
                pass
            print("\n[LOOP] ✅  Task completato secondo VLM!")
            _exit_reason = "success"
            break

        step0 = plan.steps[0]
        print(f"[LOOP] Prossimo step: {step0.primitive}({step0.args})")

        # Prevent phantom pick: skip pick if already holding an object
        if step0.primitive == "pick":
            if hybrid_session is not None:
                already_holding = hybrid_session.is_holding
            else:
                last_pick  = max((i for i, s in enumerate(completed_steps) if s.startswith("pick")),  default=-1)
                last_place = max((i for i, s in enumerate(completed_steps) if s.startswith("place") or s.startswith("stack")), default=-1)
                already_holding = last_pick > last_place
            if already_holding:
                print(f"[WARN] Phantom pick detected (already holding) — skipping")
                completed_steps.append(f"skip_pick({step0.args.get('object','?')})")
                continue

        # Prevent phantom place: skip place if the gripper should be empty
        # (no pick in completed_steps since last place/gripper_open)
        if step0.primitive == "place":
            if hybrid_session is not None:
                gripper_empty = not hybrid_session.is_holding
            else:
                last_pick = max(
                    (i for i, s in enumerate(completed_steps) if s.startswith("pick")),
                    default=-1
                )
                last_place = max(
                    (i for i, s in enumerate(completed_steps) if s.startswith("place")),
                    default=-1
                )
                gripper_empty = last_pick < last_place
            if gripper_empty:
                # Count consecutive skip_place to detect stuck loop
                consecutive_skips = sum(
                    1 for s in reversed(completed_steps)
                    if s.startswith("skip_place")
                    ) if completed_steps else 0
                if consecutive_skips >= 2:
                    print(f"[LOOP] ✅ {consecutive_skips} phantom places consecutivi → "
                          "task considerato completato (oggetto già depositato)")
                    _exit_reason = "success"
                    break
                print(f"[WARN] Phantom place detected (no pick since last place) — skipping")
                completed_steps.append(f"skip_place({step0.args.get('object','?')})")
                continue

        # Phase 2: VLM object names are passed directly to DINO as queries.
        # ground_names() was a Phase 1 step for oracle name matching and is no longer called.
        from copy import deepcopy
        plan_grounded = deepcopy(plan)

        # 5c. DINO pose estimation — primary source is the overview camera.
        # The overview D435i operates within its optimal depth range (0.8-1.5 m)
        # with stable extrinsic calibration and a full view of the workspace.
        # The wrist camera is not used as the DINO source: its ~0.3 m depth is
        # borderline for D435i and hand-eye calibration is less reliable.
        # Falls back to wrist camera if overview is unavailable.
        _data_dir = str(_REPO_ROOT / "data")
        _dino_detections: list = []
        step0 = plan_grounded.steps[0] if plan_grounded.steps else None
        if step0:
            try:
                import numpy as _np
                from vlm.perception import PerceptionModule

                # Select primary camera source for DINO
                if _using_overview and _OV_K is not None and _OV_CTB is not None:
                    det_img_all   = image_vlm   # scene_overview.png — full workspace view
                    det_K_all     = _OV_K
                    det_ctb_all   = _OV_CTB
                    src_label_all = "overview"
                else:
                    # Fallback: wrist camera
                    _cam = PerceptionModule.load_camera_data(_data_dir)
                    if _cam:
                        det_img_all, det_K_all, det_ctb_all = image, _cam[0], _cam[1]
                        src_label_all = "wrist"
                    else:
                        det_img_all = det_K_all = det_ctb_all = None
                        src_label_all = "none"

                # Collect object names referenced in the current step
                names_to_estimate = {}
                for _key in ("target", "object", "location", "container"):
                    _n = step0.args.get(_key, "")
                    if _n and _n not in _INFRA and _n not in names_to_estimate:
                        names_to_estimate[_n] = _key

                # If currently holding an object, skip DINO for it regardless of step.
                # The held object is inside the gripper and not visible in the overview.
                if holding:
                    if hybrid_session is not None:
                        _held = hybrid_session.holding_object()
                    else:
                        _held = next(
                            (s[5:].split(",")[0].rstrip(")")
                             for s in reversed(completed_steps)
                             if s.startswith("pick(")),
                            None,
                        )
                    if _held and _held in names_to_estimate:
                        names_to_estimate.pop(_held)
                        print(f"[LOOP] Holding '{_held}' — skip DINO (in gripper, not visible)")

                _dino_detections = []
                _vlm_name_matches: list[str] = []
                _vlm_sim_snaps: list[str] = []
                for name, name_key in names_to_estimate.items():
                    # SIM-ONLY: fuzzy name match against Gazebo model names.
                    # Disabled under --perception-only (Session 25).
                    # On the real robot gazebo_poses is empty → this block never executes.
                    _gz_name_match = None
                    if (
                        not args.perception_only
                        and gazebo_poses
                        and name not in gazebo_poses
                    ):
                        from planner.dino_localisation import (
                            fuzzy_gazebo_name_match,
                            world_to_plink0,
                        )

                        _gz_name_match = fuzzy_gazebo_name_match(name, gazebo_poses)
                        if _gz_name_match is not None:
                            _gp = gazebo_poses[_gz_name_match]
                            _resolved_pose = world_to_plink0(
                                float(_gp["x"]),
                                float(_gp["y"]),
                                float(_gp.get("z", 0.795)),
                            )
                            _resolved_pose = {
                                "x": _resolved_pose["x"],
                                "y": _resolved_pose["y"],
                                "z": 0.025,
                            }
                            print(
                                f"[LOOP] NameMatch: '{name}' → '{_gz_name_match}' "
                                "(Gazebo pose, no DINO needed)"
                            )
                            _vlm_name_matches.append(name)
                            _last_dino_est[name] = dict(_resolved_pose)
                            _last_dino_est[_gz_name_match] = dict(_resolved_pose)
                            _publish_perception_pose(
                                args,
                                _gz_name_match,
                                _resolved_pose["x"],
                                _resolved_pose["y"],
                                _resolved_pose["z"],
                            )
                            step0.args = dict(step0.args)
                            step0.args[name_key] = _gz_name_match
                            continue  # skip DINO for this name

                    if det_img_all is None or det_K_all is None:
                        print(f"[LOOP] No camera for '{name}' — skip")
                        continue

                    # Load depth array for real-robot depth-based unprojection.
                    _depth_arr = _load_depth_array(src_label_all)

                    pose_est = perception.get_pose(
                        name, det_img_all, det_K_all, det_ctb_all,
                        vlm_description=name.replace("_", " "),
                        depth_image=_depth_arr,
                    )
                    if perception._last_detection:
                        _dino_detections.append(perception._last_detection.copy())
                    if pose_est:
                        from planner.dino_localisation import (
                            nearest_gazebo_snap,
                            refine_z_with_height,
                        )
                        from planner.hybrid_runtime import (
                            GAZEBO_LOCATION_MODELS as _GZ_LOCS_VLM,
                        )

                        _height_m = _estimate_object_height(
                            perception._last_detection,
                            (pose_est["x"], pose_est["y"], pose_est["z"]),
                            det_K_all,
                            det_ctb_all,
                        )
                        _used_depth = bool(
                            getattr(perception, "_last_used_depth", False)
                        )
                        _z_ref = refine_z_with_height(
                            float(pose_est["z"]),
                            _height_m,
                            used_depth=_used_depth,
                        )
                        print(
                            f"[LOOP] DINO [{src_label_all}]: '{name}' → "
                            f"({pose_est['x']:.3f},{pose_est['y']:.3f},{_z_ref:.3f})"
                            + (" depth" if _used_depth else "")
                        )

                        # SIM-ONLY snap — disabled under --perception-only.
                        # Exclude location models so a cup is not snapped to shelf_b.
                        _pub_x = float(pose_est["x"])
                        _pub_y = float(pose_est["y"])
                        _pub_z = float(_z_ref)
                        _gz_resolved = None
                        if not args.perception_only and gazebo_poses:
                            _snap = nearest_gazebo_snap(
                                _pub_x,
                                _pub_y,
                                gazebo_poses,
                                exclude=_GZ_LOCS_VLM,
                            )
                            if _snap is not None:
                                _gz_resolved, _best_d, _snapped = _snap
                                print(
                                    f"[LOOP] SIM snap: "
                                    f"DINO({pose_est['x']:.3f},{pose_est['y']:.3f},"
                                    f"{_z_ref:.3f})"
                                    f" → oracle '{_gz_resolved}' "
                                    f"({_snapped['x']:.3f},{_snapped['y']:.3f},"
                                    f"{_snapped['z']:.3f}) "
                                    f"Δxy={_best_d*100:.1f}cm"
                                )
                                _vlm_sim_snaps.append(name)
                                _pub_x = _snapped["x"]
                                _pub_y = _snapped["y"]
                                _pub_z = _snapped["z"]

                        _last_dino_est[name] = {
                            "x": _pub_x, "y": _pub_y, "z": _pub_z
                        }
                        if _height_m is not None:
                            print(f"[LOOP] height est: {_height_m*100:.1f} cm")
                        _publish_perception_pose(
                            args, name, _pub_x, _pub_y, _pub_z, height_m=_height_m
                        )
                        if _gz_resolved and _gz_resolved != name:
                            _last_dino_est[_gz_resolved] = {
                                "x": _pub_x, "y": _pub_y, "z": _pub_z
                            }
                            _publish_perception_pose(
                                args, _gz_resolved, _pub_x, _pub_y, _pub_z,
                                height_m=_height_m,
                            )
                            step0.args = dict(step0.args)
                            step0.args[name_key] = _gz_resolved
                    else:
                        print(
                            f"[LOOP] DINO [{src_label_all}]: '{name}' "
                            "non rilevato — oracle fallback"
                        )

                if hybrid_session is not None:
                    from planner.dino_localisation import (
                        pose_provenance_label,
                        summarize_shortcuts,
                    )

                    hybrid_session.perception_provenance = {
                        "perception_only": args.perception_only,
                        "shortcuts": summarize_shortcuts(
                            perception_only=args.perception_only,
                            name_match=_vlm_name_matches,
                            sim_snap=_vlm_sim_snaps,
                        ),
                        "pose_provenance": pose_provenance_label(
                            perception_only=args.perception_only,
                            name_match=_vlm_name_matches,
                            sim_snap=_vlm_sim_snaps,
                            had_dino=bool(_dino_detections or _last_dino_est),
                        ),
                    }

                # Save DINO bounding-box overlay for this iteration
                if _dino_detections and det_img_all is not None:
                    try:
                        from vlm.perception import PerceptionModule as _PM
                        _ann = _PM.draw_detections(det_img_all, _dino_detections)
                        _dino_path = _RUN_DIR / f"iter_{iteration+1:02d}_dino.png"
                        _ann.save(str(_dino_path))
                        print(f"[LOOP] DINO annotation saved: {_dino_path.name}")
                    except Exception as _ae:
                        print(f"[WARN] DINO annotation failed: {_ae}")

            except Exception as _pe:
                print(f"[WARN] pre-step perception failed: {_pe}")

        # Restore accumulated enrichment if current plan has none.
        # VLM correctly omits enrichment for repeat iterations, but generate_problem
        # needs the action definitions (e.g. pour effects) to infer the PDDL goal.
        _cur_da = plan_grounded.domain_additions
        _cur_enriched = (
            _cur_da.get("new_predicates") or _cur_da.get("new_actions") or
            _cur_da.get("new_types") or _cur_da.get("modified_preconditions")
        )
        if not _cur_enriched and _accumulated_da:
            plan_grounded.domain_additions = _accumulated_da

        # Logical location grounding (shelf → shelf_b, …) before PDDL / inject.
        if gazebo_poses:
            _base_locs = (
                hybrid_session.known_locations
                if hybrid_session is not None
                else ["table", "shelf"]
            )
            _, _alias_locs = partition_gazebo_for_hybrid(
                gazebo_poses, base_locations=_base_locs
            )
            _alias_notes = ground_plan_locations(plan_grounded, _alias_locs)
            if _alias_notes:
                print(f"[LOOP] Location alias: {', '.join(_alias_notes)}")
            if hybrid_session is not None:
                hybrid_session.known_locations = list(_alias_locs)
                if not hybrid_session._goal_locked:
                    _cmd = rewrite_command_locations(
                        hybrid_session.command or args.task, _alias_locs
                    )
                    if _cmd != hybrid_session.command:
                        print(f"[LOOP] Goal command rewrite: {_cmd!r}")
                        hybrid_session.command = _cmd

        # Generate PDDL + save comprehensive debug info for this iteration
        pddl_str = ""
        used_hybrid = False
        try:
            from planner.problem_generator import generate_problem
            if hybrid_session is not None and hybrid_session.should_use_hybrid(plan_grounded):
                # Partition Gazebo models: shelf_b etc. are locations, not items.
                _item_poses, _locs = partition_gazebo_for_hybrid(
                    gazebo_poses,
                    base_locations=hybrid_session.known_locations,
                )
                hybrid_session.known_locations = list(_locs)
                _on_surface = _infer_on_surface(_item_poses, args.world)
                oracle_scene = None
                if _item_poses:
                    oracle_scene = scene_from_xyz_poses(
                        _item_poses,
                        known_locations=hybrid_session.known_locations,
                        on_surface=_on_surface,
                        gripper_empty=not hybrid_session.is_holding,
                        holding=hybrid_session.holding_object(),
                        domain_template=plan_grounded.domain_template,
                    )
                from planner.dino_localisation import (
                    plink0_to_world,
                    poses_for_init_from_estimates,
                )

                dino_poses_plink = poses_for_init_from_estimates(
                    _last_dino_est,
                    exclude=hybrid_session.known_locations,
                    frame="plink0",
                )
                dino_poses = {
                    n: plink0_to_world(p["x"], p["y"], p["z"])
                    for n, p in dino_poses_plink.items()
                }
                # Honest dino / perception-only: infer (on …) from perceived z.
                if (
                    args.perception_only
                    or hybrid_session.scene_source == SceneSource.DINO
                ) and dino_poses:
                    _on_surface_dino = _infer_on_surface(dino_poses, args.world)
                else:
                    _on_surface_dino = _on_surface
                dino_scene = scene_from_dino_payload(
                    [
                        d
                        for d in _dino_detections
                        if (d.get("name") or d.get("label") or "")
                        not in hybrid_session.known_locations
                    ],
                    poses=dino_poses or None,
                    on_surface=_on_surface_dino,
                    known_locations=hybrid_session.known_locations,
                    domain_template=plan_grounded.domain_template,
                )
                if oracle_scene is None and dino_scene is None:
                    print("[WARN] Hybrid ON but no oracle/DINO scene — falling back to legacy PDDL")
                    pddl_str = generate_problem(plan_grounded)
                elif (
                    hybrid_session.scene_source == SceneSource.DINO
                    and dino_scene is None
                ):
                    print(
                        "[WARN] scene_source=dino but no DINO scene — "
                        "falling back to legacy PDDL"
                    )
                    pddl_str = generate_problem(plan_grounded)
                elif (
                    hybrid_session.scene_source == SceneSource.ORACLE
                    and oracle_scene is None
                ):
                    print(
                        "[WARN] scene_source=oracle but no oracle scene — "
                        "falling back to legacy PDDL"
                    )
                    pddl_str = generate_problem(plan_grounded)
                else:
                    pddl_str, _fused = hybrid_session.generate_hybrid_problem(
                        plan_grounded,
                        oracle_scene=oracle_scene,
                        dino_scene=dino_scene,
                        problem_name=f"loop_iter_{iteration+1}",
                    )
                    used_hybrid = True
                    print(
                        f"[LOOP] Hybrid PDDL "
                        f"(scene_source={hybrid_session.scene_source.value}, "
                        f"goal_backend={hybrid_session.goal_backend_used}, "
                        f"holding={hybrid_session.holding_object()!r})"
                    )
            else:
                if hybrid_session is not None:
                    print(
                        "[LOOP] Hybrid session active but plan not hybrid-compatible "
                        "(enrichment / unknown primitive for template "
                        f"{getattr(plan_grounded, 'domain_template', '?')!r}) — legacy PDDL"
                    )
                pddl_str = generate_problem(plan_grounded)
            print("\n  PDDL PROBLEM:")
            for line in pddl_str.splitlines():
                print(f"    {line}")
            print()
        except Exception as _pe:
            pddl_str = f"# generation failed: {_pe}"
            used_hybrid = False
            print(f"[WARN] PDDL generation failed: {_pe}")

        # Save per-iteration debug package to run directory
        _iter_n = iteration + 1
        try:
            import json as _dbg_json
            from pathlib import Path as _PPath

            # 1. VLM plan JSON (raw + grounded)
            # Full remaining plan (all steps, before extracting current step)
            _full_plan_dict  = _dbg_json.loads(_current_plan.to_json()) if _current_plan else {}
            # Current step only (what gets executed this iteration)
            _plan_raw_dict   = _dbg_json.loads(plan.to_json())
            _plan_grnd_dict  = _dbg_json.loads(plan_grounded.to_json())

            # 2. PDDL domain content
            _domain_path = (_REPO_ROOT / "pddl" / "domains" /
                            f"{plan.domain_template}.pddl")
            _domain_str = (_domain_path.read_text()
                           if _domain_path.exists() else "# domain file not found")

            # 3. Comprehensive debug JSON
            _debug = {
                "iteration":       _iter_n,
                "task":            args.task,
                "world":           getattr(args, "world", "unknown"),
                "completed_steps": completed_steps,
                "vlm_time_s":      round(vlm_time, 2),
                "full_remaining_plan": _full_plan_dict,   # all remaining steps
                "plan_raw":        _plan_raw_dict,        # current step only
                "plan_grounded":   _plan_grnd_dict,
                "domain_template": plan.domain_template,
                "domain_additions": plan.domain_additions,
                "pddl_problem":    pddl_str,
                "step_primitive":  step0.primitive if step0 else None,
                "step_args":       dict(step0.args) if step0 else {},
                "dino_estimates":  dict(_last_dino_est),
                "placed_at":       {k: list(v) for k, v in _placed_at.items()},
                "using_overview_cam": _using_overview,
                "hybrid_used":     used_hybrid,
                "hybrid_problem_gen": (
                    None
                    if hybrid_session is None
                    else hybrid_session.metrics_snapshot()
                ),
                # Session 14: oracle vs DINO vs fused side-by-side (when hybrid ran).
                "scene_compare": (
                    None
                    if hybrid_session is None or not used_hybrid
                    else hybrid_session.last_scene_compare
                ),
            }
            _iter_dir = _RUN_DIR / f"iter_{_iter_n:02d}"
            _iter_dir.mkdir(exist_ok=True)

            (_iter_dir / "debug.json").write_text(
                _dbg_json.dumps(_debug, indent=2, ensure_ascii=False))
            (_iter_dir / "full_remaining_plan.json").write_text(
                _dbg_json.dumps(_full_plan_dict, indent=2, ensure_ascii=False))
            (_iter_dir / "plan_current_step.json").write_text(
                _dbg_json.dumps(_plan_raw_dict, indent=2, ensure_ascii=False))
            (_iter_dir / "plan_grounded.json").write_text(
                _dbg_json.dumps(_plan_grnd_dict, indent=2, ensure_ascii=False))
            (_iter_dir / "problem.pddl").write_text(pddl_str)
            (_iter_dir / f"domain_{plan.domain_template}.pddl").write_text(_domain_str)

            # Move wrist snapshot into iter subfolder
            import shutil as _shutil
            _wrist_src = _RUN_DIR / f"iter_{_iter_n:02d}_wrist.png"
            if _wrist_src.exists():
                _shutil.move(str(_wrist_src), str(_iter_dir / "wrist.png"))
            _annot_src = _RUN_DIR / f"iter_{_iter_n:02d}_annotated.png"
            if _annot_src.exists():
                _shutil.move(str(_annot_src), str(_iter_dir / "overview_annotated.png"))
            # Also save current overview image
            _ov_src = _REPO_ROOT / "data" / "scene_overview.png"
            if _ov_src.exists():
                _shutil.copy2(str(_ov_src), str(_iter_dir / "overview.png"))

        except Exception as _save_err:
            print(f"[WARN] Debug save failed: {_save_err}")

        # 6. Serialize + inject.
        # Full PDDL pipeline (no direct flag): orchestrator runs FastDownward to
        # validate the single-step plan before dispatch.  The problem_generator
        # infers the live robot state from the VLM plan structure:
        #   - pick steps   → object starts on a surface
        #   - place steps without prior pick → arm is already holding the object
        #   - pour/tilt steps without prior pick → arm is already holding the source
        # This makes single-step validation correct for all mid-task states.
        payload = json.dumps({
            "command":  args.task,
            "vlm_plan": json.loads(plan_grounded.to_json()),
        })
        bash_cmd = (
            "source /opt/ros/humble/setup.bash && "
            "source /workspace/ros2_ws/install/setup.bash && "
            "python3 /workspace/scripts/_publish_plan.py"
        )
        inject_result = subprocess.run(
            docker_cmd + ["bash", "-c", bash_cmd],
            input=payload.encode(),
            capture_output=True,
        )
        if inject_result.returncode != 0:
            print(f"[FAIL] Injection failed: {inject_result.stderr.decode().strip()}")
            _exit_reason = "fail"
            break
        print(f"[OK]   Step injected.")

        # 7. Wait for step completion
        print("[LOOP] Attendo completamento step...")
        result = _wait_step_complete(args, timeout=60, min_seq=_last_seq + 1)
        if "seq" in result:
            _last_seq = result["seq"]

        # Build step description — include original VLM name + grounded PDDL name
        # so the VLM can match its own terminology with the completed action.
        s0_orig = plan.steps[0]
        s0_grnd = plan_grounded.steps[0]
        # Always use ORIGINAL VLM names in completed_steps context.
        # The Gazebo resolution (glass→coffee_cup) is sim-internal — VLM should
        # see its own names so it recognises completed steps correctly.
        obj_orig = s0_orig.args.get("object", s0_orig.args.get("target", "?"))
        loc_orig = s0_orig.args.get("location", "")
        step_desc = (f"{s0_orig.primitive}({obj_orig}, {loc_orig})"
                     if loc_orig else f"{s0_orig.primitive}({obj_orig})")
        if result.get("success"):
            # Detect look_at loop: same look_at repeated → replan instead of break
            if s0_grnd.primitive == "look_at" and step_desc in completed_steps:
                print(f"[WARN] look_at('{obj_orig}') già eseguito — "
                      "DINO non riesce a trovare l'oggetto → replan")
                _current_plan = None
                _last_failed_step = f"look_at({obj_orig}) — object not detectable"
                _replan_count += 1
                if hybrid_session is not None:
                    hybrid_session.invalidate_goal()
                continue

            completed_steps.append(step_desc)
            print(f"[OK]   Step completato: {step_desc}")

            if hybrid_session is not None:
                # Prefer grounded args so tracker symbols match SceneState / oracle.
                if hybrid_session.verifier_enabled:
                    _pre = hybrid_session.last_fused_scene
                    _action = completed_action_from_step(s0_grnd)
                    _symbolic = (
                        _LoopStateVerifier().expect(_pre, _action)
                        if _pre is not None
                        else None
                    )
                    # Refresh oracle poses after execution for observed SceneState.
                    _post_all = {
                        k: v
                        for k, v in _get_gazebo_models(args).items()
                        if not any(
                            s in k
                            for s in (
                                "panda",
                                "franka",
                                "ground",
                                "wall",
                                "floor",
                                "camera",
                                "sun",
                            )
                        )
                    }
                    _post_poses, _post_locs = partition_gazebo_for_hybrid(
                        _post_all,
                        base_locations=hybrid_session.known_locations,
                    )
                    hybrid_session.known_locations = list(_post_locs)
                    _on_surface_post = _infer_on_surface(_post_poses, args.world)
                    _held = (
                        _symbolic.robot.holding
                        if _symbolic is not None
                        else hybrid_session.holding_object()
                    )
                    _obs_oracle = None
                    if _post_poses:
                        # Robot from symbolic post-action (executor succeeded);
                        # poses/relations from fresh oracle (may still be stale → YELLOW).
                        _obs_oracle = scene_from_xyz_poses(
                            _post_poses,
                            known_locations=hybrid_session.known_locations,
                            on_surface=_on_surface_post,
                            gripper_empty=_held is None,
                            holding=_held,
                            domain_template=plan_grounded.domain_template,
                        )
                    from planner.dino_localisation import (
                        plink0_to_world as _p2w,
                        poses_for_init_from_estimates as _poses_init,
                    )

                    _verify_dino_plink = _poses_init(
                        _last_dino_est,
                        exclude=hybrid_session.known_locations,
                        frame="plink0",
                    )
                    _obs_dino = scene_from_dino_payload(
                        [
                            d
                            for d in _dino_detections
                            if (d.get("name") or d.get("label") or "")
                            not in hybrid_session.known_locations
                        ],
                        poses={
                            n: _p2w(p["x"], p["y"], p["z"])
                            for n, p in _verify_dino_plink.items()
                        }
                        or None,
                        on_surface=_on_surface_post,
                        known_locations=hybrid_session.known_locations,
                        domain_template=plan_grounded.domain_template,
                    )
                    # Perception for verify follows scene_source (dino-only = real-world-like).
                    _obs_primary = (
                        _obs_dino
                        if hybrid_session.scene_source == SceneSource.DINO
                        else _obs_oracle
                        if hybrid_session.scene_source == SceneSource.ORACLE
                        else _obs_oracle or _obs_dino
                    )
                    if _obs_primary is not None and _symbolic is not None:
                        from planner.problem_generator.init_generator.schema import (
                            Meta as _Meta,
                            SceneState as _SS,
                        )

                        _src_tag = hybrid_session.scene_source.value
                        _observed = _SS(
                            objects=list(_obs_primary.objects)
                            or list(_symbolic.objects),
                            locations=list(_obs_primary.locations)
                            or list(_symbolic.locations),
                            relations=list(_obs_primary.relations),
                            robot=_symbolic.robot,
                            domain_template=_symbolic.domain_template,
                            frame_id=_symbolic.frame_id,
                            meta=_Meta(
                                sources_used=[_src_tag, "fusion"],
                                fusion_notes=[
                                    f"post-step observe: robot from expect, "
                                    f"relations from {_src_tag}"
                                ],
                            ),
                        )
                    elif _symbolic is not None:
                        _observed = _symbolic
                    else:
                        _observed = hybrid_session.build_scene_state(
                            oracle_scene=_obs_oracle,
                            dino_scene=_obs_dino,
                        )

                    _outcome = hybrid_session.process_completed_step(
                        s0_grnd,
                        observed=_observed,
                        pre_scene=_pre,
                        success_flag=True,
                        oracle_facts=_obs_oracle,
                        dino_facts=_obs_dino,
                        images=[image_vlm, image],
                    )
                    print(
                        f"[LOOP] Verifier {_outcome.verdict} "
                        f"(tracker_updated={_outcome.tracker_updated}, "
                        f"vlm_calls={_outcome.vlm_calls}, "
                        f"session_vlm_total="
                        f"{hybrid_session.vlm_call_count}, "
                        f"verdicts={hybrid_session.verdict_counts})"
                    )
                    if _outcome.replan:
                        _replan_count += 1
                        if _replan_count >= args.max_replans:
                            print(
                                f"[LOOP] ❌ Troppi replan ({_replan_count}) — task abortito"
                            )
                            _exit_reason = "abort"
                            break
                        print(
                            f"[LOOP] ⚠️  Verifier RED → Replan #{_replan_count}"
                        )
                        _current_plan = None
                        _last_failed_step = (
                            f"{step_desc} — verifier RED: "
                            + "; ".join(_outcome.mismatches[:3])
                        )
                        continue
                else:
                    hybrid_session.note_completed(s0_grnd)

            # Track place destinations for annotation markers
            if s0_orig.primitive == "place":
                obj_placed = s0_orig.args.get("object", "")
                loc_placed = s0_grnd.args.get("location", "")
                if obj_placed and loc_placed in _last_dino_est:
                    _loc_pose = _last_dino_est[loc_placed]
                    if isinstance(_loc_pose, dict):
                        px, py = float(_loc_pose["x"]), float(_loc_pose["y"])
                    else:
                        px, py = float(_loc_pose[0]), float(_loc_pose[1])
                    _placed_at[obj_placed] = (px, py)
                    print(f"[LOOP] Annotation: '{obj_placed}' placed at "
                          f"({px:.2f},{py:.2f}) → ✓ marker added to future images")
        else:
            # ── REPLANNING ON FAILURE ────────────────────────────────────────
            print(f"[FAIL] Step fallito: {step_desc}")
            _replan_count += 1
            if _replan_count >= args.max_replans:
                print(f"[LOOP] ❌ Troppi replan ({_replan_count}) — task abortito")
                _exit_reason = "abort"
                break
            print(f"[LOOP] ⚠️  Replan #{_replan_count} — rigenero piano completo...")
            _current_plan     = None          # force full replan next iteration
            _last_failed_step = step_desc     # context for VLM
            if hybrid_session is not None:
                if hybrid_session.verifier_enabled:
                    # Record RED via verifier path (executor failure).
                    _pre = hybrid_session.last_fused_scene
                    if _pre is not None:
                        _fail_obs = hybrid_session.build_scene_state(
                            oracle_scene=_pre,
                        )
                        hybrid_session.process_completed_step(
                            s0_grnd,
                            observed=_fail_obs,
                            pre_scene=_pre,
                            success_flag=False,
                        )
                    else:
                        hybrid_session.invalidate_goal()
                else:
                    # Goal regenerated on next hybrid PDDL build; tracker keeps mid-task facts.
                    hybrid_session.invalidate_goal()
            # Do NOT break — continue to next iteration which will replan
        # NOTE: task_complete from orchestrator = last step of CURRENT plan done.
        # In closed-loop, task completion is determined by the VLM (next iteration
        # returns complete=true or 0 steps), not by step count.  Do NOT break here.
    else:
        print(f"\n[WARN] Limite massimo di {args.max_steps} step raggiunto.")
        if _exit_reason == "max_steps":
            pass  # already default

    _wall_s = round(time.monotonic() - _loop_t0, 2)
    _hybrid_metrics = (
        None if hybrid_session is None else hybrid_session.metrics_snapshot()
    )

    # ── Did the goal actually hold in the world? ─────────────────────────────
    # The loop only knows that the executor stopped reporting errors, which is
    # not the same as the task being done: a release that drops the object still
    # looks like a clean run. Re-read the poses and check the goal facts.
    # Plan-only never moved the world, so skip the check (a violated goal here
    # would only say "the scene is still the initial one").
    if args.plan_only:
        _goal_check = None
    else:
        _goal_check = _check_goal_in_world(
            args,
            goal_facts=(
                (_hybrid_metrics or {}).get("goal_facts")
                # A failing run replans last, and replanning clears goal_facts.
                or (_hybrid_metrics or {}).get("last_goal_facts")
            ),
            world_name=args.world,
        )
    if _goal_check is not None:
        _verdict = _goal_check["status"]
        _line = _goal_check.get("summary", "")
        if _verdict == "satisfied":
            print(f"[OK]   Goal verified in world: {_line}")
        elif _verdict == "violated":
            print(f"[FAIL] Goal NOT satisfied in world: {_line}")
        else:
            print(f"[WARN] Goal not verifiable from poses: {_line}")
        # Only ever downgrade: a loop that aborted stays aborted even if the
        # world happens to satisfy the goal, but a "success" that the world
        # contradicts must not be reported as one.
        if _verdict == "violated" and _exit_reason == "success":
            print(
                "[LOOP] ⚠️  Downgrading success → goal_violated "
                "(executor reported OK but the world disagrees)"
            )
            _exit_reason = "goal_violated"
        elif _verdict == "satisfied" and _exit_reason != "success":
            print(
                f"[LOOP] ℹ️  Goal holds in the world even though the loop ended "
                f"as {_exit_reason!r} — executor reported a failure it recovered from"
            )

    _success = _exit_reason == "success"
    print(f"\n[LOOP] Steps completati: {completed_steps}")
    from planner.problem_generator.goal_generator.domain_view import (
        enrichment_authored_payload as _enrichment_authored_payload,
    )

    _enrichment_authored = _enrichment_authored_payload(
        enrichment_additions,
        skills=enrichment_skills,
        domain_path=enrichment_domain_path,
        reused=enrichment_reused,
    )
    # Save completed steps + Session 17 summary.json for mini-eval harness
    with open(str(_RUN_DIR / "run_info.txt"), "a") as _rf:
        _rf.write(f"steps:     {completed_steps}\n")
        _rf.write(f"n_steps:   {len(completed_steps)}\n")
        _rf.write(f"exit:      {_exit_reason}\n")
        _rf.write(f"success:   {_success}\n")
        _rf.write(f"replans:   {_replan_count}\n")
        _rf.write(f"wall_s:    {_wall_s}\n")
        _rf.write(
            "goal_check: "
            f"{'n/a' if _goal_check is None else _goal_check['status']}\n"
        )
    _summary = {
        "task": args.task,
        "world": _world_tag,
        "control": control_mode.value,
        "domain_template": domain_template,
        "hybrid": _hybrid_flag or "off",
        "goal_backend": _goal_backend_flag,
        "scene_source": (
            None
            if hybrid_session is None
            else hybrid_session.scene_source.value
        ),
        "perception_only": bool(args.perception_only),
        "localisation_errors": (
            None
            if hybrid_session is None
            else hybrid_session.last_localisation_report
        ),
        "success": _success,
        "exit_reason": _exit_reason,
        "n_steps": len(completed_steps),
        "completed_steps": list(completed_steps),
        "replan_count": _replan_count,
        "wall_time_s": _wall_s,
        "max_steps": args.max_steps,
        "max_replans": args.max_replans,
        # None = never measured (legacy path / no geometry), so a missing check
        # is not silently read as a passing one.
        "goal_check": _goal_check,
        "hybrid_problem_gen": _hybrid_metrics,
        "vlm_call_count": (
            0 if _hybrid_metrics is None else int(_hybrid_metrics.get("vlm_call_count") or 0)
        ),
        "verdict_counts": (
            {"GREEN": 0, "YELLOW": 0, "RED": 0}
            if _hybrid_metrics is None
            else dict(_hybrid_metrics.get("verdict_counts") or {})
        ),
        "goal_backend_used": (
            None if _hybrid_metrics is None else _hybrid_metrics.get("goal_backend_used")
        ),
        "goal_fallback_count": (
            0
            if _hybrid_metrics is None
            else int(_hybrid_metrics.get("goal_fallback_count") or 0)
        ),
        # ── Session 30: online-enrichment loop metrics ───────────────────────
        "enrichment_used": bool(enrichment_ctx),
        "enrichment_actions": list(
            enrichment_ctx.action_names if enrichment_ctx else ()
        ),
        "enrichment_skills": list(enrichment_skills),
        "enrichment_authored": _enrichment_authored,
        "domain_persisted": enrichment_domain_path,
        "domain_reused": enrichment_reused,
        "domain_select_backend": domain_select_backend,
        "refuse_reason": refuse_reason,
        "selection_reason": selection_reason or None,
        "domain_completeness": domain_completeness,
        "needed_skills": list(needed_skills),
        "plan_only": bool(args.plan_only),
        "fd_actions": list(_fd_plan_actions),
        "fd_primitives": list(_fd_plan_prims),
        "n_plan_actions": len(_fd_plan_actions),
        "pddl_problem": _fd_pddl_problem,
        "pddl_init": _fd_pddl_init,
        "pddl_goal": _fd_pddl_goal,
        "pddl_domain": _fd_pddl_domain,
        "init_facts": list(_fd_init_facts),
        "run_dir": str(_RUN_DIR.relative_to(_REPO_ROOT)),
    }
    from planner.call_timings import attach_snapshot

    attach_snapshot(_summary)
    try:
        (_RUN_DIR / "summary.json").write_text(
            json.dumps(_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[LOOP] summary.json → {_RUN_DIR.relative_to(_REPO_ROOT)}/summary.json")
    except Exception as _sum_err:
        print(f"[WARN] summary.json write failed: {_sum_err}")
    print(f"[LOOP] Debug images saved in: {_RUN_DIR.relative_to(_REPO_ROOT)}")

    # Generate self-contained HTML report for this run
    try:
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location(
            "_generate_report",
            str(_REPO_ROOT / "scripts" / "_generate_report.py"),
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _report = _mod.generate_html_report(_RUN_DIR)
        print(f"[LOOP] Report HTML: {_report.relative_to(_REPO_ROOT)}")
    except Exception as _re:
        print(f"[WARN] Report generation failed: {_re}")


if __name__ == "__main__":
    main()
