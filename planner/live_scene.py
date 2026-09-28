"""
Shared live-scene spine (D10 extract from ``scripts/run_loop_host.py``).

Host imports the underscored helpers and calls them as before (bit-equivalent).
Baseline scripts call :func:`acquire_live_scene` for the same DINO / adapter
path, then ``fair_compact_scene``. No enricher, pruner, or domain select.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent

_WORLD_SURFACES: dict[str, dict] = {}
_WORLD_PROP_CACHE: dict[str, list[str]] = {}

# Same infrastructure filter as run_loop_host FD capture (never DINO targets).
SCENE_INFRA = frozenset(
    {
        "floor",
        "room",
        "ground_plane",
        "sun",
        "robot_pedestal",
        "overview_camera",
        "table",
        "workbench",
        # workshop furniture (not graspable props)
        "workshop_room",
        "central_workbench",
        "assembly_bench",
        "parts_shelf",
        "tool_cabinet_left",
        "back_pegboard",
        "parts_bin",
        "toolbox",
    }
)

_SHARED_PERCEPTION = None

def _docker(container: str, use_sudo: bool) -> list[str]:
    return (["sudo", "docker"] if use_sudo else ["docker"]) + ["exec", "-i", container]

def _run_in_container(args, bash_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        _docker(args.container, args.sudo_docker) + ["bash", "-c", bash_cmd],
        capture_output=True, timeout=timeout,
    )

def _pre_scan(args) -> bool:
    """Move arm to scan pose before capture."""
    print("[LOOP] Pre-scan: moving arm to scan pose...")
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_pre_scan.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=40)
    output = r.stdout.decode().strip()
    if output:
        for line in output.splitlines():
            print(f"       {line}")
    if r.returncode == 0:
        print("[OK]   Scan pose reached.")
        return True
    err = r.stderr.decode().strip()
    if err:
        print(f"[WARN] Pre-scan: {err}")
    print("[WARN] Pre-scan failed — continuing anyway (will use fallback camera)")
    return False

def _capture(args) -> Path | None:
    """Capture image from wrist camera."""
    scene_path = _REPO_ROOT / "data" / "scene.png"
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_capture_scene.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=15)
    output = r.stdout.decode().strip()
    if r.returncode != 0:
        print(f"[FAIL] Capture failed: {r.stderr.decode().strip()}")
        return None
    # Print capture output (includes which camera topic was used)
    for line in output.splitlines():
        print(f"       {line}")
    return scene_path

def _get_gazebo_models(args) -> dict:
    """Get Gazebo scene objects and their positions."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_get_model_states.py"
    )
    r = _run_in_container(args, bash_cmd, timeout=10)
    if r.returncode == 0:
        try:
            return json.loads(r.stdout.decode().strip()).get("models", {})
        except Exception:
            pass
    return {}

def _read_overview_pose_from_world(world_name: str):
    """
    Parse the world SDF file and extract the overview_camera model pose.
    Returns (x, y, z, roll, pitch, yaw) or None if not found.
    """
    import xml.etree.ElementTree as ET
    from pathlib import Path
    world_path = (Path(__file__).resolve().parent.parent /
                  "ros2_ws/src/vlm_robot_planner_bringup/worlds" /
                  f"{world_name}.world")
    if not world_path.exists():
        return None
    try:
        tree = ET.parse(str(world_path))
        for model in tree.iter("model"):
            if model.get("name") == "overview_camera":
                pose_el = model.find("pose")
                if pose_el is not None and pose_el.text:
                    vals = list(map(float, pose_el.text.split()))
                    if len(vals) == 6:
                        return vals   # [x, y, z, roll, pitch, yaw]
    except Exception:
        pass
    return None

def _world_surfaces(world_name: str) -> dict:
    """Placement surfaces of a world, parsed from its SDF once per process."""
    if world_name not in _WORLD_SURFACES:
        from planner.support_surfaces import load_world_surfaces

        _WORLD_SURFACES[world_name] = load_world_surfaces(
            _REPO_ROOT / "ros2_ws/src/vlm_robot_planner_bringup/worlds"
            / f"{world_name}.world"
        )
    return _WORLD_SURFACES[world_name]

def _infer_on_surface(item_poses: dict, world_name: str) -> dict:
    """Map each item to the surface actually holding it.

    Items no surface claims keep ``table`` so the problem still states where they
    are, which also means a world whose SDF we cannot read behaves exactly as
    before this inference existed.
    """
    from planner.support_surfaces import on_relations

    surfaces = _world_surfaces(world_name)
    if not surfaces:
        return {name: "table" for name in item_poses}
    return on_relations(item_poses, surfaces.values(), default="table")

def _get_scene_objects(world_name: str) -> list[str]:
    """
    Read all named objects from the world SDF and return their names.
    Used to inject the exact object names into the VLM prompt so the model
    generates correct oracle-compatible names regardless of task wording.
    Skips structural models (walls, floor, pedestal, cameras, furniture).
    Cached per world name (SDF is static for the process lifetime).
    """
    if world_name in _WORLD_PROP_CACHE:
        return list(_WORLD_PROP_CACHE[world_name])
    import xml.etree.ElementTree as ET
    _SKIP = {
        "sun", "ground_plane", "floor", "room", "wall_back", "wall_left",
        "wall_right", "robot_pedestal", "overview_camera", "ceiling_lamp",
        "wall_cabinet_l", "wall_cabinet_r", "fridge", "stove", "kitchen_table",
        "chair_north", "chair_south", "chair_east", "counter", "workbench",
        "desk", "side_table", "laptop_stand", "monitor_stand", "cabinet",
        "shelf_b", "bookshelf", "sofa", "plant", "trash_can", "office_chair",
        "coffee_table", "safety_cone",
        "workshop_room", "central_workbench", "assembly_bench", "parts_shelf",
        "tool_cabinet_left", "back_pegboard", "parts_bin", "toolbox",
    }
    world_path = (Path(__file__).resolve().parent.parent /
                  "ros2_ws/src/vlm_robot_planner_bringup/worlds" /
                  f"{world_name}.world")
    if not world_path.exists():
        _WORLD_PROP_CACHE[world_name] = []
        return []
    try:
        tree = ET.parse(str(world_path))
        names = []
        for model in tree.iter("model"):
            n = model.get("name", "")
            if n and n not in _SKIP:
                names.append(n)
        for inc in tree.iter("include"):
            name_el = inc.find("name")
            n = name_el.text.strip() if name_el is not None and name_el.text else ""
            if n and n not in _SKIP:
                names.append(n)
        out = sorted(set(names))
        _WORLD_PROP_CACHE[world_name] = out
        return list(out)
    except Exception:
        _WORLD_PROP_CACHE[world_name] = []
        return []

def _get_overview_cam_data(world_name: str = "office"):
    """
    Compute K and cam_to_base for the OVERVIEW camera.
    Reads pose from the world SDF file — update the world file to recalibrate.
    The overview camera is STATIC so this is computed once at startup.
    Returns (K, cam_to_base) or (None, None) on error.
    """
    try:
        import numpy as np, math

        # ── Read pose from world file ─────────────────────────────────────────
        pose = _read_overview_pose_from_world(world_name)
        if pose is None:
            # Fallback: hardcoded default
            pose = [1.0, 0.7, 1.5, 0.0, 0.68, -2.19]
            print(f"[WARN] overview_camera not found in {world_name}.world — using default")
        else:
            print(f"[INFO] Overview cam pose from {world_name}.world: "
                  f"pos=({pose[0]:.2f},{pose[1]:.2f},{pose[2]:.2f}) "
                  f"rpy=({pose[3]:.2f},{pose[4]:.2f},{pose[5]:.2f})")

        _POS  = np.array(pose[:3])
        _RPY  = tuple(pose[3:])
        _W, _H, _FOV = 640, 480, 1.047
        _ROBOT_BASE = np.array([0.20, 0.0, 0.770])

        # ── Intrinsics — prefer actual K from camera_info topic ───────────────
        from pathlib import Path as _Path
        ov_info_path = _Path(__file__).resolve().parent.parent / "data" / "overview_camera_info.json"
        if ov_info_path.exists():
            import json as _json
            with open(str(ov_info_path)) as _f:
                K = np.array(_json.load(_f)["K"])
            print(f"[INFO] Overview K from camera_info: fx={K[0,0]:.1f}")
        else:
            fx = fy = _W / (2.0 * math.tan(_FOV / 2.0))
            K = np.array([[fx, 0, _W/2.0], [0, fy, _H/2.0], [0, 0, 1.0]])
            print(f"[INFO] Overview K computed from FOV: fx={K[0,0]:.1f} (run calibration first)")

        # ── Rotation: SDF RPY → world-to-OpenCV-camera ───────────────────────
        def _rpy(r, p, y):
            Rx = np.array([[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]])
            Ry = np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
            Rz = np.array([[math.cos(y),-math.sin(y),0],[math.sin(y),math.cos(y),0],[0,0,1]])
            return Rz @ Ry @ Rx

        R_W_G = _rpy(*_RPY)       # world → Gazebo link (cols = cam axes in world)
        # Gazebo cam: +X=optical; OpenCV cam: +Z=optical
        R_C_G = np.array([[0,-1,0],[0,0,-1],[1,0,0]])  # Gazebo +Y=left → OpenCV -X
        R_world_to_cam = R_C_G @ R_W_G.T   # world → OpenCV camera

        # ── cam_to_base (camera → panda_link0) ───────────────────────────────
        # Simulation: always use the world SDF pose (camera is a Gazebo model).
        # A leftover data/overview_camera_pose.json from another world/real robot
        # must not override it — that file is only for the real-robot path.
        sdf_pose = _read_overview_pose_from_world(world_name)
        if sdf_pose is not None:
            R_cam_to_world = R_world_to_cam.T
            cam_pos_in_base = _POS - _ROBOT_BASE
            cam_to_base = np.eye(4)
            cam_to_base[:3, :3] = R_cam_to_world
            cam_to_base[:3,  3] = cam_pos_in_base
            print("[INFO] Overview cam_to_base from world SDF")
            return K, cam_to_base

        from pathlib import Path as _Path2
        import json as _json2
        _ov_pose_path = _Path2(__file__).resolve().parent.parent / "data" / "overview_camera_pose.json"
        if _ov_pose_path.exists():
            with open(str(_ov_pose_path)) as _f2:
                cam_to_base = np.array(_json2.load(_f2)["cam_to_base"])
            print("[INFO] Overview cam_to_base from overview_camera_pose.json (TF-based)")
            return K, cam_to_base

        R_cam_to_world = R_world_to_cam.T
        cam_pos_in_base = _POS - _ROBOT_BASE
        cam_to_base = np.eye(4)
        cam_to_base[:3, :3] = R_cam_to_world
        cam_to_base[:3,  3] = cam_pos_in_base
        return K, cam_to_base
    except Exception as _e:
        print(f"[WARN] overview cam calibration failed: {_e}")
        return None, None

def _estimate_object_height(
    detection: dict | None,
    obj_xyz: tuple,
    K,
    ctb,
) -> float | None:
    """Estimate object height from DINO bbox using the pinhole model.

    H ≈ bbox_height_px × dist(camera, object) / fy

    Works for both overview camera and wrist camera (ctb changes with arm pose).
    Phase 2 improvement: replace with depth-channel measurement from RealSense
    (sample depth at multiple rows of the bbox → more accurate, handles tilt).

    Returns None if inputs are unavailable or the estimate is out of range.
    """
    if detection is None or K is None or ctb is None:
        return None
    try:
        import numpy as np

        bbox_h_px = detection["box"][3] - detection["box"][1]   # y2 - y1
        if bbox_h_px < 5:   # < 5 pixels → unreliable
            return None
        cam_origin = ctb[:3, 3]                                 # camera in panda_link0
        dist = float(np.linalg.norm(np.array(obj_xyz) - cam_origin))
        fy = float(K[1, 1])
        h = bbox_h_px * dist / fy
        return h if 0.02 < h < 0.60 else None   # sanity: 2 cm – 60 cm
    except Exception:
        return None

def _load_depth_array(src_label: str):
    """Load wrist/overview depth ``.npy`` when present (uint16 mm)."""
    import numpy as np

    path = {
        "wrist": _REPO_ROOT / "data" / "depth.npy",
        "overview": _REPO_ROOT / "data" / "depth_overview.npy",
    }.get(src_label)
    if path is None or not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception:
        return None

def _run_dino_scene_sweep(
    *,
    perception,
    image,
    K,
    ctb,
    src_label: str,
    catalog_names: list[str],
    gazebo_poses: dict,
    perception_only: bool = False,
    snap_exclude: Iterable[str] | None = None,
) -> dict:
    """
    Session 25: one overview (or wrist) pass over every catalog prop.

    Returns detections + raw/published poses + shortcut / localisation metadata.
    Does not consult ``step0.args`` — FD control has no step sketch.
    Does **not** publish poses (caller publishes after the FD plan is known).
    """
    from planner.dino_localisation import (
        build_localisation_report,
        fuzzy_gazebo_name_match,
        nearest_gazebo_snap,
        pose_provenance_label,
        refine_z_with_height,
        summarize_shortcuts,
        world_to_plink0,
    )

    detections: list = []
    poses_raw: dict[str, dict[str, float]] = {}
    poses_published: dict[str, dict[str, float]] = {}
    name_match_used: list[str] = []
    sim_snap_used: list[str] = []
    depth_arr = _load_depth_array(src_label)
    used_depth_any = False
    _snap_skip = set(snap_exclude or ())

    for name in catalog_names:
        # SIM-ONLY NameMatch — only for *aliases* not already Gazebo keys.
        # Disabled under --perception-only. Catalog keys from Gazebo never hit
        # this (fuzzy returns None when name ∈ gazebo_poses).
        if not perception_only and gazebo_poses:
            gz_match = fuzzy_gazebo_name_match(name, gazebo_poses)
            if gz_match is not None:
                gp = gazebo_poses[gz_match]
                resolved = world_to_plink0(
                    float(gp["x"]), float(gp["y"]), float(gp.get("z", 0.795))
                )
                # Historical NameMatch published z=0.025 (table grasp height).
                resolved_pub = {
                    "x": resolved["x"],
                    "y": resolved["y"],
                    "z": 0.025,
                }
                print(
                    f"[LOOP] NameMatch: '{name}' → '{gz_match}' "
                    "(Gazebo pose, no DINO needed)"
                )
                name_match_used.append(name)
                poses_published[name] = dict(resolved_pub)
                poses_published[gz_match] = dict(resolved_pub)
                continue

        if image is None or K is None or ctb is None:
            print(f"[LOOP] No camera for '{name}' — skip")
            continue

        pose_est = perception.get_pose(
            name,
            image,
            K,
            ctb,
            vlm_description=name.replace("_", " "),
            depth_image=depth_arr,
        )
        det = None
        if perception._last_detection:
            det = perception._last_detection.copy()
            detections.append(det)

        if not pose_est:
            print(f"[LOOP] DINO [{src_label}]: '{name}' non rilevato")
            continue

        # Honest depth flag: only True when unprojection actually succeeded.
        used_depth = bool(getattr(perception, "_last_used_depth", False))
        if used_depth:
            used_depth_any = True

        height_m = _estimate_object_height(
            det,
            (pose_est["x"], pose_est["y"], pose_est["z"]),
            K,
            ctb,
        )
        z_ref = refine_z_with_height(
            float(pose_est["z"]),
            height_m,
            used_depth=used_depth,
        )
        raw = {
            "x": float(pose_est["x"]),
            "y": float(pose_est["y"]),
            "z": float(z_ref),
        }
        poses_raw[name] = dict(raw)
        print(
            f"[LOOP] DINO [{src_label}]: '{name}' → "
            f"({raw['x']:.3f},{raw['y']:.3f},{raw['z']:.3f})"
            + (" depth" if used_depth else "")
            + (f" height={height_m*100:.1f}cm" if height_m is not None else "")
        )

        pub = dict(raw)
        if not perception_only and gazebo_poses:
            snap = nearest_gazebo_snap(
                pub["x"], pub["y"], gazebo_poses, exclude=_snap_skip
            )
            if snap is not None:
                gz_resolved, best_d, snapped = snap
                print(
                    f"[LOOP] SIM snap: "
                    f"DINO({raw['x']:.3f},{raw['y']:.3f},{raw['z']:.3f})"
                    f" → oracle '{gz_resolved}' "
                    f"({snapped['x']:.3f},{snapped['y']:.3f},{snapped['z']:.3f}) "
                    f"Δxy={best_d*100:.1f}cm"
                )
                sim_snap_used.append(name)
                pub = dict(snapped)
                if gz_resolved != name:
                    poses_published[gz_resolved] = dict(pub)

        poses_published[name] = dict(pub)

    shortcuts = summarize_shortcuts(
        perception_only=perception_only,
        name_match=name_match_used,
        sim_snap=sim_snap_used,
    )
    # Report only catalog props (not every furniture model without a DINO hit).
    loc_report = build_localisation_report(
        poses_raw,
        {
            n: {"x": float(p["x"]), "y": float(p["y"]), "z": float(p["z"])}
            for n, p in (gazebo_poses or {}).items()
            if isinstance(p, dict) and "x" in p
        },
        dino_frame="plink0",
        oracle_frame="world",
        only_names=catalog_names,
    )
    for row in loc_report:
        row["ik_without_snap"] = None

    # perception_only → raw DINO only; else snapped/NameMatch poses for :init.
    init_poses = dict(poses_raw if perception_only else poses_published)

    return {
        "detections": detections,
        "poses_raw": poses_raw,
        "poses_published": poses_published,
        "poses_for_init": init_poses,
        "shortcuts": shortcuts,
        "localisation_errors": loc_report,
        "pose_provenance": pose_provenance_label(
            perception_only=perception_only,
            name_match=name_match_used,
            sim_snap=sim_snap_used,
            had_dino=bool(detections or poses_raw or poses_published),
        ),
        "src_label": src_label,
        "used_depth": used_depth_any,
        "depth_available": depth_arr is not None,
    }

def _print_localisation_report(rows: list[dict]) -> None:
    """Human-readable DINO-vs-oracle Δxy/Δz table."""
    if not rows:
        return
    print("[LOOP] Localisation error (DINO raw vs oracle, world frame):")
    print(f"       {'name':<16} {'Δxy_cm':>8} {'Δz_cm':>8}  notes")
    for row in rows:
        if not row.get("has_both"):
            miss = "no-dino" if row.get("dino") is None else "no-oracle"
            print(f"       {row['name']:<16} {'—':>8} {'—':>8}  {miss}")
            continue
        print(
            f"       {row['name']:<16} "
            f"{row['delta_xy_cm']:>8.2f} {row['delta_z_cm']:>8.2f}"
        )


@dataclass
class DockerLoopArgs:
    """Minimal host-args stand-in for docker exec helpers."""

    container: str = "vlm_ros2"
    sudo_docker: bool = False


@dataclass
class LiveSceneResult:
    scene: Any
    detections: list = field(default_factory=list)
    poses_for_init: dict = field(default_factory=dict)
    gazebo_poses: dict = field(default_factory=dict)
    using_overview: bool = False
    pre_scan_ok: bool = False
    provenance: dict = field(default_factory=dict)


def get_shared_perception():
    """Process-wide PerceptionModule (DINO), same object R0 loads."""
    global _SHARED_PERCEPTION
    if _SHARED_PERCEPTION is None:
        from vlm.perception import PerceptionModule

        print("[live-scene] Loading PerceptionModule…")
        _SHARED_PERCEPTION = PerceptionModule()
        _SHARED_PERCEPTION.load()
        print("[live-scene] PerceptionModule ready.")
    return _SHARED_PERCEPTION


def acquire_live_scene(
    *,
    world: str,
    container: str = "vlm_ros2",
    scene_source: str = "dino",
    perception_only: bool = True,
    sudo_docker: bool = False,
    do_pre_scan: bool = True,
    perception: Any | None = None,
) -> LiveSceneResult:
    """
    One-shot R0 perception: pre-scan + capture + DINO sweep → SceneState.

    Same helpers as ``run_loop_host`` ``control=fd``. Does not call the
    enricher, pruner, or ``resolve_domain_for_task``. ``domain_template`` on
    the returned scene is cleared (baselines strip it in ``fair_compact_scene``
    anyway).
    """
    from planner.dino_localisation import (
        catalog_prop_names,
        plink0_to_world,
        poses_for_init_from_estimates,
    )
    from planner.hybrid_runtime import (
        GAZEBO_LOCATION_MODELS,
        HybridProblemSession,
        SceneSource,
        partition_gazebo_for_hybrid,
        resolve_scene_source,
        scene_from_dino_payload,
        scene_from_xyz_poses,
    )
    from PIL import Image as PilImage

    args = DockerLoopArgs(container=container, sudo_docker=sudo_docker)
    source = resolve_scene_source(scene_source)
    pre_scan_ok = False
    try:
        if do_pre_scan:
            pre_scan_ok = _pre_scan(args)
            time.sleep(1.0)
        image_path = _capture(args)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"live capture failed ({exc}). "
            "Start the sim, or pass --mock-scene for the fixture SceneState."
        ) from exc
    if image_path is None:
        raise RuntimeError(
            "live capture failed (no scene.png). "
            "Start the sim, or pass --mock-scene for the fixture SceneState."
        )

    image = PilImage.open(image_path).convert("RGB")
    ov_k, ov_ctb = _get_overview_cam_data(world)
    ov_path = _REPO_ROOT / "data" / "scene_overview.png"
    using_overview = ov_path.is_file() and ov_k is not None
    if using_overview:
        image_vlm = PilImage.open(str(ov_path)).convert("RGB")
        src_label = "overview"
        k, ctb = ov_k, ov_ctb
    else:
        image_vlm = image
        src_label = "wrist"
        k, ctb = ov_k, ov_ctb
        print("[WARN] overview camera unavailable — wrist / missing K")

    raw_gz = _get_gazebo_models(args)
    gazebo_poses = {k: v for k, v in raw_gz.items() if k not in SCENE_INFRA}
    print(f"[LOOP] Scene objects: {list(gazebo_poses.keys())}")

    item_poses, known_locations = partition_gazebo_for_hybrid(
        gazebo_poses, base_locations=["table", "shelf"]
    )
    on_oracle = _infer_on_surface(item_poses, world) if item_poses else {}

    oracle_scene = None
    if item_poses:
        oracle_scene = scene_from_xyz_poses(
            item_poses,
            known_locations=known_locations,
            on_surface=on_oracle,
            gripper_empty=True,
            holding=None,
            domain_template=None,
        )

    dino_scene = None
    detections: list = []
    poses_for_init: dict = {}
    provenance: dict = {}
    need_dino = source in {SceneSource.DINO, SceneSource.FUSED}
    if need_dino:
        if not using_overview or k is None or ctb is None:
            raise RuntimeError(
                "scene_source=dino requires a successful overview capture. "
                "Start the sim, or pass --mock-scene for the fixture SceneState."
            )
        perc = perception if perception is not None else get_shared_perception()
        catalog = catalog_prop_names(
            gazebo_poses=gazebo_poses,
            world_props=_get_scene_objects(world),
            location_models=GAZEBO_LOCATION_MODELS,
            infra=SCENE_INFRA,
            held=None,
        )
        print(
            f"[LOOP] FD DINO scene sweep ({len(catalog)} props, "
            f"src={src_label}, perception_only={perception_only})"
        )
        sweep = _run_dino_scene_sweep(
            perception=perc,
            image=image_vlm,
            K=k,
            ctb=ctb,
            src_label=src_label,
            catalog_names=catalog,
            gazebo_poses=gazebo_poses,
            perception_only=perception_only,
            snap_exclude=GAZEBO_LOCATION_MODELS,
        )
        detections = list(sweep.get("detections") or [])
        poses_for_init = dict(sweep.get("poses_for_init") or {})
        _print_localisation_report(list(sweep.get("localisation_errors") or []))
        provenance = {
            "perception_only": perception_only,
            "shortcuts": sweep.get("shortcuts"),
            "pose_provenance": sweep.get("pose_provenance"),
            "depth_available": sweep.get("depth_available"),
            "src_label": sweep.get("src_label"),
        }
        dino_poses_plink = poses_for_init_from_estimates(
            poses_for_init,
            exclude=list(known_locations) + list(GAZEBO_LOCATION_MODELS),
            frame="plink0",
        )
        dino_poses_world = {
            n: plink0_to_world(p["x"], p["y"], p["z"])
            for n, p in dino_poses_plink.items()
        }
        use_dino_surfaces = perception_only or source == SceneSource.DINO
        if use_dino_surfaces and dino_poses_world:
            on_dino = _infer_on_surface(dino_poses_world, world)
        else:
            on_dino = on_oracle
        dino_scene = scene_from_dino_payload(
            [
                d
                for d in detections
                if (d.get("name") or d.get("label") or "")
                not in known_locations
                and (d.get("name") or d.get("label") or "")
                not in GAZEBO_LOCATION_MODELS
            ],
            poses=dino_poses_world or None,
            on_surface=on_dino,
            known_locations=known_locations,
            domain_template=None,
        )
        if source == SceneSource.DINO and dino_scene is None:
            raise RuntimeError(
                "scene_source=dino requires a successful DINO sweep. "
                "Pass --mock-scene for the fixture SceneState."
            )

    if source == SceneSource.ORACLE and oracle_scene is None:
        raise RuntimeError(
            "scene_source=oracle but no Gazebo models. "
            "Pass --mock-scene for the fixture SceneState."
        )
    if dino_scene is None and oracle_scene is None:
        raise RuntimeError(
            "live scene empty (no oracle/DINO). "
            "Pass --mock-scene for the fixture SceneState."
        )

    session = HybridProblemSession(
        scene_source=source,
        domain_template="",
        known_locations=list(known_locations),
        command="",
    )
    scene = session.build_scene_state(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
    )
    scene.domain_template = None
    return LiveSceneResult(
        scene=scene,
        detections=detections,
        poses_for_init=poses_for_init,
        gazebo_poses=gazebo_poses,
        using_overview=using_overview,
        pre_scan_ok=pre_scan_ok,
        provenance=provenance,
    )

