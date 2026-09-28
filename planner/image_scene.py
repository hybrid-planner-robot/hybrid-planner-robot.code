"""Image → VLM inventory → DINO poses → SceneState (PoC, not the golden battery)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from planner.hybrid_runtime import scene_from_dino_payload
from planner.problem_generator.init_generator.schema import LocationFact, SceneState
from vlm.inventory import (
    SceneInventory,
    apply_task_buckets,
    default_support_id,
    ensure_support_locations,
    list_objects_from_image,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class InventorySweep:
    """VLM names + DINO metric poses for hybrid FD / real-robot inject."""

    scene: SceneState
    inventory: SceneInventory
    detections: list[dict[str, Any]]
    poses_plink0: dict[str, dict[str, float]]


def release_cuda() -> None:
    """Drop unused GPU tensors so the text LLM can load after Qwen-VL."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def fallback_camera(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Identity extrinsics + pinhole K from image size. Not metric."""
    w, h = image.size
    fx = fy = float(max(w, h))
    k = np.array([[fx, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    return k, np.eye(4)


def _load_k_and_pose(info_path: Path, pose_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not (info_path.is_file() and pose_path.is_file()):
        return None
    k = np.array(json.loads(info_path.read_text(encoding="utf-8"))["K"], dtype=np.float64)
    cam_to_base = np.array(
        json.loads(pose_path.read_text(encoding="utf-8"))["cam_to_base"],
        dtype=np.float64,
    )
    return k, cam_to_base


def load_camera_matrices(
    camera_dir: Path | None,
    *,
    prefer_overview: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load K + cam_to_base.

    Overview files (``overview_camera_info.json`` / ``overview_camera_pose.json``)
    match the fixed stand camera used for inventory photos. Wrist
    ``camera_info.json`` / ``camera_pose.json`` are the fallback.
    """
    directory = Path(camera_dir) if camera_dir is not None else (_REPO_ROOT / "data")
    if not directory.is_absolute():
        directory = _REPO_ROOT / directory
    overview = _load_k_and_pose(
        directory / "overview_camera_info.json",
        directory / "overview_camera_pose.json",
    )
    wrist = _load_k_and_pose(
        directory / "camera_info.json",
        directory / "camera_pose.json",
    )
    if prefer_overview:
        return overview if overview is not None else wrist
    return wrist if wrist is not None else overview


def execution_poses_plink0(
    scene: SceneState,
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    """panda_link0 xyz for pick/place inject (items + DINO'd containers)."""
    poses: dict[str, dict[str, float]] = {}
    for obj in scene.objects:
        if obj.pose is None:
            continue
        pos = obj.pose.position
        poses[obj.name] = {"x": float(pos.x), "y": float(pos.y), "z": float(pos.z)}
    for det in detections or []:
        name = str(det.get("name") or "").strip()
        pose = det.get("pose")
        if not name or not isinstance(pose, dict) or "x" not in pose:
            continue
        poses[name] = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
        }
    return poses


def load_execution_poses(
    scene_path: Path,
    scene: SceneState,
) -> dict[str, dict[str, float]]:
    """Read ``execution_poses`` from an oracle_mock dump, else object poses."""
    extra: dict[str, dict[str, float]] = {}
    if scene_path.is_file():
        try:
            data = json.loads(scene_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        raw = data.get("execution_poses") or {}
        if isinstance(raw, dict):
            for name, pose in raw.items():
                if isinstance(pose, dict) and "x" in pose:
                    extra[str(name)] = {
                        "x": float(pose["x"]),
                        "y": float(pose["y"]),
                        "z": float(pose["z"]),
                    }
    merged = execution_poses_plink0(scene)
    merged.update(extra)
    return merged


def scene_to_oracle_mock(
    scene: SceneState,
    *,
    execution_poses: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Dump a SceneState as ``oracle_mock_v1`` for ``run_loop_host --mock-scene``."""
    objects: list[dict[str, Any]] = []
    for obj in scene.objects:
        loc = obj.location
        row: dict[str, Any] = {"name": obj.name}
        if loc:
            row["location"] = loc
        if obj.pose is not None:
            row["pose"] = obj.pose.to_dict()
        objects.append(row)
    loc_rows: list[dict[str, Any]] = []
    for loc in scene.locations:
        row = {
            "name": loc.name,
            "reachable": True if loc.reachable is None else bool(loc.reachable),
        }
        if loc.type and loc.type != "location":
            row["type"] = loc.type
        if loc.open is not None:
            row["open"] = bool(loc.open)
        loc_rows.append(row)
    if not loc_rows:
        loc_rows = [{"name": "support", "reachable": True}]
        for row in objects:
            row.setdefault("location", "support")
    payload: dict[str, Any] = {
        "format": "oracle_mock_v1",
        "frame_id": scene.frame_id,
        "gripper_empty": bool(scene.robot.gripper_empty),
        "holding": scene.robot.holding,
        "objects": objects,
        "locations": loc_rows,
    }
    poses = dict(execution_poses or execution_poses_plink0(scene))
    if poses:
        payload["execution_poses"] = poses
    return payload


def apply_inventory_containers(
    scene: SceneState, inventory: SceneInventory
) -> SceneState:
    """Mark inventory containers as PDDL ``container`` locations.

    Open/closed comes from the VLM photo inventory, not from the symbol name.
    """
    from dataclasses import replace

    container_ids = set(inventory.container_ids())
    if not container_ids:
        return scene
    new_objects = [obj for obj in scene.objects if obj.name not in container_ids]
    by_loc = {loc.name: loc for loc in scene.locations}
    for cid in container_ids:
        existing = by_loc.get(cid)
        is_open = inventory.container_is_open(cid)
        if existing is None:
            by_loc[cid] = LocationFact(
                name=cid,
                type="container",
                reachable=True,
                open=is_open,
                source="vlm",
                confidence=1.0,
            )
        else:
            by_loc[cid] = replace(
                existing,
                type="container",
                open=is_open,
            )
    return replace(
        scene,
        objects=new_objects,
        locations=sorted(by_loc.values(), key=lambda loc: loc.name),
    )


def _detection_from_pose(
    ent_id: str,
    pose: dict[str, float] | None,
    last_detection: Any,
) -> dict[str, Any]:
    if isinstance(last_detection, dict) and last_detection.get("name"):
        det = dict(last_detection)
    else:
        det = {"name": ent_id, "box": [0, 0, 1, 1], "score": 0.4}
    if isinstance(pose, dict) and "x" in pose:
        xyz = {
            "x": float(pose["x"]),
            "y": float(pose["y"]),
            "z": float(pose["z"]),
        }
        det["pose"] = xyz
    return det


def localize_inventory(
    inventory: SceneInventory,
    *,
    perception: Any,
    image: Image.Image,
    k: np.ndarray,
    cam_to_base: np.ndarray,
    depth_image: np.ndarray | None = None,
) -> tuple[SceneState, list[dict[str, Any]]]:
    """
    Run GroundingDINO ``get_pose`` for inventory items and containers.

    Items become PDDL objects. Containers are DINO'd for execution poses
    (place-in-container needs the bowl xyz) then promoted to locations.
    Support surfaces are not DINO-queried; they seed ``known_locations``.
    ``(on item loc)`` uses the static workspace slab (table/cloth), not a
    portable destination such as notebook/tray, unless that is the only
    non-container location.
    Misses still enter the scene (on the VLM surface, no pose).
    """
    inventory = ensure_support_locations(inventory)
    loc_ids = list(inventory.location_ids())
    support = default_support_id(inventory)
    object_dets: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    object_poses: dict[str, dict[str, float]] = {}
    on_surface: dict[str, str] = {}
    seen: set[str] = set()

    def _query(ent_id: str, query: str) -> dict[str, Any]:
        pose = perception.get_pose(
            ent_id,
            image,
            k,
            cam_to_base,
            vlm_description=query,
            depth_image=depth_image,
        )
        det = _detection_from_pose(
            ent_id, pose if isinstance(pose, dict) else None,
            getattr(perception, "_last_detection", None),
        )
        return det

    for ent in inventory.objects:
        if ent.id in seen or ent.id in loc_ids:
            continue
        seen.add(ent.id)
        det = _query(ent.id, ent.query)
        object_dets.append(det)
        detections.append(det)
        pose = det.get("pose")
        if isinstance(pose, dict) and "x" in pose:
            object_poses[ent.id] = {
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "z": float(pose["z"]),
            }
        on_surface[ent.id] = support

    for ent in inventory.containers:
        if ent.id in seen:
            continue
        seen.add(ent.id)
        detections.append(_query(ent.id, ent.query))

    scene = scene_from_dino_payload(
        object_dets,
        poses=object_poses or None,
        on_surface=on_surface,
        known_locations=loc_ids,
    )
    if scene is None:
        raise RuntimeError("DINO produced an empty SceneState from the inventory")
    return apply_inventory_containers(scene, inventory), detections


def scene_from_inventory_only(inventory: SceneInventory) -> SceneState:
    """Symbolic scene (all items on the VLM surface) when DINO is skipped."""
    inventory = ensure_support_locations(inventory)
    support = default_support_id(inventory)
    detections = [
        {"name": ent.id, "box": [0, 0, 1, 1], "score": 1.0}
        for ent in inventory.objects
    ]
    on_surface = {ent.id: support for ent in inventory.objects}
    scene = scene_from_dino_payload(
        detections,
        on_surface=on_surface,
        known_locations=list(inventory.location_ids()),
    )
    if scene is None:
        raise RuntimeError("inventory produced an empty SceneState")
    return apply_inventory_containers(scene, inventory)


def perceive_image(
    image: Image.Image,
    *,
    task: str = "",
    planner: Any | None = None,
    perception: Any | None = None,
    inventory: SceneInventory | None = None,
    camera_dir: Path | None = None,
    depth_image: np.ndarray | None = None,
    skip_dino: bool = False,
    prefer_overview: bool = False,
) -> tuple[SceneState, SceneInventory, list[dict[str, Any]]]:
    """
    VLM names (unless ``inventory`` is given) then optional DINO localization.

    Golden planning is unchanged: the returned SceneState is meant to be dumped
    as ``oracle_mock_v1`` and fed to ``run_loop_host --mock-scene``.
    """
    if inventory is None:
        if planner is None:
            raise ValueError("pass planner or a precomputed inventory")
        inventory = list_objects_from_image(image, planner, task=task)
    else:
        inventory = apply_task_buckets(inventory, task)
    inventory = ensure_support_locations(inventory)

    if skip_dino or perception is None:
        return scene_from_inventory_only(inventory), inventory, []

    loaded = load_camera_matrices(camera_dir, prefer_overview=prefer_overview)
    if loaded is None:
        k, ctb = fallback_camera(image)
    else:
        k, ctb = loaded
    scene, detections = localize_inventory(
        inventory,
        perception=perception,
        image=image,
        k=k,
        cam_to_base=ctb,
        depth_image=depth_image,
    )
    return scene, inventory, detections


def sweep_from_perceive(
    scene: SceneState,
    inventory: SceneInventory,
    detections: list[dict[str, Any]],
) -> InventorySweep:
    return InventorySweep(
        scene=scene,
        inventory=inventory,
        detections=detections,
        poses_plink0=execution_poses_plink0(scene, detections),
    )
