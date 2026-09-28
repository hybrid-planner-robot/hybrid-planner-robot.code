"""Session 25 — DINO localisation helpers for perception-driven ``:init``.

Pure (no GPU / ROS) utilities shared by the closed loop:

* catalog-driven prop name selection (world / Gazebo, not step args)
* SIM NameMatch + snap (gated by ``perception_only``)
* panda_link0 ↔ Gazebo-world frame conversion for support inference
* DINO-vs-oracle localisation error rows for the paper figure
* ``fused_from`` provenance extensions (requested vs actually fed)
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# Gazebo world origin of panda_link0 (same constants as run_loop_host).
ROBOT_BASE_XY: tuple[float, float] = (0.20, 0.0)
ROBOT_BASE_Z: float = 0.770

# Snap DINO xy to nearest Gazebo model within this radius (metres).
SNAP_RADIUS_M: float = 0.15

# Hardcoded fallback used before Session 25 when depth / height were unavailable.
FALLBACK_WORLD_Z: float = 0.82


def plink0_to_world(
    x: float,
    y: float,
    z: float,
    *,
    base_xy: tuple[float, float] = ROBOT_BASE_XY,
    base_z: float = ROBOT_BASE_Z,
) -> dict[str, float]:
    """Convert a panda_link0 point to Gazebo world coordinates."""
    return {
        "x": float(x) + float(base_xy[0]),
        "y": float(y) + float(base_xy[1]),
        "z": float(z) + float(base_z),
    }


def world_to_plink0(
    x: float,
    y: float,
    z: float,
    *,
    base_xy: tuple[float, float] = ROBOT_BASE_XY,
    base_z: float = ROBOT_BASE_Z,
) -> dict[str, float]:
    """Convert a Gazebo world point to panda_link0 coordinates."""
    return {
        "x": float(x) - float(base_xy[0]),
        "y": float(y) - float(base_xy[1]),
        "z": float(z) - float(base_z),
    }


def as_pose_xyz(value: Any) -> dict[str, float] | None:
    """Normalize ``(x,y[,z])`` / ``{x,y,z}`` to a pose dict (z defaults 0)."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        try:
            return {
                "x": float(value["x"]),
                "y": float(value["y"]),
                "z": float(value.get("z", 0.0)),
            }
        except (KeyError, TypeError, ValueError):
            return None
    try:
        seq = tuple(value)
        if len(seq) < 2:
            return None
        z = float(seq[2]) if len(seq) >= 3 else 0.0
        return {"x": float(seq[0]), "y": float(seq[1]), "z": z}
    except (TypeError, ValueError):
        return None


def catalog_prop_names(
    *,
    gazebo_poses: Mapping[str, Any] | None = None,
    world_props: Sequence[str] | None = None,
    location_models: Iterable[str] | None = None,
    infra: Iterable[str] | None = None,
    held: str | None = None,
) -> list[str]:
    """
    Graspable prop names for a scene-level DINO sweep.

    Prefers live Gazebo model names; falls back to the world SDF prop catalog.
    Location / infrastructure models and the held object are excluded.
    """
    locs = {str(x) for x in (location_models or ())}
    skip = {str(x) for x in (infra or ())} | locs
    if held:
        skip.add(str(held))

    names: list[str] = []
    seen: set[str] = set()
    for src in (gazebo_poses or {},):
        for name in src:
            n = str(name)
            if n in skip or n in seen:
                continue
            seen.add(n)
            names.append(n)
    if not names:
        for name in world_props or ():
            n = str(name)
            if n in skip or n in seen:
                continue
            seen.add(n)
            names.append(n)
    return sorted(names)


def fuzzy_gazebo_name_match(
    name: str,
    gazebo_poses: Mapping[str, Any],
) -> str | None:
    """
    SIM-ONLY NameMatch: unique substring match of ``name`` against Gazebo keys.

    Returns the Gazebo model name, or ``None`` when ambiguous / absent.
    """
    if not gazebo_poses or name in gazebo_poses:
        return None
    needle = name.lower().replace("_", "")
    candidates: list[str] = []
    for gz in gazebo_poses:
        gz_l = gz.lower().replace("_", "")
        if needle in gz_l or gz_l in needle:
            candidates.append(gz)
    if len(candidates) == 1:
        return candidates[0]
    return None


def nearest_gazebo_snap(
    x: float,
    y: float,
    gazebo_poses: Mapping[str, Mapping[str, float]],
    *,
    radius_m: float = SNAP_RADIUS_M,
    base_xy: tuple[float, float] = ROBOT_BASE_XY,
    exclude: Iterable[str] | None = None,
) -> tuple[str, float, dict[str, float]] | None:
    """
    SIM-ONLY snap: nearest Gazebo model within ``radius_m`` of panda_link0 (x,y).

    ``exclude`` typically holds location / furniture model names so a cup is
    not snapped onto ``shelf_b``.

    Returns ``(gazebo_name, delta_xy_m, pose_plink0)`` or ``None``.
    """
    if not gazebo_poses:
        return None
    skip = {str(n) for n in (exclude or ())}
    best_name: str | None = None
    best_d = float("inf")
    best_pose: dict[str, float] | None = None
    for gz, gp in gazebo_poses.items():
        if gz in skip:
            continue
        try:
            gx = float(gp["x"]) - float(base_xy[0])
            gy = float(gp["y"]) - float(base_xy[1])
            gz_z = float(gp.get("z", ROBOT_BASE_Z + 0.025)) - ROBOT_BASE_Z
        except (KeyError, TypeError, ValueError):
            continue
        d = ((x - gx) ** 2 + (y - gy) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best_name = gz
            best_pose = {"x": gx, "y": gy, "z": max(gz_z, 0.0)}
    if best_name is None or best_pose is None or best_d >= radius_m:
        return None
    return best_name, float(best_d), best_pose


def localisation_error_row(
    name: str,
    dino_xyz: Mapping[str, float] | None,
    oracle_xyz: Mapping[str, float] | None,
    *,
    same_frame: bool = True,
) -> dict[str, Any]:
    """
    One paper-figure row: DINO vs oracle Δxy / Δz in centimetres.

    Poses must be in the same frame when ``same_frame`` is True (caller converts).
    """
    row: dict[str, Any] = {
        "name": name,
        "dino": as_pose_xyz(dino_xyz),
        "oracle": as_pose_xyz(oracle_xyz),
        "delta_xy_cm": None,
        "delta_z_cm": None,
        "has_both": False,
    }
    d = row["dino"]
    o = row["oracle"]
    if d is None or o is None:
        return row
    if not same_frame:
        return row
    row["has_both"] = True
    row["delta_xy_cm"] = round(
        (((d["x"] - o["x"]) ** 2 + (d["y"] - o["y"]) ** 2) ** 0.5) * 100.0,
        2,
    )
    row["delta_z_cm"] = round((d["z"] - o["z"]) * 100.0, 2)
    return row


def build_localisation_report(
    dino_poses: Mapping[str, Any],
    oracle_poses: Mapping[str, Any],
    *,
    dino_frame: str = "plink0",
    oracle_frame: str = "world",
    only_names: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Per-object localisation error table (DINO vs oracle).

    Converts frames so both sides are compared in world coordinates.
    When ``only_names`` is set, restrict rows to that set (avoids flooding the
    table with every furniture model that has no DINO hit).
    """
    if only_names is not None:
        names = sorted({str(n) for n in only_names})
    else:
        names = sorted(set(dino_poses) | set(oracle_poses))
    rows: list[dict[str, Any]] = []
    for name in names:
        d_raw = as_pose_xyz(dino_poses.get(name))
        o_raw = as_pose_xyz(oracle_poses.get(name))
        if dino_frame == "plink0" and d_raw is not None:
            d_world = plink0_to_world(d_raw["x"], d_raw["y"], d_raw["z"])
        else:
            d_world = d_raw
        if oracle_frame == "plink0" and o_raw is not None:
            o_world = plink0_to_world(o_raw["x"], o_raw["y"], o_raw["z"])
        else:
            o_world = o_raw
        row = localisation_error_row(name, d_world, o_world)
        row["dino_frame"] = dino_frame
        row["oracle_frame"] = oracle_frame
        rows.append(row)
    return rows


def refine_z_with_height(
    z_plink0: float,
    height_m: float | None,
    *,
    used_depth: bool,
) -> float:
    """
    Prefer depth-derived ``z``; else raise ray-plane table ``z`` by half height.

    ``_estimate_object_height`` returns object extent, not a world z — used only
    when depth unprojection was unavailable.
    """
    if used_depth or height_m is None:
        return float(z_plink0)
    return float(z_plink0) + 0.5 * float(height_m)


def poses_for_init_from_estimates(
    last_dino_est: Mapping[str, Any],
    *,
    exclude: Iterable[str] | None = None,
    fallback_z: float = FALLBACK_WORLD_Z,
    frame: str = "plink0",
) -> dict[str, dict[str, float]]:
    """
    Build ``{name: {x,y,z}}`` for ``scene_from_dino_payload``.

    Accepts legacy ``(x, y)`` tuples (fill ``fallback_z`` in the requested frame)
    or full xyz mappings / 3-tuples.
    """
    skip = {str(x) for x in (exclude or ())}
    out: dict[str, dict[str, float]] = {}
    for name, raw in last_dino_est.items():
        if name in skip:
            continue
        pose = as_pose_xyz(raw)
        if pose is None:
            continue
        # Legacy (x, y) only — no real z recorded.
        if isinstance(raw, (tuple, list)) and len(raw) == 2:
            if frame == "world":
                pose = {"x": pose["x"], "y": pose["y"], "z": float(fallback_z)}
            else:
                # fallback_z is world-ish historically (0.82); convert for plink0.
                pose = {
                    "x": pose["x"],
                    "y": pose["y"],
                    "z": float(fallback_z) - ROBOT_BASE_Z,
                }
        out[str(name)] = pose
    return out


def init_fed_by_label(*, oracle: bool, dino: bool) -> str:
    """Compact label for what actually entered fusion / ``:init``."""
    if dino and oracle:
        return "fused"
    if dino:
        return "dino"
    if oracle:
        return "oracle"
    return "none"


def extend_fused_from_provenance(
    fused_from: Mapping[str, Any] | None,
    *,
    requested: str,
    perception_only: bool = False,
    shortcuts: Mapping[str, Any] | None = None,
    pose_provenance: str | None = None,
) -> dict[str, Any]:
    """
    Extend ``fused_from`` so a run cannot claim DINO provenance it did not have.

    Adds ``requested``, ``init_fed_by``, optional shortcut / pose provenance.
    """
    base = dict(fused_from or {})
    oracle = bool(base.get("oracle"))
    dino = bool(base.get("dino"))
    base["requested"] = str(requested)
    base["init_fed_by"] = init_fed_by_label(oracle=oracle, dino=dino)
    base["perception_only"] = bool(perception_only)
    if shortcuts is not None:
        base["shortcuts"] = dict(shortcuts)
    if pose_provenance is not None:
        base["pose_provenance"] = str(pose_provenance)
    return base


def summarize_shortcuts(
    *,
    perception_only: bool,
    name_match: Sequence[str] | None = None,
    sim_snap: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compact shortcut usage for provenance / localisation reports."""
    nm = [str(x) for x in (name_match or ())]
    snap = [str(x) for x in (sim_snap or ())]
    return {
        "perception_only": bool(perception_only),
        "name_match": nm,
        "name_match_count": len(nm),
        "sim_snap": snap,
        "sim_snap_count": len(snap),
        "any_oracle_substitute": (not perception_only) and bool(nm or snap),
    }


def pose_provenance_label(
    *,
    perception_only: bool,
    name_match: Sequence[str] | None = None,
    sim_snap: Sequence[str] | None = None,
    had_dino: bool,
) -> str:
    """Honest label for where metric poses feeding ``:init`` came from."""
    if not had_dino:
        return "none"
    if perception_only:
        return "dino_raw"
    nm = bool(name_match)
    snap = bool(sim_snap)
    if nm and snap:
        return "mixed_shortcuts"
    if nm:
        return "name_match"
    if snap:
        return "dino_snapped"
    return "dino_raw"
