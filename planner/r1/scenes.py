"""World → oracle_mock_v1 fixture (shared by host mock-scene and the battery)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO = Path(__file__).resolve().parents[2]
_SCENE_DIR = _REPO / "tests" / "fixtures" / "llm_plan"

DEFAULT_MOCK_SCENE = _SCENE_DIR / "wood_cube_table.json"

# Place destinations that are not graspable items in the v3 mock (like tabletop
# ``shelf``). Live DINO still sees them; the dump promotes them to locations.
DEFAULT_PLACE_LOCATIONS: dict[str, tuple[str, ...]] = {
    "tabletop": ("shelf",),
    "kitchen": ("target_tray", "tray"),
    "workshop": ("metal_tray",),
}

# Not in the overview FOV (far shelf). DINO still fires on the plate blob.
DEFAULT_SKIP_NAMES: dict[str, tuple[str, ...]] = {
    "workshop": ("small_box", "small_box2"),
}


def mock_scene_file_for_world(world: str | None) -> Path:
    """Return the constant object dump for ``world`` (tabletop / kitchen / workshop)."""
    name = str(world or "tabletop").strip() or "tabletop"
    path = _SCENE_DIR / f"{name}.json"
    if path.is_file():
        return path
    return DEFAULT_MOCK_SCENE


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a[:4])
    bx0, by0, bx1, by1 = (float(v) for v in b[:4])
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms_detections(
    detections: Sequence[Mapping[str, Any]],
    *,
    iou: float,
) -> list[dict[str, Any]]:
    """Keep the highest-scoring box when two detections overlap."""
    ranked = sorted(
        (dict(d) for d in detections),
        key=lambda d: float(d.get("score") or 0.0),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for det in ranked:
        box = det.get("box") or []
        if len(box) < 4:
            kept.append(det)
            continue
        if any(_box_iou(box, k.get("box") or []) >= iou for k in kept if len(k.get("box") or []) >= 4):
            continue
        kept.append(det)
    return kept


def detections_to_oracle_mock(
    detections: Sequence[Mapping[str, Any]],
    *,
    on_surface: Mapping[str, str] | None = None,
    locations: Iterable[str] | None = None,
    place_locations: Sequence[str] = (),
    min_score: float = 0.0,
    nms_iou: float | None = None,
    skip_names: Iterable[str] = (),
    frame_id: str = "panda_link0",
    note: str | None = None,
) -> dict[str, Any]:
    """Freeze a DINO sweep as the v3 ``oracle_mock_v1`` inventory (names + ``on``)."""
    skip = {str(n).strip() for n in skip_names if str(n).strip()}
    dets = [
        dict(d)
        for d in detections
        if float(d.get("score") or 0.0) >= float(min_score)
        and str(d.get("name") or d.get("label") or "").strip()
        and str(d.get("name") or d.get("label") or "").strip() not in skip
    ]
    if nms_iou is not None:
        dets = nms_detections(dets, iou=float(nms_iou))

    loc_names: list[str] = []
    seen_loc: set[str] = set()
    for name in list(locations or ()) + ["table"] + list(place_locations):
        n = str(name).strip()
        if not n or n in seen_loc:
            continue
        seen_loc.add(n)
        loc_names.append(n)

    surface = dict(on_surface or {})
    objects: list[dict[str, str]] = []
    seen_obj: set[str] = set()
    for det in sorted(dets, key=lambda d: str(d.get("name") or d.get("label") or "")):
        name = str(det.get("name") or det.get("label") or "").strip()
        if name in seen_obj or name in seen_loc:
            continue
        seen_obj.add(name)
        objects.append({"name": name, "location": str(surface.get(name) or "table")})

    payload: dict[str, Any] = {
        "format": "oracle_mock_v1",
        "frame_id": frame_id,
        "gripper_empty": True,
        "objects": objects,
        "locations": [{"name": n, "reachable": True} for n in loc_names],
    }
    if note:
        payload["note"] = note
    return payload
