"""Support-surface inference from 3D poses.

A PDDL ``(on <item> <location>)`` fact claims to say which surface actually
holds an item. The closed loop used to assert ``table`` for every graspable
object, which is true for the initial tabletop layout and silently wrong the
moment something sits on a shelf — including a replan issued *after* a
successful place, which would re-assert that the object is still on the table.

Given the geometry of the known surfaces this module derives the relation from
the item's pose instead. The same inference answers "did the task actually
succeed?", so ``check_goal_facts`` reuses it rather than trusting the executor's
own report that its last primitive returned without an error.

Surface geometry is read from the world SDF rather than hardcoded, so adding a
world or moving a shelf needs no code change. Stdlib only — no ROS, no Gazebo —
so it runs in CI.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

__all__ = [
    "SupportSurface",
    "GoalCheck",
    "DEFAULT_XY_MARGIN",
    "DEFAULT_MAX_CLEARANCE",
    "DEFAULT_MIN_CLEARANCE",
    "SURFACE_LINK_NAME",
    "as_xyz",
    "infer_support_surface",
    "on_relations",
    "check_goal_facts",
    "surfaces_from_sdf",
    "load_world_surfaces",
]

# An item may overhang its surface, and poses carry noise, so allow a little
# slack outside the footprint before rejecting a candidate surface.
DEFAULT_XY_MARGIN = 0.03      # metres

# A resting object's centre sits above the surface by half its height, so this
# covers props up to ~0.24 m tall; measured clearances for the tabletop props are
# 0.00-0.06 m. Anything higher is in the gripper, and a looser bound really does
# misreport held objects: a cup 0.20 m above the shelf was being called `on` it.
DEFAULT_MAX_CLEARANCE = 0.12  # metres

# Slightly negative to tolerate pose noise on thin objects.
DEFAULT_MIN_CLEARANCE = -0.01  # metres

# The worlds in this repo name the placement slab of a furniture model
# ``surface`` (both the table and the shelves do). That convention is what lets
# the footprint be read from the SDF instead of maintained by hand.
SURFACE_LINK_NAME = "surface"


@dataclass(frozen=True)
class SupportSurface:
    """A horizontal placement surface, in the same frame as the item poses.

    ``top_z`` is the height of the surface the item rests *on*, not the model
    origin: for a Gazebo box that is the link centre plus half the box height.
    """

    name: str
    center_x: float
    center_y: float
    top_z: float
    half_x: float
    half_y: float

    def contains_xy(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        return (
            abs(x - self.center_x) <= self.half_x + margin
            and abs(y - self.center_y) <= self.half_y + margin
        )


@dataclass
class GoalCheck:
    """Outcome of comparing PDDL goal facts against observed poses.

    ``status`` is ``satisfied`` only when at least one fact was checkable and
    every checkable fact held. Predicates this module cannot evaluate from poses
    alone (``holding``, ``camera-aimed-at``, …) are reported as unverifiable
    rather than silently counted as satisfied.
    """

    status: str  # "satisfied" | "violated" | "unverifiable"
    facts: list[dict] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return self.status == "satisfied"

    @property
    def violated(self) -> bool:
        return self.status == "violated"

    def as_dict(self) -> dict:
        return {"status": self.status, "facts": list(self.facts)}

    def summary_line(self) -> str:
        parts = []
        for f in self.facts:
            fact = f"({f['predicate']} {' '.join(f['args'])})"
            if f["verdict"] == "ok":
                parts.append(f"{fact} ✓")
            elif f["verdict"] == "unverifiable":
                parts.append(f"{fact} ?")
            else:
                parts.append(f"{fact} ✗ ({f.get('observed') or 'nothing'})")
        return "; ".join(parts)


def as_xyz(pose) -> tuple[float, float, float] | None:
    """Accept ``(x, y, z)`` or ``{"x": …, "y": …, "z": …}``."""
    if pose is None:
        return None
    if isinstance(pose, Mapping):
        try:
            return float(pose["x"]), float(pose["y"]), float(pose["z"])
        except (KeyError, TypeError, ValueError):
            return None
    try:
        x, y, z = (float(v) for v in tuple(pose)[:3])
    except (TypeError, ValueError):
        return None
    return x, y, z


def infer_support_surface(
    pose,
    surfaces: Iterable[SupportSurface],
    *,
    xy_margin: float = DEFAULT_XY_MARGIN,
    max_clearance: float = DEFAULT_MAX_CLEARANCE,
    min_clearance: float = DEFAULT_MIN_CLEARANCE,
    default: str | None = None,
) -> str | None:
    """Name the surface holding ``pose``, or ``default`` if none qualifies.

    A surface qualifies when the item's (x, y) falls inside its footprint and
    the item sits just above it. Surfaces often nest — a shelf standing on a
    table contains the same (x, y) as the table — so the highest qualifying
    surface wins.
    """
    xyz = as_xyz(pose)
    if xyz is None:
        return default
    x, y, z = xyz

    best_name: str | None = None
    best_top = float("-inf")
    for surface in surfaces:
        if not surface.contains_xy(x, y, margin=xy_margin):
            continue
        clearance = z - surface.top_z
        if clearance < min_clearance or clearance > max_clearance:
            continue
        if surface.top_z > best_top:
            best_name, best_top = surface.name, surface.top_z

    return best_name if best_name is not None else default


def on_relations(
    item_poses: Mapping[str, object],
    surfaces: Iterable[SupportSurface],
    *,
    default: str | None = None,
    **kwargs,
) -> dict[str, str]:
    """Map each item to the surface holding it, skipping unsupported items."""
    surfaces = list(surfaces)
    out: dict[str, str] = {}
    for name, pose in item_poses.items():
        loc = infer_support_surface(pose, surfaces, default=default, **kwargs)
        if loc:
            out[name] = loc
    return out


def check_goal_facts(
    goal_facts: Sequence[Sequence[str]],
    item_poses: Mapping[str, object],
    surfaces: Iterable[SupportSurface],
    **kwargs,
) -> GoalCheck:
    """Check PDDL goal facts against observed poses.

    Only ``on`` is decidable from poses; anything else is reported as
    unverifiable so a run is never credited with a success this module did not
    actually confirm.
    """
    surfaces = list(surfaces)
    results: list[dict] = []
    any_checked = False
    any_failed = False

    for fact in goal_facts or ():
        parts = [str(p) for p in fact]
        if not parts:
            continue
        predicate, args = parts[0], parts[1:]
        entry: dict = {"predicate": predicate, "args": args}

        if predicate == "on" and len(args) == 2:
            item, expected = args
            observed = infer_support_surface(
                item_poses.get(item), surfaces, **kwargs
            )
            entry["observed"] = observed
            if item not in item_poses:
                entry["verdict"] = "unverifiable"
                entry["reason"] = f"no observed pose for {item!r}"
            elif observed == expected:
                entry["verdict"] = "ok"
                any_checked = True
            else:
                entry["verdict"] = "failed"
                any_checked = True
                any_failed = True
        else:
            entry["verdict"] = "unverifiable"
            entry["reason"] = f"{predicate!r} not decidable from poses"

        results.append(entry)

    if any_failed:
        status = "violated"
    elif any_checked:
        status = "satisfied"
    else:
        status = "unverifiable"
    return GoalCheck(status=status, facts=results)


# ── Reading surface geometry from a world SDF ────────────────────────────────


def _parse_pose(element) -> tuple[float, float, float]:
    """Translation part of an SDF ``<pose>``; rotation is ignored."""
    if element is None or not (element.text or "").strip():
        return 0.0, 0.0, 0.0
    try:
        vals = [float(v) for v in element.text.split()]
    except ValueError:
        return 0.0, 0.0, 0.0
    vals += [0.0] * (3 - len(vals))
    return vals[0], vals[1], vals[2]


def _parse_box_size(link) -> tuple[float, float, float] | None:
    for tag in ("collision", "visual"):
        for node in link.iter(tag):
            box = node.find("./geometry/box/size")
            if box is not None and (box.text or "").strip():
                try:
                    sx, sy, sz = (float(v) for v in box.text.split()[:3])
                except ValueError:
                    continue
                return sx, sy, sz
    return None


def surfaces_from_sdf(sdf_text: str) -> dict[str, SupportSurface]:
    """Extract placement surfaces from world SDF text.

    Only models carrying a box-shaped link named ``surface`` are returned, which
    is what distinguishes furniture from graspable props and walls. Model
    rotation is ignored: the worlds here keep surfaces axis-aligned, and a
    rotated footprint would need the full transform.
    """
    try:
        root = ET.fromstring(sdf_text)
    except ET.ParseError:
        return {}

    out: dict[str, SupportSurface] = {}
    for model in root.iter("model"):
        name = (model.get("name") or "").strip()
        if not name:
            continue
        link = next(
            (
                l
                for l in model.iter("link")
                if (l.get("name") or "") == SURFACE_LINK_NAME
            ),
            None,
        )
        if link is None:
            continue
        size = _parse_box_size(link)
        if size is None:
            continue

        mx, my, mz = _parse_pose(model.find("pose"))
        lx, ly, lz = _parse_pose(link.find("pose"))
        sx, sy, sz = size
        out[name] = SupportSurface(
            name=name,
            center_x=mx + lx,
            center_y=my + ly,
            top_z=mz + lz + sz / 2.0,
            half_x=sx / 2.0,
            half_y=sy / 2.0,
        )
    return out


def load_world_surfaces(world_path: str | Path) -> dict[str, SupportSurface]:
    """``surfaces_from_sdf`` for a world file; empty dict if unreadable."""
    path = Path(world_path)
    if not path.exists():
        return {}
    try:
        return surfaces_from_sdf(path.read_text())
    except OSError:
        return {}
