"""
Compact SceneState views for goal-LLM prompts.

Omits bulky poses / orientations by default (design §3.3.1). Never includes
camera images — the goal model is text-only.
"""

from __future__ import annotations

import json
from typing import Any

from ..init_generator.schema import SceneState

__all__ = ["compact_scene_dict", "compact_scene_json"]


def compact_scene_dict(
    scene: SceneState,
    *,
    include_poses: bool = False,
    include_meta: bool = False,
) -> dict[str, Any]:
    """
    Serialize a prompt-friendly SceneState subset.

    Always includes object/location names, relations, and robot facts.
    Poses are omitted unless ``include_poses=True``.
    """
    objects: list[dict[str, Any]] = []
    for obj in scene.objects:
        entry: dict[str, Any] = {"name": obj.name, "type": obj.type}
        if obj.location is not None:
            entry["location"] = obj.location
        if obj.clear is not None:
            entry["clear"] = obj.clear
        if obj.reachable is not None:
            entry["reachable"] = obj.reachable
        if include_poses and obj.pose is not None:
            entry["pose"] = obj.pose.to_dict()
        objects.append(entry)

    locations: list[dict[str, Any]] = []
    for loc in scene.locations:
        entry = {"name": loc.name, "type": loc.type}
        if loc.reachable is not None:
            entry["reachable"] = loc.reachable
        if loc.open is not None:
            entry["open"] = loc.open
        locations.append(entry)

    relations = [
        {"predicate": rel.predicate, "args": list(rel.args)}
        for rel in scene.relations
    ]

    out: dict[str, Any] = {
        "objects": objects,
        "locations": locations,
        "relations": relations,
        "robot": {
            "gripper_empty": scene.robot.gripper_empty,
            "holding": scene.robot.holding,
            "camera_aimed_at": scene.robot.camera_aimed_at,
        },
    }
    if scene.domain_template is not None:
        out["domain_template"] = scene.domain_template
    if include_meta and (scene.meta.sources_used or scene.meta.fusion_notes):
        out["meta"] = scene.meta.to_dict()
    return out


def compact_scene_json(
    scene: SceneState,
    *,
    include_poses: bool = False,
    include_meta: bool = False,
    indent: int | None = 2,
) -> str:
    """JSON string of :func:`compact_scene_dict` for prompt injection."""
    return json.dumps(
        compact_scene_dict(
            scene,
            include_poses=include_poses,
            include_meta=include_meta,
        ),
        indent=indent,
    )
