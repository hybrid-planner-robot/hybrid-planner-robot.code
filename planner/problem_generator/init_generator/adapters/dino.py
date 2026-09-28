"""
Production DINO adapter: GroundingDINO detection payloads → ``SceneState``.

Wraps detection shapes produced by ``vlm.perception.PerceptionModule`` without
importing torch/GPU code.  Offline tests load frozen snapshots from ``fixtures/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..schema import (
    LocationFact,
    Meta,
    ObjectFact,
    Orientation,
    Pose,
    Position,
    RelationFact,
    RobotFacts,
    SceneState,
    SceneStateError,
)

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

_DEFAULT_FRAME = "panda_link0"
_DEFAULT_ROBOT_CONFIDENCE = 0.9
_LOCATION_HINTS = frozenset({
    "shelf", "table", "surface", "platform", "stand",
    "rack", "tray", "bin", "ground", "floor",
})


def _normalize_dino_name(label: str) -> str:
    """Map noisy DINO labels to snake_case PDDL names."""
    name = label.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _pose_from_xyz(xyz: dict[str, float]) -> Pose:
    return Pose(
        position=Position(x=float(xyz["x"]), y=float(xyz["y"]), z=float(xyz["z"])),
        orientation=Orientation(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def _pose_from_dict(data: dict[str, Any] | None) -> Pose | None:
    if not data:
        return None
    if "position" in data:
        return Pose.from_dict(data)
    if "x" in data and "y" in data and "z" in data:
        return _pose_from_xyz(data)
    return None


class DinoAdapter:
    """Convert ``PerceptionModule`` detection payloads → ``SceneState``."""

    FORMAT = "dino_detections_v1"

    @classmethod
    def from_detections(
        cls,
        detections: list[dict[str, Any]],
        *,
        poses: dict[str, dict[str, float]] | None = None,
        on_surface: dict[str, str] | None = None,
        known_locations: list[str] | None = None,
        frame_id: str = _DEFAULT_FRAME,
        domain_template: str | None = None,
    ) -> SceneState:
        """
        Map production detection records to ``SceneState``.

        Each detection follows ``PerceptionModule._last_detection`` /
        ``draw_detections`` shape::

            {"name": str, "box": [x0, y0, x1, y1], "score": float}

        Optional ``poses`` maps normalized object names to ``get_pose()`` output
        ``{"x", "y", "z"}``.  ``on_surface`` supplies ``(on obj loc)`` when
        spatial reasoning is unavailable (typical in sim loop).
        """
        known = set(known_locations or [])
        pose_map = poses or {}
        surface_map = on_surface or {}

        objects: list[ObjectFact] = []
        relations: list[RelationFact] = []
        location_names: set[str] = set(known)

        for det in detections:
            raw_name = det.get("name") or det.get("label")
            if not raw_name:
                raise SceneStateError("detection missing name/label")

            score = float(det.get("score", det.get("confidence", 0.5)))
            name = _normalize_dino_name(str(raw_name))

            if name in known or cls._looks_like_location(name):
                location_names.add(name)
                continue

            surface = det.get("on_surface") or surface_map.get(name)
            pose = _pose_from_dict(det.get("pose")) or _pose_from_dict(pose_map.get(name))

            objects.append(
                ObjectFact(
                    name=name,
                    type="item",
                    pose=pose,
                    location=surface,
                    clear=True,
                    reachable=True,
                    source="dino",
                    confidence=score,
                )
            )
            if surface:
                location_names.add(surface)
                relations.append(
                    RelationFact(
                        predicate="on",
                        args=[name, surface],
                        source="dino",
                        confidence=score,
                    )
                )

        locations = [
            LocationFact(
                name=loc_name,
                type="location",
                reachable=True,
                source="dino" if loc_name not in known else "mock",
                confidence=0.9 if loc_name in known else 0.65,
            )
            for loc_name in sorted(location_names)
        ]

        scene = SceneState(
            schema_version="1.0",
            frame_id=frame_id,
            domain_template=domain_template,
            objects=objects,
            locations=locations,
            relations=relations,
            robot=RobotFacts(
                gripper_empty=True,
                holding=None,
                camera_aimed_at=None,
                source="dino",
                confidence=_DEFAULT_ROBOT_CONFIDENCE,
            ),
            meta=Meta(sources_used=["dino"]),
        )
        scene.validate()
        return scene

    @classmethod
    def from_detect_output(
        cls,
        boxes: dict[str, list[list[float]]],
        *,
        scores: dict[str, float] | None = None,
        poses: dict[str, dict[str, float]] | None = None,
        on_surface: dict[str, str] | None = None,
        known_locations: list[str] | None = None,
        frame_id: str = _DEFAULT_FRAME,
        domain_template: str | None = None,
    ) -> SceneState:
        """
        Map raw ``PerceptionModule._detect()`` output to ``SceneState``.

        Uses the highest-scoring box per name when ``scores`` is omitted (0.5).
        """
        score_map = scores or {}
        detections: list[dict[str, Any]] = []
        for name, box_list in boxes.items():
            if not box_list:
                continue
            box = box_list[0]
            detections.append(
                {
                    "name": name,
                    "box": box,
                    "score": score_map.get(name, 0.5),
                }
            )
        return cls.from_detections(
            detections,
            poses=poses,
            on_surface=on_surface,
            known_locations=known_locations,
            frame_id=frame_id,
            domain_template=domain_template,
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> SceneState:
        """Load a frozen ``dino_detections_v1`` fixture (no GPU required)."""
        fixture_path = Path(
            path) if path else _FIXTURES_DIR / "dino_detection_payload.json"
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT:
            raise SceneStateError(
                f"expected format {cls.FORMAT!r}, got {data.get('format')!r}"
            )

        return cls.from_detections(
            data.get("detections", []),
            poses=data.get("poses"),
            on_surface=data.get("on_surface"),
            known_locations=data.get("known_locations"),
            frame_id=data.get("frame_id", _DEFAULT_FRAME),
            domain_template=data.get("domain_template"),
        )

    @staticmethod
    def _looks_like_location(name: str) -> bool:
        tokens = name.replace("_", " ").split()
        return any(token in _LOCATION_HINTS for token in tokens)
