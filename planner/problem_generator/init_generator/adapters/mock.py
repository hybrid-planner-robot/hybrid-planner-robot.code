"""
Mock adapters for offline testing without ROS/Gazebo/GPU.

Each adapter loads a source-specific fixture JSON file and returns a validated
canonical ``SceneState``.
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
    Pose,
    RelationFact,
    RobotFacts,
    SceneState,
    SceneStateError,
)

_MOCK_DIR = Path(__file__).resolve().parent.parent / "mock"


def _load_fixture(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_dino_name(label: str) -> str:
    """Map noisy DINO labels to snake_case PDDL names."""
    name = label.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name


class OracleMockAdapter:
    """Convert oracle-style mock fixtures → SceneState."""

    FORMAT = "oracle_mock_v1"

    @classmethod
    def load(cls, path: str | Path | None = None) -> SceneState:
        data = _load_fixture(path or _MOCK_DIR / "oracle_table_scene.json")
        if data.get("format") != cls.FORMAT:
            raise SceneStateError(
                f"expected format {cls.FORMAT!r}, got {data.get('format')!r}"
            )

        objects: list[ObjectFact] = []
        relations: list[RelationFact] = []

        for obj in data.get("objects", []):
            name = obj["name"]
            location = obj.get("location")
            pose = Pose.from_dict(obj.get("pose")) if obj.get("pose") else None

            objects.append(
                ObjectFact(
                    name=name,
                    type="item",
                    pose=pose,
                    location=location,
                    clear=True,
                    reachable=True,
                    source="oracle",
                    confidence=1.0,
                )
            )
            if location:
                relations.append(
                    RelationFact(
                        predicate="on",
                        args=[name, location],
                        source="oracle",
                        confidence=1.0,
                    )
                )

        locations = [
            LocationFact(
                name=loc["name"],
                type=loc.get("type", "location"),
                reachable=loc.get("reachable", True),
                open=loc.get("open"),
                source="oracle",
                confidence=1.0,
            )
            for loc in data.get("locations", [])
        ]

        gripper_empty = bool(data.get("gripper_empty", True))
        scene = SceneState(
            schema_version="1.0",
            frame_id=data.get("frame_id", "panda_link0"),
            domain_template=data.get("domain_template"),
            objects=objects,
            locations=locations,
            relations=relations,
            robot=RobotFacts(
                gripper_empty=gripper_empty,
                holding=None if gripper_empty else data.get("holding"),
                camera_aimed_at=data.get("camera_aimed_at"),
                source="oracle",
                confidence=1.0,
            ),
            meta=Meta(sources_used=["oracle"]),
        )
        scene.validate()
        return scene


class DinoMockAdapter:
    """Convert DINO-style noisy detection fixtures → SceneState."""

    FORMAT = "dino_mock_v1"

    @classmethod
    def load(cls, path: str | Path | None = None) -> SceneState:
        data = _load_fixture(path or _MOCK_DIR / "dino_noisy_detections.json")
        if data.get("format") != cls.FORMAT:
            raise SceneStateError(
                f"expected format {cls.FORMAT!r}, got {data.get('format')!r}"
            )

        known_locations = set(data.get("known_locations", []))
        objects: list[ObjectFact] = []
        relations: list[RelationFact] = []
        location_names: set[str] = set(known_locations)

        for det in data.get("detections", []):
            label = det["label"]
            confidence = float(det["confidence"])
            name = _normalize_dino_name(label)
            on_surface = det.get("on_surface")
            pose = Pose.from_dict(det.get("pose")) if det.get("pose") else None

            if on_surface is None and name in known_locations:
                location_names.add(name)
                continue

            objects.append(
                ObjectFact(
                    name=name,
                    type="item",
                    pose=pose,
                    location=on_surface,
                    clear=True,
                    reachable=True,
                    source="dino",
                    confidence=confidence,
                )
            )
            if on_surface:
                location_names.add(on_surface)
                relations.append(
                    RelationFact(
                        predicate="on",
                        args=[name, on_surface],
                        source="dino",
                        confidence=confidence,
                    )
                )

        locations = [
            LocationFact(
                name=loc_name,
                type="location",
                reachable=True,
                source="dino" if loc_name not in known_locations else "mock",
                confidence=0.9 if loc_name in known_locations else 0.65,
            )
            for loc_name in sorted(location_names)
        ]

        scene = SceneState(
            schema_version="1.0",
            frame_id=data.get("frame_id", "panda_link0"),
            domain_template=data.get("domain_template"),
            objects=objects,
            locations=locations,
            relations=relations,
            robot=RobotFacts(
                gripper_empty=True,
                holding=None,
                camera_aimed_at=None,
                source="dino",
                confidence=0.9,
            ),
            meta=Meta(sources_used=["dino"]),
        )
        scene.validate()
        return scene


class TrackerMockAdapter:
    """Convert mid-task tracker mock fixtures → SceneState."""

    FORMAT = "tracker_mock_v1"

    @classmethod
    def load(cls, path: str | Path | None = None) -> SceneState:
        data = _load_fixture(path or _MOCK_DIR / "holding_mid_task.json")
        if data.get("format") != cls.FORMAT:
            raise SceneStateError(
                f"expected format {cls.FORMAT!r}, got {data.get('format')!r}"
            )

        holding = data.get("holding")
        gripper_empty = bool(data.get("gripper_empty", holding is None))

        objects = [
            ObjectFact(
                name=obj["name"],
                type=obj.get("type", "item"),
                clear=obj.get("clear"),
                reachable=obj.get("reachable"),
                source="tracker",
                confidence=1.0,
            )
            for obj in data.get("objects", [])
        ]

        locations = [
            LocationFact(
                name=loc["name"],
                type=loc.get("type", "location"),
                reachable=loc.get("reachable", True),
                open=loc.get("open"),
                source="mock",
                confidence=1.0,
            )
            for loc in data.get("locations", [])
        ]

        relations: list[RelationFact] = []
        for rel in data.get("relations", []):
            relations.append(RelationFact.from_dict(rel))

        scene = SceneState(
            schema_version="1.0",
            frame_id=data.get("frame_id", "panda_link0"),
            domain_template=data.get("domain_template"),
            objects=objects,
            locations=locations,
            relations=relations,
            robot=RobotFacts(
                gripper_empty=gripper_empty,
                holding=holding,
                camera_aimed_at=data.get("camera_aimed_at"),
                source="tracker",
                confidence=1.0,
            ),
            meta=Meta(sources_used=["tracker", "mock"]),
        )
        scene.validate()
        return scene
