"""
Canonical SceneState schema for hybrid problem generation.

See ``docs/hybrid_problem_generator_design.md`` §2 for the JSON contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SourceTag = str

VALID_SOURCES: frozenset[str] = frozenset(
    {"oracle", "dino", "tracker", "vlm", "fusion", "mock", "manual"}
)

SCHEMA_VERSION = "1.0"


class SceneStateError(ValueError):
    """Invalid SceneState payload or invariant violation."""


@dataclass
class Position:
    x: float
    y: float
    z: float

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Position | None:
        if data is None:
            return None
        return cls(x=float(data["x"]), y=float(data["y"]), z=float(data["z"]))

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}


@dataclass
class Orientation:
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Orientation | None:
        if data is None:
            return None
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            z=float(data["z"]),
            w=float(data["w"]),
        )

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z, "w": self.w}


@dataclass
class Pose:
    position: Position
    orientation: Orientation

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Pose | None:
        if data is None:
            return None
        position = Position.from_dict(data.get("position"))
        orientation = Orientation.from_dict(data.get("orientation"))
        if position is None or orientation is None:
            raise SceneStateError("pose requires position and orientation")
        return cls(position=position, orientation=orientation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "orientation": self.orientation.to_dict(),
        }


@dataclass
class ObjectFact:
    name: str
    source: SourceTag
    confidence: float
    type: str = "item"
    pose: Pose | None = None
    location: str | None = None
    clear: bool | None = None
    reachable: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObjectFact:
        return cls(
            name=data["name"],
            type=data.get("type", "item"),
            pose=Pose.from_dict(data.get("pose")),
            location=data.get("location"),
            clear=data.get("clear"),
            reachable=data.get("reachable"),
            source=data["source"],
            confidence=float(data["confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "source": self.source,
            "confidence": self.confidence,
        }
        if self.pose is not None:
            out["pose"] = self.pose.to_dict()
        if self.location is not None:
            out["location"] = self.location
        if self.clear is not None:
            out["clear"] = self.clear
        if self.reachable is not None:
            out["reachable"] = self.reachable
        return out


@dataclass
class LocationFact:
    name: str
    source: SourceTag
    confidence: float
    type: str = "location"
    reachable: bool | None = None
    open: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocationFact:
        return cls(
            name=data["name"],
            type=data.get("type", "location"),
            reachable=data.get("reachable"),
            open=data.get("open"),
            source=data["source"],
            confidence=float(data["confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "source": self.source,
            "confidence": self.confidence,
        }
        if self.reachable is not None:
            out["reachable"] = self.reachable
        if self.open is not None:
            out["open"] = self.open
        return out


@dataclass
class RelationFact:
    predicate: str
    args: list[str]
    source: SourceTag
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationFact:
        return cls(
            predicate=data["predicate"],
            args=list(data["args"]),
            source=data["source"],
            confidence=float(data["confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate,
            "args": list(self.args),
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class RobotFacts:
    gripper_empty: bool
    holding: str | None
    camera_aimed_at: str | None
    source: SourceTag
    confidence: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotFacts:
        return cls(
            gripper_empty=bool(data["gripper_empty"]),
            holding=data.get("holding"),
            camera_aimed_at=data.get("camera_aimed_at"),
            source=data["source"],
            confidence=float(data["confidence"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gripper_empty": self.gripper_empty,
            "holding": self.holding,
            "camera_aimed_at": self.camera_aimed_at,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class Meta:
    sources_used: list[str] = field(default_factory=list)
    fusion_notes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Meta:
        if not data:
            return cls()
        return cls(
            sources_used=list(data.get("sources_used", [])),
            fusion_notes=list(data.get("fusion_notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sources_used": list(self.sources_used),
            "fusion_notes": list(self.fusion_notes),
        }


@dataclass
class SceneState:
    objects: list[ObjectFact]
    locations: list[LocationFact]
    relations: list[RelationFact]
    robot: RobotFacts
    schema_version: str = SCHEMA_VERSION
    timestamp: str | None = None
    frame_id: str = "panda_link0"
    domain_template: str | None = None
    meta: Meta = field(default_factory=Meta)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneState:
        if "schema_version" not in data:
            raise SceneStateError("schema_version is required")
        if "objects" not in data:
            raise SceneStateError("objects is required")
        if "locations" not in data:
            raise SceneStateError("locations is required")
        if "relations" not in data:
            raise SceneStateError("relations is required")
        if "robot" not in data:
            raise SceneStateError("robot is required")

        scene = cls(
            schema_version=data["schema_version"],
            timestamp=data.get("timestamp"),
            frame_id=data.get("frame_id", "panda_link0"),
            domain_template=data.get("domain_template"),
            objects=[ObjectFact.from_dict(o) for o in data["objects"]],
            locations=[LocationFact.from_dict(loc) for loc in data["locations"]],
            relations=[RelationFact.from_dict(r) for r in data["relations"]],
            robot=RobotFacts.from_dict(data["robot"]),
            meta=Meta.from_dict(data.get("meta")),
        )
        scene.validate()
        return scene

    @classmethod
    def from_json(cls, text: str) -> SceneState:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: str | Path) -> SceneState:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "objects": [o.to_dict() for o in self.objects],
            "locations": [loc.to_dict() for loc in self.locations],
            "relations": [r.to_dict() for r in self.relations],
            "robot": self.robot.to_dict(),
        }
        if self.timestamp is not None:
            out["timestamp"] = self.timestamp
        if self.domain_template is not None:
            out["domain_template"] = self.domain_template
        if self.meta.sources_used or self.meta.fusion_notes:
            out["meta"] = self.meta.to_dict()
        return out

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path, *, indent: int | None = 2) -> None:
        Path(path).write_text(self.to_json(indent=indent), encoding="utf-8")

    def validate(self) -> None:
        """Enforce schema invariants."""
        if self.schema_version != SCHEMA_VERSION:
            raise SceneStateError(
                f"unsupported schema_version: {self.schema_version!r}"
            )

        for obj in self.objects:
            _validate_source(obj.source, "object", obj.name)
            _validate_confidence(obj.confidence, "object", obj.name)

        for loc in self.locations:
            _validate_source(loc.source, "location", loc.name)
            _validate_confidence(loc.confidence, "location", loc.name)

        for rel in self.relations:
            _validate_source(rel.source, "relation", rel.predicate)
            _validate_confidence(rel.confidence, "relation", rel.predicate)

        _validate_source(self.robot.source, "robot", "robot")
        _validate_confidence(self.robot.confidence, "robot", "robot")

        holding = self.robot.holding
        if holding is None and not self.robot.gripper_empty:
            raise SceneStateError(
                "robot invariant violated: holding is null but gripper_empty is false"
            )
        if holding is not None and self.robot.gripper_empty:
            raise SceneStateError(
                "robot invariant violated: holding is set but gripper_empty is true"
            )


def _validate_source(source: str, kind: str, name: str) -> None:
    if source not in VALID_SOURCES:
        raise SceneStateError(f"invalid source {source!r} on {kind} {name!r}")


def _validate_confidence(confidence: float, kind: str, name: str) -> None:
    if not 0.0 <= confidence <= 1.0:
        raise SceneStateError(
            f"confidence {confidence} out of range [0,1] on {kind} {name!r}"
        )
