"""
Production oracle adapter: ``WorldState`` → canonical ``SceneState``.

Wraps ``simulation.oracle.world_state.WorldState`` without modifying the oracle
module.  Offline tests load frozen snapshots from ``fixtures/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simulation.oracle.world_state import (
    ObjectState,
    Orientation as OracleOrientation,
    Pose as OraclePose,
    Position as OraclePosition,
    WorldState,
)

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

_DEFAULT_CONFIDENCE = 1.0
_DEFAULT_FRAME = "panda_link0"


def _oracle_pose_to_schema(oracle_pose: OraclePose) -> Pose:
    p = oracle_pose.position
    q = oracle_pose.orientation
    return Pose(
        position=Position(x=p.x, y=p.y, z=p.z),
        orientation=Orientation(x=q.x, y=q.y, z=q.z, w=q.w),
    )


def _world_state_from_fixture(data: dict[str, Any]) -> WorldState:
    objects: list[ObjectState] = []
    for obj in data.get("objects", []):
        pose_data = obj.get("pose")
        if pose_data is None:
            raise SceneStateError(f"object {obj.get('name')!r} missing pose")
        pos = pose_data["position"]
        ori = pose_data["orientation"]
        pose = OraclePose(
            position=OraclePosition(x=pos["x"], y=pos["y"], z=pos["z"]),
            orientation=OracleOrientation(
                x=ori["x"], y=ori["y"], z=ori["z"], w=ori["w"]
            ),
        )
        objects.append(
            ObjectState(
                name=obj["name"],
                pose=pose,
                location=obj.get("location") or "",
            )
        )
    return WorldState(
        objects=objects,
        gripper_empty=bool(data.get("gripper_empty", True)),
    )


class OracleAdapter:
    """Convert Gazebo oracle ``WorldState`` snapshots → ``SceneState``."""

    FORMAT = "world_state_v1"

    @classmethod
    def from_world_state(
        cls,
        world_state: WorldState,
        *,
        locations: list[LocationFact] | list[dict[str, Any]] | None = None,
        known_locations: list[str] | None = None,
        frame_id: str = _DEFAULT_FRAME,
        domain_template: str | None = None,
        holding: str | None = None,
        camera_aimed_at: str | None = None,
        confidence: float = _DEFAULT_CONFIDENCE,
    ) -> SceneState:
        """
        Map a live or reconstructed ``WorldState`` to ``SceneState``.

        ``WorldState`` from ``GazeboOracle.get_world_state()`` provides poses
        only; pass ``locations`` / ``known_locations`` when symbolic placement
        is known from the task or planner context.
        """
        location_facts = cls._build_locations(
            locations=locations,
            known_locations=known_locations,
            object_states=world_state.objects,
            confidence=confidence,
        )

        objects: list[ObjectFact] = []
        relations: list[RelationFact] = []

        for obj in world_state.objects:
            location = obj.location or None
            objects.append(
                ObjectFact(
                    name=obj.name,
                    type="item",
                    pose=_oracle_pose_to_schema(obj.pose),
                    location=location,
                    clear=True if location else None,
                    reachable=True,
                    source="oracle",
                    confidence=confidence,
                )
            )
            if location:
                relations.append(
                    RelationFact(
                        predicate="on",
                        args=[obj.name, location],
                        source="oracle",
                        confidence=confidence,
                    )
                )

        gripper_empty = world_state.gripper_empty if holding is None else False
        if holding is not None:
            gripper_empty = False

        scene = SceneState(
            schema_version="1.0",
            frame_id=frame_id,
            domain_template=domain_template,
            objects=objects,
            locations=location_facts,
            relations=relations,
            robot=RobotFacts(
                gripper_empty=gripper_empty,
                holding=holding,
                camera_aimed_at=camera_aimed_at,
                source="oracle",
                confidence=confidence,
            ),
            meta=Meta(sources_used=["oracle"]),
        )
        scene.validate()
        return scene

    @classmethod
    def load(cls, path: str | Path | None = None) -> SceneState:
        """Load a frozen ``world_state_v1`` fixture (no Gazebo required)."""
        fixture_path = Path(path) if path else _FIXTURES_DIR / "oracle_world_state.json"
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        if data.get("format") != cls.FORMAT:
            raise SceneStateError(
                f"expected format {cls.FORMAT!r}, got {data.get('format')!r}"
            )

        world_state = _world_state_from_fixture(data)
        locations = data.get("locations")
        return cls.from_world_state(
            world_state,
            locations=locations,
            known_locations=data.get("known_locations"),
            frame_id=data.get("frame_id", _DEFAULT_FRAME),
            domain_template=data.get("domain_template"),
            holding=data.get("holding"),
            camera_aimed_at=data.get("camera_aimed_at"),
        )

    @staticmethod
    def _build_locations(
        *,
        locations: list[LocationFact] | list[dict[str, Any]] | None,
        known_locations: list[str] | None,
        object_states: list[ObjectState],
        confidence: float,
    ) -> list[LocationFact]:
        if locations:
            out: list[LocationFact] = []
            for loc in locations:
                if isinstance(loc, LocationFact):
                    out.append(loc)
                else:
                    out.append(
                        LocationFact(
                            name=loc["name"],
                            type=loc.get("type", "location"),
                            reachable=loc.get("reachable"),
                            open=loc.get("open"),
                            source="oracle",
                            confidence=confidence,
                        )
                    )
            return out

        names: set[str] = set(known_locations or [])
        for obj in object_states:
            if obj.location:
                names.add(obj.location)

        return [
            LocationFact(
                name=name,
                type="location",
                reachable=True,
                source="oracle",
                confidence=confidence,
            )
            for name in sorted(names)
        ]
