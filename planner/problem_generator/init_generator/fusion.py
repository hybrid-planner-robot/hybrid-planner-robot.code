"""
Deterministic fusion of perception sources + StateTracker into SceneState.

Precedence (Session 7):
  1. Tracker wins for robot facts and recently moved object relations.
  2. Oracle preferred over DINO for poses / ``on`` in sim-like runs.
  3. Otherwise higher confidence; tie-break oracle > dino > vlm.
  4. Decisions recorded in ``meta.fusion_notes``.

See ``docs/hybrid_problem_generator_design.md`` §3.2.2.
"""

from __future__ import annotations

from typing import Any

from planner.state_tracker import StateTracker

from .schema import (
    LocationFact,
    Meta,
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)

_FUSION_SOURCE = "fusion"
_SUBJECT_RELATIONS = frozenset({"on", "stacked-on", "in-container", "holding", "camera-aimed-at"})
_SOURCE_RANK: dict[str, int] = {
    "oracle": 4,
    "dino": 3,
    "vlm": 2,
    "tracker": 5,
    "fusion": 1,
    "mock": 2,
    "manual": 2,
}


def relation_subject_key(rel: RelationFact) -> tuple[str, str]:
    """Key for relation conflicts on the primary manipulated object."""
    if rel.predicate in _SUBJECT_RELATIONS and rel.args:
        return (rel.predicate, rel.args[0])
    return (rel.predicate, "|".join(rel.args))


# Back-compat alias used inside this module.
_relation_subject_key = relation_subject_key


def _source_rank(source: str) -> int:
    return _SOURCE_RANK.get(source, 0)


def _prefer_relation(
    current: RelationFact,
    candidate: RelationFact,
    *,
    sim_like: bool,
) -> RelationFact:
    """Pick the winning relation fact between two candidates."""
    if sim_like:
        if current.source == "oracle" and candidate.source == "dino":
            return current
        if current.source == "dino" and candidate.source == "oracle":
            return candidate

    if candidate.confidence > current.confidence:
        return candidate
    if candidate.confidence < current.confidence:
        return current

    if _source_rank(candidate.source) > _source_rank(current.source):
        return candidate
    return current


def _prefer_object(
    current: ObjectFact,
    candidate: ObjectFact,
    *,
    sim_like: bool,
) -> ObjectFact:
    if sim_like and current.source == "oracle" and candidate.source == "dino":
        return current
    if sim_like and current.source == "dino" and candidate.source == "oracle":
        return candidate

    if candidate.confidence > current.confidence:
        return candidate
    if candidate.confidence < current.confidence:
        return current

    if _source_rank(candidate.source) > _source_rank(current.source):
        return candidate
    return current


def _prefer_location(
    current: LocationFact,
    candidate: LocationFact,
    *,
    sim_like: bool,
) -> LocationFact:
    if sim_like and current.source == "oracle" and candidate.source == "dino":
        return current
    if sim_like and current.source == "dino" and candidate.source == "oracle":
        return candidate

    if candidate.confidence > current.confidence:
        return candidate
    if candidate.confidence < current.confidence:
        return current

    if _source_rank(candidate.source) > _source_rank(current.source):
        return candidate
    return current


def _extract_tracker(
    tracker: StateTracker | SceneState | dict[str, Any] | None,
) -> tuple[RobotFacts | None, list[RelationFact]]:
    if tracker is None:
        return None, []

    if isinstance(tracker, StateTracker):
        partial = tracker.as_partial_scene()
        robot = RobotFacts.from_dict(partial["robot"])
        relations = [RelationFact.from_dict(r) for r in partial["relations"]]
        return robot, relations

    if isinstance(tracker, SceneState):
        return tracker.robot, list(tracker.relations)

    if isinstance(tracker, dict):
        robot_data = tracker.get("robot")
        robot = RobotFacts.from_dict(robot_data) if robot_data else None
        relations = [
            RelationFact.from_dict(r) for r in tracker.get("relations", [])
        ]
        return robot, relations

    return None, []


def _default_robot() -> RobotFacts:
    return RobotFacts(
        gripper_empty=True,
        holding=None,
        camera_aimed_at=None,
        source=_FUSION_SOURCE,
        confidence=1.0,
    )


def _merge_robot(
    *,
    tracker_robot: RobotFacts | None,
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
) -> RobotFacts:
    if tracker_robot is not None:
        return RobotFacts(
            gripper_empty=tracker_robot.gripper_empty,
            holding=tracker_robot.holding,
            camera_aimed_at=tracker_robot.camera_aimed_at,
            source=tracker_robot.source,
            confidence=tracker_robot.confidence,
        )

    if oracle_scene is not None:
        return RobotFacts(
            gripper_empty=oracle_scene.robot.gripper_empty,
            holding=oracle_scene.robot.holding,
            camera_aimed_at=oracle_scene.robot.camera_aimed_at,
            source=_FUSION_SOURCE,
            confidence=oracle_scene.robot.confidence,
        )

    if dino_scene is not None:
        return RobotFacts(
            gripper_empty=dino_scene.robot.gripper_empty,
            holding=dino_scene.robot.holding,
            camera_aimed_at=dino_scene.robot.camera_aimed_at,
            source=_FUSION_SOURCE,
            confidence=dino_scene.robot.confidence,
        )

    return _default_robot()


def _detect_name_mismatches(
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
    notes: list[str],
) -> None:
    if oracle_scene is None or dino_scene is None:
        return

    oracle_names = {o.name for o in oracle_scene.objects}
    dino_names = {o.name for o in dino_scene.objects}
    only_dino = dino_names - oracle_names
    only_oracle = oracle_names - dino_names

    for name in sorted(only_dino):
        loose = name.replace("_", "")
        alias_hits = [n for n in only_oracle if n.replace("_", "") == loose]
        if alias_hits:
            notes.append(
                f"name mismatch: dino {name!r} vs oracle {alias_hits[0]!r} (not merged)"
            )
        else:
            notes.append(f"name mismatch: dino-only detection {name!r}")

    for name in sorted(only_oracle - dino_names):
        if not any(
            n.replace("_", "") == name.replace("_", "") for n in only_dino
        ):
            notes.append(f"name mismatch: oracle-only object {name!r}")


def _merge_objects(
    scenes: list[SceneState],
    *,
    sim_like: bool,
    notes: list[str],
) -> list[ObjectFact]:
    by_name: dict[str, ObjectFact] = {}

    for scene in scenes:
        for obj in scene.objects:
            if obj.name not in by_name:
                by_name[obj.name] = obj
                continue
            prev = by_name[obj.name]
            chosen = _prefer_object(prev, obj, sim_like=sim_like)
            if chosen is prev and obj is not prev:
                notes.append(
                    f"object {obj.name!r}: kept {prev.source} over {obj.source}"
                )
            elif chosen is obj and obj is not prev:
                notes.append(
                    f"object {obj.name!r}: kept {obj.source} over {prev.source}"
                )
            by_name[obj.name] = chosen

    return sorted(by_name.values(), key=lambda o: o.name)


def _merge_locations(
    scenes: list[SceneState],
    *,
    sim_like: bool,
) -> list[LocationFact]:
    by_name: dict[str, LocationFact] = {}
    for scene in scenes:
        for loc in scene.locations:
            if loc.name not in by_name:
                by_name[loc.name] = loc
            else:
                by_name[loc.name] = _prefer_location(
                    by_name[loc.name], loc, sim_like=sim_like
                )
    return sorted(by_name.values(), key=lambda loc: loc.name)


def _merge_relations(
    scenes: list[SceneState],
    tracker_relations: list[RelationFact],
    *,
    sim_like: bool,
    holding: str | None,
    notes: list[str],
) -> list[RelationFact]:
    merged: dict[tuple[str, str], RelationFact] = {}

    for scene in scenes:
        for rel in scene.relations:
            key = _relation_subject_key(rel)
            if key not in merged:
                merged[key] = rel
            else:
                merged[key] = _prefer_relation(
                    merged[key], rel, sim_like=sim_like
                )

    for rel in tracker_relations:
        key = _relation_subject_key(rel)
        if key in merged:
            notes.append(
                f"tracker override: {rel.predicate} {rel.args} "
                f"replaces {merged[key].source}"
            )
        merged[key] = RelationFact(
            predicate=rel.predicate,
            args=list(rel.args),
            source="tracker",
            confidence=rel.confidence,
        )

    if holding:
        on_key = ("on", holding)
        if on_key in merged:
            notes.append(f"removed {on_key} because tracker holds {holding!r}")
            del merged[on_key]

    return sorted(
        merged.values(),
        key=lambda r: (_relation_subject_key(r), r.predicate, tuple(r.args)),
    )


def _collect_scenes(
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
) -> list[SceneState]:
    scenes: list[SceneState] = []
    if oracle_scene is not None:
        scenes.append(oracle_scene)
    if dino_scene is not None:
        scenes.append(dino_scene)
    return scenes


def _perception_scenes(
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
) -> list[SceneState]:
    """Oracle + DINO scenes used for relation fusion (not tracker SceneState)."""
    return _collect_scenes(oracle_scene, dino_scene)


def _entity_scenes(
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
    tracker: StateTracker | SceneState | dict[str, Any] | None,
) -> list[SceneState]:
    scenes = _collect_scenes(oracle_scene, dino_scene)
    if isinstance(tracker, SceneState):
        scenes.append(tracker)
    return scenes


def _sources_used(
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
    tracker_robot: RobotFacts | None,
    tracker_relations: list[RelationFact],
    vlm_patch: SceneState | None,
) -> list[str]:
    used: list[str] = []
    if oracle_scene is not None:
        used.append("oracle")
    if dino_scene is not None:
        used.append("dino")
    if tracker_robot is not None or tracker_relations:
        used.append("tracker")
    if vlm_patch is not None:
        used.append("vlm")
    used.append("fusion")
    return used


def _has_deterministic_source(
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
    tracker_robot: RobotFacts | None,
    tracker_relations: list[RelationFact],
) -> bool:
    return (
        oracle_scene is not None
        or dino_scene is not None
        or tracker_robot is not None
        or bool(tracker_relations)
    )


def _vlm_patch_intends_robot(vlm_patch: SceneState) -> bool:
    """True when the patch explicitly corrects robot facts (not a default empty)."""
    if "robot_patch" in vlm_patch.meta.fusion_notes:
        return True
    robot = vlm_patch.robot
    return (
        robot.holding is not None
        or robot.camera_aimed_at is not None
        or not robot.gripper_empty
    )


def _apply_vlm_patch(
    *,
    objects: list[ObjectFact],
    locations: list[LocationFact],
    relations: list[RelationFact],
    robot: RobotFacts,
    vlm_patch: SceneState,
    notes: list[str],
) -> tuple[list[ObjectFact], list[LocationFact], list[RelationFact], RobotFacts]:
    """
    Merge a VLM correction patch into an already-fused deterministic scene.

    Relations from the patch win by subject key. A relation with
    ``confidence == 0.0`` removes that subject key (stale-fact clear).
    Robot facts apply only when the patch intends a robot correction and the
    current robot is not tracker-owned (tracker still wins).
    """
    notes.append("vlm_patch applied")

    by_name: dict[str, ObjectFact] = {o.name: o for o in objects}
    for obj in vlm_patch.objects:
        tagged = ObjectFact(
            name=obj.name,
            type=obj.type,
            pose=obj.pose,
            location=obj.location,
            clear=obj.clear,
            reachable=obj.reachable,
            source="vlm",
            confidence=obj.confidence,
        )
        if obj.name not in by_name:
            by_name[obj.name] = tagged
            notes.append(f"vlm_patch added object {obj.name!r}")
        else:
            by_name[obj.name] = tagged
            notes.append(f"vlm_patch updated object {obj.name!r}")
    objects_out = sorted(by_name.values(), key=lambda o: o.name)

    by_loc: dict[str, LocationFact] = {loc.name: loc for loc in locations}
    for loc in vlm_patch.locations:
        by_loc[loc.name] = LocationFact(
            name=loc.name,
            type=loc.type,
            reachable=loc.reachable,
            open=loc.open,
            source="vlm",
            confidence=loc.confidence,
        )
    locations_out = sorted(by_loc.values(), key=lambda loc: loc.name)

    merged_rel: dict[tuple[str, str], RelationFact] = {
        _relation_subject_key(rel): rel for rel in relations
    }
    for rel in vlm_patch.relations:
        key = _relation_subject_key(rel)
        if rel.confidence <= 0.0:
            if key in merged_rel:
                notes.append(
                    f"vlm_patch removed relation {rel.predicate} {list(rel.args)}"
                )
                del merged_rel[key]
            continue
        if key in merged_rel:
            notes.append(
                f"vlm_patch relation {rel.predicate} {rel.args} "
                f"replaces {merged_rel[key].source}"
            )
        else:
            notes.append(f"vlm_patch relation {rel.predicate} {rel.args}")
        merged_rel[key] = RelationFact(
            predicate=rel.predicate,
            args=list(rel.args),
            source="vlm",
            confidence=rel.confidence,
        )
    relations_out = sorted(
        merged_rel.values(),
        key=lambda r: (_relation_subject_key(r), r.predicate, tuple(r.args)),
    )

    robot_out = robot
    if not _vlm_patch_intends_robot(vlm_patch):
        pass
    elif robot.source == "tracker":
        notes.append("vlm_patch robot ignored (tracker wins)")
    else:
        robot_out = RobotFacts(
            gripper_empty=vlm_patch.robot.gripper_empty,
            holding=vlm_patch.robot.holding,
            camera_aimed_at=vlm_patch.robot.camera_aimed_at,
            source="vlm",
            confidence=vlm_patch.robot.confidence,
        )
        notes.append("vlm_patch robot applied")

    return objects_out, locations_out, relations_out, robot_out


def apply_vlm_patch_to_scene(
    base: SceneState,
    vlm_patch: SceneState,
) -> SceneState:
    """
    Apply a VLM patch onto an already-fused SceneState (YELLOW correction).

    Does not require re-running full oracle/DINO fusion. Tracker-owned robot
    facts on ``base`` are preserved.
    """
    notes = list(base.meta.fusion_notes)
    objects, locations, relations, robot = _apply_vlm_patch(
        objects=list(base.objects),
        locations=list(base.locations),
        relations=list(base.relations),
        robot=base.robot,
        vlm_patch=vlm_patch,
        notes=notes,
    )
    sources = list(base.meta.sources_used)
    if "vlm" not in sources:
        # Keep fusion last if present.
        if sources and sources[-1] == "fusion":
            sources = sources[:-1] + ["vlm", "fusion"]
        else:
            sources.append("vlm")

    scene = SceneState(
        schema_version=base.schema_version,
        timestamp=base.timestamp,
        frame_id=base.frame_id,
        domain_template=base.domain_template or vlm_patch.domain_template,
        objects=objects,
        locations=locations,
        relations=relations,
        robot=robot,
        meta=Meta(sources_used=sources, fusion_notes=notes),
    )
    scene.validate()
    return scene


class FusionEngine:
    """Fuse oracle / DINO / tracker (and optional VLM patch) into SceneState."""

    def merge(
        self,
        *,
        oracle_scene: SceneState | None = None,
        dino_scene: SceneState | None = None,
        tracker: StateTracker | SceneState | dict[str, Any] | None = None,
        vlm_patch: SceneState | None = None,
        sim_like: bool = True,
    ) -> SceneState:
        """
        Merge inputs into one canonical SceneState.

        Requires at least one deterministic source (oracle, DINO, or tracker).
        ``vlm_patch`` may only *patch* an already-fused scene — never sole init.
        """
        tracker_robot, tracker_relations = _extract_tracker(tracker)
        notes: list[str] = []
        _detect_name_mismatches(oracle_scene, dino_scene, notes)

        has_deterministic = _has_deterministic_source(
            oracle_scene, dino_scene, tracker_robot, tracker_relations
        )
        if not has_deterministic:
            if vlm_patch is not None:
                raise ValueError(
                    "vlm_patch cannot be the sole init source; "
                    "provide oracle, dino, and/or tracker"
                )
            raise ValueError("merge requires at least one scene source or tracker")

        robot = _merge_robot(
            tracker_robot=tracker_robot,
            oracle_scene=oracle_scene,
            dino_scene=dino_scene,
        )

        entity_scenes = _entity_scenes(oracle_scene, dino_scene, tracker)
        objects = _merge_objects(entity_scenes, sim_like=sim_like, notes=notes)
        locations = _merge_locations(entity_scenes, sim_like=sim_like)
        relations = _merge_relations(
            _perception_scenes(oracle_scene, dino_scene),
            tracker_relations,
            sim_like=sim_like,
            holding=robot.holding,
            notes=notes,
        )

        frame_id = "panda_link0"
        domain_template: str | None = None
        for scene in entity_scenes:
            frame_id = scene.frame_id
            if scene.domain_template:
                domain_template = scene.domain_template
                break

        if vlm_patch is not None:
            objects, locations, relations, robot = _apply_vlm_patch(
                objects=objects,
                locations=locations,
                relations=relations,
                robot=robot,
                vlm_patch=vlm_patch,
                notes=notes,
            )
            if vlm_patch.domain_template and not domain_template:
                domain_template = vlm_patch.domain_template

        scene = SceneState(
            schema_version="1.0",
            frame_id=frame_id,
            domain_template=domain_template,
            objects=objects,
            locations=locations,
            relations=relations,
            robot=robot,
            meta=Meta(
                sources_used=_sources_used(
                    oracle_scene,
                    dino_scene,
                    tracker_robot,
                    tracker_relations,
                    vlm_patch,
                ),
                fusion_notes=notes,
            ),
        )
        scene.validate()
        return scene
