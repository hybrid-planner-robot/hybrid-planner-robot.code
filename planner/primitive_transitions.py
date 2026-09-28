"""
Shared symbolic effects for standard manipulation primitives.

Single source of truth for ``StateTracker.apply`` and ``StateVerifier.expect``.
See ``docs/hybrid_problem_generator_design.md`` §3.2.4 and §4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from planner.enrichment_effects import EnrichmentContext
    from planner.problem_generator.init_generator.schema import RelationFact, RobotFacts

PICK_PRIMITIVES = frozenset({"pick", "unstack", "pick-from-container"})
PLACE_ON_PRIMITIVES = frozenset({"place"})
STACK_PRIMITIVES = frozenset({"stack"})
PLACE_IN_PRIMITIVES = frozenset({"place-in-container"})
LOOK_PRIMITIVES = frozenset({"look-at", "look_at"})

# Catalog skills that only exist in a domain after online enrichment
# (Sessions 28–30). They have ROS bodies, so their manipulation effects are
# modelled here; the symbolic fluent each one asserts is task-specific and
# comes from ``EnrichmentContext`` (see planner/enrichment_effects.py).
CATALOG_ENRICHMENT_PRIMITIVES = frozenset({"pour", "tilt", "stir", "cut"})

KNOWN_PRIMITIVES = (
    PICK_PRIMITIVES
    | PLACE_ON_PRIMITIVES
    | STACK_PRIMITIVES
    | PLACE_IN_PRIMITIVES
    | LOOK_PRIMITIVES
)

_RELATION_PREDICATES = frozenset({"on", "stacked-on", "in-container"})
_TRACKER_SOURCE = "tracker"
_EXPECTED_SOURCE = "fusion"
_DEFAULT_CONFIDENCE = 1.0


@dataclass(frozen=True)
class CompletedAction:
    """Lightweight completed primitive for tracker / verifier use."""

    primitive: str
    args: dict[str, Any] = field(default_factory=dict)
    success_flag: bool = True


@dataclass(frozen=True)
class MovedRecord:
    """Last known placement for an object moved by the tracker."""

    predicate: str  # "on" | "stacked-on" | "in-container"
    support: str


@dataclass(frozen=True)
class TrackerDelta:
    """Incremental update produced by one primitive for StateTracker."""

    robot: RobotFacts
    moved_set: dict[str, MovedRecord]
    moved_remove: frozenset[str]
    applied: bool
    # Enriched-action fluents to remember, e.g. ``("poured", "bottle", "glass")``.
    facts_add: tuple[tuple[str, ...], ...] = ()


def normalize_primitive(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def arg(args: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def has_local_model(
    primitive: str,
    enrichment: "EnrichmentContext | None" = None,
) -> bool:
    """
    True when this primitive's symbolic effect is known without asking a VLM.

    Enriched catalog actions qualify **only** through ``enrichment``: the fluent
    they assert is task-specific, so without that context (or for an open-vocab
    action with no ROS body) there is still no local model.
    """
    if normalize_primitive(primitive) in KNOWN_PRIMITIVES:
        return True
    if enrichment is None:
        return False
    return enrichment.is_executable(primitive)


def involved_objects(action: CompletedAction) -> set[str]:
    args = action.args or {}
    names: set[str] = set()
    for key in (
        "object",
        "top",
        "i",
        "target",
        "bot",
        "bottom",
        "on",
        "container",
        "location",
        # Enriched catalog actions (pour source→target, stir container, …).
        "source",
        "src",
        "dst",
        "tool",
    ):
        value = arg(args, key)
        if value:
            names.add(value)
    return names


def released_object(
    args: Mapping[str, Any],
    pre_robot: RobotFacts,
) -> str | None:
    """Object a releasing catalog skill (``pour``) lets go of."""
    return arg(args, "source", "src", "object", "i", "from") or pre_robot.holding


def relation_scope(
    action: CompletedAction,
    enrichment: "EnrichmentContext | None" = None,
) -> set[str]:
    """
    Objects whose placement the script can predict after ``action``.

    Everything ``involved_objects`` reports, minus the vessel a releasing
    catalog skill puts down: ``pour`` ends with an open gripper somewhere near
    the source pose, and the script has no honest expectation about the
    resulting ``(on …)`` fact. Scoring it would manufacture a mismatch on every
    successful pour, so that object is left to perception instead.
    """
    names = involved_objects(action)
    effect = enrichment.effect_for(action.primitive) if enrichment else None
    if effect is not None and effect.releases_object:
        released = arg(action.args or {}, "source", "src", "object", "i", "from")
        names.discard(released or "")
    return names


def apply_robot_transition(
    pre: RobotFacts,
    action: CompletedAction,
    *,
    enrichment: "EnrichmentContext | None" = None,
) -> RobotFacts:
    """Return post-action robot facts; unchanged when args are incomplete."""
    from planner.problem_generator.init_generator.schema import RobotFacts

    primitive = normalize_primitive(action.primitive)
    args = dict(action.args) if action.args else {}

    effect = enrichment.effect_for(action.primitive) if enrichment else None
    if effect is not None:
        if not effect.releases_object:
            return pre
        if not released_object(args, pre):
            return pre
        return RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=pre.camera_aimed_at,
            source=pre.source,
            confidence=pre.confidence,
        )

    if primitive in PICK_PRIMITIVES:
        obj = arg(args, "object", "top", "i")
        if not obj:
            return pre
        return RobotFacts(
            gripper_empty=False,
            holding=obj,
            camera_aimed_at=pre.camera_aimed_at,
            source=pre.source,
            confidence=pre.confidence,
        )

    if primitive in PLACE_ON_PRIMITIVES | STACK_PRIMITIVES | PLACE_IN_PRIMITIVES:
        obj = _resolve_place_object(args, pre, primitive)
        if not obj:
            return pre
        if primitive in PLACE_ON_PRIMITIVES and not arg(args, "location", "l"):
            return pre
        if primitive in STACK_PRIMITIVES and not arg(args, "bot", "bottom", "on"):
            return pre
        if primitive in PLACE_IN_PRIMITIVES and not arg(
            args, "container", "c", "location"
        ):
            return pre
        return RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=pre.camera_aimed_at,
            source=pre.source,
            confidence=pre.confidence,
        )

    if primitive in LOOK_PRIMITIVES:
        target = arg(args, "target", "object", "i")
        if not target:
            return pre
        return RobotFacts(
            gripper_empty=pre.gripper_empty,
            holding=pre.holding,
            camera_aimed_at=target,
            source=pre.source,
            confidence=pre.confidence,
        )

    return pre


def apply_relation_transition(
    pre_relations: list[RelationFact],
    action: CompletedAction,
    *,
    holding: str | None = None,
    source: str = _EXPECTED_SOURCE,
    confidence: float = _DEFAULT_CONFIDENCE,
    enrichment: "EnrichmentContext | None" = None,
) -> list[RelationFact]:
    """Return post-action relations for standard and enriched-catalog primitives."""
    from planner.problem_generator.init_generator.schema import RelationFact

    primitive = normalize_primitive(action.primitive)
    args = dict(action.args) if action.args else {}
    relations = list(pre_relations)

    effect = enrichment.effect_for(action.primitive) if enrichment else None
    if effect is not None:
        fact = enrichment.fact_for(action.primitive, args)
        if effect.releases_object:
            obj = released_object(args, _robot_from_holding(holding))
            if obj:
                # Where the released vessel ends up is a perception question,
                # so drop the stale placement instead of inventing a new one.
                relations = _remove_object_relations(relations, obj)
        if fact:
            relations.append(
                RelationFact(
                    predicate=fact[0],
                    args=list(fact[1:]),
                    source=source,
                    confidence=confidence,
                )
            )
        return relations

    if primitive in PICK_PRIMITIVES:
        obj = arg(args, "object", "top", "i")
        if not obj:
            return relations
        return _remove_object_relations(relations, obj)

    if primitive in PLACE_ON_PRIMITIVES:
        obj = _resolve_place_object(args, _robot_from_holding(holding), primitive)
        loc = arg(args, "location", "l")
        if not obj or not loc:
            return relations
        relations = _remove_object_relations(relations, obj)
        relations.append(
            RelationFact(
                predicate="on",
                args=[obj, loc],
                source=source,
                confidence=confidence,
            )
        )
        return relations

    if primitive in STACK_PRIMITIVES:
        obj = _resolve_place_object(args, _robot_from_holding(holding), primitive)
        bot = arg(args, "bot", "bottom", "on")
        if not obj or not bot:
            return relations
        relations = _remove_object_relations(relations, obj)
        relations.append(
            RelationFact(
                predicate="stacked-on",
                args=[obj, bot],
                source=source,
                confidence=confidence,
            )
        )
        return relations

    if primitive in PLACE_IN_PRIMITIVES:
        obj = _resolve_place_object(args, _robot_from_holding(holding), primitive)
        container = arg(args, "container", "c", "location")
        if not obj or not container:
            return relations
        relations = _remove_object_relations(relations, obj)
        relations.append(
            RelationFact(
                predicate="in-container",
                args=[obj, container],
                source=source,
                confidence=confidence,
            )
        )
        return relations

    return relations


def compute_tracker_delta(
    pre_robot: RobotFacts,
    moved: Mapping[str, MovedRecord],
    action: CompletedAction,
    *,
    enrichment: "EnrichmentContext | None" = None,
) -> TrackerDelta:
    """Compute StateTracker update for one successfully completed primitive."""
    primitive = normalize_primitive(action.primitive)
    args = dict(action.args) if action.args else {}

    effect = enrichment.effect_for(action.primitive) if enrichment else None
    if effect is not None:
        fact = enrichment.fact_for(action.primitive, args)
        robot = apply_robot_transition(pre_robot, action, enrichment=enrichment)
        released = (
            released_object(args, pre_robot) if effect.releases_object else None
        )
        if fact is None and released is None:
            return TrackerDelta(pre_robot, {}, frozenset(), applied=False)
        return TrackerDelta(
            robot=_as_tracker_robot(robot),
            moved_set={},
            moved_remove=frozenset({released} if released else ()),
            applied=True,
            facts_add=(fact,) if fact else (),
        )

    if primitive in PICK_PRIMITIVES:
        obj = arg(args, "object", "top", "i")
        if not obj:
            return TrackerDelta(pre_robot, {}, frozenset(), applied=False)
        robot = apply_robot_transition(pre_robot, action)
        return TrackerDelta(
            robot=_as_tracker_robot(robot),
            moved_set={},
            moved_remove=frozenset({obj}),
            applied=True,
        )

    if primitive in PLACE_ON_PRIMITIVES:
        obj = _resolve_place_object(args, pre_robot, primitive)
        loc = arg(args, "location", "l")
        if not obj or not loc:
            return TrackerDelta(pre_robot, {}, frozenset(), applied=False)
        robot = apply_robot_transition(pre_robot, action)
        return TrackerDelta(
            robot=_as_tracker_robot(robot),
            moved_set={obj: MovedRecord(predicate="on", support=loc)},
            moved_remove=frozenset(),
            applied=True,
        )

    if primitive in STACK_PRIMITIVES:
        obj = _resolve_place_object(args, pre_robot, primitive)
        bot = arg(args, "bot", "bottom", "on")
        if not obj or not bot:
            return TrackerDelta(pre_robot, {}, frozenset(), applied=False)
        robot = apply_robot_transition(pre_robot, action)
        return TrackerDelta(
            robot=_as_tracker_robot(robot),
            moved_set={obj: MovedRecord(predicate="stacked-on", support=bot)},
            moved_remove=frozenset(),
            applied=True,
        )

    if primitive in PLACE_IN_PRIMITIVES:
        obj = _resolve_place_object(args, pre_robot, primitive)
        container = arg(args, "container", "c", "location")
        if not obj or not container:
            return TrackerDelta(pre_robot, {}, frozenset(), applied=False)
        robot = apply_robot_transition(pre_robot, action)
        return TrackerDelta(
            robot=_as_tracker_robot(robot),
            moved_set={
                obj: MovedRecord(predicate="in-container", support=container)
            },
            moved_remove=frozenset(),
            applied=True,
        )

    if primitive in LOOK_PRIMITIVES:
        target = arg(args, "target", "object", "i")
        if not target:
            return TrackerDelta(pre_robot, {}, frozenset(), applied=False)
        robot = apply_robot_transition(pre_robot, action)
        return TrackerDelta(
            robot=_as_tracker_robot(robot),
            moved_set={},
            moved_remove=frozenset(),
            applied=True,
        )

    return TrackerDelta(pre_robot, {}, frozenset(), applied=False)


def _resolve_place_object(
    args: Mapping[str, Any],
    pre_robot: RobotFacts,
    primitive: str,
) -> str | None:
    if primitive in PLACE_ON_PRIMITIVES:
        return arg(args, "object", "i") or pre_robot.holding
    if primitive in STACK_PRIMITIVES:
        return arg(args, "object", "top", "i") or pre_robot.holding
    if primitive in PLACE_IN_PRIMITIVES:
        return arg(args, "object", "i") or pre_robot.holding
    return None


def _robot_from_holding(holding: str | None) -> RobotFacts:
    from planner.problem_generator.init_generator.schema import RobotFacts

    return RobotFacts(
        gripper_empty=holding is None,
        holding=holding,
        camera_aimed_at=None,
        source=_TRACKER_SOURCE,
        confidence=_DEFAULT_CONFIDENCE,
    )


def _as_tracker_robot(robot: RobotFacts) -> RobotFacts:
    from planner.problem_generator.init_generator.schema import RobotFacts

    return RobotFacts(
        gripper_empty=robot.gripper_empty,
        holding=robot.holding,
        camera_aimed_at=robot.camera_aimed_at,
        source=_TRACKER_SOURCE,
        confidence=_DEFAULT_CONFIDENCE,
    )


def _remove_object_relations(
    relations: list[RelationFact],
    obj: str,
) -> list[RelationFact]:
    return [
        rel
        for rel in relations
        if not (
            rel.predicate in _RELATION_PREDICATES
            and len(rel.args) >= 1
            and rel.args[0] == obj
        )
    ]
