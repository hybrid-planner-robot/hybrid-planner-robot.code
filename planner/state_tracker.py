"""
Mid-task robot / symbolic state tracker for closed-loop iterations.

Persists gripper / holding / camera facts (and optionally last moved object
locations) across successfully completed primitives without re-perceiving
from vision every step.

See ``docs/hybrid_problem_generator_design.md`` §3.1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping, Protocol, runtime_checkable

from planner.primitive_transitions import (
    CompletedAction,
    MovedRecord,
    compute_tracker_delta,
)

if TYPE_CHECKING:
    from planner.enrichment_effects import EnrichmentContext
    from planner.problem_generator.init_generator.schema import RobotFacts

_DEFAULT_CONFIDENCE = 1.0
_TRACKER_SOURCE = "tracker"


@runtime_checkable
class ActionLike(Protocol):
    """Anything with PlanStep / CompletedAction shape."""

    primitive: str
    args: Mapping[str, Any]


def _default_robot(initial: RobotFacts | None = None) -> RobotFacts:
    from planner.problem_generator.init_generator.schema import RobotFacts

    if initial is None:
        return RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source=_TRACKER_SOURCE,
            confidence=_DEFAULT_CONFIDENCE,
        )
    return RobotFacts(
        gripper_empty=initial.gripper_empty,
        holding=initial.holding,
        camera_aimed_at=initial.camera_aimed_at,
        source=_TRACKER_SOURCE,
        confidence=float(initial.confidence),
    )


class StateTracker:
    """
    Persist robot / mid-task symbolic facts across closed-loop iterations.

    Call ``apply`` only for successfully completed primitives.
    """

    def __init__(
        self,
        initial: RobotFacts | None = None,
        *,
        enrichment: "EnrichmentContext | None" = None,
    ) -> None:
        self._robot = _default_robot(initial)
        self._moved: dict[str, MovedRecord] = {}
        self._facts: list[tuple[str, ...]] = []
        self._enrichment = enrichment

    @property
    def enrichment(self) -> "EnrichmentContext | None":
        """Enriched actions in play for this task (Session 30), if any."""
        return self._enrichment

    @enrichment.setter
    def enrichment(self, context: "EnrichmentContext | None") -> None:
        self._enrichment = context

    def reset(self, initial: RobotFacts | None = None) -> None:
        """Clear history; optionally seed robot facts."""
        self._robot = _default_robot(initial)
        self._moved.clear()
        self._facts.clear()

    def apply(self, action: ActionLike | CompletedAction) -> None:
        """
        Update tracker from a successfully completed primitive.

        Delegates symbolic effects to ``primitive_transitions``. Enriched
        catalog actions are applied only when this tracker carries the task's
        ``EnrichmentContext``; anything else (open-vocab, unknown) is a no-op.
        """
        completed = (
            action
            if isinstance(action, CompletedAction)
            else CompletedAction(
                primitive=action.primitive,
                args=dict(action.args) if action.args else {},
            )
        )
        delta = compute_tracker_delta(
            self._robot,
            self._moved,
            completed,
            enrichment=self._enrichment,
        )
        if not delta.applied:
            return
        self._robot = delta.robot
        for key in delta.moved_remove:
            self._moved.pop(key, None)
        for key, record in delta.moved_set.items():
            self._moved[key] = record
        for fact in delta.facts_add:
            if fact not in self._facts:
                self._facts.append(fact)

    def snapshot(self) -> RobotFacts:
        """Return current gripper_empty, holding, camera_aimed_at (source=tracker)."""
        from planner.problem_generator.init_generator.schema import RobotFacts

        return RobotFacts(
            gripper_empty=self._robot.gripper_empty,
            holding=self._robot.holding,
            camera_aimed_at=self._robot.camera_aimed_at,
            source=_TRACKER_SOURCE,
            confidence=self._robot.confidence,
        )

    def moved_objects(self) -> dict[str, str]:
        """
        object_name → last known support (location, bottom item, or container).

        Used by fusion when perception lags behind execution.
        """
        return {name: record.support for name, record in self._moved.items()}

    def moved_relations(self) -> list[tuple[str, str, str]]:
        """Return (predicate, object, support) tuples for tracked placements."""
        return [
            (record.predicate, name, record.support)
            for name, record in self._moved.items()
        ]

    def enrichment_facts(self) -> list[tuple[str, ...]]:
        """
        Fluents asserted by completed enriched actions, e.g. ``("poured", …)``.

        Perception cannot see these, so the tracker is their only carrier into
        the next ``:init`` — without them FD would re-plan the same pour forever.
        """
        return list(self._facts)

    def as_partial_scene(self) -> dict[str, Any]:
        """Robot block (+ moved / enrichment relations) suitable for Fusion.merge."""
        relations = [
            {
                "predicate": pred,
                "args": [obj, support],
                "source": _TRACKER_SOURCE,
                "confidence": _DEFAULT_CONFIDENCE,
            }
            for pred, obj, support in self.moved_relations()
        ]
        relations.extend(
            {
                "predicate": fact[0],
                "args": list(fact[1:]),
                "source": _TRACKER_SOURCE,
                "confidence": _DEFAULT_CONFIDENCE,
            }
            for fact in self._facts
        )
        return {
            "robot": self.snapshot().to_dict(),
            "relations": relations,
        }
