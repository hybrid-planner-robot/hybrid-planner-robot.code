"""
Compare expected vs observed SceneState after each executed primitive.

See ``docs/hybrid_problem_generator_design.md`` §3.2 (StateVerifier) and §4.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import TYPE_CHECKING, Callable

from planner.primitive_transitions import (
    CompletedAction,
    apply_relation_transition,
    apply_robot_transition,
    arg,
    has_local_model,
    involved_objects,
    normalize_primitive,
    relation_scope,
)

if TYPE_CHECKING:
    from planner.enrichment_effects import EnrichmentContext
from planner.problem_generator.init_generator.schema import (
    Meta,
    RelationFact,
    RobotFacts,
    SceneState,
)

from .types import Verdict, VerificationResult

_RELATION_PREDICATES = frozenset({"on", "stacked-on", "in-container"})

# Scoring weights (§4.4) — tunable in Session 12.
_ROBOT_HOLDING_WEIGHT = 0.20
_ROBOT_GRIPPER_WEIGHT = 0.15
_ROBOT_CAMERA_WEIGHT = 0.10
_RELATION_STEP_WEIGHT = 0.20
_RELATION_MAX_WEIGHT = 0.40
_PRESENCE_WEIGHT = 0.15

VlmCallback = Callable[..., SceneState]


def _call_vlm_callback(
    callback: VlmCallback,
    expected: SceneState,
    observed: SceneState,
    action: CompletedAction,
    mismatches: list[str],
) -> SceneState:
    """Invoke callback; pass mismatches when the callable accepts a 4th arg."""
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        params = {}
    if len(params) >= 4:
        return callback(expected, observed, action, mismatches)
    return callback(expected, observed, action)


class StateVerifier:
    """Script-first post-action verification with VLM gating on YELLOW only."""

    def __init__(
        self,
        *,
        green_threshold: float = 0.15,
        red_threshold: float = 0.60,
        vlm_callback: VlmCallback | None = None,
        enrichment: "EnrichmentContext | None" = None,
    ) -> None:
        self.green_threshold = green_threshold
        self.red_threshold = red_threshold
        self.vlm_callback = vlm_callback
        # Session 30: enriched catalog actions in play for this task. Without
        # it an enriched primitive has no local model, exactly as before.
        self.enrichment = enrichment

    def expect(self, pre: SceneState, action: CompletedAction) -> SceneState:
        """
        Symbolic post-action state from ``pre`` and ``action``.

        Pure function for standard primitives and for enriched catalog actions
        covered by ``enrichment``; other unknowns leave ``pre`` unchanged (the
        verifier may then classify YELLOW when observation diverges).
        """
        if not has_local_model(action.primitive, self.enrichment):
            return deepcopy(pre)

        robot = apply_robot_transition(pre.robot, action, enrichment=self.enrichment)
        relations = apply_relation_transition(
            pre.relations,
            action,
            holding=pre.robot.holding,
            enrichment=self.enrichment,
        )
        return SceneState(
            schema_version=pre.schema_version,
            timestamp=pre.timestamp,
            frame_id=pre.frame_id,
            domain_template=pre.domain_template,
            objects=list(pre.objects),
            locations=list(pre.locations),
            relations=relations,
            robot=RobotFacts(
                gripper_empty=robot.gripper_empty,
                holding=robot.holding,
                camera_aimed_at=robot.camera_aimed_at,
                source="fusion",
                confidence=1.0,
            ),
            meta=Meta(
                sources_used=list(pre.meta.sources_used) + ["fusion"],
                fusion_notes=list(pre.meta.fusion_notes),
            ),
        )

    def score(
        self,
        expected: SceneState,
        observed: SceneState,
        *,
        action: CompletedAction | None = None,
    ) -> tuple[float, list[str]]:
        """Return normalized mismatch score in [0, 1] and human-readable diffs."""
        mismatches: list[str] = []
        total = 0.0
        involved = involved_objects(action) if action else set()
        scoped = relation_scope(action, self.enrichment) if action else set()

        total += self._score_robot(expected.robot, observed.robot, mismatches)
        total += self._score_relations(
            expected.relations,
            observed.relations,
            scoped,
            mismatches,
        )
        total += self._score_presence(expected, observed, involved, mismatches)

        return min(1.0, total), mismatches

    def verify(
        self,
        pre: SceneState,
        action: CompletedAction,
        observed: SceneState,
    ) -> VerificationResult:
        """expect → score → classify; invoke VLM stub only on YELLOW."""
        if not action.success_flag:
            expected = self.expect(pre, action)
            return VerificationResult(
                verdict="RED",
                mismatch_score=1.0,
                expected=expected,
                observed=observed,
                mismatches=["executor reported success_flag=False"],
                request_vlm=False,
                replan=True,
            )

        if not has_local_model(action.primitive, self.enrichment):
            score, mismatches = self.score(pre, observed, action=action)
            mismatches = ["no local model for enrichment primitive"] + mismatches
            if score <= self.green_threshold:
                return VerificationResult(
                    verdict="GREEN",
                    mismatch_score=score,
                    expected=deepcopy(pre),
                    observed=observed,
                    mismatches=mismatches,
                    request_vlm=False,
                    replan=False,
                )
            return self._yellow_result(
                deepcopy(pre),
                observed,
                action,
                min(score, self.red_threshold),
                mismatches,
            )

        expected = self.expect(pre, action)
        hard_red = self._hard_red_reasons(expected, observed, action)
        if hard_red:
            return VerificationResult(
                verdict="RED",
                mismatch_score=1.0,
                expected=expected,
                observed=observed,
                mismatches=hard_red,
                request_vlm=False,
                replan=True,
            )

        score, mismatches = self.score(expected, observed, action=action)
        verdict = self._classify(score)

        unverifiable = self._unverifiable_fluent(action)
        if verdict == "YELLOW" or (unverifiable and verdict == "GREEN"):
            # A clean score on an enriched action still only proves the
            # manipulation happened; no adapter can confirm the fluent itself.
            # Send it through the VLM gate rather than claiming GREEN off the
            # action effect alone (design §4.5). With no VLM wired,
            # ``_yellow_result`` re-scores and lands back on GREEN, so this
            # costs a second look, never a stalled loop.
            return self._yellow_result(
                expected,
                observed,
                action,
                score,
                mismatches,
                notes=[unverifiable] if unverifiable else [],
            )

        return VerificationResult(
            verdict=verdict,
            mismatch_score=score,
            expected=expected,
            observed=observed,
            mismatches=mismatches,
            request_vlm=False,
            replan=verdict == "RED",
        )

    def _unverifiable_fluent(self, action: CompletedAction) -> str | None:
        """Message when this action asserts a fluent perception cannot confirm."""
        if self.enrichment is None:
            return None
        effect = self.enrichment.effect_for(action.primitive)
        if effect is None or effect.perceptually_verifiable:
            return None
        fact = self.enrichment.fact_for(action.primitive, action.args or {})
        fluent = fact[0] if fact else normalize_primitive(action.primitive)
        return (
            f"enrichment fluent {fluent!r} is not perceptually verifiable "
            "(asserted from the action effect)"
        )

    def _classify(self, score: float) -> Verdict:
        if score <= self.green_threshold:
            return "GREEN"
        if score <= self.red_threshold:
            return "YELLOW"
        return "RED"

    def _yellow_result(
        self,
        expected: SceneState,
        observed: SceneState,
        action: CompletedAction,
        score: float,
        mismatches: list[str],
        *,
        notes: list[str] | None = None,
    ) -> VerificationResult:
        """
        Call VLM at most once, merge correction, then re-score once.

        Resolved → GREEN; still above ``red_threshold`` → RED; else stay YELLOW.
        Never invokes VLM a second time for the same action verification.

        ``notes`` survive into every verdict: they record *why* the script could
        not decide on its own (e.g. an unverifiable enrichment fluent), which
        stays true even when the re-score comes back GREEN.
        """
        notes = list(notes or [])
        mismatches = notes + list(mismatches)
        corrected = observed
        if self.vlm_callback is not None:
            corrected = _call_vlm_callback(
                self.vlm_callback,
                expected,
                observed,
                action,
                mismatches,
            )

        # Re-verify once without another VLM call.
        hard_red = self._hard_red_reasons(expected, corrected, action)
        if hard_red:
            return VerificationResult(
                verdict="RED",
                mismatch_score=1.0,
                expected=expected,
                observed=corrected,
                mismatches=list(mismatches) + hard_red,
                request_vlm=False,
                replan=True,
            )

        score2, mismatches2 = self.score(expected, corrected, action=action)
        verdict2 = self._classify(score2)
        merged_mismatches = list(dict.fromkeys([*mismatches, *mismatches2]))
        if verdict2 == "GREEN":
            return VerificationResult(
                verdict="GREEN",
                mismatch_score=score2,
                expected=expected,
                observed=corrected,
                mismatches=notes + mismatches2,
                request_vlm=False,
                replan=False,
            )
        if verdict2 == "RED":
            return VerificationResult(
                verdict="RED",
                mismatch_score=score2,
                expected=expected,
                observed=corrected,
                mismatches=merged_mismatches,
                request_vlm=False,
                replan=True,
            )
        return VerificationResult(
            verdict="YELLOW",
            mismatch_score=score2,
            expected=expected,
            observed=corrected,
            mismatches=merged_mismatches,
            request_vlm=True,
            replan=False,
        )

    def _hard_red_reasons(
        self,
        expected: SceneState,
        observed: SceneState,
        action: CompletedAction,
    ) -> list[str]:
        reasons: list[str] = []

        exp_h = expected.robot.holding
        obs_h = observed.robot.holding
        if exp_h and obs_h and exp_h != obs_h:
            reasons.append(
                f"holding contradiction: expected {exp_h!r}, observed {obs_h!r}"
            )

        critical = self._critical_objects(action)
        observed_names = {obj.name for obj in observed.objects}
        for name in critical:
            if name not in observed_names:
                reasons.append(f"critical object {name!r} missing from observation")

        reasons.extend(
            self._failed_manipulation_reasons(expected, observed, action)
        )

        return reasons

    @staticmethod
    def _failed_manipulation_reasons(
        expected: SceneState,
        observed: SceneState,
        action: CompletedAction,
    ) -> list[str]:
        """Hard RED when a pick/place clearly did not change the world."""
        primitive = normalize_primitive(action.primitive)
        args = action.args or {}

        if primitive in {"pick", "unstack", "pick-from-container"}:
            obj = arg(args, "object", "top", "i")
            if not obj:
                return []
            if (
                expected.robot.holding == obj
                and observed.robot.holding is None
                and observed.robot.gripper_empty
            ):
                still_placed = any(
                    rel.predicate in _RELATION_PREDICATES
                    and rel.args
                    and rel.args[0] == obj
                    for rel in observed.relations
                )
                if still_placed:
                    return [f"pick of {obj!r} failed: object still on support"]
        return []

    @staticmethod
    def _critical_objects(action: CompletedAction) -> set[str]:
        primitive = normalize_primitive(action.primitive)
        args = action.args or {}
        if primitive in {"pick", "unstack", "pick-from-container"}:
            obj = arg(args, "object", "top", "i")
            return {obj} if obj else set()
        return set()

    @staticmethod
    def _score_robot(
        expected: RobotFacts,
        observed: RobotFacts,
        mismatches: list[str],
    ) -> float:
        score = 0.0
        if expected.holding != observed.holding:
            score += _ROBOT_HOLDING_WEIGHT
            mismatches.append(
                f"robot.holding: expected {expected.holding!r}, "
                f"observed {observed.holding!r}"
            )
        if expected.gripper_empty != observed.gripper_empty:
            score += _ROBOT_GRIPPER_WEIGHT
            mismatches.append(
                f"robot.gripper_empty: expected {expected.gripper_empty}, "
                f"observed {observed.gripper_empty}"
            )
        if expected.camera_aimed_at != observed.camera_aimed_at:
            score += _ROBOT_CAMERA_WEIGHT
            mismatches.append(
                f"robot.camera_aimed_at: expected {expected.camera_aimed_at!r}, "
                f"observed {observed.camera_aimed_at!r}"
            )
        return score

    @staticmethod
    def _score_relations(
        expected: list[RelationFact],
        observed: list[RelationFact],
        involved: set[str],
        mismatches: list[str],
    ) -> float:
        if not involved:
            return 0.0

        def _keys(relations: list[RelationFact]) -> set[tuple[str, tuple[str, ...]]]:
            keys: set[tuple[str, tuple[str, ...]]] = set()
            for rel in relations:
                if rel.predicate not in _RELATION_PREDICATES:
                    continue
                if not rel.args or rel.args[0] not in involved:
                    continue
                keys.add((rel.predicate, tuple(rel.args)))
            return keys

        exp_keys = _keys(expected)
        obs_keys = _keys(observed)
        diff_count = len(exp_keys.symmetric_difference(obs_keys))
        if diff_count == 0:
            return 0.0

        missing = exp_keys - obs_keys
        extra = obs_keys - exp_keys
        for key in sorted(missing):
            mismatches.append(f"relation missing: {key[0]} {' '.join(key[1])}")
        for key in sorted(extra):
            mismatches.append(f"relation extra: {key[0]} {' '.join(key[1])}")

        return min(_RELATION_MAX_WEIGHT, diff_count * _RELATION_STEP_WEIGHT)

    @staticmethod
    def _score_presence(
        expected: SceneState,
        observed: SceneState,
        involved: set[str],
        mismatches: list[str],
    ) -> float:
        if not involved:
            return 0.0
        observed_names = {obj.name for obj in observed.objects}
        missing = sorted(name for name in involved if name not in observed_names)
        if not missing:
            return 0.0
        for name in missing:
            mismatches.append(f"object not detected: {name!r}")
        return min(_PRESENCE_WEIGHT, len(missing) * _PRESENCE_WEIGHT)
