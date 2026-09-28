"""Mock VLM fusion client for offline tests (no GPU / no network)."""

from __future__ import annotations

from typing import Callable

from planner.primitive_transitions import involved_objects
from planner.problem_generator.init_generator.fusion import relation_subject_key
from planner.problem_generator.init_generator.schema import RelationFact, SceneState

from .client import VlmFusionRequest, patch_to_scene

PatchFn = Callable[[VlmFusionRequest], SceneState]


def _default_resolving_patch(request: VlmFusionRequest) -> SceneState:
    """
    Build a patch that aligns observed relations with expected for involved
    objects (adds expected facts; clears stale subject keys with confidence 0).
    """
    involved = involved_objects(request.action)
    patch_rels: list[RelationFact] = []
    expected_keys = {
        relation_subject_key(rel) for rel in request.expected.relations
    }

    for rel in request.expected.relations:
        if involved and rel.args and rel.args[0] not in involved:
            continue
        patch_rels.append(
            RelationFact(
                predicate=rel.predicate,
                args=list(rel.args),
                source="vlm",
                confidence=max(rel.confidence, 0.75),
            )
        )

    for rel in request.observed.relations:
        key = relation_subject_key(rel)
        if key in expected_keys:
            continue
        if involved and rel.args and rel.args[0] not in involved:
            continue
        patch_rels.append(
            RelationFact(
                predicate=rel.predicate,
                args=list(rel.args),
                source="vlm",
                confidence=0.0,
            )
        )

    return patch_to_scene(
        relations=patch_rels,
        robot=request.expected.robot,
    )


class MockVlmFusionClient:
    """
    Deterministic stand-in for the vision VLM.

    ``call_count`` is asserted by Session 10 tests (GREEN/RED → 0, YELLOW → 1).
    """

    def __init__(
        self,
        *,
        patch: SceneState | None = None,
        patch_fn: PatchFn | None = None,
    ) -> None:
        self._patch = patch
        self._patch_fn = patch_fn
        self.call_count = 0
        self.requests: list[VlmFusionRequest] = []

    def disambiguate(self, request: VlmFusionRequest) -> SceneState:
        self.call_count += 1
        self.requests.append(request)
        if self._patch_fn is not None:
            return self._patch_fn(request)
        if self._patch is not None:
            return self._patch
        return _default_resolving_patch(request)
