"""
VLM fusion client: images + deterministic facts → relation patch SceneState.

Production wiring can swap in a real vision VLM; CI uses ``MockVlmFusionClient``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from planner.primitive_transitions import CompletedAction
from planner.problem_generator.init_generator.fusion import apply_vlm_patch_to_scene
from planner.problem_generator.init_generator.schema import (
    Meta,
    RelationFact,
    RobotFacts,
    SceneState,
)

_RELATION_PREDICATES = frozenset(
    {"on", "stacked-on", "in-container", "holding", "camera-aimed-at"}
)


@dataclass
class VlmFusionRequest:
    """Inputs for a single bounded VLM disambiguation call."""

    expected: SceneState
    observed: SceneState
    action: CompletedAction
    mismatches: list[str] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)
    oracle_facts: SceneState | None = None
    dino_facts: SceneState | None = None
    tracker_facts: SceneState | None = None


class VlmFusionClient(Protocol):
    """Vision VLM that returns a SceneState patch (relations / corrections)."""

    def disambiguate(self, request: VlmFusionRequest) -> SceneState:
        """Return a patch SceneState to merge into ``request.observed``."""
        ...


def _extract_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group())
    if not isinstance(data, dict):
        raise ValueError("VLM patch JSON must be an object")
    return data


def _parse_relation_entry(
    item: Any,
    *,
    confidence: float,
    known_symbols: frozenset[str] | None,
) -> RelationFact:
    if isinstance(item, (list, tuple)) and len(item) >= 1:
        pred = str(item[0]).strip()
        args = [str(a).strip() for a in item[1:]]
        conf = confidence
    elif isinstance(item, dict):
        pred = str(item.get("predicate", "")).strip()
        args = [str(a).strip() for a in item.get("args", [])]
        conf = float(item.get("confidence", confidence))
    else:
        raise ValueError(f"invalid VLM relation entry: {item!r}")

    if pred not in _RELATION_PREDICATES:
        raise ValueError(f"disallowed VLM relation predicate: {pred!r}")
    if known_symbols is not None:
        for arg in args:
            if arg not in known_symbols:
                raise ValueError(f"unknown symbol in VLM patch: {arg!r}")
    return RelationFact(
        predicate=pred,
        args=args,
        source="vlm",
        confidence=conf,
    )


def parse_vlm_patch_json(
    raw: str,
    *,
    known_symbols: frozenset[str] | None = None,
    confidence: float = 0.75,
) -> SceneState:
    """
    Parse VLM JSON into a patch SceneState.

    Accepted shapes::

        {"relations": [["on", "red_cup", "shelf"], ...]}
        {"relations": [{"predicate": "on", "args": ["red_cup", "shelf"], ...}]}
        {"remove_relations": [["on", "red_cup"], ...]}

    Optional ``robot`` object may include ``holding`` / ``gripper_empty`` /
    ``camera_aimed_at`` corrections.
    """
    data = _extract_json_object(raw)
    relations_raw = data.get("relations", [])
    if not isinstance(relations_raw, list):
        raise ValueError("VLM patch 'relations' must be a list")

    relations: list[RelationFact] = [
        _parse_relation_entry(
            item, confidence=confidence, known_symbols=known_symbols
        )
        for item in relations_raw
    ]

    remove_raw = data.get("remove_relations", [])
    if remove_raw and not isinstance(remove_raw, list):
        raise ValueError("VLM patch 'remove_relations' must be a list")
    for item in remove_raw or []:
        rel = _parse_relation_entry(
            item, confidence=0.0, known_symbols=known_symbols
        )
        relations.append(
            RelationFact(
                predicate=rel.predicate,
                args=list(rel.args),
                source="vlm",
                confidence=0.0,
            )
        )

    notes = ["parsed VLM patch JSON"]
    robot_data = data.get("robot")
    if isinstance(robot_data, dict):
        holding = robot_data.get("holding")
        gripper_empty = robot_data.get("gripper_empty")
        if gripper_empty is None:
            gripper_empty = holding is None
        robot = RobotFacts(
            gripper_empty=bool(gripper_empty),
            holding=holding,
            camera_aimed_at=robot_data.get("camera_aimed_at"),
            source="vlm",
            confidence=float(robot_data.get("confidence", confidence)),
        )
        notes.append("robot_patch")
    else:
        robot = RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="vlm",
            confidence=confidence,
        )

    return SceneState(
        objects=[],
        locations=[],
        relations=relations,
        robot=robot,
        meta=Meta(sources_used=["vlm"], fusion_notes=notes),
    )


def patch_to_scene(
    *,
    relations: Sequence[RelationFact] | None = None,
    robot: RobotFacts | None = None,
    confidence: float = 0.75,
) -> SceneState:
    """Build a minimal patch SceneState from typed facts."""
    rels = [
        RelationFact(
            predicate=r.predicate,
            args=list(r.args),
            source="vlm",
            confidence=r.confidence,
        )
        for r in (relations or [])
    ]
    notes: list[str] = []
    if robot is None:
        robot_out = RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="vlm",
            confidence=confidence,
        )
    else:
        robot_out = RobotFacts(
            gripper_empty=robot.gripper_empty,
            holding=robot.holding,
            camera_aimed_at=robot.camera_aimed_at,
            source="vlm",
            confidence=robot.confidence,
        )
        notes.append("robot_patch")
    return SceneState(
        objects=[],
        locations=[],
        relations=rels,
        robot=robot_out,
        meta=Meta(sources_used=["vlm"], fusion_notes=notes),
    )


def build_prompt(request: VlmFusionRequest) -> tuple[str, str]:
    """Build (system, user) text for a vision+text VLM disambiguation call."""
    system = (
        "You resolve ambiguous robot manipulation scene relations. "
        "Return ONLY JSON: "
        '{"relations": [["predicate", "arg1", ...], ...]}. '
        "Allowed predicates: on, stacked-on, in-container. "
        "Use only object/location names from the provided facts. "
        "Do not invent objects. Prefer correcting the single mismatched relation."
    )

    def _compact(scene: SceneState | None, label: str) -> str:
        if scene is None:
            return f"{label}: (none)"
        rels = [
            f"({r.predicate} {' '.join(r.args)})@{r.source}"
            for r in scene.relations
        ]
        robot = scene.robot.to_dict()
        return (
            f"{label}:\n"
            f"  robot={robot}\n"
            f"  relations={rels or []}"
        )

    user_parts = [
        f"action: {request.action.primitive} {request.action.args}",
        f"mismatches: {request.mismatches}",
        _compact(request.expected, "expected"),
        _compact(request.observed, "observed"),
        _compact(request.oracle_facts, "oracle"),
        _compact(request.dino_facts, "dino"),
        _compact(request.tracker_facts, "tracker"),
        f"images_attached: {len(request.images)}",
    ]
    return system, "\n".join(user_parts)


def build_vlm_callback(
    client: VlmFusionClient,
    *,
    images: Sequence[Any] | None = None,
    oracle_facts: SceneState | None = None,
    dino_facts: SceneState | None = None,
    tracker_facts: SceneState | None = None,
) -> Callable[..., SceneState]:
    """
    Build a ``StateVerifier.vlm_callback`` that calls ``client`` once and merges.

    The callback applies the returned patch onto ``observed`` via
    ``apply_vlm_patch_to_scene`` (never replaces the full scene with VLM alone).
    Accepts optional 4th ``mismatches`` argument from the verifier.
    """
    image_list = list(images or [])

    def _callback(
        expected: SceneState,
        observed: SceneState,
        action: CompletedAction,
        mismatches: Sequence[str] | None = None,
    ) -> SceneState:
        request = VlmFusionRequest(
            expected=expected,
            observed=observed,
            action=action,
            mismatches=list(mismatches or []),
            images=image_list,
            oracle_facts=oracle_facts,
            dino_facts=dino_facts,
            tracker_facts=tracker_facts,
        )
        patch = client.disambiguate(request)
        return apply_vlm_patch_to_scene(observed, patch)

    return _callback
