"""Tests for VLM hybrid fusion on YELLOW only (Session 10)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.primitive_transitions import CompletedAction
from planner.problem_generator.init_generator.builder import build_scene
from planner.problem_generator.init_generator.fusion import (
    FusionEngine,
    apply_vlm_patch_to_scene,
)
from planner.problem_generator.init_generator.schema import (
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)
from planner.problem_generator.init_generator.vlm_fusion import (
    LiveVlmFusionClient,
    MockVlmFusionClient,
    build_vlm_callback,
    parse_vlm_patch_json,
    patch_to_scene,
)
from planner.state_verifier import StateVerifier


def _table_scene(*, holding: str | None = None) -> SceneState:
    gripper_empty = holding is None
    relations = []
    if holding is None:
        relations.append(
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="oracle",
                confidence=1.0,
            )
        )
    return SceneState(
        objects=[
            ObjectFact(
                name="red_cup",
                source="oracle",
                confidence=1.0,
                location="table",
            ),
        ],
        locations=[],
        relations=relations,
        robot=RobotFacts(
            gripper_empty=gripper_empty,
            holding=holding,
            camera_aimed_at=None,
            source="oracle",
            confidence=1.0,
        ),
    )


def _stale_pick_observation(expected: SceneState, pre: SceneState) -> SceneState:
    """Robot matches expected after pick, but stale (on …) remains."""
    return SceneState(
        objects=list(expected.objects),
        locations=list(expected.locations),
        relations=list(pre.relations),
        robot=expected.robot,
    )


def test_vlm_not_called_on_green():
    client = MockVlmFusionClient()
    verifier = StateVerifier(vlm_callback=build_vlm_callback(client))
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = verifier.expect(pre, action)

    result = verifier.verify(pre, action, expected)

    assert result.verdict == "GREEN"
    assert client.call_count == 0
    assert result.request_vlm is False


def test_vlm_not_called_on_red():
    client = MockVlmFusionClient()
    verifier = StateVerifier(vlm_callback=build_vlm_callback(client))
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"}, success_flag=False)

    result = verifier.verify(pre, action, pre)

    assert result.verdict == "RED"
    assert client.call_count == 0
    assert result.request_vlm is False


def test_vlm_called_once_on_yellow_and_resolves_to_green():
    client = MockVlmFusionClient()
    verifier = StateVerifier(vlm_callback=build_vlm_callback(client, images=["frame.jpg"]))
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = verifier.expect(pre, action)
    observed = _stale_pick_observation(expected, pre)

    result = verifier.verify(pre, action, observed)

    assert client.call_count == 1
    assert result.verdict == "GREEN"
    assert result.request_vlm is False
    assert not any(
        r.predicate == "on" and r.args[0] == "red_cup" for r in result.observed.relations
    )
    assert client.requests[0].images == ["frame.jpg"]
    assert client.requests[0].action.primitive == "pick"


def test_live_vlm_fusion_client_with_injected_complete_fn():
    """Session 16: LiveVlmFusionClient is thin glue over complete_fn / planner."""

    def complete(system: str, user: str, images) -> str:
        assert "resolve" in system.lower() or "JSON" in system
        assert "mismatches" in user
        assert images == ["frame.jpg"]
        return (
            '{"relations": [], "robot": {"holding": "red_cup", "gripper_empty": false},'
            ' "remove_relations": [["on", "red_cup", "table"]]}'
        )

    client = LiveVlmFusionClient(complete_fn=complete)
    verifier = StateVerifier(
        vlm_callback=build_vlm_callback(client, images=["frame.jpg"])
    )
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = verifier.expect(pre, action)
    observed = _stale_pick_observation(expected, pre)

    result = verifier.verify(pre, action, observed)

    assert client.call_count == 1
    assert result.verdict == "GREEN"
    assert client.last_raw is not None


def test_vlm_not_called_on_green_with_live_client():
    client = LiveVlmFusionClient(
        complete_fn=lambda s, u, i: (_ for _ in ()).throw(AssertionError("no VLM"))
    )
    verifier = StateVerifier(vlm_callback=build_vlm_callback(client))
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = verifier.expect(pre, action)

    result = verifier.verify(pre, action, expected)

    assert result.verdict == "GREEN"
    assert client.call_count == 0


def test_vlm_yellow_unresolved_stays_yellow_without_second_call():
    """If the patch does not fix the mismatch, stay YELLOW — still max 1 call."""
    noop_patch = patch_to_scene(relations=[])  # relation-only empty → no change
    client = MockVlmFusionClient(patch=noop_patch)
    verifier = StateVerifier(vlm_callback=build_vlm_callback(client))
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = verifier.expect(pre, action)
    observed = _stale_pick_observation(expected, pre)

    result = verifier.verify(pre, action, observed)

    assert client.call_count == 1
    assert result.verdict == "YELLOW"
    assert result.request_vlm is True


def test_vlm_red_mismatch_no_call_even_with_callback():
    client = MockVlmFusionClient()
    verifier = StateVerifier(vlm_callback=build_vlm_callback(client))
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    # Unchanged world after pick → hard RED / large mismatch, no VLM.
    result = verifier.verify(pre, action, pre)

    assert result.verdict == "RED"
    assert client.call_count == 0


def test_fusion_applies_vlm_patch_relations():
    oracle = _table_scene()
    # Same subject key ("on", "red_cup") → VLM replaces oracle table placement.
    patch = parse_vlm_patch_json('{"relations": [["on", "red_cup", "shelf"]]}')

    scene = FusionEngine().merge(oracle_scene=oracle, vlm_patch=patch)

    assert "vlm" in scene.meta.sources_used
    assert any("vlm_patch applied" in n for n in scene.meta.fusion_notes)
    on_pairs = {
        (r.args[0], r.args[1])
        for r in scene.relations
        if r.predicate == "on" and len(r.args) >= 2
    }
    assert ("red_cup", "shelf") in on_pairs
    assert ("red_cup", "table") not in on_pairs
    red_on = next(r for r in scene.relations if r.predicate == "on" and r.args[0] == "red_cup")
    assert red_on.source == "vlm"


def test_vlm_cannot_be_sole_init_source():
    patch = parse_vlm_patch_json('{"relations": [["on", "red_cup", "table"]]}')
    with pytest.raises(ValueError, match="sole init source"):
        FusionEngine().merge(vlm_patch=patch)


def test_apply_vlm_patch_to_scene_preserves_tracker_robot():
    base = SceneState(
        objects=[
            ObjectFact(name="red_cup", source="oracle", confidence=1.0),
        ],
        locations=[],
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="oracle",
                confidence=1.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=False,
            holding="red_cup",
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    patch = patch_to_scene(
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="vlm",
                confidence=0.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="vlm",
            confidence=0.9,
        ),
    )

    corrected = apply_vlm_patch_to_scene(base, patch)

    assert corrected.robot.source == "tracker"
    assert corrected.robot.holding == "red_cup"
    assert not any(r.predicate == "on" for r in corrected.relations)
    assert any("tracker wins" in n for n in corrected.meta.fusion_notes)


def test_build_scene_with_vlm_patch_keeps_deterministic_base():
    oracle = _table_scene()
    patch = parse_vlm_patch_json('{"relations": [["on", "red_cup", "shelf"]]}')
    scene = build_scene(oracle_scene=oracle, vlm_patch=patch)

    assert "oracle" in scene.meta.sources_used
    assert "vlm" in scene.meta.sources_used
    assert "fusion" in scene.meta.sources_used


def test_parse_vlm_patch_json_with_facts_context():
    patch = parse_vlm_patch_json(
        '{"relations": [["on", "red_cup", "shelf"]]}',
        known_symbols=frozenset({"red_cup", "shelf", "table"}),
    )
    assert len(patch.relations) == 1
    assert patch.relations[0].args == ["red_cup", "shelf"]


def test_yellow_handler_passes_oracle_dino_tracker_facts_to_client():
    oracle = _table_scene()
    dino = SceneState(
        objects=list(oracle.objects),
        locations=[],
        relations=list(oracle.relations),
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="dino",
            confidence=0.7,
        ),
    )
    tracker = SceneState(
        objects=[],
        locations=[],
        relations=[],
        robot=RobotFacts(
            gripper_empty=False,
            holding="red_cup",
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    client = MockVlmFusionClient()
    callback = build_vlm_callback(
        client,
        images=["cam.png"],
        oracle_facts=oracle,
        dino_facts=dino,
        tracker_facts=tracker,
    )
    verifier = StateVerifier(vlm_callback=callback)
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = verifier.expect(pre, action)
    observed = _stale_pick_observation(expected, pre)

    verifier.verify(pre, action, observed)

    assert client.call_count == 1
    req = client.requests[0]
    assert req.oracle_facts is oracle
    assert req.dino_facts is dino
    assert req.tracker_facts is tracker
    assert req.images == ["cam.png"]
    assert req.mismatches  # verifier forwards mismatch strings
