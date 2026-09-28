"""Tests for StateVerifier and GREEN/YELLOW/RED gating (Session 8)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.state_tracker import CompletedAction
from planner.problem_generator.init_generator.schema import (
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)
from planner.state_verifier import StateVerifier


def _table_scene(*, holding: str | None = None) -> SceneState:
    gripper_empty = holding is None
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
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="oracle",
                confidence=1.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=gripper_empty,
            holding=holding,
            camera_aimed_at=None,
            source="oracle",
            confidence=1.0,
        ),
    )


def test_expect_pick_removes_on_relation():
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    expected = StateVerifier().expect(pre, action)

    assert expected.robot.holding == "red_cup"
    assert expected.robot.gripper_empty is False
    assert not any(
        r.predicate == "on" and r.args[0] == "red_cup" for r in expected.relations
    )


def test_expect_place_adds_on_relation():
    pre = _table_scene(holding="red_cup")
    action = CompletedAction("place", {"object": "red_cup", "location": "shelf"})
    expected = StateVerifier().expect(pre, action)

    assert expected.robot.gripper_empty is True
    assert expected.robot.holding is None
    assert ("on", "red_cup", "shelf") in {
        (r.predicate, r.args[0], r.args[1]) for r in expected.relations
    }


def test_verify_green_after_pick():
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    verifier = StateVerifier()
    expected = verifier.expect(pre, action)

    result = verifier.verify(pre, action, expected)

    assert result.verdict == "GREEN"
    assert result.mismatch_score <= verifier.green_threshold
    assert result.request_vlm is False
    assert result.replan is False


def test_verify_yellow_partial_relation_mismatch():
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    verifier = StateVerifier()
    expected = verifier.expect(pre, action)

    # Robot matches but stale (on red_cup table) relation remains.
    observed = SceneState(
        objects=list(expected.objects),
        locations=list(expected.locations),
        relations=list(pre.relations),
        robot=expected.robot,
    )

    result = verifier.verify(pre, action, observed)

    assert result.verdict == "YELLOW"
    assert verifier.green_threshold < result.mismatch_score <= verifier.red_threshold
    assert result.request_vlm is True
    assert result.replan is False


def test_verify_red_executor_failure():
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"}, success_flag=False)
    observed = _table_scene()

    result = StateVerifier().verify(pre, action, observed)

    assert result.verdict == "RED"
    assert result.replan is True
    assert result.request_vlm is False
    assert any("success_flag=False" in m for m in result.mismatches)


def test_verify_red_holding_contradiction():
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    verifier = StateVerifier()
    expected = verifier.expect(pre, action)

    observed = SceneState(
        objects=list(expected.objects),
        locations=list(expected.locations),
        relations=list(expected.relations),
        robot=RobotFacts(
            gripper_empty=False,
            holding="blue_box",
            camera_aimed_at=None,
            source="oracle",
            confidence=1.0,
        ),
    )

    result = verifier.verify(pre, action, observed)

    assert result.verdict == "RED"
    assert result.replan is True
    assert any("holding contradiction" in m for m in result.mismatches)


def test_verify_red_large_mismatch():
    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    observed = _table_scene()  # unchanged — clear failure

    result = StateVerifier().verify(pre, action, observed)

    assert result.verdict == "RED"
    assert result.mismatch_score > StateVerifier().red_threshold
    assert result.replan is True


def test_vlm_stub_not_called_on_green():
    calls: list[str] = []

    def _vlm_stub(expected, observed, action):
        calls.append(action.primitive)
        return observed

    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    verifier = StateVerifier(vlm_callback=_vlm_stub)
    expected = verifier.expect(pre, action)

    verifier.verify(pre, action, expected)

    assert calls == []


def test_vlm_stub_not_called_on_red():
    calls: list[str] = []

    def _vlm_stub(expected, observed, action):
        calls.append(action.primitive)
        return observed

    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"}, success_flag=False)
    verifier = StateVerifier(vlm_callback=_vlm_stub)

    verifier.verify(pre, action, pre)

    assert calls == []


def test_vlm_stub_called_on_yellow():
    calls: list[str] = []

    def _vlm_stub(expected, observed, action):
        calls.append(action.primitive)
        return observed

    pre = _table_scene()
    action = CompletedAction("pick", {"object": "red_cup"})
    verifier = StateVerifier(vlm_callback=_vlm_stub)
    expected = verifier.expect(pre, action)
    observed = SceneState(
        objects=list(expected.objects),
        locations=list(expected.locations),
        relations=list(pre.relations),
        robot=expected.robot,
    )

    result = verifier.verify(pre, action, observed)

    assert result.verdict == "YELLOW"
    assert calls == ["pick"]
    assert result.request_vlm is True


def test_enrichment_primitive_no_local_model_yellow_on_divergence():
    pre = _table_scene(holding="red_cup")
    action = CompletedAction("pour", {"source": "red_cup", "target": "bowl"})
    observed = _table_scene()  # diverges from pre

    result = StateVerifier().verify(pre, action, observed)

    assert result.verdict == "YELLOW"
    assert result.request_vlm is True
    assert any("no local model" in m for m in result.mismatches)


def test_enrichment_primitive_no_local_model_green_when_unchanged():
    pre = _table_scene(holding="red_cup")
    action = CompletedAction("pour", {"source": "red_cup", "target": "bowl"})

    result = StateVerifier().verify(pre, action, pre)

    assert result.verdict == "GREEN"
    assert result.request_vlm is False
