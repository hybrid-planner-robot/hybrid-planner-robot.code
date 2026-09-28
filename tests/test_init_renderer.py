"""Tests for InitRenderer: SceneState → PDDL :init facts."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.init_generator.adapters.mock import (
    DinoMockAdapter,
    OracleMockAdapter,
    TrackerMockAdapter,
)
from planner.problem_generator.init_generator.renderer import InitRenderer
from planner.problem_generator.init_generator.schema import (
    LocationFact,
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)


def _facts(scene: SceneState) -> list[tuple]:
    return InitRenderer().render_facts(scene)


def _section(scene: SceneState) -> str:
    return InitRenderer().render_section(scene)


def test_oracle_fixture_on_clear_gripper_empty():
    scene = OracleMockAdapter.load()
    facts = _facts(scene)
    assert ("on", "red_cup", "table") in facts
    assert ("on", "blue_box", "table") in facts
    assert ("clear", "red_cup") in facts
    assert ("clear", "blue_box") in facts
    assert ("gripper-empty",) in facts
    assert ("holding", "red_cup") not in facts
    # Session 23: reachable is not emitted (dropped from complete domains).
    assert not any(f[0] == "reachable" for f in facts)


def test_dino_fixture_normalized_on_relations():
    scene = DinoMockAdapter.load()
    facts = _facts(scene)
    assert ("on", "red_cup", "table") in facts
    assert ("on", "blu_box", "table") in facts
    assert ("gripper-empty",) in facts


def test_tracker_holding_fixture():
    scene = TrackerMockAdapter.load()
    facts = _facts(scene)
    assert ("holding", "red_cup") in facts
    assert scene.robot.camera_aimed_at == "red_cup"
    assert not any(f[0] == "camera-aimed-at" for f in facts)
    assert ("clear", "red_cup") in facts
    assert ("gripper-empty",) not in facts
    assert not any(f[0] == "on" for f in facts)


def test_camera_aimed_at_never_emitted_in_init():
    """:init must not contain camera-aimed-at; look-at has to achieve it."""
    scene = SceneState(
        objects=[ObjectFact(name="cup", source="mock", confidence=1.0, clear=True)],
        locations=[LocationFact(name="table", source="mock", confidence=1.0)],
        relations=[
            RelationFact(predicate="on", args=["cup", "table"], source="mock", confidence=1.0),
            RelationFact(
                predicate="camera-aimed-at",
                args=["cup"],
                source="tracker",
                confidence=1.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at="cup",
            source="tracker",
            confidence=1.0,
        ),
    )
    facts = _facts(scene)
    assert ("camera-aimed-at", "cup") not in facts
    assert not any(f[0] == "camera-aimed-at" for f in facts)
    assert "(camera-aimed-at" not in _section(scene)


def test_render_section_structure():
    scene = OracleMockAdapter.load()
    section = _section(scene)
    assert section.startswith("  (:init")
    assert section.endswith("  )")
    assert "(on red_cup table)" in section
    assert "(gripper-empty)" in section


def test_clear_omitted_when_false():
    scene = SceneState(
        objects=[
            ObjectFact(name="cup", source="mock", confidence=1.0, clear=False),
        ],
        locations=[LocationFact(name="table", source="mock", confidence=1.0)],
        relations=[RelationFact(predicate="on", args=["cup", "table"], source="mock", confidence=1.0)],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
    )
    facts = _facts(scene)
    assert ("clear", "cup") not in facts


def test_reachable_field_never_emitted():
    """Schema may still carry reachable; InitRenderer never maps it to PDDL."""
    scene = SceneState(
        objects=[
            ObjectFact(
                name="cup",
                source="mock",
                confidence=1.0,
                clear=True,
                reachable=True,
            ),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
    )
    facts = _facts(scene)
    assert ("reachable", "cup") not in facts
    assert ("reachable", "table") not in facts


def test_container_open_closed():
    scene = SceneState(
        objects=[],
        locations=[
            LocationFact(name="bin", source="mock", confidence=1.0, type="container", open=True),
            LocationFact(name="box", source="mock", confidence=1.0, type="container", open=False),
        ],
        relations=[],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
        domain_template="containers_manipulation",
    )
    facts = _facts(scene)
    assert ("open", "bin") in facts
    assert ("closed", "box") in facts


def test_stacked_on_and_in_container_relations():
    scene = SceneState(
        objects=[
            ObjectFact(name="plate", source="mock", confidence=1.0, clear=True),
            ObjectFact(name="bowl", source="mock", confidence=1.0, clear=True),
        ],
        locations=[LocationFact(name="table", source="mock", confidence=1.0)],
        relations=[
            RelationFact(predicate="on", args=["plate", "table"], source="mock", confidence=1.0),
            RelationFact(
                predicate="stacked-on",
                args=["bowl", "plate"],
                source="mock",
                confidence=1.0,
            ),
            RelationFact(
                predicate="in-container",
                args=["spoon", "drawer"],
                source="mock",
                confidence=1.0,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
    )
    facts = _facts(scene)
    assert ("stacked-on", "bowl", "plate") in facts
    assert ("in-container", "spoon", "drawer") in facts


def test_skips_on_for_held_object():
    scene = SceneState(
        objects=[ObjectFact(name="cup", source="mock", confidence=1.0, clear=True)],
        locations=[LocationFact(name="table", source="mock", confidence=1.0)],
        relations=[
            RelationFact(predicate="on", args=["cup", "table"], source="mock", confidence=1.0),
        ],
        robot=RobotFacts(
            gripper_empty=False,
            holding="cup",
            camera_aimed_at=None,
            source="tracker",
            confidence=1.0,
        ),
    )
    facts = _facts(scene)
    assert ("holding", "cup") in facts
    assert ("on", "cup", "table") not in facts


def test_unknown_relation_warns():
    scene = SceneState(
        objects=[],
        locations=[],
        relations=[
            RelationFact(
                predicate="poured",
                args=["a", "b"],
                source="vlm",
                confidence=0.9,
            ),
        ],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
    )
    with pytest.warns(UserWarning, match="unknown relation predicate"):
        facts = _facts(scene)
    assert facts == [("gripper-empty",)]
