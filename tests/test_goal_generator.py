"""Tests for GoalGenerator (rule-based + LLM text + local LLM) and GoalRenderer."""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.goal_generator.backends.llm import LLMGoalGenerator
from planner.problem_generator.goal_generator.backends.local_llm import (
    DEFAULT_GOAL_LLM_MODEL_ID,
    DEFAULT_PERCEPTUAL_VLM_MODEL_ID,
    LocalLLMGoalGenerator,
)
from planner.problem_generator.goal_generator.backends.rule_based import (
    RuleBasedGoalGenerator,
    quantified_on_facts,
)
from planner.problem_generator.goal_generator.prompts import list_prompt_domains, load_goal_prompt
from planner.problem_generator.goal_generator.renderer import GoalRenderer
from planner.problem_generator.goal_generator.scene_compact import compact_scene_dict
from planner.problem_generator.init_generator.schema import (
    LocationFact,
    ObjectFact,
    Orientation,
    Pose,
    Position,
    RelationFact,
    RobotFacts,
    SceneState,
)

_OBJECTS = ["red_cup", "blue_box", "bowl", "plate"]
_LOCATIONS = ["table", "shelf", "drawer"]


@pytest.fixture
def generator() -> RuleBasedGoalGenerator:
    return RuleBasedGoalGenerator()


@pytest.fixture
def renderer() -> GoalRenderer:
    return GoalRenderer()


def test_pick_maps_to_holding(generator):
    result = generator.generate("pick up the red cup", _OBJECTS)
    assert result.ok is True
    assert result.facts == [("holding", "red_cup")]
    assert result.backend == "rule_based"


def test_pick_snake_case(generator):
    result = generator.generate("pick red_cup", _OBJECTS)
    assert result.ok is True
    assert result.facts == [("holding", "red_cup")]


def test_place_on_maps_to_on(generator):
    result = generator.generate(
        "place red_cup on shelf",
        _OBJECTS,
        locations=_LOCATIONS,
    )
    assert result.ok is True
    assert result.facts == [("on", "red_cup", "shelf")]


def test_put_on_the_table(generator):
    result = generator.generate(
        "put the blue box on the table",
        _OBJECTS,
        locations=_LOCATIONS,
    )
    assert result.ok is True
    assert result.facts == [("on", "blue_box", "table")]


def test_look_at_maps_to_camera_aimed_at(generator):
    result = generator.generate("look at red_cup", _OBJECTS)
    assert result.ok is True
    assert result.facts == [("camera-aimed-at", "red_cup")]


def test_stack_on_stacking_domain(generator):
    result = generator.generate(
        "stack bowl on plate",
        _OBJECTS,
        domain_template="manipulation_stacking",
    )
    assert result.ok is True
    assert result.facts == [("stacked-on", "bowl", "plate")]


def test_stack_rejected_on_base_domain(generator):
    result = generator.generate(
        "stack bowl on plate",
        _OBJECTS,
        domain_template="manipulation_base",
    )
    assert result.ok is False
    assert result.facts == []
    assert "stacked-on" in (result.error or "")


def test_in_container_containers_domain(generator):
    result = generator.generate(
        "put red_cup in drawer",
        _OBJECTS,
        locations=_LOCATIONS,
        domain_template="containers_manipulation",
    )
    assert result.ok is True
    assert result.facts == [("in-container", "red_cup", "drawer")]


def test_all_writing_tools_on_the_book(generator):
    objects = ["black_marker", "red_marker", "black_pen"]
    locations = ["blue_tablecloth", "book"]
    command = "all writing tools must be on the book"
    facts = quantified_on_facts(command, objects, locations)
    assert facts == [
        ("on", "black_marker", "book"),
        ("on", "red_marker", "book"),
        ("on", "black_pen", "book"),
    ]
    result = generator.generate(command, objects, locations=locations)
    assert result.ok is True
    assert set(result.facts) == set(facts)


def test_quantified_on_resolves_dest_still_typed_as_object():
    """Coverage probes split ``book`` as an item; dest still has to resolve."""
    facts = quantified_on_facts(
        "all writing tools must be on the book",
        ["black_marker", "red_marker", "black_pen", "book"],
        ["blue_tablecloth"],
    )
    assert facts == [
        ("on", "black_marker", "book"),
        ("on", "red_marker", "book"),
        ("on", "black_pen", "book"),
    ]


def test_belongs_on_paraphrase_maps_to_on(generator):
    result = generator.generate(
        "the wooden cube belongs on the shelf, not the table",
        ["wood_cube", "red_cup"],
        locations=["table", "shelf"],
    )
    assert result.ok is True
    assert result.facts == [("on", "wood_cube", "shelf")]


def test_belongs_on_multiword_location(generator):
    result = generator.generate(
        "the hammer belongs on the metal tray, not the table",
        ["hammer", "screwdriver"],
        locations=["table", "metal_tray"],
    )
    assert result.ok is True
    assert result.facts == [("on", "hammer", "metal_tray")]


def test_place_on_tray_aliases_to_target_tray(generator):
    result = generator.generate(
        "place the mug on the tray",
        ["mug", "cup"],
        locations=["table", "target_tray"],
    )
    assert result.ok is True
    assert result.facts == [("on", "mug", "target_tray")]


def test_unknown_object_fails_gracefully(generator):
    result = generator.generate("pick green_bottle", _OBJECTS)
    assert result.ok is False
    assert result.facts == []
    assert "unknown object" in (result.error or "")


def test_unsupported_command_fails_gracefully(generator):
    result = generator.generate("make me a sandwich", _OBJECTS, locations=_LOCATIONS)
    assert result.ok is False
    assert result.facts == []
    assert "unsupported command" in (result.error or "")


def test_empty_command_fails(generator):
    result = generator.generate("   ", _OBJECTS)
    assert result.ok is False
    assert result.error == "empty command"


def test_renderer_single_fact(renderer):
    section = renderer.render_section([("on", "red_cup", "shelf")])
    assert "  (:goal" in section
    assert "(on red_cup shelf)" in section
    assert "(and" not in section


def test_renderer_multiple_facts(renderer):
    section = renderer.render_section(
        [("on", "red_cup", "shelf"), ("on", "blue_box", "table")]
    )
    assert "(and" in section
    assert "(on red_cup shelf)" in section
    assert "(on blue_box table)" in section


def test_renderer_empty_facts_legacy_placeholder(renderer):
    section = renderer.render_section([])
    assert "(gripper-empty)" in section


# ── LLM text backend (mocked; no GPU / network) ─────────────────────────────


def _mock_complete_holding(system: str, user: str) -> str:
    assert "allowed_predicates" in user
    assert "holding" in user
    assert "red_cup" in user
    assert "pick" in user.lower() or "red cup" in user.lower()
    return json.dumps({"facts": [["holding", "red_cup"]]})


def test_llm_pick_with_mocked_client():
    gen = LLMGoalGenerator(complete_fn=_mock_complete_holding, fallback_rule_based=False)
    result = gen.generate("pick up the red cup", _OBJECTS)
    assert result.ok is True
    assert result.backend == "llm"
    assert result.facts == [("holding", "red_cup")]
    assert result.raw is not None


def test_llm_place_on_with_mocked_client():
    def complete(system: str, user: str) -> str:
        assert "on" in user
        assert "shelf" in user
        return '{"facts": [["on", "red_cup", "shelf"]]}'

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=False)
    result = gen.generate(
        "place red_cup on shelf",
        _OBJECTS,
        locations=_LOCATIONS,
    )
    assert result.ok is True
    assert result.facts == [("on", "red_cup", "shelf")]


def test_llm_rejects_disallowed_predicate():
    def complete(system: str, user: str) -> str:
        return json.dumps({"facts": [["flying", "red_cup"]]})

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=False)
    result = gen.generate("pick red_cup", _OBJECTS)
    assert result.ok is False
    assert "disallowed predicate" in (result.error or "")


def test_llm_rejects_unknown_symbol():
    def complete(system: str, user: str) -> str:
        return json.dumps({"facts": [["holding", "green_bottle"]]})

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=False)
    result = gen.generate("pick green bottle", _OBJECTS)
    assert result.ok is False
    assert "unknown symbol" in (result.error or "")


def test_llm_falls_back_to_rule_based_on_bad_json():
    def complete(system: str, user: str) -> str:
        return "not json at all"

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=True)
    result = gen.generate("pick red_cup", _OBJECTS)
    assert result.ok is True
    assert result.facts == [("holding", "red_cup")]
    assert result.backend == "llm"
    assert "llm_fallback_rule_based" in (result.error or "")


def test_llm_no_fallback_propagates_error():
    def complete(system: str, user: str) -> str:
        raise RuntimeError("network down")

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=False)
    result = gen.generate("pick red_cup", _OBJECTS)
    assert result.ok is False
    assert "LLM call failed" in (result.error or "")


def test_llm_prompt_includes_constraints():
    captured: dict[str, str] = {}

    def complete(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps({"facts": [["holding", "red_cup"]]})

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=False)
    gen.generate(
        "pick red_cup",
        _OBJECTS,
        locations=_LOCATIONS,
        domain_template="manipulation_base",
    )
    assert "Return ONLY a JSON object" in captured["system"]
    assert "allowed_predicates: camera-aimed-at, holding, on" in captured["user"]
    assert "objects: red_cup, blue_box, bowl, plate" in captured["user"]
    assert "locations: table, shelf, drawer" in captured["user"]


def test_llm_parses_fenced_json():
    def complete(system: str, user: str) -> str:
        return '```json\n{"facts": [["camera-aimed-at", "red_cup"]]}\n```'

    gen = LLMGoalGenerator(complete_fn=complete, fallback_rule_based=False)
    result = gen.generate("look at red_cup", _OBJECTS)
    assert result.ok is True
    assert result.facts == [("camera-aimed-at", "red_cup")]


def test_llm_empty_command():
    gen = LLMGoalGenerator(
        complete_fn=lambda s, u: "{}",
        fallback_rule_based=False,
    )
    result = gen.generate("  ", _OBJECTS)
    assert result.ok is False
    assert result.error == "empty command"


@pytest.mark.llm
def test_llm_live_gemini_smoke():
    """Optional live smoke when GOOGLE_API_KEY is set."""
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    pytest.importorskip("google.genai")

    gen = LLMGoalGenerator(fallback_rule_based=True)
    result = gen.generate(
        "pick up the red cup",
        _OBJECTS,
        locations=_LOCATIONS,
    )
    assert result.ok is True
    assert result.backend == "llm"
    assert result.facts
    assert result.facts[0][0] in {"holding", "on", "camera-aimed-at"}


# ── Local LLM backend (mocked generate_fn; no GPU) ──────────────────────────


def _table_scene() -> SceneState:
    return SceneState(
        objects=[
            ObjectFact(
                name="red_cup",
                source="mock",
                confidence=1.0,
                location="table",
                clear=True,
                pose=Pose(
                    position=Position(0.4, 0.0, 0.8),
                    orientation=Orientation(0.0, 0.0, 0.0, 1.0),
                ),
            ),
            ObjectFact(
                name="blue_box",
                source="mock",
                confidence=1.0,
                location="table",
                clear=True,
            ),
        ],
        locations=[
            LocationFact(name="table", source="mock", confidence=1.0, reachable=True),
            LocationFact(name="shelf", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[
            RelationFact(
                predicate="on",
                args=["red_cup", "table"],
                source="mock",
                confidence=1.0,
            ),
            RelationFact(
                predicate="on",
                args=["blue_box", "table"],
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
        domain_template="manipulation_base",
    )


def test_default_goal_model_not_larger_than_perceptual_vlm():
    """§3.3.1: default goal LLM family/size must stay ≤ default VLM (8B)."""
    assert "1.5B" in DEFAULT_GOAL_LLM_MODEL_ID or "3B" in DEFAULT_GOAL_LLM_MODEL_ID
    assert "8B" in DEFAULT_PERCEPTUAL_VLM_MODEL_ID
    assert DEFAULT_GOAL_LLM_MODEL_ID != DEFAULT_PERCEPTUAL_VLM_MODEL_ID


def test_compact_scene_omits_poses_by_default():
    scene = _table_scene()
    compact = compact_scene_dict(scene)
    assert "pose" not in compact["objects"][0]
    assert compact["objects"][0]["name"] == "red_cup"
    assert compact["objects"][0]["location"] == "table"
    assert compact["relations"][0] == {"predicate": "on", "args": ["red_cup", "table"]}
    assert compact["robot"]["gripper_empty"] is True


def test_compact_scene_can_include_poses():
    scene = _table_scene()
    compact = compact_scene_dict(scene, include_poses=True)
    assert "pose" in compact["objects"][0]


def test_prompt_templates_cover_manipulation_base():
    domains = list_prompt_domains()
    assert "manipulation_base" in domains
    assert "manipulation_base.v1" not in domains
    text = load_goal_prompt("manipulation_base")
    assert "holding" in text
    assert '"facts"' in text
    # Active default is hardened v2 (refusal / anti-hallucination).
    assert '{"facts": []}' in text or "REFUSE" in text


def test_prompt_version_pins_v1_and_v2():
    from planner.problem_generator.goal_generator.prompts import list_prompt_versions

    versions = list_prompt_versions("manipulation_base")
    assert "v1" in versions and "v2" in versions
    v1 = load_goal_prompt("manipulation_base", prompt_id="v1")
    v2 = load_goal_prompt("manipulation_base", prompt_id="v2")
    assert "holding" in v1 and "holding" in v2
    # v2 adds refusal / referring-expression guidance beyond the 9b baseline.
    assert "REFUSE" in v2 or '{"facts": []}' in v2
    assert len(v2) > len(v1)


def test_local_llm_pick_with_mocked_generate_fn():
    captured: dict[str, str] = {}

    def generate(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return json.dumps({"facts": [["holding", "red_cup"]]})

    gen = LocalLLMGoalGenerator(generate_fn=generate, fallback_rule_based=False)
    result = gen.generate(
        "pick up the red cup",
        _OBJECTS,
        locations=_LOCATIONS,
        scene_state=_table_scene(),
    )
    assert result.ok is True
    assert result.backend == "local_llm"
    assert result.facts == [("holding", "red_cup")]
    assert "compact_scene_state" in captured["user"]
    assert "red_cup" in captured["user"]
    assert "allowed_predicates" in captured["user"]
    assert "domain_actions:" in captured["user"]
    assert '"pose"' not in captured["user"]
    assert "holding" in captured["system"]


def test_local_llm_place_on_with_mocked_generate_fn():
    def generate(system: str, user: str) -> str:
        return json.dumps({"facts": [["on", "red_cup", "shelf"]]})

    gen = LocalLLMGoalGenerator(generate_fn=generate, fallback_rule_based=False)
    result = gen.generate(
        "place red_cup on shelf",
        _OBJECTS,
        locations=_LOCATIONS,
        scene_state=_table_scene(),
    )
    assert result.ok is True
    assert result.facts == [("on", "red_cup", "shelf")]


def test_local_llm_quantified_on_skips_model():
    """Do not let the goal LLM drop one of the quantified objects."""

    def generate(system: str, user: str) -> str:
        raise AssertionError("quantified on-facts must not call the text LLM")

    scene = SceneState(
        objects=[
            ObjectFact(name="black_marker", source="mock", confidence=1.0, location="blue_tablecloth", clear=True),
            ObjectFact(name="red_marker", source="mock", confidence=1.0, location="blue_tablecloth", clear=True),
            ObjectFact(name="black_pen", source="mock", confidence=1.0, location="blue_tablecloth", clear=True),
        ],
        locations=[
            LocationFact(name="blue_tablecloth", source="mock", confidence=1.0, reachable=True),
            LocationFact(name="book", source="mock", confidence=1.0, reachable=True),
        ],
        relations=[],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="mock",
            confidence=1.0,
        ),
        domain_template="manipulation_base",
    )
    gen = LocalLLMGoalGenerator(generate_fn=generate, fallback_rule_based=False)
    result = gen.generate(
        "all writing tools must be on the book",
        ["black_marker", "red_marker", "black_pen"],
        locations=["blue_tablecloth", "book"],
        scene_state=scene,
    )
    assert result.ok is True
    assert result.backend == "local_llm"
    assert set(result.facts) == {
        ("on", "black_marker", "book"),
        ("on", "red_marker", "book"),
        ("on", "black_pen", "book"),
    }


def test_local_llm_requires_scene_state_without_fallback():
    gen = LocalLLMGoalGenerator(
        generate_fn=lambda s, u: "{}",
        fallback_rule_based=False,
    )
    result = gen.generate("pick red_cup", _OBJECTS)
    assert result.ok is False
    assert "scene_state is required" in (result.error or "")


def test_local_llm_missing_scene_falls_back_to_rule_based():
    gen = LocalLLMGoalGenerator(
        generate_fn=lambda s, u: "{}",
        fallback_rule_based=True,
    )
    result = gen.generate("pick red_cup", _OBJECTS)
    assert result.ok is True
    assert result.facts == [("holding", "red_cup")]
    assert "local_llm_fallback_rule_based" in (result.error or "")


def test_local_llm_rejects_disallowed_predicate():
    def generate(system: str, user: str) -> str:
        return json.dumps({"facts": [["flying", "red_cup"]]})

    gen = LocalLLMGoalGenerator(generate_fn=generate, fallback_rule_based=False)
    result = gen.generate(
        "pick red_cup",
        _OBJECTS,
        locations=_LOCATIONS,
        scene_state=_table_scene(),
    )
    assert result.ok is False
    assert "disallowed predicate" in (result.error or "")


def test_local_llm_rejects_unknown_symbol():
    def generate(system: str, user: str) -> str:
        return json.dumps({"facts": [["holding", "green_bottle"]]})

    gen = LocalLLMGoalGenerator(generate_fn=generate, fallback_rule_based=False)
    result = gen.generate(
        "pick green bottle",
        _OBJECTS,
        locations=_LOCATIONS,
        scene_state=_table_scene(),
    )
    assert result.ok is False
    assert "unknown symbol" in (result.error or "")


def test_local_llm_falls_back_on_bad_json():
    def generate(system: str, user: str) -> str:
        return "not json at all"

    gen = LocalLLMGoalGenerator(generate_fn=generate, fallback_rule_based=True)
    result = gen.generate(
        "pick red_cup",
        _OBJECTS,
        locations=_LOCATIONS,
        scene_state=_table_scene(),
    )
    assert result.ok is True
    assert result.facts == [("holding", "red_cup")]
    assert "local_llm_fallback_rule_based" in (result.error or "")


def test_local_llm_empty_command():
    gen = LocalLLMGoalGenerator(
        generate_fn=lambda s, u: "{}",
        fallback_rule_based=False,
    )
    result = gen.generate("  ", _OBJECTS, scene_state=_table_scene())
    assert result.ok is False
    assert result.error == "empty command"


@pytest.mark.llm
@pytest.mark.gpu
def test_local_llm_live_transformers_smoke():
    """
    Optional live local smoke when weights are available.

    Skip unless ``VLMRP_GOAL_LLM_LIVE=1`` (avoids accidental multi-GB downloads
    in normal ``-m llm`` runs that only intend Gemini).
    """
    if os.environ.get("VLMRP_GOAL_LLM_LIVE") != "1":
        pytest.skip("set VLMRP_GOAL_LLM_LIVE=1 to run local transformers smoke")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    model_id = os.environ.get("VLMRP_GOAL_LLM_MODEL", DEFAULT_GOAL_LLM_MODEL_ID)
    gen = LocalLLMGoalGenerator(model_id=model_id, fallback_rule_based=True)
    result = gen.generate(
        "pick up the red cup",
        _OBJECTS,
        locations=_LOCATIONS,
        scene_state=_table_scene(),
    )
    assert result.ok is True
    assert result.backend == "local_llm"
    assert result.facts
    assert result.facts[0][0] in {"holding", "on", "camera-aimed-at"}
