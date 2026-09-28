"""Tests for hybrid loop runtime (Session 11 Phase A / A+)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.hybrid_runtime import (
    ENV_GOAL_BACKEND,
    ENV_HYBRID_FLAG,
    ENV_SCENE_SOURCE,
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    SceneSource,
    _TEMPLATE_EXTRA_PRIMITIVES,
    create_session_from_env,
    plan_hybrid_compatible,
    plan_mvp_compatible,
    resolve_goal_backend,
    resolve_hybrid_mode,
    resolve_scene_source,
    scene_source_skips_oracle,
    scene_from_xyz_poses,
)
from planner.problem_generator import generate_problem
from planner.problem_generator.init_generator.adapters.mock import (
    DinoMockAdapter,
    OracleMockAdapter,
)
from vlm.planner import PlanStep, VLMPlan


def _place_plan() -> VLMPlan:
    return VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": "shelf"},
            ),
        ],
        raw_output="",
        domain_template="manipulation_base",
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, HybridMode.OFF),
        ("", HybridMode.OFF),
        ("0", HybridMode.OFF),
        ("false", HybridMode.OFF),
        ("off", HybridMode.OFF),
        ("1", HybridMode.MVP),
        ("mvp", HybridMode.MVP),
        ("true", HybridMode.MVP),
        ("full", HybridMode.FULL),
        ("weird", HybridMode.OFF),
    ],
)
def test_resolve_hybrid_mode(raw, expected):
    assert resolve_hybrid_mode(raw) == expected
    if raw is None:
        assert resolve_hybrid_mode(env={}) == HybridMode.OFF
    else:
        assert resolve_hybrid_mode(env={ENV_HYBRID_FLAG: raw}) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, GoalBackend.RULE_BASED),
        ("", GoalBackend.RULE_BASED),
        ("rule_based", GoalBackend.RULE_BASED),
        ("local_llm", GoalBackend.LOCAL_LLM),
        ("local", GoalBackend.LOCAL_LLM),
    ],
)
def test_resolve_goal_backend(raw, expected):
    assert resolve_goal_backend(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, SceneSource.FUSED),
        ("", SceneSource.FUSED),
        ("fused", SceneSource.FUSED),
        ("dino", SceneSource.DINO),
        ("dino_only", SceneSource.DINO),
        ("real", SceneSource.DINO),
        ("inventory", SceneSource.INVENTORY),
        ("vlm", SceneSource.INVENTORY),
        ("vlm_inventory", SceneSource.INVENTORY),
        ("oracle", SceneSource.ORACLE),
        ("gt", SceneSource.ORACLE),
    ],
)
def test_resolve_scene_source(raw, expected):
    assert resolve_scene_source(raw) == expected
    if raw is None:
        assert resolve_scene_source(env={}) == SceneSource.FUSED
    else:
        assert resolve_scene_source(env={ENV_SCENE_SOURCE: raw}) == expected


def test_session_dino_only_ignores_oracle_in_fusion():
    """scene_source=dino → :init from DINO (+ tracker), not Gazebo oracle."""
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        scene_source=SceneSource.DINO,
        command="place red_cup on shelf",
    )
    pddl, fused = session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, dino_scene=dino
    )
    assert session.scene_source == SceneSource.DINO
    assert "dino" in (fused.meta.sources_used if fused.meta else [])
    assert "oracle" not in (fused.meta.sources_used if fused.meta else [])
    # Side-by-side compare still retains both raw streams for diagnostics.
    assert session.last_scene_compare is not None
    assert session.last_scene_compare["scene_source"] == "dino"
    assert session.last_scene_compare["fused_from"]["dino"] is True
    assert session.last_scene_compare["fused_from"]["oracle"] is False
    assert session.last_scene_compare["fused_from"]["requested"] == "dino"
    assert session.last_scene_compare["fused_from"]["init_fed_by"] == "dino"
    assert "(on red_cup shelf)" in pddl  # rule-based goal still works
    snap = session.metrics_snapshot()
    assert snap["scene_source"] == "dino"


def test_session_inventory_ignores_oracle_in_fusion():
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        scene_source=SceneSource.INVENTORY,
        command="place red_cup on shelf",
    )
    _, fused = session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, dino_scene=dino
    )
    assert scene_source_skips_oracle(SceneSource.INVENTORY)
    assert "oracle" not in (fused.meta.sources_used if fused.meta else [])
    assert session.last_scene_compare["fused_from"]["requested"] == "inventory"
    assert session.last_scene_compare["fused_from"]["oracle"] is False


def test_session_oracle_only_ignores_dino_in_fusion():
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        scene_source=SceneSource.ORACLE,
        command="place red_cup on shelf",
    )
    _, fused = session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, dino_scene=dino
    )
    assert "oracle" in (fused.meta.sources_used if fused.meta else [])
    assert "dino" not in (fused.meta.sources_used if fused.meta else [])
    assert session.last_scene_compare["fused_from"]["oracle"] is True
    assert session.last_scene_compare["fused_from"]["dino"] is False


def test_create_session_from_env_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV_HYBRID_FLAG, raising=False)
    assert create_session_from_env(command="pick red_cup") is None


def test_create_session_from_env_on(monkeypatch):
    monkeypatch.setenv(ENV_HYBRID_FLAG, "1")
    monkeypatch.setenv(ENV_GOAL_BACKEND, "local_llm")
    session = create_session_from_env(command="pick red_cup")
    assert session is not None
    assert session.mode == HybridMode.MVP
    assert session.goal_backend == GoalBackend.LOCAL_LLM


def test_plan_mvp_compatible_accepts_declared_enrichment():
    """Session 19: enrichment plans are hybrid-compatible when new_actions declares the primitive."""
    plan = _place_plan()
    assert plan_mvp_compatible(plan) is True
    plan.domain_additions = {"new_actions": [{"name": "pour"}]}
    # pour is not in the plan steps here, so it's still compatible
    assert plan_mvp_compatible(plan) is True


def test_plan_mvp_rejects_undeclared_pour_primitive():
    """pour in steps but NOT declared in domain_additions → rejected."""
    plan = VLMPlan(
        goal="pour",
        steps=[PlanStep(primitive="pour", args={"object": "cup"})],
        raw_output="",
        domain_template="manipulation_base",
    )
    assert plan_mvp_compatible(plan) is False


def test_plan_mvp_accepts_declared_pour_primitive():
    """Session 19: pour in steps AND declared in domain_additions → accepted."""
    plan = VLMPlan(
        goal="pour",
        steps=[PlanStep(primitive="pour", args={"object": "cup"})],
        raw_output="",
        domain_template="manipulation_base",
        domain_additions={"new_actions": [{"name": "pour"}]},
    )
    assert plan_mvp_compatible(plan) is True


def test_session_goal_once_and_tracker_holding():
    oracle = OracleMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
    )
    pddl0, _ = session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, problem_name="t0"
    )
    assert "(on red_cup shelf)" in pddl0
    first_goal = list(session.goal_facts)

    session.note_completed(PlanStep(primitive="pick", args={"object": "red_cup"}))
    assert session.is_holding

    place_only = VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": "shelf"},
            )
        ],
        raw_output="",
        domain_template="manipulation_base",
    )
    pddl1, _ = session.generate_hybrid_problem(
        place_only, oracle_scene=oracle, problem_name="t1"
    )
    assert session.goal_facts == first_goal
    assert "(holding red_cup)" in pddl1


def test_invalidate_goal_keeps_the_last_goal_for_the_end_of_run_check():
    """A failing run replans last, so the goal check needs the surviving copy."""
    oracle = OracleMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
    )
    session.generate_hybrid_problem(
        _place_plan(), oracle_scene=oracle, problem_name="t0"
    )
    goal = list(session.goal_facts)
    assert goal

    session.invalidate_goal()
    assert session.goal_facts is None
    assert session.last_goal_facts == goal

    snap = session.metrics_snapshot()
    assert snap["goal_facts"] == []
    assert snap["last_goal_facts"] == [list(f) for f in goal]


def test_session_local_llm_goal_backend_mocked():
    oracle = OracleMockAdapter.load()

    def gen(system: str, user: str) -> str:
        assert "compact_scene_state" in user
        return json.dumps({"facts": [["on", "red_cup", "shelf"]]})

    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.LOCAL_LLM,
        command="place red_cup on shelf",
        local_generate_fn=gen,
    )
    pddl, _ = session.generate_hybrid_problem(_place_plan(), oracle_scene=oracle)
    assert session.goal_backend_used == "local_llm"
    assert "(on red_cup shelf)" in pddl


def test_session_local_llm_falls_back_to_rule_based():
    oracle = OracleMockAdapter.load()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.LOCAL_LLM,
        command="pick red_cup",
        local_generate_fn=lambda s, u: "not-json",
    )
    pddl, _ = session.generate_hybrid_problem(
        VLMPlan(
            goal="pick red_cup",
            steps=[PlanStep(primitive="pick", args={"object": "red_cup"})],
            raw_output="",
            domain_template="manipulation_base",
        ),
        oracle_scene=oracle,
    )
    assert "(holding red_cup)" in pddl
    assert session.goal_backend_used in {"local_llm", "rule_based"}
    assert session.goal_fallback_count == 1
    snap = session.metrics_snapshot()
    assert snap["goal_fallback_count"] == 1
    assert snap["goal_backend"] == "local_llm"


def test_flag_off_generate_problem_unchanged():
    """Default facade path must stay legacy regardless of hybrid module."""
    plan = _place_plan()
    legacy = generate_problem(plan)
    assert "(on red_cup shelf)" in legacy


def test_scene_from_xyz_poses():
    scene = scene_from_xyz_poses(
        {"red_cup": {"x": 0.4, "y": 0.0, "z": 0.8}},
        known_locations=["table"],
        on_surface={"red_cup": "table"},
        domain_template="manipulation_base",
    )
    assert scene.objects[0].name == "red_cup"
    assert any(r.predicate == "on" for r in scene.relations)


def test_partition_gazebo_treats_shelf_b_as_location():
    from planner.hybrid_runtime import (
        ground_plan_locations,
        partition_gazebo_for_hybrid,
        rewrite_command_locations,
        resolve_location_alias,
    )

    poses = {
        "red_cup": {"x": 0.5, "y": 0.1, "z": 0.8},
        "shelf_b": {"x": 0.5, "y": -0.25, "z": 0.78},
        "blue_box": {"x": 0.5, "y": -0.1, "z": 0.8},
    }
    items, locs = partition_gazebo_for_hybrid(
        poses, base_locations=["table", "shelf"]
    )
    assert "shelf_b" not in items
    assert "red_cup" in items and "blue_box" in items
    assert "shelf_b" in locs and "table" in locs
    assert "shelf" not in locs  # superseded by shelf_b
    assert "drawer" not in locs  # no shipped world has one — do not invent it

    # Explicit base still may list drawer for containers worlds that spawn one.
    _, locs_with_drawer = partition_gazebo_for_hybrid(
        poses, base_locations=["table", "shelf", "drawer"]
    )
    assert "drawer" in locs_with_drawer

    assert resolve_location_alias("shelf", locs) == "shelf_b"
    cmd = rewrite_command_locations("place the red cup on the shelf", locs)
    assert "shelf_b" in cmd

    tray_locs = ["table", "target_tray"]
    assert resolve_location_alias("tray", tray_locs) == "target_tray"
    tray_cmd = rewrite_command_locations("place the mug on the tray", tray_locs)
    assert "target_tray" in tray_cmd
    _, kitchen_locs = partition_gazebo_for_hybrid(
        {
            "mug": {"x": 0.4, "y": 0.0, "z": 0.8},
            "target_tray": {"x": 0.6, "y": 0.0, "z": 0.8},
        },
        base_locations=["table", "tray"],
    )
    assert "target_tray" in kitchen_locs
    assert "tray" not in kitchen_locs

    plan = _place_plan()
    notes = ground_plan_locations(plan, locs)
    assert notes == ["shelf→shelf_b"]
    assert plan.steps[1].args["location"] == "shelf_b"

    scene = scene_from_xyz_poses(
        items,
        known_locations=locs,
        on_surface={n: "table" for n in items},
    )
    assert "shelf_b" not in {o.name for o in scene.objects}
    assert "shelf_b" in {loc.name for loc in scene.locations}
    assert ("shelf_b", "table") not in {
        (r.args[0], r.args[1]) for r in scene.relations if r.predicate == "on"
    }


def test_smoke_script_mock_exits_zero():
    from importlib.util import module_from_spec, spec_from_file_location

    path = Path(__file__).resolve().parent.parent / "scripts" / "smoke_hybrid_pipeline.py"
    spec = spec_from_file_location("smoke_hybrid_pipeline", path)
    mod = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.run_mock_smoke(goal_backend="rule_based") == 0
    assert mod.run_mock_smoke(goal_backend="local_llm") == 0
    assert mod.run_full_verifier_smoke() == 0
    assert mod.run_live_goal_smoke(dry=True, write_fixture=True) == 0
    assert mod.run_live_verifier_smoke(dry=True, write_fixture=True) == 0


def test_full_mode_green_updates_tracker_without_vlm():
    from planner.problem_generator.init_generator.vlm_fusion import MockVlmFusionClient
    from planner.state_verifier import StateVerifier

    client = MockVlmFusionClient()
    session = HybridProblemSession(
        mode=HybridMode.FULL,
        command="place red_cup on shelf",
        vlm_fusion_client=client,
    )
    oracle = OracleMockAdapter.load()
    _, pre = session.generate_hybrid_problem(_place_plan(), oracle_scene=oracle)
    pick = PlanStep(primitive="pick", args={"object": "red_cup"})
    expected = StateVerifier().expect(pre, pick)

    out = session.process_completed_step(
        pick, observed=expected, pre_scene=pre, success_flag=True
    )
    assert out.verdict == "GREEN"
    assert out.tracker_updated is True
    assert out.vlm_calls == 0
    assert client.call_count == 0
    assert session.is_holding


def test_full_mode_yellow_resolves_with_one_vlm_call():
    from planner.problem_generator.init_generator.schema import SceneState
    from planner.problem_generator.init_generator.vlm_fusion import MockVlmFusionClient
    from planner.state_verifier import StateVerifier

    client = MockVlmFusionClient()
    session = HybridProblemSession(
        mode=HybridMode.FULL,
        command="place red_cup on shelf",
        vlm_fusion_client=client,
    )
    oracle = OracleMockAdapter.load()
    _, pre = session.generate_hybrid_problem(_place_plan(), oracle_scene=oracle)
    pick = PlanStep(primitive="pick", args={"object": "red_cup"})
    expected = StateVerifier().expect(pre, pick)
    stale = SceneState(
        objects=list(expected.objects),
        locations=list(expected.locations),
        relations=list(pre.relations),
        robot=expected.robot,
    )

    out = session.process_completed_step(
        pick, observed=stale, pre_scene=pre, success_flag=True
    )
    assert out.verdict == "GREEN"
    assert out.vlm_calls == 1
    assert client.call_count == 1
    assert out.tracker_updated is True


def test_full_mode_red_replans_without_vlm_or_tracker():
    from planner.problem_generator.init_generator.vlm_fusion import MockVlmFusionClient

    client = MockVlmFusionClient()
    session = HybridProblemSession(
        mode=HybridMode.FULL,
        command="place red_cup on shelf",
        vlm_fusion_client=client,
    )
    oracle = OracleMockAdapter.load()
    _, pre = session.generate_hybrid_problem(_place_plan(), oracle_scene=oracle)
    assert session.goal_facts is not None

    out = session.process_completed_step(
        PlanStep(primitive="pick", args={"object": "red_cup"}),
        observed=pre,
        pre_scene=pre,
        success_flag=False,
    )
    assert out.verdict == "RED"
    assert out.replan is True
    assert out.tracker_updated is False
    assert out.vlm_calls == 0
    assert client.call_count == 0
    assert session.goal_facts is None
    assert not session.is_holding


def test_mvp_mode_skips_verifier():
    from planner.problem_generator.init_generator.vlm_fusion import MockVlmFusionClient

    client = MockVlmFusionClient()
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="pick red_cup",
        vlm_fusion_client=client,
    )
    oracle = OracleMockAdapter.load()
    session.generate_hybrid_problem(
        VLMPlan(
            goal="pick red_cup",
            steps=[PlanStep(primitive="pick", args={"object": "red_cup"})],
            raw_output="",
            domain_template="manipulation_base",
        ),
        oracle_scene=oracle,
    )
    out = session.process_completed_step(
        PlanStep(primitive="pick", args={"object": "red_cup"}),
        observed=oracle,
        success_flag=True,
    )
    assert out.verdict == "GREEN"
    assert out.tracker_updated is True
    assert client.call_count == 0
    assert session.verifier_enabled is False


# ---------------------------------------------------------------------------
# Session 18 — domain / template selection
# ---------------------------------------------------------------------------

def _stacking_plan() -> VLMPlan:
    """Minimal stacking plan: unstack + stack, template=manipulation_stacking."""
    return VLMPlan(
        goal="stack bowl on plate",
        steps=[
            PlanStep(primitive="unstack", args={"object": "bowl", "from": "shelf"}),
            PlanStep(primitive="stack", args={"object": "bowl", "on": "plate"}),
        ],
        raw_output="",
        domain_template="manipulation_stacking",
    )


def _containers_plan() -> VLMPlan:
    """Minimal containers plan: pick-from-container + place-in-container."""
    return VLMPlan(
        goal="place cup in drawer",
        steps=[
            PlanStep(primitive="pick-from-container", args={"object": "cup", "container": "box"}),
            PlanStep(primitive="place-in-container", args={"object": "cup", "container": "drawer"}),
        ],
        raw_output="",
        domain_template="containers_manipulation",
    )


def _navigation_plan() -> VLMPlan:
    """Minimal navigation plan: navigate-to + pick."""
    return VLMPlan(
        goal="pick cup from kitchen",
        steps=[
            PlanStep(primitive="navigate-to", args={"location": "kitchen"}),
            PlanStep(primitive="pick", args={"object": "cup"}),
        ],
        raw_output="",
        domain_template="navigation_manipulation",
    )


@pytest.mark.parametrize(
    "plan_fn,expected",
    [
        (_place_plan, True),        # base template — still compatible
        (_stacking_plan, True),     # stacking primitives now accepted
        (_containers_plan, True),   # containers primitives now accepted
        (_navigation_plan, True),   # navigation primitive now accepted
    ],
)
def test_plan_hybrid_compatible_non_base_templates(plan_fn, expected):
    """Session 18: non-base template plans accepted by the hybrid gate."""
    assert plan_hybrid_compatible(plan_fn()) is expected


def test_plan_hybrid_compatible_enrichment_accepted_with_declared_actions():
    """Session 19: enrichment with declared new_actions is hybrid-compatible."""
    plan = _stacking_plan()
    plan.domain_additions = {"new_actions": [{"name": "pour"}]}
    assert plan_hybrid_compatible(plan) is True


def test_plan_mvp_compatible_alias_still_works():
    """plan_mvp_compatible must remain a working alias for backward compat."""
    assert plan_mvp_compatible(_place_plan()) is True
    assert plan_mvp_compatible(_stacking_plan()) is True


def test_plan_hybrid_compatible_unknown_primitive_rejected():
    """Unknown primitives (enrichment-like) are still blocked."""
    plan = VLMPlan(
        goal="pour water",
        steps=[PlanStep(primitive="pour", args={"object": "cup"})],
        raw_output="",
        domain_template="manipulation_base",
    )
    assert plan_hybrid_compatible(plan) is False


@pytest.mark.parametrize(
    "template,expected_domain",
    [
        ("manipulation_base",       "manipulation-base"),
        ("manipulation_stacking",   "manipulation-stacking"),
        ("containers_manipulation", "manipulation-containers"),
        ("navigation_manipulation", "manipulation-navigation"),
    ],
)
def test_hybrid_problem_uses_correct_domain_name(template, expected_domain):
    """
    Session 18: generate_problem_hybrid resolves the correct :domain name from
    plan.domain_template via DOMAIN_TEMPLATE_TO_NAME.
    """
    from planner.problem_generator import generate_problem
    from planner.problem_generator.init_generator.adapters.mock import OracleMockAdapter

    oracle = OracleMockAdapter.load()
    oracle.domain_template = template
    plan = VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(primitive="place", args={"object": "red_cup", "location": "shelf"}),
        ],
        raw_output="",
        domain_template=template,
    )
    pddl = generate_problem(plan, scene_state=oracle, use_hybrid=True)
    assert f"(:domain {expected_domain})" in pddl


def _make_scene(objects: list[str], locations: list[str], template: str) -> "SceneState":
    """Build a minimal SceneState for Session 18 template tests."""
    from planner.problem_generator.init_generator.schema import (
        ObjectFact,
        LocationFact,
        RobotFacts,
        SceneState as SS,
    )

    return SS(
        objects=[ObjectFact(name=o, source="oracle", confidence=1.0) for o in objects],
        locations=[LocationFact(name=loc, source="oracle", confidence=1.0) for loc in locations],
        relations=[],
        robot=RobotFacts(
            gripper_empty=True,
            holding=None,
            camera_aimed_at=None,
            source="oracle",
            confidence=1.0,
        ),
        domain_template=template,
    )


def test_stacking_plan_hybrid_pddl_uses_stacking_domain():
    """
    Session 18: a stacking plan (stack/unstack) produces (:domain manipulation-stacking)
    in hybrid mode without falling back to legacy.
    """
    scene = _make_scene(["bowl", "plate"], ["shelf", "table"], "manipulation_stacking")
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="stack bowl on plate",
        domain_template="manipulation_stacking",
    )
    pddl, _ = session.generate_hybrid_problem(
        _stacking_plan(),
        oracle_scene=scene,
        problem_name="s18_stacking",
    )
    assert "(:domain manipulation-stacking)" in pddl


def test_containers_plan_hybrid_pddl_uses_containers_domain():
    """
    Session 18: a containers plan produces (:domain manipulation-containers)
    in hybrid mode.
    """
    scene = _make_scene(["cup"], ["drawer", "box", "table"], "containers_manipulation")
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="place cup in drawer",
        domain_template="containers_manipulation",
    )
    pddl, _ = session.generate_hybrid_problem(
        _containers_plan(),
        oracle_scene=scene,
        problem_name="s18_containers",
    )
    assert "(:domain manipulation-containers)" in pddl
