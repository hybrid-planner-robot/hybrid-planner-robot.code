"""
Session 30 — the enriched-domain loop, end to end and offline.

Session 29 stopped at a persisted PDDL file; here the enriched action has to
survive the whole round trip: FD plans it, the executor runs it, the verifier
judges it, the tracker remembers it, and the next ``:init`` carries the fluent
so the goal is finally satisfied.

Everything is mocked: the single text LLM is the same canned ``generate_fn``
Session 29 uses (``tests/fixtures/domain_llm/``), Fast Downward is a stub
unless the real binary happens to be on PATH, and nothing is written outside
``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.domain_llm import LLMGoalBinder, parse_binding_payload
from planner.enrichment_effects import (
    CATALOG_SKILL_EFFECTS,
    EnrichmentContext,
    catalog_skill_effect,
    ground_enrichment_goal,
    step_args_for_catalog_action,
)
from planner.hybrid_runtime import (
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    SceneSource,
    enrichment_provenance,
    make_domain_stub_plan,
    plan_hybrid_compatible,
)
from planner.online_enrichment import (
    EnrichmentStatus,
    SelectionBackend,
    make_domain_selector,
    make_goal_binder,
    make_online_enricher,
    resolve_domain_for_task,
)
from planner.plan_parser import normalize_to_primitives, parse_plan
from planner.primitive_transitions import (
    CompletedAction,
    has_local_model,
    relation_scope,
)
from planner.problem_generator.init_generator.renderer import InitRenderer
from planner.problem_generator.init_generator.schema import (
    ObjectFact,
    RelationFact,
    RobotFacts,
    SceneState,
)
from planner.state_tracker import StateTracker
from planner.state_verifier import StateVerifier

from test_domain_llm import MockTextLLM

_POUR_ADDITIONS = {
    "new_types": [],
    "new_predicates": ["(poured ?src - item ?dst - item)"],
    "new_actions": [
        {
            "name": "pour",
            "parameters": "(?src - item ?dst - item)",
            "precondition": "(and (holding ?src) (camera-aimed-at ?dst))",
            "effect": "(and (poured ?src ?dst) (clear ?src))",
        }
    ],
    "modified_preconditions": {},
}

_LEVITATE_ADDITIONS = {
    "new_types": [],
    "new_predicates": ["(floating ?i - item)"],
    "new_actions": [
        {
            "name": "levitate",
            "parameters": "(?i - item)",
            "precondition": "(gripper-empty)",
            "effect": "(and (floating ?i))",
        }
    ],
    "modified_preconditions": {},
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _pour_context() -> EnrichmentContext:
    return EnrichmentContext.from_domain_additions(_POUR_ADDITIONS)


def _scene(
    *,
    holding: str | None = None,
    poured: bool = False,
    bottle_on: str | None = "table",
) -> SceneState:
    relations: list[RelationFact] = []
    if bottle_on and holding != "bottle":
        relations.append(
            RelationFact(
                predicate="on",
                args=["bottle", bottle_on],
                source="oracle",
                confidence=1.0,
            )
        )
    relations.append(
        RelationFact(
            predicate="on", args=["glass", "table"], source="oracle", confidence=1.0
        )
    )
    if poured:
        relations.append(
            RelationFact(
                predicate="poured",
                args=["bottle", "glass"],
                source="tracker",
                confidence=1.0,
            )
        )
    return SceneState(
        objects=[
            ObjectFact(name="bottle", source="oracle", confidence=1.0, location="table"),
            ObjectFact(name="glass", source="oracle", confidence=1.0, location="table"),
        ],
        locations=[],
        relations=relations,
        robot=RobotFacts(
            gripper_empty=holding is None,
            holding=holding,
            camera_aimed_at="glass",
            source="oracle",
            confidence=1.0,
        ),
        domain_template="manipulation_base",
    )


def _binder(mapping: dict[str, list[str]] | None = None) -> LLMGoalBinder:
    """Goal binder backed by a canned reply per action name."""
    table = mapping or {"pour": ["bottle", "glass"]}

    def generate(system: str, user: str) -> str:
        for line in user.splitlines():
            if line.startswith("action:"):
                name = line.split(":", 1)[1].strip()
                if name in table:
                    return json.dumps({"bindings": table[name]})
        return json.dumps({"refuse": True, "reason": "unknown action"})

    return LLMGoalBinder(generate_fn=generate)


# ── 1. Catalog effects come from the ROS bodies ──────────────────────────────


def test_only_pour_releases_the_gripper():
    assert catalog_skill_effect("pour").releases_object is True
    for skill in ("tilt", "stir", "cut"):
        assert catalog_skill_effect(skill).releases_object is False
    # None of them can be confirmed by perception — that drives the YELLOW gate.
    assert not any(e.perceptually_verifiable for e in CATALOG_SKILL_EFFECTS.values())


def test_enriched_action_has_no_local_model_without_context():
    action = CompletedAction("pour", {"source": "bottle", "target": "glass"})
    assert has_local_model("pour") is False
    assert has_local_model("pour", _pour_context()) is True
    # An action the catalog cannot execute stays unmodelled even with a context.
    open_vocab = EnrichmentContext.from_domain_additions(_LEVITATE_ADDITIONS)
    assert has_local_model("levitate", open_vocab) is False
    assert relation_scope(action, _pour_context()) == {"glass"}


def test_context_reads_fluent_from_the_authored_effect():
    context = _pour_context()
    assert context.predicates() == frozenset({"poured"})
    assert context.fact_for("pour", {"source": "bottle", "target": "glass"}) == (
        "poured",
        "bottle",
        "glass",
    )
    # Missing arguments must not produce a half-bound fact.
    assert context.fact_for("pour", {"source": "bottle"}) is None


# ── 2. Tracker + verifier hooks ──────────────────────────────────────────────


def test_tracker_records_the_fluent_and_the_release():
    tracker = StateTracker(enrichment=_pour_context())
    tracker.apply(CompletedAction("pick", {"object": "bottle"}))
    assert tracker.snapshot().holding == "bottle"

    tracker.apply(CompletedAction("pour", {"source": "bottle", "target": "glass"}))
    snap = tracker.snapshot()
    assert snap.holding is None and snap.gripper_empty is True
    assert tracker.enrichment_facts() == [("poured", "bottle", "glass")]
    assert {
        (r["predicate"], tuple(r["args"]))
        for r in tracker.as_partial_scene()["relations"]
    } == {("poured", ("bottle", "glass"))}


def test_tracker_ignores_enriched_actions_without_context():
    tracker = StateTracker()
    tracker.apply(CompletedAction("pick", {"object": "bottle"}))
    tracker.apply(CompletedAction("pour", {"source": "bottle", "target": "glass"}))
    assert tracker.snapshot().holding == "bottle"
    assert tracker.enrichment_facts() == []


def test_clean_pour_is_yellow_then_green_and_never_a_silent_green():
    verifier = StateVerifier(enrichment=_pour_context())
    pre = _scene(holding="bottle")
    action = CompletedAction("pour", {"source": "bottle", "target": "glass"})
    observed = _scene(holding=None)

    result = verifier.verify(pre, action, observed)
    # No VLM wired: the gate re-scores and settles on GREEN, but the reason the
    # script could not decide alone is kept in the record.
    assert result.verdict == "GREEN"
    assert any("not perceptually verifiable" in m for m in result.mismatches)


def test_pour_that_did_not_let_go_stays_yellow():
    seen: list[str] = []

    def vlm(expected, observed, action, mismatches):
        seen.append(action.primitive)
        return observed  # VLM confirms the robot is still holding the bottle

    verifier = StateVerifier(enrichment=_pour_context(), vlm_callback=vlm)
    result = verifier.verify(
        _scene(holding="bottle"),
        CompletedAction("pour", {"source": "bottle", "target": "glass"}),
        _scene(holding="bottle"),
    )
    assert result.verdict == "YELLOW"
    assert result.request_vlm is True
    assert seen == ["pour"]


def test_verifier_untouched_for_standard_primitives():
    verifier = StateVerifier(enrichment=_pour_context())
    result = verifier.verify(
        _scene(),
        CompletedAction("pick", {"object": "bottle"}),
        _scene(holding="bottle", bottle_on=None),
    )
    assert result.verdict == "GREEN"
    assert not any("perceptually verifiable" in m for m in result.mismatches)


# ── 3. The fluent reaches :init and :goal ────────────────────────────────────


def test_init_renderer_emits_declared_enrichment_predicates():
    scene = _scene(poured=True)
    with pytest.warns(UserWarning, match="unknown relation predicate"):
        # Undeclared predicates keep warning — that is what catches typos.
        assert ("poured", "bottle", "glass") not in InitRenderer().render_facts(scene)
    facts = InitRenderer(extra_predicates=frozenset({"poured"})).render_facts(scene)
    assert ("poured", "bottle", "glass") in facts


def test_goal_binding_prefers_the_llm_for_a_paraphrase():
    context = _pour_context()
    command = "get me something to drink"  # names neither vessel
    symbols = ["bottle", "glass", "table"]

    assert ground_enrichment_goal(context, command, symbols) == []
    assert ground_enrichment_goal(context, command, symbols, binder=_binder()) == [
        ("_raw_fact", "(poured bottle glass)")
    ]


def test_goal_binding_falls_back_to_command_order():
    context = _pour_context()
    bound = ground_enrichment_goal(
        context,
        "pour the bottle into the glass",
        ["glass", "bottle", "table"],
        binder=_binder({}),  # model declines
    )
    assert bound == [("_raw_fact", "(poured bottle glass)")]


def test_binding_payload_rejects_objects_outside_the_scene():
    with pytest.raises(Exception):
        parse_binding_payload(
            json.dumps({"bindings": ["carafe", "glass"]}),
            arity=2,
            scene_symbols=["bottle", "glass"],
        )
    assert (
        parse_binding_payload(
            json.dumps({"refuse": True, "reason": "no vessel"}),
            arity=2,
            scene_symbols=["bottle", "glass"],
        )
        is None
    )


# ── 4. Hybrid stays on; open-vocab is flagged ────────────────────────────────


def test_catalog_enrichment_keeps_hybrid_on():
    plan = make_domain_stub_plan(
        "get me something to drink",
        "manipulation_base",
        domain_additions=_POUR_ADDITIONS,
    )
    assert enrichment_provenance(plan) == "catalog"
    assert plan_hybrid_compatible(plan) is True

    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="get me something to drink",
        enrichment=_pour_context(),
    )
    assert session.should_use_hybrid(plan) is True
    assert session.enrichment_used is True


def test_open_vocab_enrichment_is_labelled_for_ablation():
    plan = make_domain_stub_plan(
        "levitate the cup",
        "manipulation_base",
        domain_additions=_LEVITATE_ADDITIONS,
    )
    assert enrichment_provenance(plan) == "open_vocab"
    # Still hybrid-compatible (Session 19 behaviour), just not the paper path.
    assert plan_hybrid_compatible(plan) is True


def test_fd_action_arguments_are_named_from_the_catalog_signature():
    prims = normalize_to_primitives(parse_plan(["(pour bottle glass)"]))
    assert [(p.name, p.args) for p in prims] == [("pour", ["bottle", "glass"])]
    assert step_args_for_catalog_action("pour", ["bottle", "glass"]) == {
        "source": "bottle",
        "target": "glass",
    }
    assert step_args_for_catalog_action("stir", ["bowl"]) == {"container": "bowl"}


# ── 5. End to end: select → enrich → persist → plan → execute → verify ───────


class _StubFD:
    """Fast Downward stand-in: replies from the goal facts in the problem."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def solve_from_strings(self, domain: str, problem: str) -> list[str]:
        self.problems.append(problem)
        assert "(:action pour" in domain, "FD must receive the enriched domain"
        if "(poured bottle glass)" in _goal_block(problem):
            if "(poured bottle glass)" in _init_block(problem):
                return []  # already satisfied
            return ["(pick bottle table)", "(pour bottle glass)"]
        return ["(pick bottle table)"]


def _init_block(problem: str) -> str:
    start = problem.index("(:init")
    return problem[start : problem.index("(:goal")]


def _goal_block(problem: str) -> str:
    return problem[problem.index("(:goal") :]


@pytest.mark.parametrize(
    "command",
    [
        "get me something to drink",  # paraphrase: never says "pour"
        "i am thirsty",  # English paraphrase
    ],
)
def test_paraphrase_runs_the_enriched_loop_end_to_end(tmp_path, command):
    llm = MockTextLLM()

    resolution = resolve_domain_for_task(
        command,
        enrichment_enabled=True,
        selector=make_domain_selector(SelectionBackend.LLM, generate_fn=llm),
        enricher=make_online_enricher(
            SelectionBackend.LLM, generate_fn=llm, enriched_dir=tmp_path
        ),
        scene_symbols=("bottle", "glass", "table"),
    )
    enrichment = resolution.enrichment
    assert enrichment is not None
    assert enrichment.status == EnrichmentStatus.ENRICHED
    assert enrichment.skills_grounded == ("pour",)
    assert Path(enrichment.domain_path).exists()

    context = EnrichmentContext.from_domain_additions(enrichment.domain_additions)
    session = HybridProblemSession(
        mode=HybridMode.FULL,
        goal_backend=GoalBackend.RULE_BASED,
        scene_source=SceneSource.ORACLE,
        command=command,
        domain_template=resolution.template,
    )
    session.set_enrichment(
        context,
        skills=enrichment.skills_grounded,
        domain_path=enrichment.domain_path,
        backend=resolution.selection.backend,
        goal_binder=_binder(),
    )

    plan = make_domain_stub_plan(
        command,
        resolution.template,
        domain_additions=enrichment.domain_additions,
    )
    assert session.should_use_hybrid(plan) is True

    # ── Plan: :goal carries the enriched fluent, FD sees the enriched domain ──
    problem, _ = session.generate_hybrid_problem(plan, oracle_scene=_scene())
    assert "(poured bottle glass)" in _goal_block(problem)
    assert "(poured bottle glass)" not in _init_block(problem)

    fd = _StubFD()
    actions = fd.solve_from_strings(enrichment.domain_text, problem)
    steps = [
        (p.name, step_args_for_catalog_action(p.name, p.args))
        for p in normalize_to_primitives(parse_plan(actions))
    ]
    assert [name for name, _ in steps] == ["pick", "pour"]

    # ── Execute + verify ─────────────────────────────────────────────────────
    session.last_fused_scene = _scene()
    pick = session.process_completed_step(
        {"primitive": "pick", "args": {"object": "bottle"}},
        observed=_scene(holding="bottle", bottle_on=None),
        pre_scene=_scene(),
    )
    assert pick.verdict == "GREEN" and pick.tracker_updated

    pour = session.process_completed_step(
        {"primitive": "pour", "args": steps[1][1]},
        observed=_scene(holding=None),
        pre_scene=_scene(holding="bottle"),
    )
    assert pour.verdict == "GREEN" and pour.tracker_updated
    assert any("not perceptually verifiable" in m for m in pour.mismatches)
    assert session.tracker.enrichment_facts() == [("poured", "bottle", "glass")]

    # ── Replan: the fluent is now in :init, so the goal holds ────────────────
    session.invalidate_goal()
    problem2, _ = session.generate_hybrid_problem(plan, oracle_scene=_scene())
    assert "(poured bottle glass)" in _init_block(problem2)
    assert fd.solve_from_strings(enrichment.domain_text, problem2) == []

    metrics = session.metrics_snapshot()
    assert metrics["enrichment_used"] is True
    assert metrics["enrichment_actions"] == ["pour"]
    assert metrics["domain_persisted"] == enrichment.domain_path
    assert metrics["domain_select_backend"] == "llm"
    assert metrics["refuse_reason"] is None
    assert metrics["enrichment_facts"] == [["poured", "bottle", "glass"]]


def test_complete_task_never_enriches_and_keeps_metrics_clean(tmp_path):
    llm = MockTextLLM()
    resolution = resolve_domain_for_task(
        "place red_cup on shelf",
        enrichment_enabled=True,
        selector=make_domain_selector(SelectionBackend.LLM, generate_fn=llm),
        enricher=make_online_enricher(
            SelectionBackend.LLM, generate_fn=llm, enriched_dir=tmp_path
        ),
    )
    assert resolution.enrichment is None or (
        resolution.enrichment.status == EnrichmentStatus.SKIPPED
    )
    assert llm.enrich_calls == 0

    session = HybridProblemSession(mode=HybridMode.MVP, command="place red_cup on shelf")
    session.set_enrichment(None, backend=resolution.selection.backend)
    metrics = session.metrics_snapshot()
    assert metrics["enrichment_used"] is False
    assert metrics["domain_persisted"] is None
    assert metrics["enrichment_facts"] == []


def test_refuse_surfaces_a_reason_and_leaves_no_domain(tmp_path):
    llm = MockTextLLM()
    resolution = resolve_domain_for_task(
        "solder the broken wire",
        enrichment_enabled=True,
        selector=make_domain_selector(SelectionBackend.LLM, generate_fn=llm),
        enricher=make_online_enricher(
            SelectionBackend.LLM, generate_fn=llm, enriched_dir=tmp_path
        ),
    )
    assert resolution.refused is True
    message = resolution.enrichment.refuse_message
    assert message and "catalog" in message.lower()
    assert list(tmp_path.glob("*.pddl")) == []

    session = HybridProblemSession(mode=HybridMode.MVP, command="solder the broken wire")
    session.set_enrichment(None, backend=resolution.selection.backend)
    session.refuse_reason = message
    assert session.metrics_snapshot()["refuse_reason"] == message


@pytest.mark.skipif(
    shutil.which("fast-downward.py") is None
    and shutil.which("fast-downward") is None,
    reason="Fast Downward not on PATH (offline suite uses the stub planner)",
)
def test_real_fd_solves_the_enriched_domain(tmp_path):
    from planner.fast_downward import FastDownwardPlanner

    llm = MockTextLLM()
    resolution = resolve_domain_for_task(
        "get me something to drink",
        enrichment_enabled=True,
        selector=make_domain_selector(SelectionBackend.LLM, generate_fn=llm),
        enricher=make_online_enricher(
            SelectionBackend.LLM, generate_fn=llm, enriched_dir=tmp_path
        ),
        scene_symbols=("bottle", "glass", "table"),
    )
    enrichment = resolution.enrichment
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        command="get me something to drink",
        domain_template=resolution.template,
    )
    session.set_enrichment(
        EnrichmentContext.from_domain_additions(enrichment.domain_additions),
        goal_binder=_binder(),
    )
    plan = make_domain_stub_plan(
        "get me something to drink",
        resolution.template,
        domain_additions=enrichment.domain_additions,
    )
    problem, _ = session.generate_hybrid_problem(plan, oracle_scene=_scene())
    actions = FastDownwardPlanner().solve_from_strings(
        enrichment.domain_text, problem
    )
    assert any(a.startswith("(pour") for a in actions)


def test_goal_binder_factory_follows_the_selection_backend():
    assert make_goal_binder(SelectionBackend.RULE_BASED) is None
    assert isinstance(make_goal_binder(SelectionBackend.LLM), LLMGoalBinder)
