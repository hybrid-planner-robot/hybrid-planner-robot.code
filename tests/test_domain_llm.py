"""
Session 29 — one text LLM for domain select / completeness / enrichment.

Everything here is offline: the single text LLM is injected as a mock
``generate_fn`` returning the canned payloads in
``tests/fixtures/domain_llm/``. No GPU, no weights, no Fast Downward (except
one opt-in solve that skips when the binary is absent).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.domain_llm import (
    DomainSelectionError,
    EnrichmentPayloadError,
    LLMDomainEnricher,
    LLMDomainSelector,
    domain_action_names,
    parse_enrichment_payload,
    parse_selection_payload,
    pddl_syntax_errors,
)
from planner.domain_store import (
    enrichment_signature,
    list_enriched_domains,
    load_base_domain,
    load_enriched_domain,
)
from planner.online_enrichment import (
    DomainCompleteness,
    EnrichmentRequest,
    EnrichmentStatus,
    SelectionBackend,
    make_domain_selector,
    make_online_enricher,
    resolve_domain_for_task,
    resolve_selection_backend,
    select_domain_rule_based,
)
from planner.problem_generator.enrichment_goal import goals_from_domain_additions
from planner.skill_catalog import ros_primitive_for_action, skill_signature
from planner.text_llm import (
    DEFAULT_TEXT_LLM_MODEL_ID,
    resolve_text_llm_model_id,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "domain_llm"
SELECT_CASES = json.loads((_FIXTURES / "select_cases.json").read_text())["cases"]
ENRICH_FIXTURES = json.loads((_FIXTURES / "enrich_cases.json").read_text())


# ── Mock single text LLM ─────────────────────────────────────────────────────


class MockTextLLM:
    """
    One mock model for both prompts, as in production.

    Selection replies are matched on the task line; enrichment replies on the
    allowed skill listed in the prompt. ``calls`` records every (system, user).
    """

    def __init__(self, *, enrich_override: dict | str | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self.enrich_override = enrich_override

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if "current_domain_pddl:" in user:
            return self._enrich(user)
        return self._select(user)

    @staticmethod
    def _select(user: str) -> str:
        task_line = ""
        for line in user.splitlines():
            if line.startswith("task:"):
                task_line = line[len("task:") :].strip().lower()
        for case in SELECT_CASES:
            if case["match"] in task_line:
                return json.dumps(case["reply"])
        return json.dumps(
            {
                "template": "manipulation_base",
                "completeness": "complete",
                "needed_skills": [],
                "reason": "mock default",
            }
        )

    def _enrich(self, user: str) -> str:
        if self.enrich_override is not None:
            override = self.enrich_override
            return override if isinstance(override, str) else json.dumps(override)
        for skill, payload in ENRICH_FIXTURES["cases"].items():
            if f"- {skill}:" in user:
                return json.dumps(payload)
        return json.dumps({"refuse": True, "reason": "mock has no payload"})

    @property
    def select_calls(self) -> int:
        return sum(1 for _, user in self.calls if "current_domain_pddl:" not in user)

    @property
    def enrich_calls(self) -> int:
        return sum(1 for _, user in self.calls if "current_domain_pddl:" in user)


def _request(
    *,
    template: str = "manipulation_base",
    command: str = "get me something to drink",
    candidates: tuple[str, ...] = ("pour",),
) -> EnrichmentRequest:
    selection = select_domain_rule_based(command)
    return EnrichmentRequest(
        template=template,
        command=command,
        scene_symbols=("bottle", "glass", "table"),
        candidate_skills=candidates,
        reason="test handoff",
        selection=selection,
    )


# ── One text LLM, one model id ───────────────────────────────────────────────


def test_single_model_id_shared_by_goal_and_domain_paths():
    from planner.problem_generator.goal_generator import DEFAULT_GOAL_LLM_MODEL_ID

    assert DEFAULT_GOAL_LLM_MODEL_ID == DEFAULT_TEXT_LLM_MODEL_ID
    assert resolve_text_llm_model_id(env={}) == DEFAULT_TEXT_LLM_MODEL_ID
    # New canonical var wins; the Session 9b goal var still works.
    assert resolve_text_llm_model_id(env={"VLMRP_GOAL_LLM_MODEL": "legacy/x"}) == (
        "legacy/x"
    )
    assert (
        resolve_text_llm_model_id(
            env={"VLMRP_TEXT_LLM_MODEL": "one/model", "VLMRP_GOAL_LLM_MODEL": "old"}
        )
        == "one/model"
    )
    assert resolve_text_llm_model_id("explicit/id", env={"VLMRP_TEXT_LLM_MODEL": "e"}) == (
        "explicit/id"
    )


def test_selection_backend_flag_defaults_to_rule_based():
    assert resolve_selection_backend(None, env={}) == SelectionBackend.RULE_BASED
    assert resolve_selection_backend("") == SelectionBackend.RULE_BASED
    assert resolve_selection_backend("llm") == SelectionBackend.LLM
    assert resolve_selection_backend("text_llm") == SelectionBackend.LLM
    assert (
        resolve_selection_backend(None, env={"VLMRP_DOMAIN_SELECT": "llm"})
        == SelectionBackend.LLM
    )
    assert isinstance(make_domain_selector(None, env={}), object)
    assert make_domain_selector(None, env={}).backend == "rule_based"
    assert make_domain_selector("llm", generate_fn=MockTextLLM()).backend == "llm"


# ── Semantic completeness (the paper claim) ──────────────────────────────────


def test_paraphrase_is_incomplete_for_llm_but_not_for_keywords():
    """'get me something to drink' never says pour — keywords miss it, the LLM does not."""
    command = "get me something to drink"
    assert select_domain_rule_based(command).completeness == DomainCompleteness.COMPLETE

    selector = LLMDomainSelector(generate_fn=MockTextLLM())
    result = selector.select(command, scene_symbols=("bottle", "glass"))
    assert result.completeness == DomainCompleteness.INCOMPLETE
    assert result.needed_skills == ("pour",)
    assert result.backend == "llm"
    assert result.error is None


@pytest.mark.parametrize(
    "command,template",
    [
        ("place red_cup on shelf", "manipulation_base"),
        ("stack blue_block on red_block", "manipulation_stacking"),
    ],
)
def test_complete_tasks_stay_complete_under_llm_selection(command, template):
    result = LLMDomainSelector(generate_fn=MockTextLLM()).select(command)
    assert result.template == template
    assert result.completeness == DomainCompleteness.COMPLETE
    assert result.needed_skills == ()


def test_selector_drops_skills_outside_the_catalog():
    result = LLMDomainSelector(generate_fn=MockTextLLM()).select("levitate the cup")
    assert result.completeness == DomainCompleteness.INCOMPLETE
    assert result.needed_skills == ()
    assert "levitate" in result.reason


def test_selector_falls_back_to_keywords_on_bad_json():
    selector = LLMDomainSelector(generate_fn=lambda s, u: "not json at all")
    result = selector.select("pour water from bottle to mug")
    assert result.backend == "rule_based_fallback"
    assert result.error is not None
    # Fallback still produces the Session 28 answer.
    assert result.completeness == DomainCompleteness.INCOMPLETE
    assert result.needed_skills == ("pour",)


def test_selector_raises_when_fallback_disabled():
    selector = LLMDomainSelector(
        generate_fn=lambda s, u: "{}", fallback_rule_based=False
    )
    with pytest.raises(DomainSelectionError):
        selector.select("pour water")


@pytest.mark.parametrize(
    "payload",
    [
        {"template": "kitchen_domain", "completeness": "complete"},
        {"template": "manipulation_base", "completeness": "maybe"},
        {"template": "manipulation_base", "completeness": "complete",
         "needed_skills": {"skill": "pour"}},
    ],
)
def test_parse_selection_payload_rejects_bad_answers(payload):
    with pytest.raises(DomainSelectionError):
        parse_selection_payload(json.dumps(payload))


def test_parse_selection_payload_normalizes_soft_answers():
    """A bare string skill is coerced; a complete verdict cannot need skills."""
    lenient = parse_selection_payload(
        json.dumps(
            {
                "template": "manipulation_base",
                "completeness": "incomplete",
                "needed_skills": "POUR",
                "reason": "fill the glass",
            }
        )
    )
    assert lenient.needed_skills == ("pour",)

    contradictory = parse_selection_payload(
        json.dumps(
            {
                "template": "manipulation_base",
                "completeness": "complete",
                "needed_skills": ["pour"],
                "reason": "contradicts itself",
            }
        )
    )
    assert contradictory.needed_skills == ()


def test_parse_selection_payload_native_place_in_container_is_complete():
    """v4 implicit place: place-in-container is already in the template."""
    result = parse_selection_payload(
        json.dumps(
            {
                "template": "containers_manipulation",
                "completeness": "incomplete",
                "needed_skills": ["place_in_container"],
                "reason": "no action can place into a container",
            }
        )
    )
    assert result.completeness.value == "complete"
    assert result.needed_skills == ()
    assert "treated as complete" in result.reason
    assert "no catalog skill matches the gap" not in result.reason


def test_parse_selection_payload_invented_skill_still_empty_gap():
    result = parse_selection_payload(
        json.dumps(
            {
                "template": "manipulation_base",
                "completeness": "incomplete",
                "needed_skills": ["levitate"],
                "reason": "needs to float",
            }
        )
    )
    assert result.completeness.value == "incomplete"
    assert result.needed_skills == ()
    assert "dropped non-catalog skills: levitate" in result.reason
    assert "no catalog skill matches the gap" in result.reason


# ── Closed-catalog enrichment ────────────────────────────────────────────────


def test_enrichment_authors_merges_and_persists(tmp_path):
    mock = MockTextLLM()
    enricher = LLMDomainEnricher(generate_fn=mock, enriched_dir=str(tmp_path))
    outcome = enricher.enrich(_request())

    assert outcome.status == EnrichmentStatus.ENRICHED
    assert outcome.skills_grounded == ("pour",)
    assert outcome.ros_primitives == ("pour",)
    assert outcome.refuse_message is None

    assert "(:action pour" in (outcome.domain_text or "")
    assert "(poured ?src - item ?dst - item)" in (outcome.domain_text or "")
    # Fixed actions survive the merge.
    assert {"pick", "place", "look-at", "pour"} <= domain_action_names(
        outcome.domain_text or ""
    )
    assert pddl_syntax_errors(outcome.domain_text or "") == []

    persisted = Path(outcome.domain_path or "")
    assert persisted.is_file()
    assert persisted.parent == tmp_path
    assert persisted.read_text() == outcome.domain_text


def test_enriched_action_maps_to_a_real_ros_primitive(tmp_path):
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(), enriched_dir=str(tmp_path)
    ).enrich(_request())

    action = (outcome.domain_additions or {})["new_actions"][0]
    primitive = ros_primitive_for_action(action["name"])
    assert primitive == "pour"
    # Signature is the ROS body's, not something the model invented.
    sig = skill_signature("pour")
    assert sig is not None and sig.roles == ("source", "target")
    assert ros_primitive_for_action("levitate") is None


def test_enrichment_goal_facts_come_from_session_19_helper(tmp_path):
    """Deliverable 4: domain file *and* goal facts via goal_from_enrichment_action."""
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(), enriched_dir=str(tmp_path)
    ).enrich(_request())

    steps = [{"primitive": "pour", "args": {"source": "bottle", "target": "glass"}}]
    goals = goals_from_domain_additions(outcome.domain_additions, steps)
    assert goals == [("_raw_fact", "(poured bottle glass)")]


def test_persisted_domain_is_reused_without_calling_the_llm(tmp_path):
    first_mock = MockTextLLM()
    first = LLMDomainEnricher(
        generate_fn=first_mock, enriched_dir=str(tmp_path)
    ).enrich(_request())
    assert first_mock.enrich_calls == 1
    assert first.reused is False

    second_mock = MockTextLLM()
    second = LLMDomainEnricher(
        generate_fn=second_mock, enriched_dir=str(tmp_path)
    ).enrich(_request(command="pour me a drink please"))

    assert second_mock.enrich_calls == 0
    assert second.reused is True
    assert second.status == EnrichmentStatus.ENRICHED
    assert second.signature == first.signature
    assert second.domain_text == first.domain_text
    assert second.domain_additions == first.domain_additions
    assert len(list_enriched_domains(tmp_path)) == 1


def test_pddl_syntax_errors_flags_predicate_arity_mismatch():
    """Live FD failure: declare arity-2 containing, use arity-1 in effect."""
    base = load_base_domain("containers_manipulation")
    # Minimal illegal domain shaped like the poisoned cache entry.
    poisoned = base.replace(
        "(:predicates",
        "(:predicates\n    (containing ?t - item ?i - item)",
        1,
    )
    # Insert a bad pour before the closing paren of the domain.
    bad_action = """
  (:action pour
    :parameters (?source - item ?target - item)
    :precondition (and (holding ?source))
    :effect (and (empty ?source) (containing ?target))
  )
"""
    poisoned = poisoned.rstrip()
    if poisoned.endswith(")"):
        poisoned = poisoned[:-1] + bad_action + "\n)\n"
    errors = pddl_syntax_errors(poisoned)
    assert any("containing" in e and "arity" in e for e in errors), errors


def test_parse_rejects_new_predicate_arity_mismatch_in_effect():
    payload = {
        "skill": "pour",
        "new_predicates": ["(containing ?t - item ?i - item)"],
        "action": {
            "name": "pour",
            "parameters": "(?source - item ?target - item)",
            "precondition": "(and (holding ?source))",
            # First positive effect must be the new fluent (goal helper);
            # arity 1 vs declared 2 is the bug under test.
            "effect": "(and (containing ?target) (empty ?source))",
        },
        "reason": "arity bug from live 7B cache",
    }
    with pytest.raises(EnrichmentPayloadError, match="arity"):
        parse_enrichment_payload(json.dumps(payload), allowed_skills=["pour"])


def test_invalid_persisted_domain_is_discarded_and_reauthored(tmp_path):
    """Reuse must not hand FD a cached domain with illegal predicate arity."""
    first = LLMDomainEnricher(
        generate_fn=MockTextLLM(), enriched_dir=str(tmp_path)
    ).enrich(_request())
    assert first.status == EnrichmentStatus.ENRICHED
    path = Path(first.domain_path or "")
    assert path.is_file()

    # Corrupt the cache the way the live kitchen run did.
    text = path.read_text(encoding="utf-8")
    if "(poured ?src - item ?dst - item)" in text:
        text = text.replace(
            "(poured ?src - item ?dst - item)",
            "(poured ?src - item ?dst - item)\n    (containing ?t - item ?i - item)",
            1,
        )
        text = text.replace(
            "(poured ?src ?dst)",
            "(containing ?dst)",
            1,
        )
    else:
        pytest.skip("unexpected mock pour shape")
    path.write_text(text, encoding="utf-8")
    assert any("arity" in e for e in pddl_syntax_errors(text))

    second_mock = MockTextLLM()
    second = LLMDomainEnricher(
        generate_fn=second_mock, enriched_dir=str(tmp_path)
    ).enrich(_request(command="pour the bottle into the glass"))

    assert second_mock.enrich_calls == 1
    assert second.reused is False
    assert second.status == EnrichmentStatus.ENRICHED
    assert "discarded invalid persisted domain" in (second.notes or "")
    assert pddl_syntax_errors(second.domain_text or "") == []
    # Cache holds only the re-authored valid domain.
    assert len(list_enriched_domains(tmp_path)) == 1


def test_signature_tracks_template_skills_and_base_domain():
    base = load_base_domain("manipulation_base")
    sig = enrichment_signature(
        template="manipulation_base", skills=["pour"], base_domain_text=base
    )
    assert sig == enrichment_signature(
        template="manipulation_base", skills=["POUR"], base_domain_text=base
    )
    assert sig != enrichment_signature(
        template="manipulation_base", skills=["stir"], base_domain_text=base
    )
    assert sig != enrichment_signature(
        template="containers_manipulation", skills=["pour"], base_domain_text=base
    )
    # Editing the fixed template invalidates the cached enrichment.
    assert sig != enrichment_signature(
        template="manipulation_base",
        skills=["pour"],
        base_domain_text=base + "\n; edited\n",
    )
    cut = enrichment_signature(
        template="manipulation_base", skills=["cut"], base_domain_text=base
    )
    assert cut == enrichment_signature(
        template="manipulation_base",
        skills=["cut"],
        base_domain_text=base,
        world="kitchen",
    )
    assert cut != enrichment_signature(
        template="manipulation_base",
        skills=["cut"],
        base_domain_text=base,
        world="workshop",
    )


def test_reload_persisted_record_carries_execution_metadata(tmp_path):
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(), enriched_dir=str(tmp_path)
    ).enrich(_request())

    record = load_enriched_domain(
        outcome.signature or "", template="manipulation_base", directory=tmp_path
    )
    assert record is not None
    assert record.skills == ("pour",)
    assert record.ros_primitives == ("pour",)
    assert record.domain_additions["new_actions"][0]["name"] == "pour"
    assert record.meta["grounded_skills"] == ["pour"]


def test_enrichment_retries_once_with_the_validation_error(tmp_path):
    replies = iter(
        [
            json.dumps(ENRICH_FIXTURES["invalid"]["wrong_arity"]),
            json.dumps(ENRICH_FIXTURES["cases"]["pour"]),
        ]
    )
    seen: list[str] = []

    def generate(system: str, user: str) -> str:
        seen.append(user)
        return next(replies)

    outcome = LLMDomainEnricher(
        generate_fn=generate, enriched_dir=str(tmp_path)
    ).enrich(_request())

    assert outcome.status == EnrichmentStatus.ENRICHED
    assert len(seen) == 2
    assert "previous answer was rejected" in seen[1]


# ── Refuse contract ──────────────────────────────────────────────────────────


def test_enricher_refuses_when_no_catalog_skill_fits(tmp_path):
    mock = MockTextLLM()
    outcome = LLMDomainEnricher(
        generate_fn=mock, enriched_dir=str(tmp_path)
    ).enrich(_request(command="solder the broken wire", candidates=()))

    assert outcome.status == EnrichmentStatus.REFUSED
    assert "Refusing rather than inventing" in (outcome.refuse_message or "")
    assert mock.enrich_calls == 0  # never asks the model to invent a motor
    assert list_enriched_domains(tmp_path) == []


@pytest.mark.parametrize(
    "case",
    [
        "open_vocab_skill",
        "name_mismatch",
        "wrong_arity",
        "unbalanced_effect",
        "negative_only_effect",
        "unbound_variable",
    ],
)
def test_invalid_payloads_refuse_and_persist_nothing(case, tmp_path):
    payload = ENRICH_FIXTURES["invalid"][case]
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(enrich_override=payload),
        enriched_dir=str(tmp_path),
        max_attempts=1,
    ).enrich(_request())

    assert outcome.status == EnrichmentStatus.REFUSED
    assert outcome.domain_text is None
    assert list_enriched_domains(tmp_path) == []


def test_model_may_decline_within_the_closed_catalog(tmp_path):
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(
            enrich_override={"refuse": True, "reason": "the bottle is sealed"}
        ),
        enriched_dir=str(tmp_path),
    ).enrich(_request())

    assert outcome.status == EnrichmentStatus.REFUSED
    assert "the bottle is sealed" in (outcome.refuse_message or "")


def test_enricher_refuses_an_action_already_in_the_domain(tmp_path):
    payload = {
        "skill": "pour",
        "action": {
            "name": "pour",
            "parameters": "(?src - item ?dst - item)",
            "precondition": "(holding ?src)",
            "effect": "(and (poured ?src ?dst))",
        },
        "new_predicates": ["(poured ?src - item ?dst - item)"],
    }
    base = load_base_domain("manipulation_base").rstrip()[:-1] + (
        "\n  (:action pour\n"
        "    :parameters (?src - item ?dst - item)\n"
        "    :precondition (holding ?src)\n"
        "    :effect (and (poured ?src ?dst))\n"
        "  )\n)\n"
    )
    request = _request()
    request = EnrichmentRequest(
        template=request.template,
        command=request.command,
        scene_symbols=request.scene_symbols,
        candidate_skills=request.candidate_skills,
        reason=request.reason,
        selection=request.selection,
        base_domain_text=base,
    )
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(enrich_override=payload),
        enriched_dir=str(tmp_path),
        max_attempts=1,
    ).enrich(request)

    assert outcome.status == EnrichmentStatus.REFUSED
    assert "already in domain" in (outcome.notes or "")


def test_effect_must_assert_a_predicate_the_action_declares():
    """
    Observed live from Qwen2.5-1.5B: a `pour` whose effect was
    `(and (holding ?dst) (clear ?dst))`. Name, arity and binding all checked
    out, but the domain gained nothing and the goal collapsed to
    `(holding cup)` — reachable with a plain pick, so FD would have "solved"
    the task without pouring.
    """
    payload = {
        "skill": "pour",
        "action": {
            "name": "pour",
            "parameters": "(?source - item ?target - item)",
            "precondition": "(and (holding ?source))",
            "effect": "(and (holding ?target) (clear ?target))",
        },
        "new_predicates": [],
    }
    with pytest.raises(EnrichmentPayloadError, match="does not declare"):
        parse_enrichment_payload(json.dumps(payload), allowed_skills=["pour"])


def test_effect_may_not_declare_parameter_types():
    """
    Observed live from Qwen2.5-7B: `stir` with effect
    `(and (pred ?x - item ?c - container) ...)`. `- type` is only legal in
    :parameters, FD rejects it, and the derived goal came out as
    `(pred can cup container)` — three arguments from a two-parameter action,
    with a type name standing in as an object.
    """
    payload = {
        "skill": "stir",
        "action": {
            "name": "stir",
            "parameters": "(?c - item)",
            "precondition": "(camera-aimed-at ?c)",
            "effect": "(and (stirred ?c - item))",
        },
        "new_predicates": ["(stirred ?c - item)"],
    }
    with pytest.raises(EnrichmentPayloadError, match="types belong in 'parameters'"):
        parse_enrichment_payload(json.dumps(payload), allowed_skills=["stir"])


def test_placeholder_predicate_names_are_rejected():
    """The 7B copied the prompt skeleton and called its new fluent `pred`."""
    payload = {
        "skill": "pour",
        "action": {
            "name": "pour",
            "parameters": "(?source - item ?target - item)",
            "precondition": "(holding ?source)",
            "effect": "(and (not (holding ?source)) (pred ?source ?target))",
        },
        "new_predicates": ["(pred ?x - item ?y - item)"],
    }
    with pytest.raises(EnrichmentPayloadError, match="placeholder"):
        parse_enrichment_payload(json.dumps(payload), allowed_skills=["pour"])


def test_enricher_refuses_a_predicate_that_already_exists(tmp_path):
    """Declaring an existing predicate as new must not launder the same trick."""
    payload = {
        "skill": "pour",
        "action": {
            "name": "pour",
            "parameters": "(?src - item ?dst - item)",
            "precondition": "(holding ?src)",
            "effect": "(and (holding ?dst))",
        },
        "new_predicates": ["(holding ?i - item)"],
    }
    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(enrich_override=payload),
        enriched_dir=str(tmp_path),
        max_attempts=1,
    ).enrich(_request())

    assert outcome.status == EnrichmentStatus.REFUSED
    assert "already in the base domain" in (outcome.notes or "")
    assert list_enriched_domains(tmp_path) == []


def test_parse_enrichment_payload_error_messages():
    with pytest.raises(EnrichmentPayloadError, match="not in the allowed set"):
        parse_enrichment_payload(
            json.dumps(ENRICH_FIXTURES["cases"]["stir"]), allowed_skills=["pour"]
        )
    with pytest.raises(EnrichmentPayloadError, match="invalid enrichment JSON"):
        parse_enrichment_payload("¯\\_(ツ)_/¯", allowed_skills=["pour"])


# ── End-to-end routing through the public API ────────────────────────────────


def test_llm_route_enriches_a_paraphrase_end_to_end(tmp_path):
    mock = MockTextLLM()
    resolution = resolve_domain_for_task(
        "get me something to drink",
        enrichment_enabled=True,
        selector=make_domain_selector("llm", generate_fn=mock),
        enricher=make_online_enricher(
            "llm", generate_fn=mock, enriched_dir=str(tmp_path)
        ),
        scene_symbols=("bottle", "glass", "table"),
    )

    assert resolution.selection.backend == "llm"
    assert resolution.selection.completeness == DomainCompleteness.INCOMPLETE
    assert resolution.used_enricher is True
    assert resolution.refused is False
    assert resolution.enrichment is not None
    assert resolution.enrichment.status == EnrichmentStatus.ENRICHED
    assert resolution.domain_text is not None
    assert "(:action pour" in resolution.domain_text
    assert mock.select_calls == 1 and mock.enrich_calls == 1


def test_llm_route_refuses_when_the_catalog_cannot_help(tmp_path):
    mock = MockTextLLM()
    resolution = resolve_domain_for_task(
        "solder the broken wire",
        enrichment_enabled=True,
        selector=make_domain_selector("llm", generate_fn=mock),
        enricher=make_online_enricher(
            "llm", generate_fn=mock, enriched_dir=str(tmp_path)
        ),
    )

    assert resolution.selection.completeness == DomainCompleteness.INCOMPLETE
    assert resolution.refused is True
    assert "Refusing rather than inventing" in (
        resolution.enrichment.refuse_message or ""
    )
    assert mock.enrich_calls == 0
    assert list_enriched_domains(tmp_path) == []


@pytest.mark.parametrize(
    "command", ["place red_cup on shelf", "stack blue_block on red_block"]
)
def test_complete_tasks_never_enrich_on_the_llm_route(command, tmp_path):
    mock = MockTextLLM()
    resolution = resolve_domain_for_task(
        command,
        enrichment_enabled=True,
        selector=make_domain_selector("llm", generate_fn=mock),
        enricher=make_online_enricher(
            "llm", generate_fn=mock, enriched_dir=str(tmp_path)
        ),
    )

    assert resolution.selection.completeness == DomainCompleteness.COMPLETE
    assert resolution.enrichment.status == EnrichmentStatus.SKIPPED
    assert resolution.used_enricher is False
    assert mock.enrich_calls == 0
    assert list_enriched_domains(tmp_path) == []


def test_enrichment_flag_off_never_calls_the_llm_enricher(tmp_path):
    mock = MockTextLLM()
    resolution = resolve_domain_for_task(
        "get me something to drink",
        enrichment_enabled=False,
        selector=make_domain_selector("llm", generate_fn=mock),
        enricher=make_online_enricher(
            "llm", generate_fn=mock, enriched_dir=str(tmp_path)
        ),
    )

    assert resolution.enrichment_enabled is False
    assert resolution.enrichment.status == EnrichmentStatus.SKIPPED
    assert mock.enrich_calls == 0
    assert list_enriched_domains(tmp_path) == []


def test_default_route_is_still_the_keyword_fallback():
    """No selector / no flags → Session 28 behaviour, byte for byte."""
    resolution = resolve_domain_for_task("pour water from bottle to mug")
    assert resolution.selection.backend == SelectionBackend.RULE_BASED.value
    assert resolution.selection == select_domain_rule_based(
        "pour water from bottle to mug"
    )
    assert resolution.enrichment.status == EnrichmentStatus.SKIPPED


def test_rule_based_backend_keeps_the_stub_enricher(tmp_path):
    resolution = resolve_domain_for_task(
        "pour water from bottle to mug",
        enrichment_enabled=True,
        selector=make_domain_selector("rule_based"),
        enricher=make_online_enricher("rule_based"),
    )
    assert resolution.selection.backend == "rule_based"
    assert resolution.enrichment.status == EnrichmentStatus.STUB
    assert resolution.enrichment.domain_text is None


# ── Optional live-ish check: the enriched domain really parses for FD ────────


@pytest.mark.skipif(
    shutil.which("fast-downward") is None,
    reason="fast-downward not on PATH (offline CI)",
)
def test_enriched_domain_solves_with_fast_downward(tmp_path):
    from planner.fast_downward import FastDownwardPlanner

    outcome = LLMDomainEnricher(
        generate_fn=MockTextLLM(), enriched_dir=str(tmp_path)
    ).enrich(_request())
    problem = """
(define (problem pour-drink)
  (:domain manipulation-base)
  (:objects bottle glass - item table - location)
  (:init (on bottle table) (on glass table) (clear bottle) (clear glass)
         (gripper-empty) (camera-aimed-at bottle))
  (:goal (poured bottle glass))
)
"""
    plan = FastDownwardPlanner().solve_from_strings(outcome.domain_text or "", problem)
    assert plan is not None
    assert any("pour" in action for action in plan)
