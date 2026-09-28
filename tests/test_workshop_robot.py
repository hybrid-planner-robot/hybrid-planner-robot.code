"""Workshop robot: disjoint catalog, isolated cut cache, select without pour."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner.domain_llm import (
    LLMDomainEnricher,
    build_selection_prompt,
    parse_selection_payload,
)
from planner.domain_store import enrichment_signature, list_enriched_domains, load_base_domain
from planner.online_enrichment import (
    DomainCompleteness,
    DomainSelectionResult,
    EnrichmentRequest,
    EnrichmentStatus,
    SelectionBackend,
    detect_needed_enrichment_skills,
    select_domain_rule_based,
)
from planner.skill_catalog import (
    CATALOG_SKILLS,
    ENRICHMENT_CANDIDATE_SKILLS,
    WORKSHOP_ENRICHMENT_CANDIDATE_SKILLS,
    enrichment_candidates_for_domain,
    enrichment_candidates_for_world,
    robot_id_for_world,
    ros_primitive_for_action,
)

from test_domain_llm import MockTextLLM


def _cut_request(command: str, world: str | None) -> EnrichmentRequest:
    selection = DomainSelectionResult(
        template="manipulation_base",
        completeness=DomainCompleteness.INCOMPLETE,
        needed_skills=("cut",),
        reason="gap",
        backend=SelectionBackend.LLM.value,
    )
    return EnrichmentRequest(
        template="manipulation_base",
        command=command,
        scene_symbols=("wood_board", "tea_box", "table"),
        candidate_skills=("cut",),
        reason="gap",
        selection=selection,
        base_domain_text=load_base_domain("manipulation_base"),
        world=world,
    )


def test_workshop_and_household_enrichment_sets_overlap_only_on_cut():
    household = enrichment_candidates_for_world(None)
    kitchen = enrichment_candidates_for_world("kitchen")
    workshop = enrichment_candidates_for_world("workshop")
    assert household == kitchen == ENRICHMENT_CANDIDATE_SKILLS
    assert workshop == WORKSHOP_ENRICHMENT_CANDIDATE_SKILLS
    assert household & workshop == {"cut"}
    assert "pour" not in workshop
    assert "drill" not in household
    assert "pour" in CATALOG_SKILLS
    assert robot_id_for_world("workshop") != robot_id_for_world("kitchen")


def test_enrichment_candidates_default_to_household():
    missing = enrichment_candidates_for_domain("manipulation_base")
    assert missing == ENRICHMENT_CANDIDATE_SKILLS
    workshop = enrichment_candidates_for_domain("manipulation_base", world="workshop")
    assert workshop == WORKSHOP_ENRICHMENT_CANDIDATE_SKILLS
    assert "pour" not in workshop
    assert "drill" in workshop


def test_rule_based_cues_are_filtered_by_robot_catalog():
    assert detect_needed_enrichment_skills("pour the can into the glass") == ("pour",)
    assert detect_needed_enrichment_skills(
        "pour the can into the glass", world="workshop"
    ) == ()
    assert detect_needed_enrichment_skills(
        "drill a hole in the metal plate"
    ) == ()
    assert detect_needed_enrichment_skills(
        "drill a hole in the metal plate", world="workshop"
    ) == ("drill",)
    assert select_domain_rule_based(
        "pour the can into the glass", world="workshop"
    ).completeness == DomainCompleteness.COMPLETE
    drill = select_domain_rule_based(
        "drill a hole in the metal plate", world="workshop"
    )
    assert drill.completeness == DomainCompleteness.INCOMPLETE
    assert drill.needed_skills == ("drill",)


def test_selection_payload_drops_household_skills_on_workshop():
    result = parse_selection_payload(
        json.dumps(
            {
                "template": "manipulation_base",
                "completeness": "incomplete",
                "needed_skills": ["pour", "drill"],
                "reason": "mixed",
            }
        ),
        world="workshop",
    )
    assert result.needed_skills == ("drill",)
    prompt = build_selection_prompt("drill a hole", (), world="workshop")
    assert "pour(" not in prompt
    assert "drill(" in prompt
    household = build_selection_prompt("pour the can", ())
    assert "pour(" in household
    assert "drill(" not in household


def test_workshop_cut_does_not_reuse_household_cache(tmp_path):
    mock = MockTextLLM()
    enricher = LLMDomainEnricher(generate_fn=mock, enriched_dir=str(tmp_path))
    first = enricher.enrich(_cut_request("cut the tea box", "kitchen"))
    assert first.status == EnrichmentStatus.ENRICHED
    assert first.reused is False
    after_kitchen = mock.enrich_calls
    second = enricher.enrich(_cut_request("cut the wood board", "workshop"))
    assert second.status == EnrichmentStatus.ENRICHED
    assert second.reused is False
    assert mock.enrich_calls == after_kitchen + 1
    assert first.signature != second.signature
    assert "workshop" in (second.domain_path or "")
    assert ros_primitive_for_action("drill") == "drill"
    household_files = list_enriched_domains(tmp_path)
    workshop_files = list_enriched_domains(tmp_path, world="workshop")
    assert len(household_files) == 1
    assert len(workshop_files) == 1
    base = load_base_domain("manipulation_base")
    assert enrichment_signature(
        template="manipulation_base", skills=["cut"], base_domain_text=base, world="kitchen"
    ) != enrichment_signature(
        template="manipulation_base", skills=["cut"], base_domain_text=base, world="workshop"
    )
