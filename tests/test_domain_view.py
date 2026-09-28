"""Tests for compact domain view helpers used by the goal LLM path."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planner.problem_generator.goal_generator.domain_view import (
    compact_domain_view,
    enrichment_authored_payload,
    filter_facts_to_predicates,
    format_domain_actions_block,
)


def test_compact_domain_view_merges_additions():
    view = compact_domain_view(
        "manipulation_base",
        domain_additions={
            "new_predicates": ["(poured ?s - item ?t - item)"],
            "new_actions": [
                {
                    "name": "pour",
                    "parameters": "(?s - item ?t - item)",
                    "precondition": "(holding ?s)",
                    "effect": "(poured ?s ?t)",
                }
            ],
        },
    )
    assert "poured" in view["predicates"]
    assert "holding" in view["predicates"]
    names = {a["name"] for a in view["actions"]}
    assert "pour" in names
    assert "pick" in names  # from base template


def test_format_domain_actions_block():
    block = format_domain_actions_block(
        [{"name": "pour", "parameters": "(?s)", "precondition": "(holding ?s)", "effect": "(poured ?s)"}]
    )
    assert "pour" in block
    assert "effect=(poured ?s)" in block


def test_filter_facts_to_predicates():
    facts = [
        ("holding", "cup"),
        ("poured", "bottle", "glass"),
        ("_raw_fact", "(tilted can)"),
        ("flying", "cup"),
    ]
    kept = filter_facts_to_predicates(facts, ["holding", "poured", "tilted"])
    assert ("holding", "cup") in kept
    assert ("poured", "bottle", "glass") in kept
    assert ("_raw_fact", "(tilted can)") in kept
    assert all(f[0] != "flying" for f in kept)


def test_enrichment_outcome_predicates_and_facts_use():
    from planner.problem_generator.goal_generator.domain_view import (
        enrichment_outcome_predicates,
        facts_use_any_predicate,
    )

    preds = enrichment_outcome_predicates(
        {
            "new_predicates": ["(transferred-liquid ?s - item ?t - item)"],
            "new_actions": [
                {
                    "name": "pour",
                    "parameters": "(?s - item ?t - item)",
                    "effect": "(and (not (holding ?s)) (transferred-liquid ?s ?t))",
                }
            ],
        }
    )
    assert preds == frozenset({"transferred-liquid"})
    assert facts_use_any_predicate(
        [("transferred-liquid", "can", "glass")], preds
    )
    assert not facts_use_any_predicate([("holding", "cup")], preds)

    payload = enrichment_authored_payload(
        {
            "new_actions": [
                {
                    "name": "tilt",
                    "parameters": "(?o - item)",
                    "precondition": "(holding ?o)",
                    "effect": "(tilted ?o)",
                }
            ],
            "new_predicates": ["(tilted ?o - item)"],
        },
        skills=["tilt"],
        domain_path="/tmp/x.pddl",
        reused=False,
    )
    assert payload is not None
    assert payload["skills"] == ["tilt"]
    assert payload["actions"][0]["name"] == "tilt"
    assert payload["new_predicates"] == ["(tilted ?o - item)"]
