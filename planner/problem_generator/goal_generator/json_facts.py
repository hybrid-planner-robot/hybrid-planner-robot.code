"""Parse and validate LLM JSON goal-fact payloads (shared by text backends)."""

from __future__ import annotations

import json
import re

from ..init_generator.renderer import PddlFact

__all__ = ["extract_json_object", "parse_goal_facts"]


def extract_json_object(raw: str) -> dict:
    """Parse a JSON object from raw model text (tolerates optional fences)."""
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
        raise ValueError("LLM response JSON must be an object")
    return data


def parse_goal_facts(
    raw: str,
    *,
    allowed: frozenset[str],
    known_symbols: frozenset[str],
) -> tuple[list[PddlFact], str | None]:
    """
    Validate JSON ``{"facts": [[pred, ...], ...]}`` against predicate/symbol
    whitelists. Returns ``(facts, None)`` on success or ``([], error)``.
    """
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        return [], f"invalid LLM JSON: {exc}"

    facts_raw = data.get("facts")
    if not isinstance(facts_raw, list) or not facts_raw:
        return [], "LLM response missing non-empty 'facts' list"

    facts: list[PddlFact] = []
    for item in facts_raw:
        if not isinstance(item, (list, tuple)) or len(item) < 1:
            return [], f"invalid fact entry: {item!r}"
        pred = str(item[0]).strip()
        args = [str(a).strip() for a in item[1:]]
        if pred not in allowed:
            return [], f"disallowed predicate in LLM output: {pred!r}"
        for arg in args:
            if arg not in known_symbols:
                return [], f"unknown symbol in LLM output: {arg!r}"
        facts.append((pred, *args))

    return facts, None
