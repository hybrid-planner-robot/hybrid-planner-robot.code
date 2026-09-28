"""
Rule-based goal generation from natural-language task commands.

No LLM / vision — pattern matching only.
"""

from __future__ import annotations

import re
from typing import Sequence

from ...init_generator.renderer import PddlFact
from ...init_generator.schema import SceneState
from ..predicates import resolve_allowed_predicates
from ..types import GoalResult

_PLACE_ON = re.compile(
    r"^(?:place|put)\s+(?P<obj>.+?)\s+on\s+(?:the\s+)?(?P<loc>.+?)\.?$",
    re.IGNORECASE,
)
_BELONGS_ON = re.compile(
    r"(?:belongs|should\s+(?:be|live|sit)|must\s+be|needs?\s+to\s+be|goes)"
    r"\s+on(?:to)?\s+(?:the\s+)?"
    r"(?P<loc>.+?)(?=\s*,|\s+not\b|\.|$)",
    re.IGNORECASE,
)
_ON_DEST_TAIL = re.compile(
    r"\bon(?:to)?\s+(?:the\s+)?(?P<loc>.+?)\s*\.?\s*$",
    re.IGNORECASE,
)
_ALL_QUANT = re.compile(r"\b(?:all|every|each)\b", re.IGNORECASE)
_WRITING_ITEM = re.compile(r"(?:pen|marker|pencil|crayon)", re.IGNORECASE)
_IN_CONTAINER = re.compile(
    r"^(?:place|put)\s+(?P<obj>.+?)\s+in(?:to)?\s+(?:the\s+)?(?P<container>.+?)\.?$",
    re.IGNORECASE,
)
_STACK_ON = re.compile(
    r"^stack\s+(?P<top>.+?)\s+on\s+(?:the\s+)?(?P<bottom>.+?)\.?$",
    re.IGNORECASE,
)
_LOOK_AT = re.compile(
    r"^(?:look_at|look at)\s+(?:the\s+)?(?P<target>.+?)\.?$",
    re.IGNORECASE,
)
_PICK = re.compile(
    r"^(?:pick(?:\s+up)?|grasp|pick-up)\s+(?:the\s+)?(?P<obj>.+?)\.?$",
    re.IGNORECASE,
)


def _normalize_phrase(phrase: str) -> str:
    text = phrase.strip().lower()
    text = re.sub(r"\bwooden\b", "wood", text)
    text = re.sub(r"^the\s+", "", text)
    text = re.sub(r"[^a-z0-9\s_]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "_")


def _resolve_entity(phrase: str, candidates: Sequence[str]) -> str | None:
    """Map a command phrase to a known symbolic name."""
    if not candidates:
        return None

    normalized = _normalize_phrase(phrase)
    candidate_set = set(candidates)

    if normalized in candidate_set:
        return normalized

    # Gazebo furniture names vs NL labels (tray ↔ target_tray, shelf ↔ shelf_b).
    _aliases = {
        "tray": ("target_tray", "tray"),
        "target_tray": ("target_tray", "tray"),
        "shelf": ("shelf_b", "shelf"),
        "shelf_b": ("shelf_b", "shelf"),
    }
    for cand in _aliases.get(normalized, ()):
        if cand in candidate_set:
            return cand

    # Longest substring match (handles "red cup" → red_cup).
    ranked = sorted(candidates, key=len, reverse=True)
    phrase_lower = _normalize_phrase(phrase).replace("_", " ")
    for name in ranked:
        variants = {name, name.replace("_", " ")}
        for variant in variants:
            if variant == normalized or variant == phrase_lower:
                return name
            if re.search(rf"\b{re.escape(variant)}\b", phrase_lower):
                return name
        # "bowl" → white_bowl (phrase is a token of the symbol).
        name_words = name.replace("_", " ")
        if re.search(rf"\b{re.escape(phrase_lower)}\b", name_words):
            return name

    return None


def quantified_on_facts(
    command: str,
    objects: Sequence[str],
    locations: Sequence[str],
) -> list[PddlFact] | None:
    """``all writing tools must be on the book`` → one ``on`` fact per tool."""
    text = str(command or "").strip()
    if not text or not _ALL_QUANT.search(text):
        return None
    match = _ON_DEST_TAIL.search(text)
    if not match:
        return None
    dest_phrase = match.group("loc")
    loc = _resolve_entity(dest_phrase, locations) or _resolve_entity(
        dest_phrase, objects
    )
    if not loc:
        return None
    items = [name for name in objects if name != loc]
    if re.search(r"\bwriting\b|\bpens?\b|\bmarkers?\b", text, re.IGNORECASE):
        writing = [name for name in items if _WRITING_ITEM.search(name)]
        if writing:
            items = writing
    if not items:
        return None
    return [("on", name, loc) for name in items]


class RuleBasedGoalGenerator:
    """Pattern-based GoalGenerator for canonical manipulation tasks."""

    backend = "rule_based"

    def generate(
        self,
        command: str,
        objects: Sequence[str],
        *,
        locations: Sequence[str] | None = None,
        domain_template: str = "manipulation_base",
        allowed_predicates: Sequence[str] | None = None,
        scene_state: SceneState | None = None,  # unused; Protocol compatibility
    ) -> GoalResult:
        del scene_state  # rule-based path is command + symbol lists only
        text = command.strip()
        if not text:
            return GoalResult(
                facts=[],
                backend=self.backend,
                ok=False,
                error="empty command",
            )

        locs = list(locations or [])
        all_items = list(objects)
        allowed = resolve_allowed_predicates(domain_template, allowed_predicates)

        facts, error = self._parse(text, all_items, locs, allowed, domain_template)
        if error:
            return GoalResult(
                facts=[],
                backend=self.backend,
                ok=False,
                error=error,
            )

        filtered = [f for f in facts if f[0] in allowed or f[0] == "_raw_fact"]
        if not filtered:
            return GoalResult(
                facts=[],
                backend=self.backend,
                ok=False,
                error=f"no goal facts allowed for domain_template={domain_template!r}",
            )

        return GoalResult(facts=filtered, backend=self.backend, ok=True)

    def _parse(
        self,
        text: str,
        objects: Sequence[str],
        locations: Sequence[str],
        allowed: frozenset[str],
        domain_template: str,
    ) -> tuple[list[PddlFact], str | None]:
        quantified = quantified_on_facts(text, objects, locations)
        if quantified and "on" in allowed:
            return quantified, None

        m = _PLACE_ON.match(text)
        if m:
            obj = _resolve_entity(m.group("obj"), objects)
            loc = _resolve_entity(m.group("loc"), locations)
            if not obj:
                return [], f"unknown object in command: {m.group('obj')!r}"
            if not loc:
                return [], f"unknown location in command: {m.group('loc')!r}"
            if "on" not in allowed:
                return [], f"'on' goal not supported for domain {domain_template!r}"
            return [("on", obj, loc)], None

        m = _BELONGS_ON.search(text)
        if m and "on" in allowed:
            loc = _resolve_entity(m.group("loc"), locations)
            obj = _resolve_entity(text, objects)
            if obj and loc:
                return [("on", obj, loc)], None

        m = _IN_CONTAINER.match(text)
        if m:
            obj = _resolve_entity(m.group("obj"), objects)
            container = _resolve_entity(m.group("container"), locations)
            if not obj:
                return [], f"unknown object in command: {m.group('obj')!r}"
            if not container:
                return [], f"unknown container in command: {m.group('container')!r}"
            if "in-container" not in allowed:
                return [], f"'in-container' goal not supported for domain {domain_template!r}"
            return [("in-container", obj, container)], None

        m = _STACK_ON.match(text)
        if m:
            top = _resolve_entity(m.group("top"), objects)
            bottom = _resolve_entity(m.group("bottom"), objects)
            if not top:
                return [], f"unknown object in command: {m.group('top')!r}"
            if not bottom:
                return [], f"unknown object in command: {m.group('bottom')!r}"
            if "stacked-on" not in allowed:
                return [], f"'stacked-on' goal not supported for domain {domain_template!r}"
            return [("stacked-on", top, bottom)], None

        m = _LOOK_AT.match(text)
        if m:
            target = _resolve_entity(m.group("target"), objects)
            if not target:
                return [], f"unknown target in command: {m.group('target')!r}"
            if "camera-aimed-at" not in allowed:
                return [], f"'camera-aimed-at' goal not supported for domain {domain_template!r}"
            return [("camera-aimed-at", target)], None

        m = _PICK.match(text)
        if m:
            obj = _resolve_entity(m.group("obj"), objects)
            if not obj:
                return [], f"unknown object in command: {m.group('obj')!r}"
            if "holding" not in allowed:
                return [], f"'holding' goal not supported for domain {domain_template!r}"
            return [("holding", obj)], None

        return [], f"unsupported command: {text!r}"
