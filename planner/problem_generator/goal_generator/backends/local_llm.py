"""
Local text-LLM GoalGenerator (production target for hybrid :goal).

Maps compact SceneState + natural-language task → PDDL goal facts using a
**local** HF/transformers (or injectable) generate function. This is NOT the
vision/planning VLM and NOT a cloud API.

Session 29: the loader lives in :mod:`planner.text_llm` and is shared with
domain selection / completeness / closed-catalog enrichment — one text LLM for
all symbolic NLP, differing only by prompt (design §3.3.1). Historic 9b–9c
constraint (goal model ≤ perceptual VLM, default ``Qwen/Qwen2.5-1.5B-Instruct``
vs ``Qwen/Qwen3-VL-8B-Instruct``) is now a joint-VRAM budget with the VLM.
"""

from __future__ import annotations

import os
from typing import Sequence

from planner.call_timings import invoke_llm
from planner.text_llm import (
    DEFAULT_PERCEPTUAL_VLM_MODEL_ID,
    DEFAULT_TEXT_LLM_MODEL_ID,
    GenerateFn,
    LocalTextClient,
    TransformersLocalClient,
    get_shared_text_client,
    resolve_text_llm_model_id,
)

from ...init_generator.schema import SceneState
from ..domain_view import format_domain_actions_block
from ..json_facts import parse_goal_facts
from ..predicates import resolve_allowed_predicates
from ..prompts import load_goal_prompt
from ..scene_compact import compact_scene_json
from ..types import GoalResult
from .rule_based import RuleBasedGoalGenerator, quantified_on_facts

# Legacy alias: the goal model *is* the single text LLM (Session 29).
DEFAULT_GOAL_LLM_MODEL_ID = DEFAULT_TEXT_LLM_MODEL_ID

__all__ = [
    "DEFAULT_GOAL_LLM_MODEL_ID",
    "DEFAULT_PERCEPTUAL_VLM_MODEL_ID",
    "GenerateFn",
    "LocalLLMGoalGenerator",
    "LocalTextClient",
    "TransformersLocalClient",
]


def _build_user_prompt(
    command: str,
    objects: Sequence[str],
    locations: Sequence[str],
    allowed: frozenset[str],
    domain_template: str,
    scene_json: str,
    *,
    domain_actions: Sequence[dict] | None = None,
) -> str:
    preds = ", ".join(sorted(allowed))
    objs = ", ".join(objects) if objects else "(none)"
    locs = ", ".join(locations) if locations else "(none)"
    actions_block = format_domain_actions_block(domain_actions or ())
    return (
        f"domain_template: {domain_template}\n"
        f"allowed_predicates: {preds}\n"
        f"domain_actions:\n{actions_block}\n"
        f"objects: {objs}\n"
        f"locations: {locs}\n"
        f"compact_scene_state:\n{scene_json}\n"
        f"command: {command.strip()}\n"
    )


def _symbols_from_scene(scene: SceneState) -> tuple[list[str], list[str]]:
    objs = [o.name for o in scene.objects]
    locs = [loc.name for loc in scene.locations]
    return objs, locs


def _merge_symbols(
    primary: Sequence[str],
    extra: Sequence[str],
) -> list[str]:
    return list(dict.fromkeys([*primary, *extra]))


class LocalLLMGoalGenerator:
    """
    Local text-LLM GoalGenerator (backend ``local_llm``).

    ``scene_state`` is **required** at ``generate`` time. Inject ``generate_fn``
    or ``client`` for tests/mocks (no GPU). When neither is provided, uses the
    shared single text LLM (:func:`planner.text_llm.get_shared_text_client`).

    On invalid JSON / unknown symbols / model failure, optionally falls back to
    ``RuleBasedGoalGenerator`` (default ``fallback_rule_based=True``).
    """

    backend = "local_llm"

    def __init__(
        self,
        *,
        generate_fn: GenerateFn | None = None,
        client: LocalTextClient | None = None,
        model_id: str | None = None,
        fallback_rule_based: bool = True,
        prompt_id: str | None = None,
    ) -> None:
        if generate_fn is not None and client is not None:
            raise ValueError("pass only one of generate_fn or client")
        self._generate_fn = generate_fn
        self._client = client
        self._model_id = resolve_text_llm_model_id(model_id)
        self._fallback_rule_based = fallback_rule_based
        # None → active default file; "v1"/"v2" pin versioned prompts for sweeps.
        env_prompt = os.environ.get("VLMRP_GOAL_LLM_PROMPT_ID")
        self._prompt_id = prompt_id if prompt_id is not None else env_prompt
        self._rule_based = RuleBasedGoalGenerator()

    def _complete(self, system: str, user: str, *, stage: str = "goal") -> str:
        def _run() -> str:
            if self._generate_fn is not None:
                return self._generate_fn(system, user)
            if self._client is not None:
                return self._client.complete(system, user)
            # Shared with domain select / enrich so the weights load at most once.
            return get_shared_text_client(self._model_id).complete(system, user)

        return invoke_llm(stage, _run)

    def generate(
        self,
        command: str,
        objects: Sequence[str],
        *,
        locations: Sequence[str] | None = None,
        domain_template: str = "manipulation_base",
        allowed_predicates: Sequence[str] | None = None,
        domain_actions: Sequence[dict] | None = None,
        scene_state: SceneState | None = None,
    ) -> GoalResult:
        text = command.strip()
        if not text:
            return GoalResult(
                facts=[],
                backend=self.backend,
                ok=False,
                error="empty command",
            )

        if scene_state is None:
            error = "scene_state is required for local_llm backend"
            if self._fallback_rule_based:
                return self._fallback(
                    text,
                    list(objects),
                    list(locations or []),
                    domain_template,
                    allowed_predicates,
                    raw=None,
                    error=error,
                )
            return GoalResult(
                facts=[],
                backend=self.backend,
                ok=False,
                error=error,
            )

        scene_objs, scene_locs = _symbols_from_scene(scene_state)
        objs = _merge_symbols(objects, scene_objs)
        locs = _merge_symbols(locations or [], scene_locs)
        if scene_state.robot.holding:
            objs = _merge_symbols(objs, [scene_state.robot.holding])
        if scene_state.robot.camera_aimed_at:
            objs = _merge_symbols(objs, [scene_state.robot.camera_aimed_at])
        loc_names = set(locs)
        for rel in scene_state.relations:
            for arg in rel.args:
                if arg in loc_names:
                    continue
                # Prefer classifying unknown args as items (locations are listed).
                objs = _merge_symbols(objs, [arg])

        allowed = resolve_allowed_predicates(domain_template, allowed_predicates)
        known = frozenset(objs) | frozenset(locs)

        quantified = quantified_on_facts(text, objs, locs)
        if quantified and all(fact[0] in allowed for fact in quantified):
            return GoalResult(
                facts=quantified,
                backend=self.backend,
                ok=True,
            )

        system_prompt = load_goal_prompt(
            domain_template, prompt_id=self._prompt_id
        )
        scene_json = compact_scene_json(scene_state, include_poses=False)
        user_prompt = _build_user_prompt(
            text,
            objs,
            locs,
            allowed,
            domain_template,
            scene_json,
            domain_actions=domain_actions,
        )

        raw: str | None = None
        try:
            raw = self._complete(system_prompt, user_prompt)
            facts, error = parse_goal_facts(
                raw, allowed=allowed, known_symbols=known
            )
        except Exception as exc:  # noqa: BLE001 — surface as GoalResult
            facts, error = [], f"local LLM call failed: {exc}"

        if error is None and facts:
            return GoalResult(
                facts=facts,
                backend=self.backend,
                raw=raw,
                ok=True,
            )

        if self._fallback_rule_based:
            return self._fallback(
                text,
                objs,
                locs,
                domain_template,
                list(allowed),
                raw=raw,
                error=error or "local LLM goal generation failed",
            )

        return GoalResult(
            facts=[],
            backend=self.backend,
            raw=raw,
            ok=False,
            error=error or "local LLM goal generation failed",
        )

    def _fallback(
        self,
        command: str,
        objects: list[str],
        locations: list[str],
        domain_template: str,
        allowed_predicates: Sequence[str] | None,
        *,
        raw: str | None,
        error: str | None,
    ) -> GoalResult:
        fallback = self._rule_based.generate(
            command,
            objects,
            locations=locations,
            domain_template=domain_template,
            allowed_predicates=(
                list(allowed_predicates) if allowed_predicates is not None else None
            ),
        )
        if fallback.ok:
            return GoalResult(
                facts=fallback.facts,
                backend=self.backend,
                raw=raw,
                ok=True,
                error=f"local_llm_fallback_rule_based: {error}",
            )
        return GoalResult(
            facts=[],
            backend=self.backend,
            raw=raw,
            ok=False,
            error=error or fallback.error or "local LLM goal generation failed",
        )
