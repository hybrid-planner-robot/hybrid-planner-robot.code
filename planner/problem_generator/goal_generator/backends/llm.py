"""
Text-LLM GoalGenerator backend (no vision).

Calls a pluggable text-completion client with a prompt constrained to
allowed PDDL predicates and known object/location symbols.  CI uses a
mock client; live Gemini smoke is gated behind ``@pytest.mark.llm``.
"""

from __future__ import annotations

import os
from typing import Callable, Protocol, Sequence

from ...init_generator.schema import SceneState
from ..json_facts import parse_goal_facts
from ..predicates import resolve_allowed_predicates
from ..types import GoalResult
from .rule_based import RuleBasedGoalGenerator
from prompts import load_prompt

CompleteFn = Callable[[str, str], str]


class TextCompletionClient(Protocol):
    """Minimal text-only completion interface (no images)."""

    def complete(self, system: str, user: str) -> str:
        ...


class GeminiTextClient:
    """Google Gemini text client via ``google-genai`` (no vision parts)."""

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._api_key = api_key
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from google import genai

        key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key not found. Set GOOGLE_API_KEY or pass api_key=."
            )
        self._client = genai.Client(api_key=key)

    def complete(self, system: str, user: str) -> str:
        from google.genai import types as gtypes

        self._ensure_client()
        model_name = (
            self.model_id
            if self.model_id.startswith("models/")
            else f"models/{self.model_id}"
        )
        response = self._client.models.generate_content(
            model=model_name,
            contents=[gtypes.Part.from_text(text=user)],
            config=gtypes.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=512,
                temperature=0.0,
            ),
        )
        return response.text or ""


_SYSTEM_PROMPT = load_prompt("goal", "cloud.md")


def _build_user_prompt(
    command: str,
    objects: Sequence[str],
    locations: Sequence[str],
    allowed: frozenset[str],
    domain_template: str,
) -> str:
    preds = ", ".join(sorted(allowed))
    objs = ", ".join(objects) if objects else "(none)"
    locs = ", ".join(locations) if locations else "(none)"
    return (
        f"domain_template: {domain_template}\n"
        f"allowed_predicates: {preds}\n"
        f"objects: {objs}\n"
        f"locations: {locs}\n"
        f"command: {command.strip()}\n"
    )


class LLMGoalGenerator:
    """
    Text-LLM GoalGenerator.

    Inject ``complete_fn`` or ``client`` for tests/mocks.  When neither is
    provided, uses ``GeminiTextClient`` (requires ``GOOGLE_API_KEY``).

    On LLM/parse failure, optionally falls back to ``RuleBasedGoalGenerator``
    (default ``fallback_rule_based=True``).
    """

    backend = "llm"

    def __init__(
        self,
        *,
        complete_fn: CompleteFn | None = None,
        client: TextCompletionClient | None = None,
        model_id: str = "gemini-2.0-flash",
        api_key: str | None = None,
        fallback_rule_based: bool = True,
    ) -> None:
        if complete_fn is not None and client is not None:
            raise ValueError("pass only one of complete_fn or client")
        self._complete_fn = complete_fn
        self._client = client
        self._model_id = model_id
        self._api_key = api_key
        self._fallback_rule_based = fallback_rule_based
        self._rule_based = RuleBasedGoalGenerator()

    def _complete(self, system: str, user: str) -> str:
        if self._complete_fn is not None:
            return self._complete_fn(system, user)
        if self._client is not None:
            return self._client.complete(system, user)
        live = GeminiTextClient(model_id=self._model_id, api_key=self._api_key)
        return live.complete(system, user)

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
        del scene_state  # cloud scaffold does not consume SceneState
        text = command.strip()
        if not text:
            return GoalResult(
                facts=[],
                backend=self.backend,
                ok=False,
                error="empty command",
            )

        locs = list(locations or [])
        objs = list(objects)
        allowed = resolve_allowed_predicates(domain_template, allowed_predicates)
        known = frozenset(objs) | frozenset(locs)

        user_prompt = _build_user_prompt(text, objs, locs, allowed, domain_template)
        raw: str | None = None
        try:
            raw = self._complete(_SYSTEM_PROMPT, user_prompt)
            facts, error = parse_goal_facts(
                raw, allowed=allowed, known_symbols=known
            )
        except Exception as exc:  # noqa: BLE001 — surface as GoalResult
            facts, error = [], f"LLM call failed: {exc}"

        if error is None and facts:
            return GoalResult(
                facts=facts,
                backend=self.backend,
                raw=raw,
                ok=True,
            )

        if self._fallback_rule_based:
            fallback = self._rule_based.generate(
                text,
                objs,
                locations=locs,
                domain_template=domain_template,
                allowed_predicates=list(allowed),
            )
            if fallback.ok:
                return GoalResult(
                    facts=fallback.facts,
                    backend=self.backend,
                    raw=raw,
                    ok=True,
                    error=f"llm_fallback_rule_based: {error}",
                )

        return GoalResult(
            facts=[],
            backend=self.backend,
            raw=raw,
            ok=False,
            error=error or "LLM goal generation failed",
        )
