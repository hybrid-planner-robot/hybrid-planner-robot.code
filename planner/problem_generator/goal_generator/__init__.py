"""Goal generation: command → PDDL goal facts (no vision VLM)."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..init_generator.renderer import PddlFact
from ..init_generator.schema import SceneState
from .backends.llm import LLMGoalGenerator
from .backends.local_llm import (
    DEFAULT_GOAL_LLM_MODEL_ID,
    DEFAULT_PERCEPTUAL_VLM_MODEL_ID,
    LocalLLMGoalGenerator,
)
from .backends.rule_based import RuleBasedGoalGenerator
from .renderer import GoalRenderer
from .scene_compact import compact_scene_dict, compact_scene_json
from .types import GoalResult

__all__ = [
    "DEFAULT_GOAL_LLM_MODEL_ID",
    "DEFAULT_PERCEPTUAL_VLM_MODEL_ID",
    "GoalGenerator",
    "GoalRenderer",
    "GoalResult",
    "LLMGoalGenerator",
    "LocalLLMGoalGenerator",
    "PddlFact",
    "RuleBasedGoalGenerator",
    "compact_scene_dict",
    "compact_scene_json",
]


class GoalGenerator(Protocol):
    """Map a task command + known symbols to PDDL goal facts."""

    def generate(
        self,
        command: str,
        objects: Sequence[str],
        *,
        locations: Sequence[str] | None = None,
        domain_template: str = "manipulation_base",
        allowed_predicates: Sequence[str] | None = None,
        scene_state: SceneState | None = None,
    ) -> GoalResult:
        """
        Parse / map the task command into PDDL goal facts constrained to
        known objects and predicates. Local LLM backends also consume a
        compact view of ``scene_state`` (required for ``local_llm``).
        """
        ...
