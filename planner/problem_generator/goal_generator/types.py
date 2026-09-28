"""Goal generator result types."""

from __future__ import annotations

from dataclasses import dataclass

from ..init_generator.renderer import PddlFact

__all__ = ["GoalResult", "PddlFact"]


@dataclass
class GoalResult:
    facts: list[PddlFact]
    backend: str
    raw: str | None = None
    ok: bool = True
    error: str | None = None
