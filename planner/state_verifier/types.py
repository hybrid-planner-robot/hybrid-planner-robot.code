"""Verification result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from planner.problem_generator.init_generator.schema import SceneState

Verdict = Literal["GREEN", "YELLOW", "RED"]


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    mismatch_score: float
    expected: SceneState
    observed: SceneState
    mismatches: list[str] = field(default_factory=list)
    request_vlm: bool = False
    replan: bool = False
