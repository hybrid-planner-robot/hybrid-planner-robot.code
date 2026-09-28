"""
PDDL problem generation facade.

Re-exports the public API used by Pipeline, orchestrator, and scripts.
Default behavior is the legacy plan-inferred init/goal path
(``legacy`` module). Hybrid SceneState generation is available via
``use_hybrid=True`` — see ``docs/hybrid_problem_generator_design.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from vlm.planner import VLMPlan

from .assembler import generate_problem_hybrid
from .init_generator.renderer import PddlFact
from .init_generator.schema import SceneState
from .legacy import (
    DOMAIN_TEMPLATE_TO_NAME,
    extract_objects_and_locations,
    generate_problem as _legacy_generate_problem,
    infer_goal_state,
    infer_init_state,
    write_problem as _legacy_write_problem,
)

__all__ = [
    "DOMAIN_TEMPLATE_TO_NAME",
    "extract_objects_and_locations",
    "generate_problem",
    "infer_goal_state",
    "infer_init_state",
    "write_problem",
]


def generate_problem(
    plan: VLMPlan,
    domain_name: str | None = None,
    problem_name: str = "generated_problem",
    *,
    scene_state: SceneState | None = None,
    goal_facts: Sequence[PddlFact] | None = None,
    use_hybrid: bool = False,
) -> str:
    """
    Generate a PDDL problem string from a VLMPlan.

    ``use_hybrid=False`` (default): legacy init/goal inference from the plan.

    ``use_hybrid=True``: ``:init`` from ``InitRenderer(scene_state)``,
    ``:goal`` from ``goal_facts`` (or ``RuleBasedGoalGenerator(plan.goal)``),
    ``:objects`` from ``scene_state`` unioned with plan symbols.
    """
    if use_hybrid:
        if scene_state is None:
            raise ValueError("use_hybrid=True requires scene_state")
        return generate_problem_hybrid(
            plan,
            scene_state,
            goal_facts,
            domain_name=domain_name,
            problem_name=problem_name,
        )
    return _legacy_generate_problem(
        plan,
        domain_name=domain_name,
        problem_name=problem_name,
    )


def write_problem(
    plan: VLMPlan,
    output_path: str | Path,
    domain_name: str | None = None,
    *,
    scene_state: SceneState | None = None,
    goal_facts: Sequence[PddlFact] | None = None,
    use_hybrid: bool = False,
) -> Path:
    """Generate and write the PDDL problem to a file."""
    content = generate_problem(
        plan,
        domain_name=domain_name,
        scene_state=scene_state,
        goal_facts=goal_facts,
        use_hybrid=use_hybrid,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path
