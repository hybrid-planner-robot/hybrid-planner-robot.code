"""
InitBuilder — fuse perception sources into canonical SceneState.

Orchestration glue between adapters / StateTracker and InitRenderer.
Does not emit PDDL.

See ``docs/hybrid_problem_generator_design.md`` §3.2.2.
"""

from __future__ import annotations

from typing import Any

from planner.state_tracker import StateTracker

from .fusion import FusionEngine
from .renderer import InitRenderer, PddlFact
from .schema import SceneState


def build_scene(
    *,
    oracle_scene: SceneState | None = None,
    dino_scene: SceneState | None = None,
    tracker: StateTracker | SceneState | dict[str, Any] | None = None,
    vlm_patch: SceneState | None = None,
    sim_like: bool = True,
    fusion: FusionEngine | None = None,
) -> SceneState:
    """
    Fuse optional oracle, DINO, and tracker inputs into one SceneState.

    ``tracker`` may be a ``StateTracker`` instance, a partial ``SceneState``,
    or a dict from ``StateTracker.as_partial_scene()``.
    """
    engine = fusion or FusionEngine()
    return engine.merge(
        oracle_scene=oracle_scene,
        dino_scene=dino_scene,
        tracker=tracker,
        vlm_patch=vlm_patch,
        sim_like=sim_like,
    )


class InitBuilder:
    """Fuse inputs → SceneState; optionally render ``:init`` facts."""

    def __init__(
        self,
        fusion: FusionEngine | None = None,
        renderer: InitRenderer | None = None,
    ) -> None:
        self._fusion = fusion or FusionEngine()
        self._renderer = renderer or InitRenderer()

    def build(
        self,
        *,
        oracle_scene: SceneState | None = None,
        dino_scene: SceneState | None = None,
        tracker: StateTracker | SceneState | dict[str, Any] | None = None,
        vlm_patch: SceneState | None = None,
        sim_like: bool = True,
    ) -> SceneState:
        """Fuse inputs → canonical SceneState (does not emit PDDL)."""
        return self._fusion.merge(
            oracle_scene=oracle_scene,
            dino_scene=dino_scene,
            tracker=tracker,
            vlm_patch=vlm_patch,
            sim_like=sim_like,
        )

    def build_init(
        self,
        *,
        oracle_scene: SceneState | None = None,
        dino_scene: SceneState | None = None,
        tracker: StateTracker | SceneState | dict[str, Any] | None = None,
        vlm_patch: SceneState | None = None,
        sim_like: bool = True,
    ) -> list[PddlFact]:
        """``build(...)`` then ``renderer.render_facts(...)``."""
        scene = self.build(
            oracle_scene=oracle_scene,
            dino_scene=dino_scene,
            tracker=tracker,
            vlm_patch=vlm_patch,
            sim_like=sim_like,
        )
        return self._renderer.render_facts(scene)
