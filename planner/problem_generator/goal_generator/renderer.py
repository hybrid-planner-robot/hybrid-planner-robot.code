"""PDDL ``:goal`` section renderer."""

from __future__ import annotations

from typing import Sequence

from ..init_generator.renderer import PddlFact, _format_fact


class GoalRenderer:
    """Render goal fact tuples as a PDDL ``(:goal ...)`` block."""

    def render_section(
        self,
        facts: Sequence[PddlFact],
        *,
        indent: str = "  ",
    ) -> str:
        """
        Return the ``(:goal ...)`` block string.

        Single fact → inline; multiple facts → ``(and ...)``.
        Empty facts → ``(gripper-empty)`` placeholder (legacy-compatible).
        """
        rendered = self._render_fact_strings(facts)
        lines = [f"{indent}(:goal"]
        inner = indent * 2

        if len(rendered) == 1:
            lines.append(f"{inner}{rendered[0]}")
        else:
            lines.append(f"{inner}(and")
            for fact in rendered:
                lines.append(f"{inner}  {fact}")
            lines.append(f"{inner})")

        lines.append(f"{indent})")
        return "\n".join(lines)

    def _render_fact_strings(self, facts: Sequence[PddlFact]) -> list[str]:
        if not facts:
            return ["(gripper-empty)  ; no explicit goal inferred"]

        on_goals = [f for f in facts if f[0] == "on"]
        holding_goals = [f for f in facts if f[0] == "holding"]
        camera_goals = [f for f in facts if f[0] == "camera-aimed-at"]
        stacked_goals = [f for f in facts if f[0] == "stacked-on"]
        container_goals = [f for f in facts if f[0] == "in-container"]
        raw_goals = [f for f in facts if f[0] == "_raw_fact"]
        other_goals = [
            f
            for f in facts
            if f[0]
            not in {
                "on",
                "holding",
                "camera-aimed-at",
                "stacked-on",
                "in-container",
                "_raw_fact",
            }
        ]

        rendered: list[str] = []
        for f in on_goals:
            rendered.append(_format_fact(f))
        for f in holding_goals:
            rendered.append(_format_fact(f))
        for f in camera_goals:
            rendered.append(_format_fact(f))
        for f in stacked_goals:
            rendered.append(_format_fact(f))
        for f in container_goals:
            rendered.append(_format_fact(f))
        for f in other_goals:
            rendered.append(_format_fact(f))
        for f in raw_goals:
            rendered.append(f[1])  # already a full PDDL fact string

        return rendered
