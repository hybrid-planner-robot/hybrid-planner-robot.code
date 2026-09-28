"""Central prompt library.

Every LLM / VLM system prompt in this repository lives under this directory.
Callers load files through ``prompt_path`` / ``load_prompt`` instead of
embedding long strings or pointing at scattered package folders.

Layout::

    prompts/
      inventory/     image inventory (item vs location vs container)
      vlm/           vision-language action-planner system prompts
      goal/          local text-LLM :goal prompts (per domain template)
      llm_plan/      llm_plan baseline stages
      llm_pddl/      llm_pddl baseline stages + toy few-shot
      domain/        domain select / enrich / bind (R0)
      r1/            R1 affordance enricher + assignment
"""

from __future__ import annotations

from pathlib import Path

DIR = Path(__file__).resolve().parent

__all__ = ["DIR", "load_prompt", "prompt_path"]


def prompt_path(*parts: str | Path) -> Path:
    """Absolute path of a file under ``prompts/``."""
    path = DIR.joinpath(*parts)
    return path


def load_prompt(*parts: str | Path) -> str:
    """Read a prompt file as UTF-8 text."""
    return prompt_path(*parts).read_text(encoding="utf-8")
