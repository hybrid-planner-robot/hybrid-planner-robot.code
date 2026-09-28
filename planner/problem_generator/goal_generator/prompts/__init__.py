"""
Domain-specialized goal prompts for the local text LLM backend.

Files live in the repository-wide ``prompts/goal/`` directory, named after
``domain_template`` (e.g. ``manipulation_base.md``).

Versioned pins:
  ``manipulation_base.v1.md`` — earlier baseline
  ``manipulation_base.v2.md`` — hardened prompts (also the active default ``.md``)
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from prompts import prompt_path

_PROMPTS_DIR = prompt_path("goal")
_DEFAULT_DOMAIN = "manipulation_base"
_VERSION_STEM = re.compile(r"^(.+)\.(v\d+)$")

__all__ = [
    "load_goal_prompt",
    "list_prompt_domains",
    "list_prompt_versions",
    "PROMPTS_DIR",
    "DEFAULT_PROMPT_ID",
]

PROMPTS_DIR = _PROMPTS_DIR
# Active default file is unversioned ``{domain}.md`` (currently v2 content).
DEFAULT_PROMPT_ID = "v2"


def list_prompt_domains() -> list[str]:
    """Return domain template names that have an active (unversioned) prompt file."""
    names: list[str] = []
    for path in _PROMPTS_DIR.glob("*.md"):
        if path.stem == "cloud":
            continue
        if _VERSION_STEM.match(path.stem):
            continue
        names.append(path.stem)
    return sorted(names)


def list_prompt_versions(domain_template: str = _DEFAULT_DOMAIN) -> list[str]:
    """Return available version ids (e.g. ``v1``, ``v2``) for a domain."""
    versions: list[str] = []
    for path in _PROMPTS_DIR.glob(f"{domain_template}.v*.md"):
        match = _VERSION_STEM.match(path.stem)
        if match:
            versions.append(match.group(2))
    return sorted(versions)


@lru_cache(maxsize=32)
def load_goal_prompt(
    domain_template: str = _DEFAULT_DOMAIN,
    prompt_id: str | None = None,
) -> str:
    """
    Load the system+few-shot prompt body for ``domain_template``.

    ``prompt_id`` selects a versioned pin for A/B sweeps:
      - ``None``: active default ``{domain}.md``
      - ``v1`` / ``v2`` / …: ``{domain}.{prompt_id}.md`` (v2 falls back to
        ``{domain}.md`` if the pin file is missing)
      - any other string: treat as an explicit stem under ``prompts/goal/``

    Falls back to ``manipulation_base`` when no domain-specific file exists.
    """
    path = _resolve_prompt_path(domain_template, prompt_id)
    if not path.is_file():
        path = _resolve_prompt_path(_DEFAULT_DOMAIN, prompt_id)
    return path.read_text(encoding="utf-8").strip() + "\n"


def _resolve_prompt_path(domain_template: str, prompt_id: str | None) -> Path:
    if prompt_id is None:
        return _PROMPTS_DIR / f"{domain_template}.md"

    pid = prompt_id.strip()
    if re.fullmatch(r"v\d+", pid):
        versioned = _PROMPTS_DIR / f"{domain_template}.{pid}.md"
        if versioned.is_file():
            return versioned
        if pid == DEFAULT_PROMPT_ID:
            return _PROMPTS_DIR / f"{domain_template}.md"
        return versioned

    stem = pid[:-3] if pid.endswith(".md") else pid
    return _PROMPTS_DIR / f"{stem}.md"
