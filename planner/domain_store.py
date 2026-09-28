"""
Base + enriched PDDL domain files on disk (Session 29).

The four fixed templates live in ``pddl/domains/``. Domains produced by
closed-catalog online enrichment are persisted in a sibling cache
(``pddl/domains/enriched/`` by default, ``VLMRP_ENRICHED_DOMAIN_DIR`` to
override) so a later task with the same signature reuses the file instead of
paying for another LLM authoring round.

Signature = template + robot identity + grounded skills + digest of the base
template text. Workshop domains live in a subdirectory so household ``cut`` is
never reused for the workshop robot. The signature also changes whenever a
fixed template is edited, so a stale enriched domain is never silently reused
against a new base.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from planner.skill_catalog import (
    DOMAIN_PDDL_ACTIONS,
    ROBOT_HOUSEHOLD,
    normalize_catalog_skill,
    robot_id_for_world,
)

ENV_ENRICHED_DOMAIN_DIR = "VLMRP_ENRICHED_DOMAIN_DIR"

_REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS_DIR = _REPO_ROOT / "pddl" / "domains"
ENRICHED_DIR_NAME = "enriched"

_SAFE_NAME = re.compile(r"[^a-z0-9_]+")

__all__ = [
    "DOMAINS_DIR",
    "ENV_ENRICHED_DOMAIN_DIR",
    "EnrichedDomainRecord",
    "base_domain_path",
    "clear_enriched_domains",
    "discard_enriched_domain",
    "enrichment_signature",
    "list_enriched_domains",
    "load_base_domain",
    "load_enriched_domain",
    "resolve_enriched_dir",
    "save_enriched_domain",
]


def base_domain_path(template: str) -> Path:
    """Path of a fixed template's ``.pddl`` file (no existence check)."""
    name = _SAFE_NAME.sub("_", str(template or "").strip().lower())
    return DOMAINS_DIR / f"{name}.pddl"


def load_base_domain(template: str) -> str:
    """
    Read one of the four fixed templates.

    Raises ``FileNotFoundError`` for unknown template names so the online path
    fails loudly instead of enriching an empty string.
    """
    if template not in DOMAIN_PDDL_ACTIONS:
        raise FileNotFoundError(
            f"unknown domain template {template!r}; "
            f"known: {sorted(DOMAIN_PDDL_ACTIONS)}"
        )
    path = base_domain_path(template)
    if not path.is_file():
        raise FileNotFoundError(f"domain template file not found: {path}")
    return path.read_text(encoding="utf-8")


def resolve_enriched_dir(
    directory: str | Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    world: str | None = None,
) -> Path:
    """Cache directory for persisted enriched domains.

    Non-household robots (workshop) write under a robot subdirectory so a
    household ``cut`` file is never reused as workshop ``cut``.
    """
    if directory is not None:
        base = Path(directory)
    else:
        source = env if env is not None else os.environ
        raw = str(source.get(ENV_ENRICHED_DOMAIN_DIR, "") or "").strip()
        base = Path(raw) if raw else DOMAINS_DIR / ENRICHED_DIR_NAME
    robot = robot_id_for_world(world)
    if robot != ROBOT_HOUSEHOLD:
        return base / robot
    return base


def enrichment_signature(
    *,
    template: str,
    skills: Sequence[str],
    base_domain_text: str,
    world: str | None = None,
) -> str:
    """Stable short id for (template, robot, grounded skills, base content)."""
    normalized = sorted({normalize_catalog_skill(s) for s in skills if str(s).strip()})
    payload = "\n".join(
        [
            str(template or ""),
            robot_id_for_world(world),
            ",".join(normalized),
            hashlib.sha256(base_domain_text.encode("utf-8")).hexdigest(),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class EnrichedDomainRecord:
    """A persisted enriched domain plus the metadata needed to reuse it."""

    signature: str
    template: str
    skills: tuple[str, ...]
    domain_text: str
    domain_path: str
    ros_primitives: tuple[str, ...] = ()
    domain_additions: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


def _paths_for(signature: str, directory: Path, template: str) -> tuple[Path, Path]:
    stem = f"{_SAFE_NAME.sub('_', template.lower())}__{signature}"
    return directory / f"{stem}.pddl", directory / f"{stem}.json"


def save_enriched_domain(
    *,
    template: str,
    skills: Sequence[str],
    domain_text: str,
    base_domain_text: str,
    ros_primitives: Sequence[str] = (),
    domain_additions: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    directory: str | Path | None = None,
    world: str | None = None,
) -> EnrichedDomainRecord:
    """Write the enriched domain + sidecar metadata; return the record."""
    target = resolve_enriched_dir(directory, world=world)
    target.mkdir(parents=True, exist_ok=True)
    signature = enrichment_signature(
        template=template,
        skills=skills,
        base_domain_text=base_domain_text,
        world=world,
    )
    pddl_path, meta_path = _paths_for(signature, target, template)

    normalized_skills = tuple(
        sorted({normalize_catalog_skill(s) for s in skills if str(s).strip()})
    )
    sidecar: dict[str, Any] = {
        "signature": signature,
        "template": template,
        "skills": list(normalized_skills),
        "ros_primitives": [str(p) for p in ros_primitives],
        "domain_additions": dict(domain_additions or {}),
        "base_sha256": hashlib.sha256(
            base_domain_text.encode("utf-8")
        ).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "world": world,
        "robot": robot_id_for_world(world),
        "meta": dict(meta or {}),
    }

    pddl_path.write_text(domain_text, encoding="utf-8")
    meta_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")

    return EnrichedDomainRecord(
        signature=signature,
        template=template,
        skills=normalized_skills,
        domain_text=domain_text,
        domain_path=str(pddl_path),
        ros_primitives=tuple(str(p) for p in ros_primitives),
        domain_additions=dict(domain_additions or {}),
        meta=dict(meta or {}),
    )


def load_enriched_domain(
    signature: str,
    *,
    template: str,
    directory: str | Path | None = None,
    world: str | None = None,
) -> EnrichedDomainRecord | None:
    """Reload a persisted enriched domain, or ``None`` when absent/corrupt."""
    target = resolve_enriched_dir(directory, world=world)
    pddl_path, meta_path = _paths_for(signature, target, template)
    if not pddl_path.is_file():
        return None

    sidecar: dict[str, Any] = {}
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                sidecar = loaded
        except (json.JSONDecodeError, OSError):
            sidecar = {}

    return EnrichedDomainRecord(
        signature=signature,
        template=str(sidecar.get("template", template)),
        skills=tuple(str(s) for s in sidecar.get("skills", []) or []),
        domain_text=pddl_path.read_text(encoding="utf-8"),
        domain_path=str(pddl_path),
        ros_primitives=tuple(str(p) for p in sidecar.get("ros_primitives", []) or []),
        domain_additions=dict(sidecar.get("domain_additions", {}) or {}),
        meta=dict(sidecar.get("meta", {}) or {}),
    )


def list_enriched_domains(
    directory: str | Path | None = None,
    *,
    world: str | None = None,
) -> list[EnrichedDomainRecord]:
    """All persisted enriched domains in the cache (sorted by file name)."""
    target = resolve_enriched_dir(directory, world=world)
    if not target.is_dir():
        return []
    records: list[EnrichedDomainRecord] = []
    for pddl_path in sorted(target.glob("*.pddl")):
        stem = pddl_path.stem
        if "__" not in stem:
            continue
        template, _, signature = stem.rpartition("__")
        record = load_enriched_domain(
            signature, template=template, directory=target, world=None
        )
        if record is not None:
            records.append(record)
    return records


def discard_enriched_domain(
    signature: str,
    *,
    template: str,
    directory: str | Path | None = None,
    world: str | None = None,
) -> bool:
    """
    Delete one cached enriched domain (``.pddl`` + sidecar ``.json``).

    Returns ``True`` when at least one file was removed. Used when a reused
    domain fails structural / arity validation so the enricher can re-author.
    """
    target = resolve_enriched_dir(directory, world=world)
    pddl_path, meta_path = _paths_for(signature, target, template)
    removed = False
    for path in (pddl_path, meta_path):
        if path.is_file():
            path.unlink()
            removed = True
    return removed


def clear_enriched_domains(
    directory: str | Path | None = None,
    *,
    world: str | None = None,
) -> int:
    """Delete cached enriched domains; return how many ``.pddl`` files went."""
    target = resolve_enriched_dir(directory, world=world)
    if not target.is_dir():
        return 0
    removed = 0
    for path in list(target.glob("*.pddl")) + list(target.glob("*.json")):
        if path.suffix == ".pddl":
            removed += 1
        path.unlink()
    return removed
