"""
Closed robot skill catalog for online domain enrichment (Sessions 28–30).

Source of truth mirrors executable primitives:
  ``ros2_ws/.../primitives/`` + ``orchestrator._prim_dispatch``.

The catalog is **per robot**, not global. Household worlds (tabletop / kitchen /
office) share one robot; workshop is a different robot with a disjoint
enrichment set. The name ``cut`` exists on both, but each robot authors and
caches its own PDDL.

Enrichment may **only** ground skills from the active robot's catalog into PDDL.
Skills already present in a fixed domain template are not enrichment candidates
for that template. Open-vocabulary motor invention is forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

# Canonical catalog names use underscore form (orchestrator dispatch keys).
# PDDL often uses hyphens; callers should normalize when comparing.

ROBOT_HOUSEHOLD = "household"
ROBOT_WORKSHOP = "workshop"

# Worlds that are not listed here share the household robot.
WORLD_TO_ROBOT: Mapping[str, str] = {
    "workshop": ROBOT_WORKSHOP,
}

# Shared stock motors already grounded in the four fixed templates.
_BASE_CATALOG_SKILLS: frozenset[str] = frozenset(
    {
        "pick",
        "place",
        "look_at",
        "navigate_to",
        "open_container",
        "close_container",
    }
)

# Household enrichment (ROS bodies exist). Keep this name for R0 / claim 2.
ENRICHMENT_CANDIDATE_SKILLS: frozenset[str] = frozenset(
    {"pour", "tilt", "stir", "cut"}
)

# Workshop enrichment is plan-only: same motor names as PDDL actions, no
# Gazebo body required. ``cut`` is independently authored, not household reuse.
WORKSHOP_ENRICHMENT_CANDIDATE_SKILLS: frozenset[str] = frozenset(
    {"cut", "drill", "paint", "clamp"}
)

# Skills the household robot can execute today (orchestrator._prim_dispatch
# keys that are real primitives — not say / gripper helpers).
CATALOG_SKILLS: frozenset[str] = _BASE_CATALOG_SKILLS | ENRICHMENT_CANDIDATE_SKILLS

WORKSHOP_CATALOG_SKILLS: frozenset[str] = (
    _BASE_CATALOG_SKILLS | WORKSHOP_ENRICHMENT_CANDIDATE_SKILLS
)

_ALL_CATALOG_SKILLS: frozenset[str] = CATALOG_SKILLS | WORKSHOP_CATALOG_SKILLS


def robot_id_for_world(world: str | None) -> str:
    """Map a sim / suite world name onto a robot catalog identity."""
    key = str(world or "").strip().lower()
    return WORLD_TO_ROBOT.get(key, ROBOT_HOUSEHOLD)


def catalog_skills_for_world(world: str | None = None) -> frozenset[str]:
    """Closed executable catalog for the robot that inhabits ``world``."""
    if robot_id_for_world(world) == ROBOT_WORKSHOP:
        return WORKSHOP_CATALOG_SKILLS
    return CATALOG_SKILLS


def enrichment_candidates_for_world(world: str | None = None) -> frozenset[str]:
    """Skills this robot may ground into PDDL (never the other robot's set)."""
    if robot_id_for_world(world) == ROBOT_WORKSHOP:
        return WORKSHOP_ENRICHMENT_CANDIDATE_SKILLS
    return ENRICHMENT_CANDIDATE_SKILLS


# Fixed-domain PDDL action names (hyphen form as in pddl/domains/*.pddl).
# open/close_container are in containers_manipulation; treat carefully — ROS
# stubs exist but the PDDL actions are already on that template.
DOMAIN_PDDL_ACTIONS: Mapping[str, frozenset[str]] = {
    "manipulation_base": frozenset({"pick", "place", "look-at"}),
    "manipulation_stacking": frozenset(
        {"pick", "place", "look-at", "stack", "unstack"}
    ),
    "containers_manipulation": frozenset(
        {
            "pick",
            "place",
            "look-at",
            "stack",
            "unstack",
            "open-container",
            "close-container",
            "pick-from-container",
            "place-in-container",
        }
    ),
    "navigation_manipulation": frozenset(
        {
            "pick",
            "place",
            "look-at",
            "stack",
            "unstack",
            "navigate-to",
        }
    ),
}

# Aliases: catalog underscore ↔ PDDL hyphen (and look_at spelling).
_CATALOG_TO_PDDL: dict[str, str] = {
    "look_at": "look-at",
    "navigate_to": "navigate-to",
    "open_container": "open-container",
    "close_container": "close-container",
    "pick_from_container": "pick-from-container",
    "place_in_container": "place-in-container",
    "pick": "pick",
    "place": "place",
    "stack": "stack",
    "unstack": "unstack",
    "pour": "pour",
    "tilt": "tilt",
    "stir": "stir",
    "cut": "cut",
    "drill": "drill",
    "paint": "paint",
    "clamp": "clamp",
}

_PDDL_TO_CATALOG: dict[str, str] = {v: k for k, v in _CATALOG_TO_PDDL.items()}


def normalize_catalog_skill(name: str) -> str:
    """Map PDDL / mixed spellings to catalog underscore form."""
    text = str(name or "").strip().lower().replace("-", "_")
    if text in _ALL_CATALOG_SKILLS:
        return text
    return text


def catalog_to_pddl_action(skill: str) -> str:
    """Canonical PDDL action spelling for a catalog skill."""
    key = normalize_catalog_skill(skill)
    return _CATALOG_TO_PDDL.get(key, key.replace("_", "-"))


def pddl_to_catalog_skill(action: str) -> str:
    """Map a PDDL action name to catalog form when known."""
    raw = str(action or "").strip().lower()
    if raw in _PDDL_TO_CATALOG:
        return _PDDL_TO_CATALOG[raw]
    return raw.replace("-", "_")


def domain_pddl_skills(template: str) -> frozenset[str]:
    """Catalog-normalized skills already present in ``template`` PDDL."""
    actions = DOMAIN_PDDL_ACTIONS.get(template, frozenset())
    return frozenset(pddl_to_catalog_skill(a) for a in actions)


def skills_already_in_template(
    template: str, skills: Sequence[str]
) -> tuple[str, ...]:
    """Subset of ``skills`` that the fixed template already defines as PDDL."""
    present = domain_pddl_skills(template)
    out: list[str] = []
    for skill in skills:
        name = normalize_catalog_skill(skill)
        if name in present and name not in out:
            out.append(name)
    return tuple(out)


def skills_missing_from_domain(
    template: str,
    world: str | None = None,
) -> frozenset[str]:
    """
    Catalog skills not yet grounded in the selected domain's PDDL.

    Uses the active robot's catalog (household unless ``world`` is workshop).
    Enrichment candidates for a template are typically
    ``skills_missing_from_domain(t, world) & enrichment_candidates_for_world(world)``.
    """
    present = domain_pddl_skills(template)
    catalog = catalog_skills_for_world(world)
    return frozenset(s for s in catalog if s not in present)


def enrichment_candidates_for_domain(
    template: str,
    world: str | None = None,
) -> frozenset[str]:
    """
    Skills the online enricher may propose for ``template`` on this robot.

    Closed set: this robot's enrichment candidates ∩ missing from the domain.
    ``world=None`` keeps the household set so existing callers are unchanged.
    """
    return skills_missing_from_domain(template, world=world) & (
        enrichment_candidates_for_world(world)
    )


# ── Executable signatures (Session 29) ───────────────────────────────────────
#
# An enriched PDDL action is only legal if it can be dispatched. The orchestrator
# reads the object name at ``_ORACLE_ARG_IDX[prim]`` of the argument list, so a
# grounded action must supply at least the arguments the ROS body consumes.
# An extra trailing argument (a surface / tool the body ignores) is tolerated;
# too few arguments is not.


@dataclass(frozen=True)
class SkillSignature:
    """How a catalog skill is executed, and what its PDDL action may look like."""

    skill: str
    ros_primitive: str  # orchestrator._prim_dispatch key
    pddl_action: str
    roles: tuple[str, ...]  # semantic role of each required argument
    description: str

    @property
    def min_params(self) -> int:
        return len(self.roles)

    @property
    def max_params(self) -> int:
        return len(self.roles) + 1


SKILL_SIGNATURES: Mapping[str, SkillSignature] = {
    sig.skill: sig
    for sig in (
        SkillSignature(
            "pick",
            "pick",
            "pick",
            ("item",),
            "close the gripper on a graspable item",
        ),
        SkillSignature(
            "place",
            "place",
            "place",
            ("item", "location"),
            "release the held item onto a surface",
        ),
        SkillSignature(
            "look_at",
            "look_at",
            "look-at",
            ("item",),
            "aim the wrist camera at an item",
        ),
        SkillSignature(
            "navigate_to",
            "navigate_to",
            "navigate-to",
            ("location",),
            "drive the base to a symbolic location",
        ),
        SkillSignature(
            "open_container",
            "open_container",
            "open-container",
            ("container",),
            "open a drawer or lid",
        ),
        SkillSignature(
            "close_container",
            "close_container",
            "close-container",
            ("container",),
            "close a drawer or lid",
        ),
        SkillSignature(
            "pour",
            "pour",
            "pour",
            ("source", "target"),
            "tilt the held source vessel over a target vessel to transfer its "
            "contents, then release the source",
        ),
        SkillSignature(
            "tilt",
            "tilt",
            "tilt",
            ("item",),
            "rotate the held item about the wrist without releasing it",
        ),
        SkillSignature(
            "stir",
            "stir",
            "stir",
            ("container",),
            "move the held tool in a circular path inside a container",
        ),
        SkillSignature(
            "cut",
            "cut",
            "cut",
            ("item",),
            "make repeated downward strokes on an item with the held tool",
        ),
        SkillSignature(
            "drill",
            "drill",
            "drill",
            ("item",),
            "bore a hole in a workpiece with the held drill",
        ),
        SkillSignature(
            "paint",
            "paint",
            "paint",
            ("source", "target"),
            "apply coating from the held source onto a target surface",
        ),
        SkillSignature(
            "clamp",
            "clamp",
            "clamp",
            ("item",),
            "secure a workpiece in a clamp so it cannot shift",
        ),
    )
}


def skill_signature(name: str) -> SkillSignature | None:
    """Executable signature for a catalog skill / PDDL action name."""
    return SKILL_SIGNATURES.get(normalize_catalog_skill(name))


def ros_primitive_for_action(action: str) -> str | None:
    """
    ``orchestrator._prim_dispatch`` key for a PDDL action name, or ``None``.

    ``None`` means the robot has no body for that action — the online path must
    refuse rather than plan with it.
    """
    sig = skill_signature(action)
    return sig.ros_primitive if sig is not None else None


def is_executable_skill(name: str) -> bool:
    """True when some robot catalog has a dispatch key for this skill / action."""
    return normalize_catalog_skill(name) in _ALL_CATALOG_SKILLS


def catalog_summary_lines(skills: frozenset[str] | None = None) -> list[str]:
    """Prompt-ready ``name(roles) — description`` lines for the closed catalog."""
    names = sorted(skills if skills is not None else CATALOG_SKILLS)
    lines: list[str] = []
    for name in names:
        sig = SKILL_SIGNATURES.get(name)
        if sig is None:
            continue
        args = ", ".join(sig.roles)
        lines.append(f"{sig.pddl_action}({args}) — {sig.description}")
    return lines
