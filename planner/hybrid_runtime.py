"""
Hybrid problem-generation runtime for the host loop (Session 11).

Flag OFF → callers keep the legacy ``generate_problem(plan)`` path.
Flag ON (``mvp`` / ``1``) → fused SceneState ``:init`` + goal facts once at
task start (rule-based by default; ``local_llm`` opt-in via Phase A+).

Flag ``full`` → Phase B: same as MVP plus ``StateVerifier`` after each step
(YELLOW → VLM patch; RED → replan; tracker updates only on GREEN / resolved
YELLOW).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from vlm.planner import VLMPlan

from planner.enrichment_effects import EnrichmentContext, ground_enrichment_goal
from planner.primitive_transitions import CompletedAction, normalize_primitive
from planner.problem_generator import generate_problem
from planner.problem_generator.enrichment_goal import goals_from_domain_additions
from planner.problem_generator.goal_generator.backends.local_llm import LocalLLMGoalGenerator
from planner.problem_generator.goal_generator.backends.rule_based import RuleBasedGoalGenerator
from planner.problem_generator.goal_generator.domain_view import (
    compact_domain_view,
    enrichment_authored_payload,
    enrichment_outcome_predicates,
    facts_use_any_predicate,
    filter_facts_to_predicates,
)
from planner.problem_generator.init_generator.adapters.dino import DinoAdapter
from planner.problem_generator.init_generator.adapters.oracle import OracleAdapter
from planner.problem_generator.init_generator.builder import build_scene
from planner.problem_generator.init_generator.renderer import PddlFact
from planner.problem_generator.init_generator.schema import RelationFact, SceneState
from planner.problem_generator.init_generator.vlm_fusion import (
    VlmFusionClient,
    build_vlm_callback,
)
from planner.state_tracker import StateTracker
from planner.state_verifier import StateVerifier, VerificationResult, Verdict

logger = logging.getLogger(__name__)

ENV_HYBRID_FLAG = "VLMRP_HYBRID_PROBLEM_GEN"
ENV_GOAL_BACKEND = "VLMRP_GOAL_BACKEND"
ENV_SCENE_SOURCE = "VLMRP_SCENE_SOURCE"
ENV_CONTROL = "VLMRP_CONTROL"
# Re-export for callers; parsing lives in online_enrichment (Session 28).
ENV_ONLINE_ENRICHMENT = "VLMRP_ONLINE_ENRICHMENT"

# Core primitives supported by all hybrid-capable templates.
MVP_PRIMITIVES = frozenset({"pick", "place", "look-at", "look_at"})

# Per-template additional primitives whose :init / :goal already have hybrid
# support (InitRenderer predicates + RuleBasedGoalGenerator rules).
# Session 18: widening the gate so non-base templates are not forced to legacy
# solely because of their domain-specific primitive names.
_TEMPLATE_EXTRA_PRIMITIVES: dict[str, frozenset[str]] = {
    "manipulation_stacking": frozenset({"stack", "unstack"}),
    "containers_manipulation": frozenset(
        {"stack", "unstack", "pick-from-container", "place-in-container",
         "open-container", "close-container",
         # underscore aliases the loop may emit
         "pick_from_container", "place_in_container",
         "open_container", "close_container"}
    ),
    "navigation_manipulation": frozenset({"navigate-to", "navigate_to"}),
}

# Union of all known hybrid-capable primitives across every template.
_ALL_HYBRID_PRIMITIVES: frozenset[str] = MVP_PRIMITIVES | frozenset(
    p for extras in _TEMPLATE_EXTRA_PRIMITIVES.values() for p in extras
)


class HybridMode(str, Enum):
    OFF = "off"
    MVP = "mvp"
    FULL = "full"  # Phase B: verifier + VLM-on-YELLOW


class GoalBackend(str, Enum):
    RULE_BASED = "rule_based"
    LOCAL_LLM = "local_llm"


class SceneSource(str, Enum):
    """
    Which perception streams feed hybrid ``:init`` fusion.

    - ``fused`` (default): oracle + DINO; ``sim_like`` prefers oracle on conflict
    - ``dino``: DINO (+ tracker) only — closer to real-world (no Gazebo GT)
    - ``inventory``: VLM names the scene, then DINO localizes those queries
      (real-robot open-vocabulary; no Gazebo catalog sweep)
    - ``oracle``: Gazebo oracle (+ tracker) only — sim ground-truth / ablation
    """

    FUSED = "fused"
    DINO = "dino"
    INVENTORY = "inventory"
    ORACLE = "oracle"


class ControlMode(str, Enum):
    """
    Closed-loop action policy (Session 22).

    - ``vlm_steps`` (default): vision VLM proposes remaining actions each iteration
    - ``fd``: Fast Downward plans from hybrid ``:init``/``:goal``; VLM is not the
      action policy. Pair with ``--hybrid mvp`` only (Session 23: ``full``+``fd``
      is refused — the FD path never calls StateVerifier).
    """

    VLM_STEPS = "vlm_steps"
    FD = "fd"


class IncompatibleControlHybridError(ValueError):
    """Raised when ``--control`` and ``--hybrid`` cannot be combined honestly."""


def control_hybrid_incompatibility(
    control: ControlMode,
    hybrid: HybridMode,
) -> str | None:
    """
    Return an error message when the flag combo would load an unused model.

    Session 23: ``--hybrid full`` enables StateVerifier + YELLOW VLM patches on
    the ``vlm_steps`` path only. The ``control=fd`` executor never calls the
    verifier, so ``full``+``fd`` would pay VRAM for a silent no-op.
    """
    if control == ControlMode.FD and hybrid == HybridMode.FULL:
        return (
            "--control fd is incompatible with --hybrid full: the FD execution "
            "path does not run StateVerifier, so the verifier VLM would be "
            "loaded but never called. Use --hybrid mvp --control fd "
            "(measured stack) or --hybrid full --control vlm_steps."
        )
    return None


def assert_control_hybrid_compatible(
    control: ControlMode,
    hybrid: HybridMode,
) -> None:
    """Raise ``IncompatibleControlHybridError`` when the combo is dishonest."""
    msg = control_hybrid_incompatibility(control, hybrid)
    if msg is not None:
        raise IncompatibleControlHybridError(msg)


def resolve_hybrid_mode(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> HybridMode:
    """
    Parse hybrid feature flag.

    Accepted truthy values: ``1``, ``true``, ``yes``, ``on``, ``mvp``.
    ``full`` selects Phase B (verifier + VLM-on-YELLOW).
    Unset / ``0`` / ``false`` / ``off`` → OFF.
    """
    raw = value if value is not None else (env or os.environ).get(ENV_HYBRID_FLAG, "")
    text = str(raw).strip().lower()
    if text in {"", "0", "false", "no", "off", "none"}:
        return HybridMode.OFF
    if text in {"full"}:
        return HybridMode.FULL
    if text in {"1", "true", "yes", "on", "mvp"}:
        return HybridMode.MVP
    # Unknown → treat as OFF to avoid surprising production enablement
    return HybridMode.OFF


def resolve_goal_backend(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> GoalBackend:
    """Parse ``VLMRP_GOAL_BACKEND`` (default ``rule_based``)."""
    raw = value if value is not None else (env or os.environ).get(ENV_GOAL_BACKEND, "")
    text = str(raw).strip().lower()
    if text in {"local_llm", "local", "local-llm"}:
        return GoalBackend.LOCAL_LLM
    return GoalBackend.RULE_BASED


def resolve_scene_source(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> SceneSource:
    """Parse ``VLMRP_SCENE_SOURCE`` (default ``fused`` = oracle+DINO)."""
    raw = value if value is not None else (env or os.environ).get(ENV_SCENE_SOURCE, "")
    text = str(raw).strip().lower()
    if text in {"dino", "dino_only", "dino-only", "perception", "real"}:
        return SceneSource.DINO
    if text in {"inventory", "vlm", "vlm_inventory", "vlm-inventory"}:
        return SceneSource.INVENTORY
    if text in {"oracle", "oracle_only", "oracle-only", "gazebo", "gt"}:
        return SceneSource.ORACLE
    return SceneSource.FUSED


def scene_source_skips_oracle(source: SceneSource) -> bool:
    """True when :init must not consult Gazebo (DINO catalog or VLM inventory)."""
    return source in {SceneSource.DINO, SceneSource.INVENTORY}


def resolve_control_mode(
    value: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
) -> ControlMode:
    """
    Parse ``VLMRP_CONTROL`` / ``--control`` (default ``vlm_steps``).

    Accepted FD aliases: ``fd``, ``fast_downward``, ``fast-downward``, ``pddl``.
    Anything else (including unset) → ``vlm_steps`` so the legacy loop stays safe.
    """
    raw = value if value is not None else (env or os.environ).get(ENV_CONTROL, "")
    text = str(raw).strip().lower()
    if text in {"fd", "fast_downward", "fast-downward", "fastdownward", "pddl"}:
        return ControlMode.FD
    return ControlMode.VLM_STEPS


def select_domain_template(command: str) -> str:
    """
    Pick a fixed domain template once per task (Session 22).

    Heuristic only. Unknown / ambiguous → ``manipulation_base``.
    Keyword matching is a known limitation (Session 23); a classifier is a
    separate decision.

    For completeness ``complete`` | ``incomplete`` and enricher routing, use
    ``planner.online_enrichment.select_domain`` / ``resolve_domain_for_task``
    (Session 28; opt-in ``VLMRP_ONLINE_ENRICHMENT``). This helper remains the
    template picker and stays backward-compatible for the default fixed path.
    """
    text = " ".join(str(command or "").lower().split())
    if not text:
        return "manipulation_base"

    if any(
        k in text
        for k in (
            "navigate to",
            "go to",
            "move to",
            "walk to",
            "drive to",
        )
    ):
        return "navigation_manipulation"

    if any(
        k in text
        for k in (
            "stack ",
            "unstack",
            "on top of",
            "stacked on",
        )
    ):
        return "manipulation_stacking"

    # Container phrasing: "in/into …" but not "on" surface goals.
    # Pour/stir/tilt + "into" is a catalog enrichment skill, not place-in-container
    # (merging pour onto containers_manipulation clashes with in-container).
    _liquid = any(
        f" {cue} " in f" {text} "
        for cue in ("pour", "pouring", "stir", "stirring", "tilt", "tilting", "decant")
    )
    if not _liquid and (
        any(
            k in text
            for k in (
                " into ",
                " inside ",
                "in the drawer",
                "in the box",
                "in the cup",
                "in the bowl",
                "in the container",
                "place-in-container",
            )
        )
        or (
            (" put " in f" {text} " or text.startswith("put ") or " place " in f" {text} ")
            and any(
                k in text
                for k in (" drawer", " box", " bowl", " container", " bin")
            )
            and " on " not in f" {text} "
        )
    ):
        return "containers_manipulation"

    return "manipulation_base"


def make_domain_stub_plan(
    command: str,
    domain_template: str = "manipulation_base",
    *,
    domain_additions: Mapping[str, Any] | None = None,
) -> VLMPlan:
    """
    VLMPlan with **no action steps** — domain + NL goal only.

    Used by ``control=fd`` so hybrid ``:init``/``:goal`` and Pipeline/FD do not
    depend on a vision-VLM action sketch.

    ``domain_additions`` (Session 30) carries the closed-catalog enrichment
    authored for this task. Passing it here is what makes the *enriched* domain
    reach Fast Downward: ``Pipeline`` merges the same additions into the base
    template with ``DomainEnricher``, the assembler uses their predicates for
    ``:init``, and ``goals_from_domain_additions`` reads them for ``:goal`` —
    one source of truth for all three.
    """
    additions = dict(domain_additions or {})
    return VLMPlan(
        goal=str(command or "").strip(),
        steps=[],
        raw_output="[control=fd] domain stub — no VLM action steps",
        domain_template=domain_template or "manipulation_base",
        domain_additions={
            "new_types": list(additions.get("new_types", []) or []),
            "new_predicates": list(additions.get("new_predicates", []) or []),
            "new_actions": list(additions.get("new_actions", []) or []),
            "modified_preconditions": dict(
                additions.get("modified_preconditions", {}) or {}
            ),
            # R1 only: unary affordance facts for :init. R0 leaves this empty.
            "init_facts": list(additions.get("init_facts", []) or []),
        },
    )


def select_scenes_for_source(
    scene_source: SceneSource,
    *,
    oracle_scene: SceneState | None,
    dino_scene: SceneState | None,
) -> tuple[SceneState | None, SceneState | None, bool]:
    """
    Filter streams for fusion and choose ``sim_like``.

    Returns ``(oracle_for_fuse, dino_for_fuse, sim_like)``.
    ``sim_like=True`` only for fused/oracle modes (oracle preferred on conflict).
    """
    if scene_source in {SceneSource.DINO, SceneSource.INVENTORY}:
        return None, dino_scene, False
    if scene_source == SceneSource.ORACLE:
        return oracle_scene, None, True
    return oracle_scene, dino_scene, True


def plan_has_enrichment(plan: VLMPlan) -> bool:
    da = plan.domain_additions or {}
    return bool(
        da.get("new_predicates")
        or da.get("new_actions")
        or da.get("new_types")
        or da.get("modified_preconditions")
    )


def enrichment_provenance(plan: VLMPlan) -> str:
    """
    Where this plan's ``domain_additions`` came from: none / catalog / open_vocab.

    ``catalog`` means every enriched action maps to a real ROS primitive, i.e.
    the Sessions 28–30 closed-catalog path. ``open_vocab`` is the legacy VLM
    invention kept for ablation only — callers should log it as such.
    """
    if not plan_has_enrichment(plan):
        return "none"
    context = EnrichmentContext.from_domain_additions(plan.domain_additions)
    if not context:
        # Predicates / types but no actions: nothing to execute either way.
        return "catalog"
    return "catalog" if context.catalog_only else "open_vocab"


def plan_hybrid_compatible(plan: VLMPlan) -> bool:
    """
    True when hybrid problem generation can handle this plan.

    Criteria (Session 19 update):
    - All *standard* primitives are in the known hybrid-capable set for the
      plan's ``domain_template``.
    - Enrichment primitives (those defined in ``domain_additions.new_actions``)
      are allowed — hybrid derives their ``:goal`` from the action effects
      via ``goals_from_domain_additions``. Init still comes from SceneState.

    Session 18: widened from MVP-only (pick/place/look_at) to include
    domain-specific primitives already supported by InitRenderer /
    RuleBasedGoalGenerator so non-base templates do not silently fall back
    to legacy solely because of their action names.
    """
    template = getattr(plan, "domain_template", "manipulation_base") or "manipulation_base"
    allowed_raw: frozenset[str] = MVP_PRIMITIVES | _TEMPLATE_EXTRA_PRIMITIVES.get(
        template, frozenset()
    )
    allowed = {normalize_primitive(p) for p in allowed_raw}

    # Enrichment action names are also permitted (Session 19).
    da = plan.domain_additions or {}
    enrichment_names: set[str] = set()
    for act in da.get("new_actions", []):
        name = act.get("name", "")
        if name:
            enrichment_names.add(normalize_primitive(name))

    for step in plan.steps or []:
        prim = normalize_primitive(getattr(step, "primitive", "") or "")
        if prim not in allowed and prim not in enrichment_names:
            return False
    return True


def plan_mvp_compatible(plan: VLMPlan) -> bool:
    """
    Alias kept for callers that used the old name.

    Previously checked only pick/place/look_at (MVP set); since Session 18
    the real gate is :func:`plan_hybrid_compatible`, which also accepts
    template-specific primitives (stack/unstack, navigate-to, …).
    """
    return plan_hybrid_compatible(plan)


def scene_from_xyz_poses(
    poses: Mapping[str, Mapping[str, float]],
    *,
    known_locations: Sequence[str] | None = None,
    on_surface: Mapping[str, str] | None = None,
    gripper_empty: bool = True,
    holding: str | None = None,
    camera_aimed_at: str | None = None,
    domain_template: str | None = None,
    frame_id: str = "panda_link0",
) -> SceneState:
    """Build an oracle-tagged SceneState from Gazebo ``{name: {x,y,z}}`` poses."""
    from simulation.oracle.world_state import (
        ObjectState,
        Orientation as OracleOrientation,
        Pose as OraclePose,
        Position as OraclePosition,
        WorldState,
    )

    surface = dict(on_surface or {})
    objects: list[ObjectState] = []
    for name, xyz in poses.items():
        loc = surface.get(name, "")
        objects.append(
            ObjectState(
                name=name,
                pose=OraclePose(
                    position=OraclePosition(
                        x=float(xyz["x"]),
                        y=float(xyz["y"]),
                        z=float(xyz["z"]),
                    ),
                    orientation=OracleOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                location=loc,
            )
        )
    world = WorldState(objects=objects, gripper_empty=gripper_empty and holding is None)
    return OracleAdapter.from_world_state(
        world,
        known_locations=list(known_locations) if known_locations else None,
        frame_id=frame_id,
        domain_template=domain_template,
        holding=holding,
        camera_aimed_at=camera_aimed_at,
    )


def infer_grasp_target(command: str, item_names: Sequence[str]) -> str | None:
    """
    Best-effort match of an NL command to a graspable scene item.

    Used by callers that still need a best-effort grasp-item match from NL.
    ``control=fd`` no longer seeds ``camera-aimed-at`` in ``:init`` from this.
    """
    text = " ".join(str(command or "").lower().replace("_", " ").split())
    if not text or not item_names:
        return None
    ranked = sorted((str(n) for n in item_names), key=len, reverse=True)
    for name in ranked:
        label = name.replace("_", " ").lower()
        if not label:
            continue
        if label in text:
            return name
        tokens = label.split()
        if len(tokens) > 1 and all(tok in text for tok in tokens):
            return name
    return None


# Look-at *as the task* (not a means to pick). Pre-scan must not seed
# ``camera-aimed-at`` here, or the goal is already true and FD returns [].
_LOOK_AT_AS_GOAL = re.compile(
    r"\b(?:look[\s_-]*at|inspect|examine)\b",
    re.IGNORECASE,
)


def command_is_look_at_goal(command: str) -> bool:
    """True when NL asks to aim the camera, not to pick/place the object."""
    text = " ".join(str(command or "").lower().replace("_", " ").split())
    return bool(text and _LOOK_AT_AS_GOAL.search(text))


# Gazebo models that are placement surfaces / furniture, not graspable items.
# Used so hybrid :init does not emit bogus facts like ``(on shelf_b table)``.
# ``drawer`` stays here so a world that *has* one is partitioned correctly, but
# it is not in the default ``known_locations`` list (no shipped world contains it).
GAZEBO_LOCATION_MODELS = frozenset(
    {
        "shelf_b",
        "shelf",
        "drawer",
        "table",
        "counter",
        "tray",
        "target_tray",
        "metal_tray",
        "workbench",
        "platform",
        "desk",
    }
)

# NL / VLM labels → preferred Gazebo symbols when present in the scene.
_LOCATION_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    ("shelf_b", "shelf"),
    ("drawer", "drawers"),
    ("target_tray", "tray"),
)


def partition_gazebo_for_hybrid(
    gazebo_poses: Mapping[str, Mapping[str, float]],
    *,
    base_locations: Sequence[str] | None = None,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """
    Split Gazebo models into graspable item poses vs symbolic locations.

    Location models (e.g. ``shelf_b``) are omitted from item poses and only
    appear in ``known_locations``. Returns ``(item_poses, known_locations)``.
    """
    base = [str(x) for x in (base_locations or ["table", "shelf"])]
    item_poses: dict[str, dict[str, float]] = {}
    scene_locations: list[str] = []
    for name, xyz in gazebo_poses.items():
        if name in GAZEBO_LOCATION_MODELS:
            if name != "table" and name not in scene_locations:
                scene_locations.append(name)
        else:
            item_poses[str(name)] = {
                "x": float(xyz["x"]),
                "y": float(xyz["y"]),
                "z": float(xyz["z"]),
            }

    known: list[str] = ["table"]
    for loc in sorted(scene_locations):
        if loc not in known:
            known.append(loc)
    for loc in base:
        if loc in known:
            continue
        # Prefer Gazebo ``shelf_b`` over abstract ``shelf`` when both would apply.
        if loc == "shelf" and "shelf_b" in known:
            continue
        if loc == "tray" and "target_tray" in known:
            continue
        known.append(loc)
    return item_poses, known


def resolve_location_alias(
    name: str,
    known_locations: Sequence[str],
) -> str:
    """Map a VLM/NL location label onto a symbol in ``known_locations`` when possible."""
    text = (name or "").strip()
    if not text:
        return text
    known = set(known_locations)
    if text in known:
        return text
    lowered = text.lower().replace(" ", "_")
    if lowered in known:
        return lowered
    for group in _LOCATION_ALIAS_GROUPS:
        if text in group or lowered in group:
            for cand in group:
                if cand in known:
                    return cand
    return text


def rewrite_command_locations(
    command: str,
    known_locations: Sequence[str],
) -> str:
    """
    Rewrite NL location words so rule-based goals match Gazebo symbols.

    Example: ``place the red cup on the shelf`` + ``shelf_b`` in scene
    → ``place the red cup on shelf_b``.
    """
    import re

    text = command or ""
    known = set(known_locations)
    if "shelf_b" in known:
        text = re.sub(r"\bthe\s+shelf\b", "shelf_b", text, flags=re.IGNORECASE)
        text = re.sub(r"\bon\s+shelf\b", "on shelf_b", text, flags=re.IGNORECASE)
        text = re.sub(r"\bto\s+shelf\b", "to shelf_b", text, flags=re.IGNORECASE)
    if "target_tray" in known:
        text = re.sub(r"\bthe\s+tray\b", "target_tray", text, flags=re.IGNORECASE)
        text = re.sub(r"\bon\s+tray\b", "on target_tray", text, flags=re.IGNORECASE)
        text = re.sub(r"\bto\s+tray\b", "to target_tray", text, flags=re.IGNORECASE)
    return text


def ground_plan_locations(
    plan: VLMPlan,
    known_locations: Sequence[str],
) -> list[str]:
    """
    Rewrite ``location`` / ``container`` args on plan steps to Gazebo symbols.

    Skips ``look_at`` ``target`` (object, not location). Returns human-readable
    ``old→new`` notes for logging.
    """
    notes: list[str] = []
    for step in plan.steps or []:
        prim = normalize_primitive(getattr(step, "primitive", "") or "")
        args = dict(getattr(step, "args", None) or {})
        changed = False
        for key in ("location", "container"):
            raw = args.get(key)
            if not isinstance(raw, str) or not raw.strip():
                continue
            new = resolve_location_alias(raw, known_locations)
            if new != raw:
                notes.append(f"{raw}→{new}")
                args[key] = new
                changed = True
        # place sometimes uses target as location synonym
        if prim == "place" and isinstance(args.get("target"), str):
            raw = args["target"]
            new = resolve_location_alias(raw, known_locations)
            if new != raw:
                notes.append(f"{raw}→{new}")
                args["target"] = new
                changed = True
        if changed:
            step.args = args
    return notes


def scene_from_dino_payload(
    detections: Sequence[Mapping[str, Any]] | None = None,
    *,
    poses: Mapping[str, Mapping[str, float]] | None = None,
    on_surface: Mapping[str, str] | None = None,
    known_locations: Sequence[str] | None = None,
    domain_template: str | None = None,
) -> SceneState | None:
    """Convert DINO detection list / pose map to SceneState (or None if empty)."""
    dets = list(detections or [])
    pose_map = dict(poses or {})
    if not dets and not pose_map:
        return None
    if not dets and pose_map:
        dets = [
            {"name": name, "box": [0, 0, 1, 1], "score": 0.7}
            for name in pose_map
        ]
    return DinoAdapter.from_detections(
        dets,
        poses=pose_map,
        on_surface=dict(on_surface or {}),
        known_locations=list(known_locations) if known_locations else None,
        domain_template=domain_template,
    )


def _scene_summary(scene: SceneState | None) -> dict[str, Any] | None:
    """Compact SceneState view for debug.json / smoke side-by-side logs."""
    if scene is None:
        return None
    on_rels = [
        list(r.args)
        for r in scene.relations
        if r.predicate == "on" and len(r.args) >= 2
    ]
    return {
        "objects": [o.name for o in scene.objects],
        "object_sources": {o.name: o.source for o in scene.objects},
        "locations": [loc.name for loc in scene.locations],
        "on": on_rels,
        "gripper_empty": scene.robot.gripper_empty,
        "holding": scene.robot.holding,
        "sources_used": list(scene.meta.sources_used) if scene.meta else [],
        "fusion_notes": list(scene.meta.fusion_notes) if scene.meta else [],
    }


def scene_compare_snapshot(
    oracle_scene: SceneState | None = None,
    dino_scene: SceneState | None = None,
    fused_scene: SceneState | None = None,
) -> dict[str, Any]:
    """Oracle vs DINO vs fused summaries (Session 14 logging)."""
    return {
        "oracle": _scene_summary(oracle_scene),
        "dino": _scene_summary(dino_scene),
        "fused": _scene_summary(fused_scene),
    }


def completed_action_from_step(step: Any) -> CompletedAction:
    """Normalize PlanStep / dict-like into ``CompletedAction`` for ``tracker.apply``."""
    if isinstance(step, CompletedAction):
        return step
    primitive = getattr(step, "primitive", None)
    args = getattr(step, "args", None)
    success_flag = getattr(step, "success_flag", True)
    if primitive is None and isinstance(step, Mapping):
        primitive = step.get("primitive", "")
        args = step.get("args", {})
        success_flag = bool(step.get("success_flag", True))
    return CompletedAction(
        primitive=str(primitive or ""),
        args=dict(args or {}),
        success_flag=bool(success_flag),
    )


@dataclass
class StepVerificationOutcome:
    """Result of post-action verification in ``full`` mode (Phase B)."""

    verdict: Verdict
    tracker_updated: bool
    replan: bool
    result: VerificationResult | None = None
    vlm_calls: int = 0
    mismatches: list[str] = field(default_factory=list)


@dataclass
class HybridProblemSession:
    """
    Per-task hybrid state: StateTracker + goal facts fixed until replan.

    In ``mvp`` mode call ``note_completed`` after a successful primitive.
    In ``full`` mode prefer ``process_completed_step`` (verifier gates the
    tracker update and may signal replan on RED).
    """

    mode: HybridMode = HybridMode.MVP
    goal_backend: GoalBackend = GoalBackend.RULE_BASED
    scene_source: SceneSource = SceneSource.FUSED
    command: str = ""
    domain_template: str = "manipulation_base"
    known_locations: list[str] = field(default_factory=lambda: ["table", "shelf"])
    tracker: StateTracker = field(default_factory=StateTracker)
    goal_facts: list[PddlFact] | None = None
    # Survives invalidate_goal() so an end-of-run goal check still knows what the
    # task was aiming at: a replan clears goal_facts, and the last thing a failing
    # run does is replan.
    last_goal_facts: list[PddlFact] | None = None
    goal_backend_used: str | None = None
    goal_error: str | None = None
    local_generate_fn: Callable[[str, str], str] | None = None
    vlm_fusion_client: VlmFusionClient | None = None
    verifier: StateVerifier | None = None
    last_fused_scene: SceneState | None = None
    last_oracle_scene: SceneState | None = None
    last_dino_scene: SceneState | None = None
    last_scene_compare: dict[str, Any] | None = None
    # Session 25: loop annotates shortcut / pose provenance before generate.
    perception_provenance: dict[str, Any] | None = None
    last_localisation_report: list[dict[str, Any]] | None = None
    verdict_counts: dict[str, int] = field(
        default_factory=lambda: {"GREEN": 0, "YELLOW": 0, "RED": 0}
    )
    vlm_call_count: int = 0
    goal_fallback_count: int = 0
    # ── Online enrichment (Sessions 28–30) ───────────────────────────────────
    # ``enrichment`` is the live loop hook: it teaches tracker and verifier what
    # an enriched action does. The rest is bookkeeping for logs / summary.json.
    enrichment: EnrichmentContext | None = None
    # Grounds enriched action parameters for :goal when there are no VLM steps.
    goal_binder: Any | None = None
    enrichment_skills: list[str] = field(default_factory=list)
    domain_persisted: str | None = None
    domain_reused: bool = False
    domain_select_backend: str | None = None
    refuse_reason: str | None = None
    # Raw ``domain_additions`` from online enrichment (for goal view + telemetry).
    domain_additions: dict[str, Any] | None = None
    _goal_locked: bool = False

    def __post_init__(self) -> None:
        if self.enrichment is not None:
            self.tracker.enrichment = self.enrichment

    @property
    def enabled(self) -> bool:
        return self.mode != HybridMode.OFF

    @property
    def enrichment_used(self) -> bool:
        """True when this run executes on an online-enriched domain."""
        return bool(self.enrichment)

    def set_enrichment(
        self,
        context: EnrichmentContext | None,
        *,
        skills: Sequence[str] = (),
        domain_path: str | None = None,
        reused: bool = False,
        backend: str | None = None,
        goal_binder: Any | None = None,
        domain_additions: Mapping[str, Any] | None = None,
    ) -> None:
        """Attach the task's enriched actions to the tracker and the verifier."""
        self.enrichment = context
        if goal_binder is not None:
            self.goal_binder = goal_binder
        self.tracker.enrichment = context
        if self.verifier is not None:
            self.verifier.enrichment = context
        self.enrichment_skills = list(skills)
        self.domain_persisted = domain_path
        self.domain_reused = reused
        if backend:
            self.domain_select_backend = backend
        if domain_additions is not None:
            self.domain_additions = dict(domain_additions)
        elif context and context.actions:
            self.domain_additions = {
                "new_actions": [dict(a) for a in context.actions.values()],
                "new_predicates": sorted(context.declared_predicates),
            }
        else:
            self.domain_additions = None

    @property
    def verifier_enabled(self) -> bool:
        """True when Phase B verification runs after each completed step."""
        return self.mode == HybridMode.FULL

    @property
    def is_holding(self) -> bool:
        snap = self.tracker.snapshot()
        return snap.holding is not None and not snap.gripper_empty

    def holding_object(self) -> str | None:
        return self.tracker.snapshot().holding

    def should_use_hybrid(self, plan: VLMPlan) -> bool:
        """
        True when hybrid problem generation should run for this plan.

        Hybrid is used when:
        - the session is enabled (mode != OFF), and
        - the plan has no enrichment domain_additions, and
        - all primitives are in the hybrid-capable set for the plan's
          domain_template (Session 18: includes stacking/containers/navigation
          domain-specific primitives, not just pick/place/look_at).
        """
        if not self.enabled:
            return False
        provenance = enrichment_provenance(plan)
        if provenance == "open_vocab":
            logger.warning(
                "hybrid: plan carries open-vocabulary domain_additions "
                "(%s) — ablation only, not the closed-catalog path",
                ", ".join(
                    str(a.get("name", "?"))
                    for a in (plan.domain_additions or {}).get("new_actions", [])
                ),
            )
        return plan_hybrid_compatible(plan)

    def invalidate_goal(self) -> None:
        """Clear cached goal (call on replan); ``last_goal_facts`` is kept."""
        self.goal_facts = None
        self.goal_backend_used = None
        self.goal_error = None
        self._goal_locked = False

    def metrics_snapshot(self) -> dict[str, Any]:
        """
        Per-run hybrid metrics for logs / smoke / Session 12 hooks.

        Includes VLM fusion call count, verifier verdict tallies, and goal
        backend choice (+ rule-based fallback count when ``local_llm`` fails).
        """
        snap: dict[str, Any] = {
            "mode": self.mode.value,
            "scene_source": self.scene_source.value,
            "goal_backend": self.goal_backend.value,
            "goal_backend_used": self.goal_backend_used,
            "goal_fallback_count": self.goal_fallback_count,
            "goal_error": self.goal_error,
            "vlm_call_count": self.vlm_call_count,
            "verdict_counts": {
                "GREEN": int(self.verdict_counts.get("GREEN", 0)),
                "YELLOW": int(self.verdict_counts.get("YELLOW", 0)),
                "RED": int(self.verdict_counts.get("RED", 0)),
            },
            "holding": self.holding_object(),
            "goal_locked": self._goal_locked,
            "goal_facts": [list(f) for f in (self.goal_facts or [])],
            "last_goal_facts": [list(f) for f in (self.last_goal_facts or [])],
            # Session 30 enrichment loop metrics.
            "enrichment_used": self.enrichment_used,
            "enrichment_actions": list(
                self.enrichment.action_names if self.enrichment else ()
            ),
            "enrichment_skills": list(self.enrichment_skills),
            "domain_persisted": self.domain_persisted,
            "domain_reused": self.domain_reused,
            "domain_select_backend": self.domain_select_backend,
            "refuse_reason": self.refuse_reason,
            "enrichment_authored": enrichment_authored_payload(
                self.domain_additions,
                skills=self.enrichment_skills,
                domain_path=self.domain_persisted,
                reused=self.domain_reused,
            ),
            "enrichment_facts": [
                list(f) for f in self.tracker.enrichment_facts()
            ],
        }
        if self.last_scene_compare is not None:
            snap["scene_compare"] = self.last_scene_compare
        if self.last_localisation_report is not None:
            snap["localisation_errors"] = list(self.last_localisation_report)
        if self.perception_provenance is not None:
            snap["perception_provenance"] = dict(self.perception_provenance)
        return snap

    def note_completed(self, step: Any) -> None:
        """Update tracker from a successfully executed step (MVP path)."""
        self.tracker.apply(completed_action_from_step(step))

    def tracker_partial_scene(self) -> SceneState:
        """Compact SceneState from tracker robot + moved relations."""
        partial = self.tracker.as_partial_scene()
        return SceneState(
            objects=[],
            locations=[],
            relations=[RelationFact.from_dict(r) for r in partial["relations"]],
            robot=self.tracker.snapshot(),
        )

    def _ensure_verifier(
        self,
        *,
        images: Sequence[Any] | None = None,
        oracle_facts: SceneState | None = None,
        dino_facts: SceneState | None = None,
        tracker_facts: SceneState | None = None,
    ) -> StateVerifier:
        client = self.vlm_fusion_client
        callback = None
        if client is not None:
            callback = build_vlm_callback(
                client,
                images=images,
                oracle_facts=oracle_facts,
                dino_facts=dino_facts,
                tracker_facts=tracker_facts or self.tracker_partial_scene(),
            )
        self.verifier = StateVerifier(
            vlm_callback=callback,
            enrichment=self.enrichment,
        )
        return self.verifier

    def process_completed_step(
        self,
        step: Any,
        *,
        observed: SceneState,
        pre_scene: SceneState | None = None,
        success_flag: bool = True,
        images: Sequence[Any] | None = None,
        oracle_facts: SceneState | None = None,
        dino_facts: SceneState | None = None,
    ) -> StepVerificationOutcome:
        """
        Post-action gate used in ``full`` mode.

        - MVP / verifier off: apply tracker immediately (GREEN-equivalent).
        - FULL: verify(expected, observed); update tracker only on GREEN
          (including YELLOW resolved by VLM); RED → replan, no tracker update;
          unresolved YELLOW → no tracker update, no replan.
        """
        action = completed_action_from_step(step)
        action = CompletedAction(
            primitive=action.primitive,
            args=dict(action.args),
            success_flag=success_flag,
        )

        if not self.verifier_enabled:
            if success_flag:
                self.note_completed(action)
            return StepVerificationOutcome(
                verdict="GREEN" if success_flag else "RED",
                tracker_updated=bool(success_flag),
                replan=not success_flag,
                mismatches=(
                    []
                    if success_flag
                    else ["executor reported success_flag=False"]
                ),
            )

        pre = pre_scene or self.last_fused_scene
        if pre is None:
            if success_flag:
                self.note_completed(action)
            return StepVerificationOutcome(
                verdict="GREEN" if success_flag else "RED",
                tracker_updated=bool(success_flag),
                replan=not success_flag,
                mismatches=[
                    "no pre_scene for verification; tracker updated without verify"
                ],
            )

        calls_before = 0
        if self.vlm_fusion_client is not None:
            calls_before = int(
                getattr(self.vlm_fusion_client, "call_count", 0) or 0
            )

        verifier = self._ensure_verifier(
            images=images,
            oracle_facts=oracle_facts,
            dino_facts=dino_facts,
        )
        result = verifier.verify(pre, action, observed)

        calls_after = calls_before
        if self.vlm_fusion_client is not None:
            calls_after = int(
                getattr(self.vlm_fusion_client, "call_count", 0) or 0
            )
        vlm_calls = max(0, calls_after - calls_before)
        self.vlm_call_count += vlm_calls
        self.verdict_counts[result.verdict] = (
            self.verdict_counts.get(result.verdict, 0) + 1
        )

        if result.verdict == "GREEN":
            self.note_completed(action)
            return StepVerificationOutcome(
                verdict="GREEN",
                tracker_updated=True,
                replan=False,
                result=result,
                vlm_calls=vlm_calls,
                mismatches=list(result.mismatches),
            )

        if result.verdict == "RED":
            self.invalidate_goal()
            return StepVerificationOutcome(
                verdict="RED",
                tracker_updated=False,
                replan=True,
                result=result,
                vlm_calls=vlm_calls,
                mismatches=list(result.mismatches),
            )

        return StepVerificationOutcome(
            verdict="YELLOW",
            tracker_updated=False,
            replan=False,
            result=result,
            vlm_calls=vlm_calls,
            mismatches=list(result.mismatches),
        )

    def ensure_goal(
        self,
        scene: SceneState,
        *,
        command: str | None = None,
        objects: Sequence[str] | None = None,
        locations: Sequence[str] | None = None,
        force: bool = False,
        plan: VLMPlan | None = None,
    ) -> list[PddlFact]:
        """
        Generate ``:goal`` once at task start (or after ``invalidate_goal``).

        The goal LLM (or rule-based fallback) is the sole authority when it
        returns facts that engage enrichment outcomes (when enrichment ran).
        It sees the *effective* domain (predicates + actions), whether stock
        or online-enriched — same prompt shape, no "enriched" label.

        Mechanical ``ground_enrichment_goal`` / step binding is used when the
        primary path is empty **or** ignores every enrichment-introduced
        predicate (e.g. ``holding(cup)`` while the authored effect is
        ``transferred-liquid``) — never merged on top of a non-empty primary.
        """
        if self._goal_locked and self.goal_facts is not None and not force:
            return list(self.goal_facts)

        text = (command if command is not None else self.command).strip()
        objs = list(objects) if objects is not None else [o.name for o in scene.objects]
        locs = (
            list(locations)
            if locations is not None
            else [loc.name for loc in scene.locations]
        )
        domain = scene.domain_template or self.domain_template
        if plan is not None and plan.domain_additions and not self.domain_additions:
            self.domain_additions = dict(plan.domain_additions)

        try:
            facts, backend_used, error = self._generate_goal_facts(
                text, objs, locs, domain, scene
            )
        except ValueError as exc:
            # An enriched task is allowed to have a request the standard goal
            # generators cannot read — "get me something to drink" names no
            # objects, and the whole point of the enrichment is that the
            # authored action supplies the goal. Re-raised below if it doesn't.
            if not self.enrichment:
                raise
            facts, backend_used, error = [], "enrichment_only", str(exc)

        outcome_preds = enrichment_outcome_predicates(self.domain_additions)
        if (
            facts
            and outcome_preds
            and not facts_use_any_predicate(facts, outcome_preds)
        ):
            # Primary goal stayed on stock fluents despite an authored outcome.
            logger.info(
                "discarding stock-only goal %s; enrichment outcomes=%s",
                facts,
                sorted(outcome_preds),
            )
            facts = []
            backend_used = (
                f"{backend_used}+ignored_stock_goal"
                if backend_used
                else "ignored_stock_goal"
            )

        # Enrichment grounding is a fallback only — never merge on top of a
        # non-empty primary goal that already uses enrichment outcomes.
        if not facts:
            enrichment_facts: list[PddlFact] = []
            if plan is not None:
                enrichment_facts = goals_from_domain_additions(
                    plan.domain_additions, plan.steps or []
                )
            if not enrichment_facts and self.enrichment:
                enrichment_facts = ground_enrichment_goal(
                    self.enrichment,
                    text,
                    list(objs) + list(locs),
                    binder=self.goal_binder,
                )
            if enrichment_facts:
                facts = list(enrichment_facts)
                backend_used = (
                    f"{backend_used}+enrichment_fallback"
                    if backend_used
                    else "enrichment_fallback"
                )

        if not facts and backend_used == "enrichment_only":
            # The command was unreadable *and* the enriched action could not be
            # bound to the scene: there is no goal to plan for. Fail here rather
            # than send an empty :goal to the planner.
            raise ValueError(
                f"goal generation failed and enrichment could not be grounded: {error}"
            )

        self.goal_facts = facts
        if facts:
            self.last_goal_facts = list(facts)
        self.goal_backend_used = backend_used
        self.goal_error = error
        self._goal_locked = True
        return list(facts)

    def _domain_text_for_goal(self) -> str | None:
        path = self.domain_persisted
        if not path:
            return None
        try:
            p = Path(path)
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("could not read enriched domain %s: %s", path, exc)
        return None

    def _generate_goal_facts(
        self,
        command: str,
        objects: list[str],
        locations: list[str],
        domain_template: str,
        scene: SceneState,
    ) -> tuple[list[PddlFact], str, str | None]:
        rule = RuleBasedGoalGenerator()
        view = compact_domain_view(
            domain_template,
            domain_text=self._domain_text_for_goal(),
            domain_additions=self.domain_additions,
        )
        allowed = list(view["predicates"])
        domain_actions = list(view["actions"])

        if self.goal_backend == GoalBackend.LOCAL_LLM:
            local = LocalLLMGoalGenerator(
                generate_fn=self.local_generate_fn,
                fallback_rule_based=True,
            )
            result = local.generate(
                command,
                objects,
                locations=locations,
                domain_template=domain_template,
                allowed_predicates=allowed,
                domain_actions=domain_actions,
                scene_state=scene,
            )
            if result.ok and result.facts:
                facts = filter_facts_to_predicates(result.facts, allowed)
                # LocalLLMGoalGenerator may already have fallen back to rule-based.
                if result.error and "fallback" in str(result.error).lower():
                    self.goal_fallback_count += 1
                return list(facts), result.backend, result.error
            # Never fall back to vision-VLM goals — try explicit rule-based.
            fallback = rule.generate(
                command,
                objects,
                locations=locations,
                domain_template=domain_template,
                allowed_predicates=allowed,
            )
            if fallback.ok:
                self.goal_fallback_count += 1
                facts = filter_facts_to_predicates(fallback.facts, allowed)
                return (
                    list(facts),
                    "rule_based",
                    f"local_llm_failed:{result.error}",
                )
            raise ValueError(
                f"goal generation failed (local_llm + rule_based): {result.error}"
            )

        result = rule.generate(
            command,
            objects,
            locations=locations,
            domain_template=domain_template,
            allowed_predicates=allowed,
        )
        if not result.ok:
            raise ValueError(f"rule-based goal failed: {result.error}")
        facts = filter_facts_to_predicates(result.facts, allowed)
        return list(facts), result.backend, None

    def build_scene_state(
        self,
        *,
        oracle_scene: SceneState | None = None,
        dino_scene: SceneState | None = None,
        sim_like: bool | None = None,
    ) -> SceneState:
        fuse_oracle, fuse_dino, auto_sim = select_scenes_for_source(
            self.scene_source,
            oracle_scene=oracle_scene,
            dino_scene=dino_scene,
        )
        use_sim = auto_sim if sim_like is None else sim_like
        scene = build_scene(
            oracle_scene=fuse_oracle,
            dino_scene=fuse_dino,
            tracker=self.tracker,
            sim_like=use_sim,
        )
        if scene.domain_template is None:
            scene.domain_template = self.domain_template
        return scene

    def generate_hybrid_problem(
        self,
        plan: VLMPlan,
        *,
        oracle_scene: SceneState | None = None,
        dino_scene: SceneState | None = None,
        problem_name: str = "hybrid_loop_problem",
        sim_like: bool | None = None,
    ) -> tuple[str, SceneState]:
        """
        Fuse scene → ensure goal → ``generate_problem(..., use_hybrid=True)``.

        Returns ``(pddl_string, fused_scene)``. Stores ``last_fused_scene`` for
        Phase B verification (pre-action state).

        ``scene_source`` filters which streams enter fusion; both sides are
        still kept in ``last_*`` / ``scene_compare`` for diagnostics.
        """
        fuse_oracle, fuse_dino, auto_sim = select_scenes_for_source(
            self.scene_source,
            oracle_scene=oracle_scene,
            dino_scene=dino_scene,
        )
        use_sim = auto_sim if sim_like is None else sim_like
        scene = self.build_scene_state(
            oracle_scene=oracle_scene,
            dino_scene=dino_scene,
            sim_like=use_sim,
        )
        # Stash raw inputs (pre-filter) for side-by-side compare / logging.
        self.last_oracle_scene = oracle_scene
        self.last_dino_scene = dino_scene
        self.last_fused_scene = scene
        compare = scene_compare_snapshot(oracle_scene, dino_scene, scene)
        compare["scene_source"] = self.scene_source.value
        # Session 25: distinguish requested scene_source from what actually
        # entered fusion / :init (see planner.dino_localisation).
        from planner.dino_localisation import extend_fused_from_provenance

        extra = self.perception_provenance if isinstance(
            self.perception_provenance, dict
        ) else None
        compare["fused_from"] = extend_fused_from_provenance(
            {
                "oracle": fuse_oracle is not None,
                "dino": fuse_dino is not None,
                "sim_like": use_sim,
            },
            requested=self.scene_source.value,
            perception_only=bool((extra or {}).get("perception_only", False)),
            shortcuts=(extra or {}).get("shortcuts"),
            pose_provenance=(extra or {}).get("pose_provenance"),
        )
        self.last_scene_compare = compare
        additions = self.domain_additions or (plan.domain_additions if plan else None)
        if additions and additions.get("init_facts"):
            from planner.r1.init_facts import apply_init_facts_from_additions

            scene = apply_init_facts_from_additions(scene, additions)
            self.last_fused_scene = scene
        goal = self.ensure_goal(
            scene,
            command=self.command or plan.goal,
            plan=plan,
        )
        pddl = generate_problem(
            plan,
            scene_state=scene,
            goal_facts=goal,
            use_hybrid=True,
            problem_name=problem_name,
        )
        return pddl, scene


def plan_fd_from_problem(
    command: str,
    *,
    domain_template: str,
    pddl_problem: str,
    fd_planner: Any | None = None,
    domains_dir: Any | None = None,
    repair_retries: int = 3,
) -> Any:
    """
    One-shot Fast Downward plan from a pre-built PDDL problem (Session 22).

    Uses a domain stub ``VLMPlan`` (empty steps) so the vision VLM is never
    consulted for actions. ``fd_planner`` may be a mock in CI.
    """
    from planner.pipeline import Pipeline

    stub = make_domain_stub_plan(command, domain_template)
    pipeline = Pipeline(
        vlm=None,
        fd_planner=fd_planner,
        domains_dir=domains_dir,
        repair_retries=repair_retries,
    )
    return pipeline.run(
        command,
        [],
        vlm_plan=stub,
        pddl_problem=pddl_problem,
    )


def create_session_from_env(
    *,
    command: str,
    domain_template: str = "manipulation_base",
    hybrid_flag: str | None = None,
    goal_backend: str | None = None,
    scene_source: str | None = None,
    known_locations: Sequence[str] | None = None,
    local_generate_fn: Callable[[str, str], str] | None = None,
    vlm_fusion_client: VlmFusionClient | None = None,
) -> HybridProblemSession | None:
    """Return a session when hybrid mode is ON, else ``None``."""
    mode = resolve_hybrid_mode(hybrid_flag)
    if mode == HybridMode.OFF:
        return None
    return HybridProblemSession(
        mode=mode,
        goal_backend=resolve_goal_backend(goal_backend),
        scene_source=resolve_scene_source(scene_source),
        command=command,
        domain_template=domain_template,
        known_locations=list(known_locations or ["table", "shelf"]),
        local_generate_fn=local_generate_fn,
        vlm_fusion_client=vlm_fusion_client,
    )
