"""
Online domain enrichment — selection completeness + Enricher contract.

Architecture (Sessions 28–30):
  1. Domain selection → ``complete`` | ``incomplete``
  2. ``incomplete`` → Domain Enricher
  3. Enricher may only ground skills from the robot skill catalog into PDDL
  4. Persist enriched domain for reuse, or refuse to the user
  5. FD plans on the resulting domain

Two interchangeable backends sit behind this API (Session 29):

``llm``
    The single text LLM (:mod:`planner.domain_llm`) judges template and
    completeness semantically and authors the PDDL. This is the paper path —
    it handles paraphrases such as "get me something to drink".
``rule_based`` (default)
    Keyword template heuristic + verb cues, kept as the CI / no-GPU fallback.
    It only recognises literal cues, so it is not the paper claim.

Opt-in: ``VLMRP_ONLINE_ENRICHMENT`` / ``--online-enrichment`` for enrichment,
``VLMRP_DOMAIN_SELECT`` / ``--domain-select`` for the LLM selector. Both default
OFF / ``rule_based`` — the fixed-domain path is unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Protocol, Sequence

from planner.skill_catalog import (
    DOMAIN_PDDL_ACTIONS,
    enrichment_candidates_for_domain,
    enrichment_candidates_for_world,
    normalize_catalog_skill,
    skills_already_in_template,
)

_LOCATION_SYMBOLS: frozenset[str] = frozenset(
    {
        "table",
        "shelf",
        "shelf_b",
        "shelf_a",
        "counter",
        "floor",
        "drawer",
        "workbench",
        "desk",
        "tray",
        "target_tray",
        "metal_tray",
        "platform",
    }
)

# Optional override for tests: (command, template, scene_symbols) -> covers?
GoalCoverageProbe = Callable[[str, str, Sequence[str]], bool]

ENV_ONLINE_ENRICHMENT = "VLMRP_ONLINE_ENRICHMENT"
ENV_DOMAIN_SELECT = "VLMRP_DOMAIN_SELECT"

# User-visible refuse contract (Session 28 documents; Session 29/30 surface it).
REFUSE_MESSAGE = (
    "Cannot solve this task: no skill in the robot catalog can address the "
    "goal gap for the selected domain. Refusing rather than inventing motor "
    "skills the robot cannot execute."
)


class DomainCompleteness(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class SelectionBackend(str, Enum):
    """Who judges template + completeness."""

    RULE_BASED = "rule_based"  # keyword heuristic — CI / no-GPU fallback
    LLM = "llm"  # the single text LLM — paper path


class EnrichmentStatus(str, Enum):
    """Outcome of an enricher handoff."""

    SKIPPED = "skipped"  # complete domain, or enrichment flag OFF
    STUB = "stub"  # handoff acknowledged, no PDDL authored (Session 28 stub)
    REFUSED = "refused"  # no catalog skill fits — user-visible refuse
    ENRICHED = "enriched"  # PDDL merged + persisted


@dataclass(frozen=True)
class DomainSelectionResult:
    """
    Result of one-shot domain selection for a task.

    ``template`` is one of the four fixed templates on disk.
    ``completeness`` is ``complete`` | ``incomplete``.
    ``reason`` is a short human-readable explanation.
    ``needed_skills`` lists catalog enrichment candidates implied by the goal
    but missing from the template PDDL (empty when complete).
    ``backend`` records who decided (``rule_based`` | ``llm`` |
    ``rule_based_fallback`` when the LLM answer was unusable), and ``error``
    carries the LLM failure that triggered a fallback.
    """

    template: str
    completeness: DomainCompleteness
    reason: str
    needed_skills: tuple[str, ...] = ()
    backend: str = SelectionBackend.RULE_BASED.value
    error: str | None = None


@dataclass(frozen=True)
class EnrichmentRequest:
    """Handoff payload from selection → Domain Enricher."""

    template: str
    command: str
    scene_symbols: tuple[str, ...]
    candidate_skills: tuple[str, ...]
    reason: str
    selection: DomainSelectionResult
    # Override the on-disk template text (tests / already-loaded domains).
    base_domain_text: str | None = None
    # Active robot world (workshop vs household). None = household.
    world: str | None = None


@dataclass(frozen=True)
class EnrichmentOutcome:
    """
    Enricher result.

    ``ENRICHED`` carries the merged ``domain_text``, the file it was persisted
    to, the catalog skills grounded, their ROS primitives, and the
    ``domain_additions`` dict that the Session 19 goal helper
    (``goals_from_domain_additions``) consumes. ``reused`` is True when the
    domain came from the persistence cache instead of a fresh LLM call.
    """

    status: EnrichmentStatus
    refuse_message: str | None = None
    domain_text: str | None = None
    domain_path: str | None = None
    skills_grounded: tuple[str, ...] = ()
    notes: str = ""
    domain_additions: dict[str, Any] | None = None
    ros_primitives: tuple[str, ...] = ()
    signature: str | None = None
    reused: bool = False


@dataclass
class DomainResolution:
    """Selection + optional enricher handoff for one task."""

    selection: DomainSelectionResult
    enrichment: EnrichmentOutcome | None = None
    enrichment_enabled: bool = False

    @property
    def template(self) -> str:
        return self.selection.template

    @property
    def domain_text(self) -> str | None:
        """Enriched domain when one was authored, else ``None`` (fixed file)."""
        if self.enrichment is None:
            return None
        return self.enrichment.domain_text

    @property
    def used_enricher(self) -> bool:
        if self.enrichment is None:
            return False
        return self.enrichment.status not in {
            EnrichmentStatus.SKIPPED,
        }

    @property
    def refused(self) -> bool:
        return (
            self.enrichment is not None
            and self.enrichment.status == EnrichmentStatus.REFUSED
        )


# Rule-based goal → enrichment-candidate skill cues (keyword / phrase).
# Order matters only for reason strings; a task may match multiple skills.
_SKILL_GOAL_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "pour",
        (
            "pour ",
            "pouring",
            "pour from",
            "pour into",
            "pour the",
            " emptied into",
        ),
    ),
    (
        "tilt",
        (
            "tilt ",
            "tilting",
            "tilt the",
            "tip the",
            "tip over",
        ),
    ),
    (
        "stir",
        (
            "stir ",
            "stirring",
            "stir the",
            "mix ",
            "mixing",
            "mix the",
        ),
    ),
    (
        "cut",
        (
            "cut ",
            "cutting",
            "cut the",
            "slice ",
            "slicing",
            "slice the",
            "chop ",
            "chopping",
        ),
    ),
    (
        "drill",
        (
            "drill ",
            "drilling",
            "drill the",
            "drill a",
            "bore ",
            "boring",
            "a hole in",
            "hole in the",
        ),
    ),
    (
        "paint",
        (
            "paint ",
            "painting",
            "paint the",
            "painted",
            "should be painted",
            "coat the",
        ),
    ),
    (
        "clamp",
        (
            "clamp ",
            "clamping",
            "clamp the",
            "secure the",
            "secured in the clamp",
            "should be clamped",
            "should be secured",
        ),
    ),
)


def resolve_online_enrichment(
    value: str | bool | None = None,
    *,
    env: dict[str, str] | None = None,
) -> bool:
    """
    Parse ``VLMRP_ONLINE_ENRICHMENT`` / CLI flag (default **OFF**).

    Accepted truthy: ``1``, ``true``, ``yes``, ``on``, ``enrich``, ``enrichment``.
    """
    if isinstance(value, bool):
        return value
    raw = value if value is not None else (env or os.environ).get(
        ENV_ONLINE_ENRICHMENT, ""
    )
    text = str(raw or "").strip().lower()
    return text in {"1", "true", "yes", "on", "enrich", "enrichment"}


def resolve_selection_backend(
    value: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> SelectionBackend:
    """
    Parse ``VLMRP_DOMAIN_SELECT`` / ``--domain-select`` (default ``rule_based``).

    Accepted LLM aliases: ``llm``, ``local_llm``, ``text_llm``, ``semantic``.
    Anything else (including unset) keeps the keyword fallback.
    """
    raw = value if value is not None else (env or os.environ).get(
        ENV_DOMAIN_SELECT, ""
    )
    text = str(raw or "").strip().lower()
    if text in {"llm", "local_llm", "local-llm", "text_llm", "text-llm", "semantic"}:
        return SelectionBackend.LLM
    return SelectionBackend.RULE_BASED


def detect_needed_enrichment_skills(
    command: str,
    world: str | None = None,
) -> tuple[str, ...]:
    """
    Rule-based scan: which enrichment-candidate skills does the NL goal imply?

    Only returns skills in this robot's enrichment set. Never invents
    open-vocabulary motors. Household cues such as pour never fire on workshop.
    """
    text = " ".join(str(command or "").lower().split())
    if not text:
        return ()
    padded = f" {text} "
    allowed = enrichment_candidates_for_world(world)
    found: list[str] = []
    for skill, cues in _SKILL_GOAL_CUES:
        if skill not in allowed:
            continue
        if any(cue in padded or cue in text for cue in cues):
            if skill not in found:
                found.append(skill)
    return tuple(found)


def format_refuse_message(
    *,
    template: str,
    command: str,
    needed_skills: Sequence[str] = (),
    detail: str = "",
    world: str | None = None,
) -> str:
    """User-visible refuse message (closed-catalog contract)."""
    parts = [REFUSE_MESSAGE]
    parts.append(f"Selected domain: {template}.")
    if command.strip():
        parts.append(f"Task: {command.strip()}.")
    if needed_skills:
        parts.append(
            "Implied skills not in domain PDDL: "
            + ", ".join(needed_skills)
            + "."
        )
    else:
        catalog = ", ".join(sorted(enrichment_candidates_for_world(world)))
        parts.append(
            "No catalog enrichment skill matched the goal "
            f"(catalog enrichment set: {catalog})."
        )
    if detail.strip():
        parts.append(detail.strip())
    return " ".join(parts)


def select_domain_rule_based(
    command: str,
    world: str | None = None,
) -> DomainSelectionResult:
    """
    Keyword fallback: pick a fixed template and flag ``complete`` | ``incomplete``.

    Criteria (see design §10.1):
      1. Choose template via the existing keyword heuristic
         (``select_domain_template``).
      2. Detect enrichment-candidate verbs in the NL goal for this robot.
      3. If any detected skill is missing from that template's PDDL →
         ``incomplete`` with ``needed_skills``.
      4. Otherwise ``complete`` — proceed on the fixed domain unchanged.

    Literal cues only: "get me something to drink" reads as complete here. Use the
    ``llm`` backend for semantic completeness (Session 29 paper path).
    """
    # Local import avoids a circular dependency with hybrid_runtime re-exports.
    from planner.hybrid_runtime import select_domain_template

    template = select_domain_template(command)
    implied = detect_needed_enrichment_skills(command, world=world)
    missing_from_domain = enrichment_candidates_for_domain(template, world=world)
    needed = tuple(s for s in implied if s in missing_from_domain)

    if needed:
        return DomainSelectionResult(
            template=template,
            completeness=DomainCompleteness.INCOMPLETE,
            reason=(
                f"goal implies catalog skill(s) {', '.join(needed)} "
                f"not grounded in {template} PDDL"
            ),
            needed_skills=needed,
        )

    if implied:
        # Implied skills already in domain (shouldn't happen for pour/… today).
        return DomainSelectionResult(
            template=template,
            completeness=DomainCompleteness.COMPLETE,
            reason=(
                f"implied skills {', '.join(implied)} already in {template}"
            ),
            needed_skills=(),
        )

    return DomainSelectionResult(
        template=template,
        completeness=DomainCompleteness.COMPLETE,
        reason=f"fixed domain {template} covers the goal verbs",
        needed_skills=(),
    )


class DomainSelector(Protocol):
    """Template + completeness judge (keyword fallback or the text LLM)."""

    backend: str

    def select(
        self,
        command: str,
        *,
        scene_symbols: Sequence[str] = (),
        world: str | None = None,
    ) -> DomainSelectionResult:
        ...


@dataclass
class RuleBasedDomainSelector:
    """Keyword selector behind the :class:`DomainSelector` interface."""

    backend: str = SelectionBackend.RULE_BASED.value

    def select(
        self,
        command: str,
        *,
        scene_symbols: Sequence[str] = (),
        world: str | None = None,
    ) -> DomainSelectionResult:
        return select_domain_rule_based(command, world=world)


def select_domain(
    command: str,
    *,
    selector: DomainSelector | None = None,
    scene_symbols: Sequence[str] = (),
    world: str | None = None,
) -> DomainSelectionResult:
    """
    Pick a fixed template and flag ``complete`` | ``incomplete``.

    Defaults to the keyword fallback so existing callers are unchanged; pass an
    ``llm`` selector (see :func:`make_domain_selector`) for the paper path.
    """
    if selector is None:
        return select_domain_rule_based(command, world=world)
    return selector.select(
        command, scene_symbols=tuple(scene_symbols), world=world
    )


class OnlineDomainEnricher(Protocol):
    """Closed-catalog PDDL author (stub, or the single text LLM)."""

    def enrich(self, request: EnrichmentRequest) -> EnrichmentOutcome:
        ...


@dataclass
class StubOnlineDomainEnricher:
    """
    Session 28 stub — records the handoff; does **not** invent PDDL.

    - Candidates nonempty → ``STUB`` (Session 29 will author + persist).
    - Candidates empty → ``REFUSED`` with the user-visible refuse contract.
    """

    calls: list[EnrichmentRequest] = field(default_factory=list)

    def enrich(self, request: EnrichmentRequest) -> EnrichmentOutcome:
        self.calls.append(request)
        # Closed catalog only — drop anything not in the catalog.
        allowed = tuple(
            normalize_catalog_skill(s)
            for s in request.candidate_skills
            if normalize_catalog_skill(s)
            in enrichment_candidates_for_domain(
                request.template, world=request.world
            )
        )
        if not allowed:
            return EnrichmentOutcome(
                status=EnrichmentStatus.REFUSED,
                refuse_message=format_refuse_message(
                    template=request.template,
                    command=request.command,
                    needed_skills=request.candidate_skills,
                    detail="Enricher stub: empty candidate set after catalog filter.",
                    world=request.world,
                ),
                notes="stub refuse — no catalog skill for goal gap",
            )
        return EnrichmentOutcome(
            status=EnrichmentStatus.STUB,
            skills_grounded=(),
            notes=(
                "Session 28 stub: handoff recorded for skills "
                f"{', '.join(allowed)}; PDDL authoring deferred to Session 29"
            ),
        )


def make_domain_selector(
    backend: SelectionBackend | str | None = None,
    *,
    generate_fn=None,
    client=None,
    model_id: str | None = None,
    env: dict[str, str] | None = None,
) -> DomainSelector:
    """
    Build a selector for ``backend`` (default from ``VLMRP_DOMAIN_SELECT``).

    ``generate_fn`` / ``client`` inject a mock text LLM for CI; without them the
    ``llm`` backend loads the shared single text model.
    """
    resolved = (
        backend
        if isinstance(backend, SelectionBackend)
        else resolve_selection_backend(backend, env=env)
    )
    if resolved == SelectionBackend.LLM:
        # Local import: domain_llm depends on the types defined above.
        from planner.domain_llm import LLMDomainSelector

        return LLMDomainSelector(
            generate_fn=generate_fn, client=client, model_id=model_id
        )
    return RuleBasedDomainSelector()


def make_online_enricher(
    backend: SelectionBackend | str | None = None,
    *,
    generate_fn=None,
    client=None,
    model_id: str | None = None,
    enriched_dir=None,
    env: dict[str, str] | None = None,
) -> OnlineDomainEnricher:
    """
    Build an enricher for ``backend``.

    ``llm`` authors + persists closed-catalog PDDL; ``rule_based`` keeps the
    no-GPU stub, which records the handoff and never invents PDDL.
    """
    resolved = (
        backend
        if isinstance(backend, SelectionBackend)
        else resolve_selection_backend(backend, env=env)
    )
    if resolved == SelectionBackend.LLM:
        from planner.domain_llm import LLMDomainEnricher

        return LLMDomainEnricher(
            generate_fn=generate_fn,
            client=client,
            model_id=model_id,
            enriched_dir=enriched_dir,
        )
    return StubOnlineDomainEnricher()


def make_goal_binder(
    backend: SelectionBackend | str | None = None,
    *,
    generate_fn=None,
    client=None,
    model_id: str | None = None,
    env: dict[str, str] | None = None,
):
    """
    Build the enriched-goal binder for ``backend`` (Session 30).

    ``llm`` grounds an enriched action's parameters with the same text LLM;
    ``rule_based`` returns ``None``, leaving the deterministic command-order
    binding in ``ground_enrichment_goal``.
    """
    resolved = (
        backend
        if isinstance(backend, SelectionBackend)
        else resolve_selection_backend(backend, env=env)
    )
    if resolved != SelectionBackend.LLM:
        return None
    from planner.domain_llm import LLMGoalBinder

    return LLMGoalBinder(
        generate_fn=generate_fn,
        client=client,
        model_id=model_id,
    )


def split_scene_symbols(
    scene_symbols: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition symbols into (objects, locations) using name hints."""
    objects: list[str] = []
    locations: list[str] = []
    for raw in scene_symbols:
        name = str(raw).strip()
        if not name:
            continue
        key = name.lower()
        if key in _LOCATION_SYMBOLS or any(
            hint in key
            for hint in (
                "shelf",
                "table",
                "counter",
                "drawer",
                "tray",
                "bowl",
                "box",
                "bin",
            )
        ):
            locations.append(name)
        else:
            objects.append(name)
    return tuple(objects), tuple(locations)


def preferred_enrichment_host_template(
    needed_skills: Sequence[str],
    *,
    current: str | None = None,
    world: str | None = None,
) -> str | None:
    """
    Choose a fixed template to host enrichment for ``needed_skills``.

    Among templates that list every needed skill as an enrichment candidate,
    prefer the one with the fewest stock PDDL actions (simplest host). Ties keep
    ``current`` when it is already optimal. No skill- or template-specific
    hardcoding — richer hosts tend to confuse authoring into reusing existing
    fluents, so the smallest capable template is the structural default.
    """
    needed = {
        normalize_catalog_skill(s)
        for s in needed_skills
        if normalize_catalog_skill(s)
    }
    if not needed:
        return None

    ranked: list[tuple[int, str]] = []
    for template, actions in DOMAIN_PDDL_ACTIONS.items():
        candidates = enrichment_candidates_for_domain(template, world=world)
        if needed <= candidates:
            ranked.append((len(actions), template))
    if not ranked:
        return None

    ranked.sort(key=lambda item: (item[0], item[1]))
    best_size = ranked[0][0]
    best = [template for size, template in ranked if size == best_size]
    if current in best:
        return current
    return best[0]


def remap_template_for_enrichment(
    selection: DomainSelectionResult,
    world: str | None = None,
) -> DomainSelectionResult:
    """
    Move enrichment onto the preferred host template when needed.

    Structural: if the selection names catalog enrichment skills, host them on
    the smallest fixed domain that can accept those skills as candidates.
    """
    host = preferred_enrichment_host_template(
        selection.needed_skills,
        current=selection.template,
        world=world,
    )
    if host is None or host == selection.template:
        return selection
    return replace(
        selection,
        template=host,
        reason=(
            f"{selection.reason} "
            f"(enrichment host remapped {selection.template} → {host}; "
            "prefer smallest capable template)"
        ),
    )


def template_covers_command(
    command: str,
    template: str,
    scene_symbols: Sequence[str] = (),
    *,
    goal_probe: GoalCoverageProbe | None = None,
    objects: Sequence[str] | None = None,
    locations: Sequence[str] | None = None,
) -> bool:
    """
    True when the existing goal helper can express ``command`` on ``template``.

    Positive evidence only: a hit means a fixed template already covers the
    phrasing (v4 ``template_complete`` / implicit place). A miss is *not* proof
    the template cannot express the goal — the helper is a regex floor, not the
    local_llm goal author used by plan-eval v4.
    """
    if goal_probe is not None:
        try:
            return bool(goal_probe(command, template, scene_symbols))
        except Exception:
            return False
    from planner.problem_generator.goal_generator.backends.rule_based import (
        RuleBasedGoalGenerator,
        _IN_CONTAINER,
        _resolve_entity,
    )
    from planner.problem_generator.goal_generator.predicates import (
        resolve_allowed_predicates,
    )

    if objects is None or locations is None:
        split_objects, split_locations = split_scene_symbols(scene_symbols)
        objects = objects if objects is not None else split_objects
        locations = locations if locations is not None else split_locations
    result = RuleBasedGoalGenerator().generate(
        command,
        objects,
        locations=locations,
        domain_template=template,
    )
    if result.ok and result.facts:
        return True
    # Quantified put-in ("edible items", "all you see") still uses
    # place-in-container when the destination is a known location.
    allowed = resolve_allowed_predicates(template)
    match = _IN_CONTAINER.match(str(command or "").strip())
    if match and "in-container" in allowed:
        container = _resolve_entity(match.group("container"), list(locations))
        if container:
            return True
    return False


def _simplest_covering_template(
    command: str,
    scene_symbols: Sequence[str] = (),
    *,
    goal_probe: GoalCoverageProbe | None = None,
) -> str | None:
    """Smallest fixed template whose stock predicates can express ``command``."""
    ranked: list[tuple[int, str]] = []
    for template, actions in DOMAIN_PDDL_ACTIONS.items():
        if template_covers_command(
            command,
            template,
            scene_symbols,
            goal_probe=goal_probe,
        ):
            ranked.append((len(actions), template))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][1]


def _skills_for_uncovered_gap(
    command: str,
    template: str,
    world: str | None = None,
) -> tuple[str, ...]:
    """Catalog enrichment skills implied by cues and missing from ``template``."""
    implied = detect_needed_enrichment_skills(command, world=world)
    allowed = enrichment_candidates_for_domain(template, world=world)
    return tuple(s for s in implied if s in allowed)


def _promote_native_template_skills(
    selection: DomainSelectionResult,
) -> DomainSelectionResult:
    """
    ``place_in_container`` / ``pick`` / ``place`` are fixed-domain actions.

    Listing them as ``needed_skills`` is a selector mix-up, not a catalog gap.
    Plan-eval v4 implicit place must stay ``complete`` and reach the goal LLM.
    """
    native = skills_already_in_template(
        selection.template, selection.needed_skills
    )
    if not native:
        return selection
    leftover = tuple(
        skill
        for skill in selection.needed_skills
        if normalize_catalog_skill(skill) not in native
    )
    if leftover or selection.completeness == DomainCompleteness.COMPLETE:
        return replace(selection, needed_skills=leftover)
    return replace(
        selection,
        completeness=DomainCompleteness.COMPLETE,
        needed_skills=(),
        reason=(
            f"{selection.reason} (template already defines "
            f"{', '.join(native)}; treated as complete)"
        ),
    )


def guard_domain_selection(
    selection: DomainSelectionResult,
    command: str,
    scene_symbols: Sequence[str] = (),
    *,
    goal_probe: GoalCoverageProbe | None = None,
    selector: DomainSelector | None = None,
    world: str | None = None,
) -> DomainSelectionResult:
    """
    Structural post-select checks, independent of which template/skill failed.

    1. If any fixed template already expresses the goal, use the simplest
       one as ``complete`` (ignore spurious catalog skills such as pour/cut
       on a put-in-container command).
    2. Native template actions listed as ``needed_skills`` (e.g.
       ``place_in_container``) are treated as ``complete``, not as a catalog
       gap — plan-eval v4 ``implicit_place`` must stay on the complete path.
    3. False-complete + catalog cue (pour/cut/…) → ``incomplete`` and enrich.
       Regex miss *without* a cue is not a refuse: revise, or keep complete
       so the goal LLM (v4) authors or refuses the ``:goal``.
    4. Remap enrichment onto the preferred host template.
    5. Incomplete + empty skills that the regex helper already covers →
       upgrade back to ``complete`` (empty-catalog refuse veto).
    """
    selection = _promote_native_template_skills(selection)

    # Coverage probes need scene symbols (or an injected probe). Without them,
    # rule-based goal helpers cannot ground objects and would false-veto every
    # complete pick/place — leave the selector's verdict alone.
    can_probe = goal_probe is not None or bool(scene_symbols)
    if can_probe:
        host = _simplest_covering_template(
            command, scene_symbols, goal_probe=goal_probe
        )
        if host is not None:
            reason = selection.reason
            if (
                selection.template != host
                or selection.completeness != DomainCompleteness.COMPLETE
                or selection.needed_skills
            ):
                reason = (
                    f"{reason} (fixed template {host} already expresses "
                    "the goal; skipped enrichment)"
                )
            return replace(
                selection,
                template=host,
                completeness=DomainCompleteness.COMPLETE,
                needed_skills=(),
                reason=reason,
            )

    covers = (
        template_covers_command(
            command,
            selection.template,
            scene_symbols,
            goal_probe=goal_probe,
        )
        if can_probe
        else None
    )

    if (
        can_probe
        and selection.completeness == DomainCompleteness.COMPLETE
        and covers is False
    ):
        needed = _skills_for_uncovered_gap(
            command, selection.template, world=world
        )
        if needed:
            # Cue table saw pour/cut/… — real catalog gap, not a paraphrase.
            selection = replace(
                selection,
                completeness=DomainCompleteness.INCOMPLETE,
                needed_skills=needed,
                reason=(
                    f"{selection.reason} "
                    "(coverage veto: selected template cannot express the goal)"
                ),
            )
        elif selector is not None:
            # Regex miss without a catalog cue: implicit place vs ungenerable.
            # Ask the selector once; do not regex-refuse (v4 goal LLM decides).
            revise = getattr(selector, "revise_after_uncovered", None)
            if callable(revise):
                try:
                    revised = revise(
                        command,
                        selection,
                        scene_symbols=scene_symbols,
                        world=world,
                    )
                except Exception:
                    revised = None
                if isinstance(revised, DomainSelectionResult):
                    selection = _promote_native_template_skills(revised)

    selection = remap_template_for_enrichment(selection, world=world)

    if (
        can_probe
        and selection.completeness == DomainCompleteness.INCOMPLETE
        and not selection.needed_skills
        and template_covers_command(
            command,
            selection.template,
            scene_symbols,
            goal_probe=goal_probe,
        )
    ):
        selection = replace(
            selection,
            completeness=DomainCompleteness.COMPLETE,
            reason=(
                f"{selection.reason} "
                "(template already expresses the goal; skipped empty-catalog refuse)"
            ),
        )

    return selection


def resolve_domain_for_task(
    command: str,
    *,
    enrichment_enabled: bool | str | None = None,
    enricher: OnlineDomainEnricher | None = None,
    selector: DomainSelector | None = None,
    scene_symbols: Sequence[str] = (),
    env: dict[str, str] | None = None,
    goal_probe: GoalCoverageProbe | None = None,
    world: str | None = None,
) -> DomainResolution:
    """
    Select domain; when enrichment is ON and selection is incomplete, hand off
    to the Domain Enricher (stub by default).

    When enrichment is OFF (default): never calls the enricher — fixed-domain
    path unchanged even if selection would be incomplete.

    Post-select guards enforce goal coverage and preferred enrichment hosts
    (see :func:`guard_domain_selection`).
    """
    enabled = resolve_online_enrichment(enrichment_enabled, env=env)
    active_selector = selector
    selection = select_domain(
        command,
        selector=active_selector,
        scene_symbols=scene_symbols,
        world=world,
    )
    selection = guard_domain_selection(
        selection,
        command,
        scene_symbols,
        goal_probe=goal_probe,
        selector=active_selector,
        world=world,
    )

    if not enabled:
        return DomainResolution(
            selection=selection,
            enrichment=EnrichmentOutcome(
                status=EnrichmentStatus.SKIPPED,
                notes="online enrichment flag OFF — fixed domain only",
            ),
            enrichment_enabled=False,
        )

    if selection.completeness == DomainCompleteness.COMPLETE:
        return DomainResolution(
            selection=selection,
            enrichment=EnrichmentOutcome(
                status=EnrichmentStatus.SKIPPED,
                notes="domain complete — enricher not called",
            ),
            enrichment_enabled=True,
        )

    # incomplete + enabled → enricher handoff
    backend = enricher if enricher is not None else StubOnlineDomainEnricher()
    # Goal-implied ∩ domain-missing, never expanded beyond the catalog. An
    # incomplete selection that names no catalog skill means the gap has no
    # executable answer, so the enricher must refuse rather than pick one.
    allowed_for_template = enrichment_candidates_for_domain(
        selection.template, world=world
    )
    request = EnrichmentRequest(
        template=selection.template,
        command=str(command or ""),
        scene_symbols=tuple(str(s) for s in scene_symbols),
        candidate_skills=tuple(
            normalize_catalog_skill(s)
            for s in selection.needed_skills
            if normalize_catalog_skill(s) in allowed_for_template
        ),
        reason=selection.reason,
        selection=selection,
        world=world,
    )
    outcome = backend.enrich(request)
    return DomainResolution(
        selection=selection,
        enrichment=outcome,
        enrichment_enabled=True,
    )
