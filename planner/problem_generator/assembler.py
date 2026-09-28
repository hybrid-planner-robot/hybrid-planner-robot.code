"""
Assemble a full PDDL problem string from init/goal sections and symbol sets.

Shared by the legacy and hybrid ``generate_problem`` paths.
See ``docs/hybrid_problem_generator_design.md`` §5.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from vlm.planner import VLMPlan

from .goal_generator.renderer import GoalRenderer
from .init_generator.renderer import InitRenderer, PddlFact
from .init_generator.schema import LocationFact, SceneState
from .legacy import DOMAIN_TEMPLATE_TO_NAME, extract_objects_and_locations


def render_objects_section(
    item_names: set[str],
    location_names: set[str],
    container_names: set[str] | None = None,
    *,
    indent: str = "  ",
) -> str:
    """Return the ``(:objects ...)`` block.

    ``container`` is a PDDL subtype of ``location``. Symbols with
    ``LocationFact.type == "container"`` must be declared as ``- container``
    (not ``- location``), or ``place-in-container`` / ``in-container`` cannot
    ground.
    """
    containers = set(container_names or ())
    surfaces = set(location_names) - containers
    items = set(item_names) - containers - surfaces
    inner = indent * 2
    lines = [f"{indent}(:objects"]
    if items:
        lines.append(f"{inner}{' '.join(sorted(items))} - item")
    if surfaces:
        lines.append(f"{inner}{' '.join(sorted(surfaces))} - location")
    if containers:
        lines.append(f"{inner}{' '.join(sorted(containers))} - container")
    lines.append(f"{indent})")
    return "\n".join(lines)


def assemble_problem(
    *,
    domain_name: str,
    problem_name: str,
    item_names: set[str],
    location_names: set[str],
    init_section: str,
    goal_section: str,
    container_names: set[str] | None = None,
) -> str:
    """Concatenate define, objects, init, and goal into a PDDL problem string."""
    lines: list[str] = [
        f"(define (problem {problem_name})",
        f"  (:domain {domain_name})",
        "",
        render_objects_section(
            item_names, location_names, container_names
        ),
        "",
        init_section,
        "",
        goal_section,
        ")",
    ]
    return "\n".join(lines)


def collect_symbols_from_scene(
    scene: SceneState,
    plan: VLMPlan | None = None,
) -> tuple[set[str], set[str], set[str]]:
    """
    Collect item, location, and container names from SceneState, unioned with
    plan symbols.

    Relation args may reference items not listed in ``scene.objects`` (e.g.
    ``in-container``); those are included as items. Destinations of
    ``in-container`` (and ``LocationFact`` / ``ObjectFact`` with
    ``type="container"``) are declared as the PDDL ``container`` subtype.
    """
    items: set[str] = {
        obj.name for obj in scene.objects if obj.type != "container"
    }
    containers: set[str] = {
        obj.name for obj in scene.objects if obj.type == "container"
    }
    containers.update(
        loc.name for loc in scene.locations if loc.type == "container"
    )
    locations: set[str] = {
        loc.name for loc in scene.locations if loc.type != "container"
    }

    location_names = locations | containers
    for rel in scene.relations:
        if rel.predicate in {"on", "in-container"} and len(rel.args) >= 2:
            items.add(rel.args[0])
            dest = rel.args[1]
            if rel.predicate == "in-container" or dest in containers:
                containers.add(dest)
                locations.discard(dest)
            elif dest not in containers:
                locations.add(dest)
        elif rel.predicate == "stacked-on" and len(rel.args) >= 2:
            items.update(rel.args[:2])
        elif rel.predicate in {"clear", "reachable", "camera-aimed-at", "holding"}:
            if rel.args:
                name = rel.args[0]
                if name in location_names:
                    pass
                else:
                    items.add(name)
        else:
            # Enrichment fluents such as ``(poured bottle glass)``: their
            # arguments must still be declared, or the problem will not parse.
            items.update(a for a in rel.args if a not in location_names)

    if scene.robot.holding:
        items.add(scene.robot.holding)
    if scene.robot.camera_aimed_at:
        items.add(scene.robot.camera_aimed_at)

    if plan is not None:
        plan_objects, plan_locations = extract_objects_and_locations(plan.steps)
        items |= plan_objects
        for loc in plan_locations:
            if loc not in containers:
                locations.add(loc)
        for step in plan.steps:
            name = (step.args or {}).get("container")
            if name:
                containers.add(str(name))
                locations.discard(str(name))
                items.discard(str(name))

    items -= containers
    items -= locations
    locations -= containers
    return items, locations, containers


def align_scene_to_goal_facts(
    scene: SceneState, facts: Sequence[PddlFact]
) -> SceneState:
    """If the goal uses an item as an ``on`` / ``in-container`` dest, retype it.

    Inventory sometimes lists a place target (notebook, plate) as an object.
    ``(on ?i - item ?l - location)`` cannot ground if that dest stays an item.
    """
    on_dests: set[str] = set()
    in_dests: set[str] = set()
    for fact in facts:
        if not fact:
            continue
        pred = fact[0]
        if pred == "on" and len(fact) >= 3:
            on_dests.add(str(fact[2]))
        elif pred == "in-container" and len(fact) >= 3:
            in_dests.add(str(fact[2]))
    promote = (on_dests | in_dests) & {obj.name for obj in scene.objects}
    if not promote:
        return scene
    remaining = [obj for obj in scene.objects if obj.name not in promote]
    if not remaining:
        return scene
    by_loc = {loc.name: loc for loc in scene.locations}
    for name in sorted(promote):
        as_container = name in in_dests
        existing = by_loc.get(name)
        if existing is None:
            by_loc[name] = LocationFact(
                name=name,
                type="container" if as_container else "location",
                reachable=True,
                source="fusion",
                confidence=1.0,
            )
        elif as_container and existing.type != "container":
            by_loc[name] = replace(existing, type="container")
    new_relations = [
        rel
        for rel in scene.relations
        if not (
            rel.predicate in {"on", "stacked-on", "in-container", "clear"}
            and rel.args
            and rel.args[0] in promote
        )
    ]
    return replace(
        scene,
        objects=remaining,
        locations=sorted(by_loc.values(), key=lambda loc: loc.name),
        relations=new_relations,
    )


def _resolve_domain_name(
    plan: VLMPlan,
    scene: SceneState,
    domain_name: str | None,
) -> str:
    if domain_name is not None:
        return domain_name
    template = scene.domain_template or plan.domain_template
    return DOMAIN_TEMPLATE_TO_NAME.get(template, "manipulation-base")


def _resolve_goal_facts(
    plan: VLMPlan,
    scene: SceneState,
    goal_facts: Sequence[PddlFact] | None,
) -> list[PddlFact]:
    if goal_facts is not None:
        return list(goal_facts)

    from .goal_generator.backends.rule_based import RuleBasedGoalGenerator

    items, locations, containers = collect_symbols_from_scene(scene, plan=None)
    domain_template = scene.domain_template or plan.domain_template
    result = RuleBasedGoalGenerator().generate(
        plan.goal,
        sorted(items),
        locations=sorted(locations | containers),
        domain_template=domain_template,
    )
    if not result.ok:
        raise ValueError(
            f"goal generation failed for command {plan.goal!r}: {result.error}"
        )
    return list(result.facts)


def generate_problem_hybrid(
    plan: VLMPlan,
    scene_state: SceneState,
    goal_facts: Sequence[PddlFact] | None = None,
    *,
    domain_name: str | None = None,
    problem_name: str = "generated_problem",
) -> str:
    """
    Build a PDDL problem from SceneState (:init) and goal facts (:goal).

    ``goal_facts`` may be omitted; then ``RuleBasedGoalGenerator`` is called
    with ``plan.goal`` and symbols from ``scene_state``.
    """
    resolved_goal = _resolve_goal_facts(plan, scene_state, goal_facts)
    scene_state = align_scene_to_goal_facts(scene_state, resolved_goal)
    items, locations, containers = collect_symbols_from_scene(scene_state, plan)
    template = (
        scene_state.domain_template
        or plan.domain_template
        or "manipulation_base"
    )
    if template != "containers_manipulation":
        # ``container`` / ``open`` exist only on the containers domain.
        locations = set(locations) | set(containers)
        containers = set()
    if scene_state.domain_template != template:
        from dataclasses import replace as _replace_scene

        scene_state = _replace_scene(scene_state, domain_template=template)
    # Session 30: the plan's own domain_additions say which enrichment
    # predicates the domain declares, so a tracked ``(poured …)`` reaches
    # ``:init`` instead of being dropped as unknown.
    from planner.enrichment_effects import predicates_from_domain_additions

    extra = predicates_from_domain_additions(plan.domain_additions)
    init_section = InitRenderer(extra_predicates=extra).render_section(scene_state)
    goal_section = GoalRenderer().render_section(resolved_goal)

    return assemble_problem(
        domain_name=_resolve_domain_name(plan, scene_state, domain_name),
        problem_name=problem_name,
        item_names=items,
        location_names=locations,
        container_names=containers,
        init_section=init_section,
        goal_section=goal_section,
    )
