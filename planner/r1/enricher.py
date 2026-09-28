"""R1 online enricher: call 1 (actions + affordances) then call 2 (assignment)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from planner.domain_enricher import DomainAdditions, DomainEnricher
from planner.domain_llm import domain_action_names, domain_predicate_names, pddl_syntax_errors
from planner.domain_store import load_base_domain
from planner.online_enrichment import (
    EnrichmentOutcome,
    EnrichmentRequest,
    EnrichmentStatus,
    format_refuse_message,
)
from planner.r1.parse import (
    ParsedR1Enrichment,
    R1AssignmentError,
    R1EnrichmentPayloadError,
    affordance_predicate_names,
    parse_r1_assignment,
    parse_r1_enrichment,
    predicate_head,
)
from planner.r1.prompts import R1_ASSIGN_SYSTEM_PROMPT, R1_ENRICH_SYSTEM_PROMPT
from planner.skill_catalog import (
    enrichment_candidates_for_domain,
    normalize_catalog_skill,
    ros_primitive_for_action,
    skill_signature,
)
from planner.call_timings import invoke_llm
from planner.text_llm import GenerateFn, LocalTextClient, get_shared_text_client


def build_r1_enrichment_prompt(
    request: EnrichmentRequest,
    *,
    base_domain_text: str,
    allowed_skills: Sequence[str],
    previous_error: str | None = None,
) -> str:
    lines: list[str] = []
    for skill in allowed_skills:
        sig = skill_signature(skill)
        if sig is None:
            continue
        lines.append(
            f"- {sig.pddl_action}: {sig.description}; "
            f"arguments {', '.join(sig.roles)} "
            f"({sig.min_params}–{sig.max_params} PDDL parameters)"
        )
    symbols = ", ".join(request.scene_symbols) or "(none reported)"
    prompt = (
        f"domain_template: {request.template}\n"
        f"allowed_skills:\n{chr(10).join(lines)}\n\n"
        f"scene_symbols: {symbols}\n"
        f"task: {request.command.strip()}\n"
        f"why_incomplete: {request.reason}\n\n"
        f"current_domain_pddl:\n{base_domain_text}\n"
    )
    if previous_error:
        prompt += (
            "\nYour previous answer was rejected: "
            f"{previous_error}\nReturn corrected JSON.\n"
        )
    return prompt


def build_r1_assignment_prompt(
    *,
    affordance_predicates: Sequence[str],
    scene_objects: Sequence[str],
    task: str,
    previous_error: str | None = None,
) -> str:
    preds = ", ".join(affordance_predicates) or "(none)"
    objs = ", ".join(scene_objects) or "(none)"
    prompt = (
        f"affordance_predicates: {preds}\n"
        f"scene_objects: {objs}\n"
        f"task: {task.strip()}\n"
    )
    if previous_error:
        prompt += (
            "\nYour previous answer was rejected: "
            f"{previous_error}\nReturn corrected JSON.\n"
        )
    return prompt


@dataclass
class R1DomainEnricher:
    """
    Closed-catalog PDDL author with unary affordances and a second LLM call
    that assigns those affordances to scene objects.

    Returns ``ENRICHED`` (merged domain + ``init_facts`` in domain_additions)
    or ``REFUSED``. Never invents a skill outside the catalog.
    """

    generate_fn: GenerateFn | None = None
    client: LocalTextClient | None = None
    model_id: str | None = None
    max_attempts: int = 2
    max_new_tokens: int = 768
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generate_fn is not None and self.client is not None:
            raise ValueError("pass only one of generate_fn or client")

    def _complete(self, system: str, user: str, *, stage: str = "r1") -> str:
        self.calls.append(user)

        def _run() -> str:
            if self.generate_fn is not None:
                return self.generate_fn(system, user)
            if self.client is not None:
                return self.client.complete(system, user)
            return get_shared_text_client(self.model_id).complete(
                system, user, max_new_tokens=self.max_new_tokens
            )

        return invoke_llm(stage, _run)

    def _refuse(
        self,
        request: EnrichmentRequest,
        *,
        detail: str,
        notes: str,
    ) -> EnrichmentOutcome:
        return EnrichmentOutcome(
            status=EnrichmentStatus.REFUSED,
            refuse_message=format_refuse_message(
                template=request.template,
                command=request.command,
                needed_skills=request.candidate_skills,
                detail=detail,
                world=request.world,
            ),
            notes=notes,
        )

    def enrich(self, request: EnrichmentRequest) -> EnrichmentOutcome:
        allowed = tuple(
            normalize_catalog_skill(s)
            for s in request.candidate_skills
            if normalize_catalog_skill(s)
            in enrichment_candidates_for_domain(
                request.template, world=request.world
            )
        )
        if not allowed:
            return self._refuse(
                request,
                detail=(
                    "No catalog skill is available for this gap on template "
                    f"{request.template}."
                ),
                notes="refuse — empty allowed set after catalog filter",
            )

        try:
            base_domain = (
                request.base_domain_text
                if request.base_domain_text is not None
                else load_base_domain(request.template)
            )
        except FileNotFoundError as exc:
            return self._refuse(
                request,
                detail=f"Base domain unavailable: {exc}",
                notes="refuse — base domain not readable",
            )

        parsed, error = self._author(request, base_domain, allowed)
        if parsed is None:
            return self._refuse(
                request,
                detail=f"R1 enricher could not author valid PDDL: {error}",
                notes=f"refuse — {error}",
            )
        if parsed.refused:
            return self._refuse(
                request,
                detail=f"Enricher declined: {parsed.reason}",
                notes="refuse — model declined within closed catalog",
            )

        merged, merge_error = self._merge(base_domain, parsed)
        if merged is None:
            return self._refuse(
                request,
                detail=f"Enriched domain failed validation: {merge_error}",
                notes=f"refuse — {merge_error}",
            )

        primitives: list[str] = []
        for act in parsed.actions:
            primitive = ros_primitive_for_action(act.action_name)
            if primitive is None:
                return self._refuse(
                    request,
                    detail=(
                        f"Action '{act.action_name}' has no ROS primitive to "
                        "dispatch."
                    ),
                    notes="refuse — action not executable",
                )
            primitives.append(primitive)

        affordances = sorted(affordance_predicate_names(parsed.new_predicates))
        item_symbols = tuple(
            s for s in request.scene_symbols
            if str(s).strip()
            and str(s).lower()
            not in {
                "table",
                "shelf",
                "shelf_b",
                "counter",
                "floor",
                "drawer",
                "workbench",
                "desk",
                "metal_tray",
            }
        )
        init_facts, assign_error = self._assign(
            request.command,
            affordances,
            item_symbols or request.scene_symbols,
        )
        if assign_error:
            return self._refuse(
                request,
                detail=f"R1 assignment failed: {assign_error}",
                notes=f"refuse — {assign_error}",
            )

        additions = {
            "new_types": list(parsed.new_types),
            "new_predicates": list(parsed.new_predicates),
            "new_actions": [dict(a.action) for a in parsed.actions],
            "modified_preconditions": {},
            "init_facts": [list(f) for f in init_facts],
        }
        notes = (
            "R1 grounded catalog skills "
            f"{', '.join(parsed.skills)} as PDDL "
            f"{', '.join(a.action_name for a in parsed.actions)} "
            f"with affordances {', '.join(affordances) or '(none)'}"
        )
        return EnrichmentOutcome(
            status=EnrichmentStatus.ENRICHED,
            domain_text=merged,
            domain_path=None,
            skills_grounded=tuple(parsed.skills),
            ros_primitives=tuple(primitives),
            domain_additions=additions,
            notes=notes,
        )

    def _author(
        self,
        request: EnrichmentRequest,
        base_domain: str,
        allowed: Sequence[str],
    ) -> tuple[ParsedR1Enrichment | None, str]:
        error = "no attempt made"
        previous: str | None = None
        for _ in range(max(1, int(self.max_attempts))):
            user = build_r1_enrichment_prompt(
                request,
                base_domain_text=base_domain,
                allowed_skills=allowed,
                previous_error=previous,
            )
            try:
                raw = self._complete(R1_ENRICH_SYSTEM_PROMPT, user, stage="r1_enrich")
                return parse_r1_enrichment(raw, allowed_skills=allowed), ""
            except R1EnrichmentPayloadError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = f"text LLM R1 enrichment call failed: {exc}"
            previous = error
        return None, error

    def _assign(
        self,
        command: str,
        affordances: Sequence[str],
        scene_objects: Sequence[str],
    ) -> tuple[list[tuple[str, str]], str]:
        error = "no attempt made"
        previous: str | None = None
        for _ in range(max(1, int(self.max_attempts))):
            user = build_r1_assignment_prompt(
                affordance_predicates=affordances,
                scene_objects=scene_objects,
                task=command,
                previous_error=previous,
            )
            try:
                raw = self._complete(R1_ASSIGN_SYSTEM_PROMPT, user, stage="r1_assign")
                facts = parse_r1_assignment(
                    raw,
                    allowed_predicates=affordances,
                    scene_symbols=scene_objects,
                )
                return facts, ""
            except R1AssignmentError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001
                error = f"text LLM R1 assignment call failed: {exc}"
            previous = error
        return [], error

    def _merge(
        self,
        base_domain: str,
        parsed: ParsedR1Enrichment,
    ) -> tuple[str | None, str]:
        already_actions = domain_action_names(base_domain)
        for act in parsed.actions:
            if act.action_name in already_actions:
                return None, f"action '{act.action_name}' already in domain"

        already_preds = domain_predicate_names(base_domain)
        clashes = [
            predicate_head(p)
            for p in parsed.new_predicates
            if predicate_head(p) in already_preds
        ]
        if clashes:
            return None, (
                f"predicate(s) already in the base domain: {', '.join(clashes)} "
                "— the action must introduce a genuinely new fluent"
            )

        result = DomainEnricher().enrich(
            base_domain,
            DomainAdditions(
                new_types=list(parsed.new_types),
                new_predicates=list(parsed.new_predicates),
                new_actions=[dict(a.action) for a in parsed.actions],
            ),
        )
        if not result.is_valid:
            return None, "; ".join(result.errors) or "structural validation failed"
        for act in parsed.actions:
            if act.action_name not in domain_action_names(result.domain_text):
                return None, (
                    f"action '{act.action_name}' was skipped during merge: "
                    + "; ".join(result.additions_skipped)
                )
        errors = pddl_syntax_errors(result.domain_text)
        if errors:
            return None, "; ".join(errors)
        return result.domain_text, ""
