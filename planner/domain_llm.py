"""
Single text LLM for domain selection, completeness, and closed-catalog
enrichment (Session 29).

Same weights as the ``:goal`` backend (:mod:`planner.text_llm`) — only the
prompt changes. Two jobs:

``LLMDomainSelector``
    NL goal + scene symbols + summaries of the four fixed templates and the
    robot skill catalog → ``DomainSelectionResult``. Semantic, so paraphrases
    that never say *pour* ("get me something to drink") are still judged
    ``incomplete`` with catalog skill ``pour``.

``LLMDomainEnricher``
    Grounds an allowed catalog skill into PDDL, merges it into the fixed
    template, syntax-checks the result (including predicate arities), maps the
    new action back to a real ROS primitive, and persists the domain for reuse.
    Reuse re-validates the cached file and discards it if illegal, then
    re-authors. If nothing in the catalog can close the gap it refuses — the
    robot never plans with a fantasy action.

Both accept an injected ``generate_fn`` / ``client`` so CI exercises the whole
path with no GPU. The keyword selector and the stub enricher remain available
through :mod:`planner.online_enrichment` as the no-GPU fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from planner.domain_enricher import DomainAdditions, DomainEnricher
from planner.domain_store import (
    discard_enriched_domain,
    enrichment_signature,
    load_base_domain,
    load_enriched_domain,
    save_enriched_domain,
)
from planner.online_enrichment import (
    DomainCompleteness,
    DomainSelectionResult,
    EnrichmentOutcome,
    EnrichmentRequest,
    EnrichmentStatus,
    SelectionBackend,
    format_refuse_message,
    select_domain_rule_based,
)
from planner.problem_generator.enrichment_goal import goal_from_enrichment_action
from planner.problem_generator.goal_generator.json_facts import extract_json_object
from planner.skill_catalog import (
    DOMAIN_PDDL_ACTIONS,
    catalog_skills_for_world,
    catalog_summary_lines,
    enrichment_candidates_for_domain,
    normalize_catalog_skill,
    ros_primitive_for_action,
    skill_signature,
    skills_already_in_template,
)
from planner.call_timings import invoke_llm
from planner.text_llm import GenerateFn, LocalTextClient, get_shared_text_client
from prompts import load_prompt

__all__ = [
    "DomainSelectionError",
    "EnrichmentPayloadError",
    "LLMDomainEnricher",
    "LLMDomainSelector",
    "ParsedEnrichment",
    "SELECT_SYSTEM_PROMPT",
    "ENRICH_SYSTEM_PROMPT",
    "parse_enrichment_payload",
    "parse_selection_payload",
    "pddl_syntax_errors",
]

TEMPLATE_NAMES: tuple[str, ...] = (
    "manipulation_base",
    "manipulation_stacking",
    "containers_manipulation",
    "navigation_manipulation",
)

TEMPLATE_SUMMARIES: Mapping[str, str] = {
    "manipulation_base": "flat surfaces only: pick / place / look-at",
    "manipulation_stacking": "adds stacking of items on items: stack / unstack",
    "containers_manipulation": (
        "adds openable containers: open/close-container, "
        "pick-from-container, place-in-container"
    ),
    "navigation_manipulation": "adds mobile base motion: navigate-to",
}


class DomainSelectionError(RuntimeError):
    """LLM selection produced no usable answer and fallback was disabled."""


class EnrichmentPayloadError(ValueError):
    """LLM enrichment payload violated the closed-catalog contract."""


# ── Prompts (files under prompts/domain/) ────────────────────────────────────

SELECT_SYSTEM_PROMPT = load_prompt("domain", "select.md")
ENRICH_SYSTEM_PROMPT = load_prompt("domain", "enrich.md")
BIND_SYSTEM_PROMPT = load_prompt("domain", "bind.md")


# ── PDDL syntax checking ─────────────────────────────────────────────────────

_COMMENT = re.compile(r";[^\n]*")
_ACTION_HEAD = re.compile(r"\(:action\s+([a-zA-Z][\w-]*)")
_TYPED_ATOM = re.compile(r"\?[\w-]+\s*-\s*[a-zA-Z]")
_PLACEHOLDER_PREDICATES = frozenset(
    {"pred", "predicate", "new-pred", "new-predicate", "fluent", "fact", "p"}
)


def strip_pddl_comments(text: str) -> str:
    """Drop ``;`` comments so paren counting is not fooled by prose."""
    return _COMMENT.sub("", text)


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


_LOGICAL_OPS = frozenset(
    {"and", "or", "not", "imply", "forall", "exists", "when"}
)
_BUILTIN_PREDICATES = frozenset({"="})


def pddl_syntax_errors(domain_text: str) -> list[str]:
    """
    Structural check of a domain: balance, required blocks, action shape,
    and predicate-arity consistency (declaration vs uses in preconditions /
    effects).

    Complements ``DomainEnricher._validate`` (which counts parens including
    comment text) and runs before anything is persisted or handed to FD.
    Catches the live failure mode where an LLM declares
    ``(containing ?t ?i)`` but writes ``(containing ?target)`` in an effect.
    """
    errors: list[str] = []
    body = strip_pddl_comments(domain_text)

    if not body.strip():
        return ["empty domain text"]
    if not _balanced(body):
        errors.append("unbalanced parentheses in domain")
    if "(define" not in body:
        errors.append("missing (define ...) header")
    for block in ("(:requirements", "(:predicates"):
        if block not in body:
            errors.append(f"missing {block} block")

    names = _ACTION_HEAD.findall(body)
    if not names:
        errors.append("domain has no actions")
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        errors.append(f"duplicate action(s): {', '.join(duplicates)}")

    for match in _ACTION_HEAD.finditer(body):
        name = match.group(1)
        block = _action_block(body, match.start())
        if block is None:
            errors.append(f"action '{name}' is not closed")
            continue
        for keyword in (":parameters", ":precondition", ":effect"):
            if keyword not in block:
                errors.append(f"action '{name}' missing {keyword}")

    # Arity / unknown-predicate gate — the check FD's translator enforces.
    declared = _predicate_arities(body)
    if declared is not None:
        for pred, arity, where in _predicate_uses(body):
            if pred in _BUILTIN_PREDICATES:
                continue
            if pred not in declared:
                errors.append(
                    f"undeclared predicate '{pred}' used in {where} "
                    f"(arity {arity})"
                )
                continue
            expected = declared[pred]
            if arity != expected:
                errors.append(
                    f"predicate '{pred}' declared arity {expected} but used "
                    f"with {arity} argument(s) in {where}"
                )
    return errors


def _top_level_sexprs(text: str) -> list[str]:
    """Top-level ``(...)`` groups in ``text`` (no nesting unwrap)."""
    out: list[str] = []
    depth = 0
    start = -1
    for i, char in enumerate(text):
        if char == "(":
            if depth == 0:
                start = i
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                out.append(text[start : i + 1])
                start = -1
            if depth < 0:
                break
    return out


def _predicate_arities(domain_body: str) -> dict[str, int] | None:
    """
    Map predicate name → arity from the ``(:predicates …)`` block.

    Arity = number of ``?vars`` in the declaration. Returns ``None`` when the
    block is missing (caller already reported that separately).
    """
    start = domain_body.find("(:predicates")
    if start < 0:
        return None
    block = _action_block(domain_body, start)
    if block is None:
        return {}
    # Drop the ``:predicates`` head; remaining top-level sexprs are declarations.
    inner = block.strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    if inner.lower().startswith(":predicates"):
        inner = inner[len(":predicates") :].strip()

    declared: dict[str, int] = {}
    for sexpr in _top_level_sexprs(inner):
        head = _predicate_head(sexpr).lower()
        if not head or head == "predicates":
            continue
        arity = len(re.findall(r"\?[a-zA-Z][\w-]*", sexpr))
        if head in declared and declared[head] != arity:
            # Keep the first; pddl_syntax_errors will still catch use mismatches.
            continue
        declared[head] = arity
    return declared


def _count_top_level_args(args_text: str) -> int:
    """Count top-level arguments of an atomic formula (vars, constants, sexprs)."""
    count = 0
    i = 0
    n = len(args_text)
    while i < n:
        char = args_text[i]
        if char.isspace():
            i += 1
            continue
        if char == "(":
            depth = 0
            while i < n:
                if args_text[i] == "(":
                    depth += 1
                elif args_text[i] == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            count += 1
            continue
        # Atom token (variable or constant); stop at whitespace / paren.
        while i < n and not args_text[i].isspace() and args_text[i] not in "()":
            i += 1
        count += 1
    return count


def _atoms_in_formula(sexpr: str) -> list[tuple[str, int]]:
    """Atomic ``(pred args…)`` under a precondition/effect formula."""
    text = sexpr.strip()
    if not text.startswith("("):
        return []
    inner = text[1:-1].strip() if text.endswith(")") else text[1:].strip()
    match = re.match(r"([a-zA-Z][\w-]*)", inner)
    if not match:
        return []
    head = match.group(1).lower()
    rest = inner[match.end() :].strip()
    if head in _LOGICAL_OPS:
        children = _top_level_sexprs(rest)
        # forall/exists: first child is the typed variable list, not a formula.
        if head in {"forall", "exists"} and children:
            children = children[1:]
        out: list[tuple[str, int]] = []
        for child in children:
            out.extend(_atoms_in_formula(child))
        return out
    return [(head, _count_top_level_args(rest))]


def _predicate_uses(domain_body: str) -> list[tuple[str, int, str]]:
    """``(predicate, arity, where)`` for every atom in action precond/effect."""
    uses: list[tuple[str, int, str]] = []
    for match in _ACTION_HEAD.finditer(domain_body):
        name = match.group(1)
        block = _action_block(domain_body, match.start())
        if block is None:
            continue
        for keyword in (":precondition", ":effect"):
            key = f"{keyword} "
            idx = block.find(keyword)
            if idx < 0:
                continue
            after = block[idx + len(keyword) :].lstrip()
            if not after.startswith("("):
                continue
            formula = _action_block(after, 0)
            if formula is None:
                continue
            for pred, arity in _atoms_in_formula(formula):
                uses.append((pred, arity, f"action '{name}' {keyword}"))
    return uses


def _action_block(text: str, start: int) -> str | None:
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def domain_action_names(domain_text: str) -> set[str]:
    """Action names defined in a domain (comments ignored)."""
    return set(_ACTION_HEAD.findall(strip_pddl_comments(domain_text)))


def domain_predicate_names(domain_text: str) -> set[str]:
    """Predicate names declared in the ``(:predicates …)`` block."""
    body = strip_pddl_comments(domain_text)
    start = body.find("(:predicates")
    if start == -1:
        return set()
    block = _action_block(body, start) or ""
    names = set(re.findall(r"\(([a-zA-Z][\w-]*)", block))
    names.discard("predicates")
    return names


# ── Selection ────────────────────────────────────────────────────────────────


def _template_catalog_block(template: str, world: str | None = None) -> str:
    actions = sorted(DOMAIN_PDDL_ACTIONS.get(template, frozenset()))
    candidates = sorted(enrichment_candidates_for_domain(template, world=world))
    return (
        f"- {template}: {TEMPLATE_SUMMARIES.get(template, '')}\n"
        f"    actions already in PDDL: {', '.join(actions) or '(none)'}\n"
        f"    enrichment candidates: {', '.join(candidates) or '(none)'}"
    )


def build_selection_prompt(
    command: str,
    scene_symbols: Sequence[str] = (),
    world: str | None = None,
) -> str:
    """User half of the select + completeness prompt."""
    templates = "\n".join(
        _template_catalog_block(t, world=world) for t in TEMPLATE_NAMES
    )
    catalog = "\n".join(
        f"- {line}"
        for line in catalog_summary_lines(catalog_skills_for_world(world))
    )
    symbols = ", ".join(str(s) for s in scene_symbols) or "(none reported)"
    return (
        "domain_templates:\n"
        f"{templates}\n\n"
        "robot_skill_catalog (closed — nothing else is executable):\n"
        f"{catalog}\n\n"
        f"scene_symbols: {symbols}\n"
        f"task: {str(command or '').strip()}\n"
    )


def parse_selection_payload(
    raw: str,
    *,
    command: str = "",
    world: str | None = None,
) -> DomainSelectionResult:
    """
    Validate a selection JSON payload against the templates and closed catalog.

    Raises ``DomainSelectionError`` when the payload cannot be trusted. Named
    skills outside the chosen template's enrichment candidates are dropped, and
    an ``incomplete`` verdict with nothing left routes to the refuse path.
    """
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DomainSelectionError(f"invalid selection JSON: {exc}") from exc

    template = str(data.get("template", "")).strip()
    if template not in TEMPLATE_NAMES:
        raise DomainSelectionError(f"unknown domain template: {template!r}")

    raw_completeness = str(data.get("completeness", "")).strip().lower()
    if raw_completeness not in {"complete", "incomplete"}:
        raise DomainSelectionError(
            f"completeness must be complete|incomplete, got {raw_completeness!r}"
        )
    completeness = DomainCompleteness(raw_completeness)

    raw_skills = data.get("needed_skills") or []
    if isinstance(raw_skills, str):
        raw_skills = [raw_skills]
    if not isinstance(raw_skills, (list, tuple)):
        raise DomainSelectionError("needed_skills must be a list")

    allowed = enrichment_candidates_for_domain(template, world=world)
    needed: list[str] = []
    dropped: list[str] = []
    for item in raw_skills:
        skill = normalize_catalog_skill(str(item))
        if skill in allowed and skill not in needed:
            needed.append(skill)
        elif skill:
            dropped.append(skill)

    native_dropped = skills_already_in_template(template, dropped)
    other_dropped = tuple(
        skill
        for skill in dropped
        if normalize_catalog_skill(skill) not in native_dropped
    )

    reason = str(data.get("reason", "")).strip() or "LLM domain selection"
    if completeness == DomainCompleteness.COMPLETE and needed:
        # A complete domain by definition needs nothing enriched.
        needed = []
    if other_dropped:
        reason = (
            f"{reason} (dropped non-catalog skills: {', '.join(other_dropped)})"
        )
    if (
        completeness == DomainCompleteness.INCOMPLETE
        and not needed
        and native_dropped
        and not other_dropped
    ):
        # place-in-container / pick / place are already in the template.
        completeness = DomainCompleteness.COMPLETE
        reason = (
            f"{reason} (template already defines "
            f"{', '.join(native_dropped)}; treated as complete)"
        )
    elif completeness == DomainCompleteness.INCOMPLETE and not needed:
        reason = f"{reason} (no catalog skill matches the gap)"

    return DomainSelectionResult(
        template=template,
        completeness=completeness,
        reason=reason,
        needed_skills=tuple(needed),
        backend=SelectionBackend.LLM.value,
    )


@dataclass
class LLMDomainSelector:
    """
    Template + completeness from the single text LLM.

    Falls back to the keyword selector when the model output is unusable
    (``fallback_rule_based=True``, the default) so a bad generation degrades to
    the Session 28 behaviour instead of failing the task.
    """

    generate_fn: GenerateFn | None = None
    client: LocalTextClient | None = None
    model_id: str | None = None
    fallback_rule_based: bool = True
    max_new_tokens: int = 256
    backend: str = SelectionBackend.LLM.value
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generate_fn is not None and self.client is not None:
            raise ValueError("pass only one of generate_fn or client")

    def _complete(self, system: str, user: str, *, stage: str = "select") -> str:
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

    def select(
        self,
        command: str,
        *,
        scene_symbols: Sequence[str] = (),
        world: str | None = None,
    ) -> DomainSelectionResult:
        user = build_selection_prompt(command, scene_symbols, world=world)
        try:
            raw = self._complete(SELECT_SYSTEM_PROMPT, user)
            return parse_selection_payload(raw, command=command, world=world)
        except DomainSelectionError as exc:
            error = str(exc)
        except Exception as exc:  # noqa: BLE001 — model / transport failure
            error = f"text LLM selection call failed: {exc}"

        if not self.fallback_rule_based:
            raise DomainSelectionError(error)

        fallback = select_domain_rule_based(command, world=world)
        return DomainSelectionResult(
            template=fallback.template,
            completeness=fallback.completeness,
            reason=f"{fallback.reason} (keyword fallback)",
            needed_skills=fallback.needed_skills,
            backend="rule_based_fallback",
            error=error,
        )

    def revise_after_uncovered(
        self,
        command: str,
        previous: DomainSelectionResult,
        *,
        scene_symbols: Sequence[str] = (),
        world: str | None = None,
    ) -> DomainSelectionResult:
        """
        One retry when the regex goal helper missed a ``complete`` phrasing.

        Not a refuse: the helper is a floor, not the v4 goal LLM. Same JSON
        contract as :meth:`select`. No task-specific examples.
        """
        base = build_selection_prompt(command, scene_symbols, world=world)
        user = (
            f"{base}\n"
            "previous_answer_rejected: true\n"
            f"previous_template: {previous.template}\n"
            f"previous_completeness: {previous.completeness.value}\n"
            f"previous_needed_skills: {list(previous.needed_skills)}\n"
            "rejection_reason: the rule-based goal helper did not parse this "
            "phrasing. If a listed enrichment candidate is required, answer "
            "incomplete with that skill. If the template's own actions already "
            "achieve the outcome (including implicit place / put-in "
            "paraphrases such as 'belongs on/to'), answer complete with empty "
            "needed_skills. Answer incomplete with an empty needed_skills "
            "list only when nothing in the catalog can close the gap.\n"
        )
        try:
            raw = self._complete(SELECT_SYSTEM_PROMPT, user, stage="select_revise")
            revised = parse_selection_payload(raw, command=command, world=world)
            return replace(
                revised,
                reason=(
                    f"{revised.reason} "
                    "(revised after coverage veto)"
                ),
            )
        except DomainSelectionError:
            return previous
        except Exception:
            return previous


# ── Goal binding for enriched actions ────────────────────────────────────────


def build_binding_prompt(
    command: str,
    action: Mapping[str, Any],
    scene_symbols: Sequence[str],
) -> str:
    """User half of the goal-binding prompt."""
    symbols = ", ".join(str(s) for s in scene_symbols) or "(none reported)"
    return (
        f"task: {str(command or '').strip()}\n"
        f"action: {action.get('name', '')}\n"
        f"parameters: {action.get('parameters', '')}\n"
        f"effect: {action.get('effect', '')}\n"
        f"scene_symbols: {symbols}\n"
    )


def parse_binding_payload(
    raw: str,
    *,
    arity: int,
    scene_symbols: Sequence[str],
) -> tuple[str, ...] | None:
    """
    Validate a binding payload; ``None`` when the model declined.

    Symbols must come from ``scene_symbols`` verbatim — the point of the closed
    catalog is lost if the goal can name objects the robot cannot see.
    """
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise DomainSelectionError(f"invalid binding JSON: {exc}") from exc

    if data.get("refuse"):
        return None

    values = data.get("bindings")
    if isinstance(values, Mapping):
        values = list(values.values())
    if not isinstance(values, (list, tuple)) or len(values) != arity:
        raise DomainSelectionError(
            f"bindings must be a list of {arity} scene symbols"
        )

    allowed = {str(s) for s in scene_symbols}
    bound = tuple(str(v).strip() for v in values)
    unknown = [b for b in bound if b not in allowed]
    if unknown:
        raise DomainSelectionError(
            f"bindings name symbols outside the scene: {', '.join(unknown)}"
        )
    return bound


@dataclass
class LLMGoalBinder:
    """
    Grounds an enriched action's parameters with the single text LLM.

    Needed for paraphrases: "get me something to drink" never names the bottle or
    the glass, so nothing in the sentence tells the planner which vessel is the
    source. The same model that chose the template and authored the PDDL picks
    the objects; when it cannot, callers fall back to the deterministic
    command-order binding and, failing that, refuse.
    """

    generate_fn: GenerateFn | None = None
    client: LocalTextClient | None = None
    model_id: str | None = None
    max_new_tokens: int = 128
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generate_fn is not None and self.client is not None:
            raise ValueError("pass only one of generate_fn or client")

    def _complete(self, system: str, user: str, *, stage: str = "bind") -> str:
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

    def bind(
        self,
        command: str,
        action: Mapping[str, Any],
        scene_symbols: Sequence[str],
        *,
        arity: int,
    ) -> tuple[str, ...] | None:
        """Symbols for each parameter, or ``None`` when binding is impossible."""
        if arity <= 0 or not scene_symbols:
            return None
        user = build_binding_prompt(command, action, scene_symbols)
        try:
            raw = self._complete(BIND_SYSTEM_PROMPT, user)
            return parse_binding_payload(
                raw, arity=arity, scene_symbols=scene_symbols
            )
        except DomainSelectionError:
            return None
        except Exception:  # noqa: BLE001 — model / transport failure
            return None


# ── Enrichment ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedEnrichment:
    """Validated closed-catalog enrichment payload."""

    skill: str
    action: dict[str, str]
    new_predicates: tuple[str, ...] = ()
    new_types: tuple[str, ...] = ()
    reason: str = ""
    refused: bool = False

    @property
    def action_name(self) -> str:
        return str(self.action.get("name", ""))


def build_enrichment_prompt(
    request: EnrichmentRequest,
    *,
    base_domain_text: str,
    allowed_skills: Sequence[str],
    previous_error: str | None = None,
) -> str:
    """User half of the enrichment prompt (closed catalog + the real domain)."""
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


def _require_sexpr(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text.startswith("(") or not text.endswith(")"):
        raise EnrichmentPayloadError(f"{label} must be a PDDL s-expression")
    if not _balanced(text):
        raise EnrichmentPayloadError(f"{label} has unbalanced parentheses")
    return text


def _predicate_head(declaration: str) -> str:
    """Name out of a declaration such as ``(poured ?src - item ?dst - item)``."""
    match = re.match(r"\(\s*([a-zA-Z][\w-]*)", declaration.strip())
    return match.group(1) if match else declaration.strip()


def parse_enrichment_payload(
    raw: str,
    *,
    allowed_skills: Sequence[str],
) -> ParsedEnrichment:
    """
    Validate an enrichment JSON payload against the closed catalog.

    Enforces: allowed skill, matching PDDL action name, parameter count within
    the ROS primitive's signature, bound variables, and a positive effect the
    Session 19 goal helper can read. Raises ``EnrichmentPayloadError``.
    """
    try:
        data = extract_json_object(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        raise EnrichmentPayloadError(f"invalid enrichment JSON: {exc}") from exc

    if data.get("refuse"):
        return ParsedEnrichment(
            skill="",
            action={},
            reason=str(data.get("reason", "")).strip() or "model refused",
            refused=True,
        )

    allowed = {normalize_catalog_skill(s) for s in allowed_skills}
    skill = normalize_catalog_skill(str(data.get("skill", "")))
    if skill not in allowed:
        raise EnrichmentPayloadError(
            f"skill {skill!r} is not in the allowed set "
            f"{sorted(allowed) or '(empty)'}"
        )
    sig = skill_signature(skill)
    if sig is None or ros_primitive_for_action(skill) is None:
        raise EnrichmentPayloadError(f"skill {skill!r} has no ROS primitive")

    action_raw = data.get("action")
    if not isinstance(action_raw, dict):
        actions = data.get("actions")
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            action_raw = actions[0]
        else:
            raise EnrichmentPayloadError("missing 'action' object")

    name = str(action_raw.get("name", "")).strip().lower()
    if normalize_catalog_skill(name) != skill:
        raise EnrichmentPayloadError(
            f"action name {name!r} does not match skill {skill!r} "
            f"(expected {sig.pddl_action!r})"
        )

    parameters = _require_sexpr(action_raw.get("parameters"), "parameters")
    precondition = _require_sexpr(action_raw.get("precondition"), "precondition")
    effect = _require_sexpr(action_raw.get("effect"), "effect")

    # `- type` is legal in :parameters and :predicates, never inside a formula.
    # A live 7B wrote `(pred ?x - item ?c - container)` as an effect atom, which
    # FD rejects and which turned into a 3-argument goal carrying the type name
    # as an object.
    for label, formula in (("precondition", precondition), ("effect", effect)):
        if _TYPED_ATOM.search(formula):
            raise EnrichmentPayloadError(
                f"{label} declares parameter types inside a formula — "
                "types belong in 'parameters', not in preconditions or effects"
            )

    declared = re.findall(r"\?([a-zA-Z][\w-]*)", parameters)
    if not (sig.min_params <= len(declared) <= sig.max_params):
        raise EnrichmentPayloadError(
            f"action '{sig.pddl_action}' needs {sig.min_params}–"
            f"{sig.max_params} parameters ({', '.join(sig.roles)}), got "
            f"{len(declared)}"
        )
    used = set(re.findall(r"\?([a-zA-Z][\w-]*)", precondition + " " + effect))
    unbound = sorted(used - set(declared))
    if unbound:
        raise EnrichmentPayloadError(
            "effect/precondition use undeclared variables: "
            + ", ".join(f"?{v}" for v in unbound)
        )
    # The Session 19 helper turns the first positive effect into the :goal fact,
    # so an action it cannot read is useless to the planner downstream.
    goal_fact = goal_from_enrichment_action(
        {"effect": effect, "parameters": parameters}, {}
    )
    if goal_fact is None:
        raise EnrichmentPayloadError(
            "effect asserts no positive predicate — no goal fact can be derived"
        )

    new_predicates = tuple(
        _require_sexpr(p, "new predicate")
        for p in (data.get("new_predicates") or [])
    )

    # The goal fact must come from a predicate this action *introduces*.
    # A live 1.5B produced `pour` with effect `(and (holding ?dst) (clear ?dst))`:
    # every check above passed, yet the domain gained nothing and the goal
    # collapsed to `(holding cup)` — satisfiable by a plain pick. An enrichment
    # that asserts only pre-existing facts is not an enrichment.
    goal_predicate = goal_fact.strip("()").split()[0]
    declared_new = {_predicate_head(p) for p in new_predicates}
    # Models copy the skeleton verbatim: a live 7B declared the placeholder
    # `(pred ?x - item ?y - item)` and made `(pred can glass)` the goal. Such a
    # fluent is "new" by every structural test and meaningless in the domain.
    if goal_predicate in _PLACEHOLDER_PREDICATES:
        raise EnrichmentPayloadError(
            f"predicate {goal_predicate!r} is a placeholder from the prompt — "
            "name the fluent after the outcome of the skill (e.g. 'poured')"
        )
    if goal_predicate not in declared_new:
        raise EnrichmentPayloadError(
            f"effect asserts {goal_predicate!r}, which the action does not "
            f"declare in new_predicates ({', '.join(sorted(declared_new)) or 'none'})"
            " — an enriched action must introduce the fact it achieves"
        )

    # Catch the live FD failure: declare (containing ?t ?i) but write
    # (containing ?target) in the effect — arity must match the declaration.
    new_arities: dict[str, int] = {}
    for decl in new_predicates:
        head = _predicate_head(decl).lower()
        new_arities[head] = len(re.findall(r"\?[a-zA-Z][\w-]*", decl))
    for label, formula in (("precondition", precondition), ("effect", effect)):
        for pred, arity in _atoms_in_formula(
            formula if formula.startswith("(") else f"({formula})"
        ):
            if pred not in new_arities:
                continue
            expected = new_arities[pred]
            if arity != expected:
                raise EnrichmentPayloadError(
                    f"{label} uses '{pred}' with {arity} argument(s) but "
                    f"new_predicates declares arity {expected}"
                )

    new_types = tuple(
        str(t).strip() for t in (data.get("new_types") or []) if str(t).strip()
    )

    return ParsedEnrichment(
        skill=skill,
        action={
            "name": sig.pddl_action,
            "parameters": parameters,
            "precondition": precondition,
            "effect": effect,
        },
        new_predicates=new_predicates,
        new_types=new_types,
        reason=str(data.get("reason", "")).strip(),
    )


@dataclass
class LLMDomainEnricher:
    """
    Closed-catalog PDDL author backed by the single text LLM.

    ``enrich`` returns ``ENRICHED`` (merged + persisted domain, plus the
    ``domain_additions`` the Session 19 goal helper consumes) or ``REFUSED``
    with the user-visible message. It never returns a domain containing an
    action the robot cannot dispatch.
    """

    generate_fn: GenerateFn | None = None
    client: LocalTextClient | None = None
    model_id: str | None = None
    enriched_dir: str | None = None
    max_attempts: int = 2
    max_new_tokens: int = 512
    reuse_persisted: bool = True
    calls: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.generate_fn is not None and self.client is not None:
            raise ValueError("pass only one of generate_fn or client")

    def _complete(self, system: str, user: str, *, stage: str = "enrich") -> str:
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
            s
            for s in dict.fromkeys(
                normalize_catalog_skill(x) for x in request.candidate_skills
            )
            if s in enrichment_candidates_for_domain(
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

        signature = enrichment_signature(
            template=request.template,
            skills=allowed,
            base_domain_text=base_domain,
            world=request.world,
        )
        discarded_invalid_cache = False
        if self.reuse_persisted:
            cached = load_enriched_domain(
                signature,
                template=request.template,
                directory=self.enriched_dir,
                world=request.world,
            )
            if cached is not None:
                cache_errors = pddl_syntax_errors(cached.domain_text)
                if not cache_errors:
                    return EnrichmentOutcome(
                        status=EnrichmentStatus.ENRICHED,
                        domain_text=cached.domain_text,
                        domain_path=cached.domain_path,
                        skills_grounded=cached.skills,
                        ros_primitives=cached.ros_primitives,
                        domain_additions=cached.domain_additions,
                        signature=signature,
                        reused=True,
                        notes=(
                            "reused persisted enriched domain "
                            f"{signature} (no LLM call)"
                        ),
                    )
                # Stale / LLM-authored illegal PDDL (e.g. arity mismatch) —
                # drop the cache entry and fall through to re-author.
                discard_enriched_domain(
                    signature,
                    template=request.template,
                    directory=self.enriched_dir,
                    world=request.world,
                )
                discarded_invalid_cache = True

        parsed, error = self._author(request, base_domain, allowed)
        if parsed is None:
            return self._refuse(
                request,
                detail=f"Enricher could not author valid PDDL: {error}",
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

        primitive = ros_primitive_for_action(parsed.action_name)
        if primitive is None:
            return self._refuse(
                request,
                detail=(
                    f"Action '{parsed.action_name}' has no ROS primitive to "
                    "dispatch."
                ),
                notes="refuse — action not executable",
            )

        additions = {
            "new_types": list(parsed.new_types),
            "new_predicates": list(parsed.new_predicates),
            "new_actions": [dict(parsed.action)],
            "modified_preconditions": {},
        }
        record = save_enriched_domain(
            template=request.template,
            skills=allowed,
            domain_text=merged,
            base_domain_text=base_domain,
            ros_primitives=(primitive,),
            domain_additions=additions,
            meta={
                "command": request.command,
                "reason": parsed.reason,
                "selection_backend": request.selection.backend,
                "grounded_skills": [parsed.skill],
            },
            directory=self.enriched_dir,
            world=request.world,
        )
        notes = (
            f"grounded catalog skill '{parsed.skill}' as PDDL action "
            f"'{parsed.action_name}' → ROS primitive '{primitive}'"
        )
        if discarded_invalid_cache:
            notes = (
                f"discarded invalid persisted domain {signature}; re-authored. "
                + notes
            )
        return EnrichmentOutcome(
            status=EnrichmentStatus.ENRICHED,
            domain_text=merged,
            domain_path=record.domain_path,
            skills_grounded=(parsed.skill,),
            ros_primitives=(primitive,),
            domain_additions=additions,
            signature=record.signature,
            notes=notes,
        )

    def _author(
        self,
        request: EnrichmentRequest,
        base_domain: str,
        allowed: Sequence[str],
    ) -> tuple[ParsedEnrichment | None, str]:
        """Prompt the LLM, retrying once with the validation error attached."""
        error = "no attempt made"
        previous: str | None = None
        for _ in range(max(1, int(self.max_attempts))):
            user = build_enrichment_prompt(
                request,
                base_domain_text=base_domain,
                allowed_skills=allowed,
                previous_error=previous,
            )
            try:
                raw = self._complete(ENRICH_SYSTEM_PROMPT, user)
                return parse_enrichment_payload(raw, allowed_skills=allowed), ""
            except EnrichmentPayloadError as exc:
                error = str(exc)
            except Exception as exc:  # noqa: BLE001 — model / transport failure
                error = f"text LLM enrichment call failed: {exc}"
            previous = error
        return None, error

    def _merge(
        self,
        base_domain: str,
        parsed: ParsedEnrichment,
    ) -> tuple[str | None, str]:
        """Apply the action to the template and syntax-check the result."""
        if parsed.action_name in domain_action_names(base_domain):
            return None, f"action '{parsed.action_name}' already in domain"

        # Complements the parse-time check: there the goal predicate had to be
        # declared as new, here it must also be absent from the base domain, so
        # redeclaring `holding` cannot smuggle a vacuous effect through.
        already = sorted(
            _predicate_head(p) for p in parsed.new_predicates
        )
        clashes = [
            name for name in already if name in domain_predicate_names(base_domain)
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
                new_actions=[dict(parsed.action)],
            ),
        )
        if not result.is_valid:
            return None, "; ".join(result.errors) or "structural validation failed"
        if parsed.action_name not in domain_action_names(result.domain_text):
            return None, (
                f"action '{parsed.action_name}' was skipped during merge: "
                + "; ".join(result.additions_skipped)
            )
        errors = pddl_syntax_errors(result.domain_text)
        if errors:
            return None, "; ".join(errors)
        return result.domain_text, ""
