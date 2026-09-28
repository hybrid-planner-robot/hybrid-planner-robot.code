#!/usr/bin/env python3
"""
Planning-only battery: perception + domain select ± enrichment + Fast Downward.

No arm execution. Scores three claims; claim 2 is never the sum of claim 1:

1. Select-only (enrichment OFF): *correct* plans on template-complete tasks,
   plus domain / :init / :goal delivered to Fast Downward.
2. Enrichment ON: *correct* plans on catalog-gap tasks only (not the
   template-complete set from claim 1).
3. Ungenerable recognition (refuse), reported separately from (1) and (2).
   A spurious pick/place plan on solder/levitate is a false plan, not a
   planning success. A plan that exists but misses gold goal/actions/domain
   is an incorrect plan, not a success.

Requires a running sim for live runs. Prefer::

    bin/eval_planning_battery.sh --dry
    bin/eval_planning_battery.sh --arms select
    bin/eval_planning_battery.sh --model Qwen/Qwen2.5-7B-Instruct
    bin/eval_planning_battery.sh --arms enrich --scene-source oracle --no-perception-only

Examples
--------
Scoring self-check (no Docker / GPU)::

    python scripts/eval_planning_battery.py --dry

Live, both arms, perception + plan only (thesis default: dino)::

    python scripts/eval_planning_battery.py --model Qwen/Qwen2.5-7B-Instruct

Oracle ``:init`` ablation (not the thesis number)::

    python scripts/eval_planning_battery.py --arms enrich \\
        --scene-source oracle --no-perception-only

Subset::

    python scripts/eval_planning_battery.py --arms select \\
        --cases explicit_place,implicit_thirsty,explicit_solder --dry

Baseline arms (opt-in; default remains select,enrich)::

    python scripts/eval_planning_battery.py --dry --arms llm_pddl,llm_plan

Offline mock through the baseline modules (no GPU; suite texts verbatim)::

    python scripts/eval_planning_battery.py --mock

Live baselines (sim + DINO, same spine as R0; not a paper default)::

    python scripts/eval_planning_battery.py --arms llm_plan \\
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import eval_enrichment_live as enrich_live  # noqa: E402
from planner.call_timings import empty_timings, from_payload  # noqa: E402
from planner.pddl_sections import split_problem  # noqa: E402
from planner.problem_generator.goal_generator.battery_scoring import (  # noqa: E402
    facts_equal,
)
from planner.domain_llm import pddl_syntax_errors  # noqa: E402
from planner.skill_catalog import (  # noqa: E402
    DOMAIN_PDDL_ACTIONS,
    catalog_to_pddl_action,
)

_DEFAULT_SUITE = (
    _REPO_ROOT / "tests" / "fixtures" / "planning_eval" / "suite_v1.json"
)
_DEFAULT_OUT = _REPO_ROOT / "data" / "planning_eval"
_TEXT_LLM_ENV = "VLMRP_TEXT_LLM_MODEL"
_ARMS = ("select", "enrich")
_BASELINE_ARMS = ("llm_pddl", "llm_plan")
_R1_ARM = "r1"
_KNOWN_ARMS = _ARMS + _BASELINE_ARMS + (_R1_ARM,)
_V2_DEFAULT_ARMS = ("llm_plan", "llm_pddl", "enrich", "r1")
_DEFAULT_SUITE_V2 = (
    _REPO_ROOT / "tests" / "fixtures" / "planning_eval" / "suite_v2_mock.json"
)
_BASELINE_SCRIPTS = {
    "llm_pddl": "run_loop_llm_pddl.py",
    "llm_plan": "run_loop_llm_plan.py",
}
_BASELINE_SCENE_DIR = _REPO_ROOT / "tests" / "fixtures" / "llm_plan"
# Live E5 decode budget: domain+problem JSON; also used for llm_plan so both
# arms share one loaded 7B (text_llm clients are keyed by model id).
_LIVE_MAX_NEW_TOKENS = 1536
_OFFICIAL_TEMPLATES = frozenset(
    {
        "manipulation_base",
        "manipulation_stacking",
        "manipulation_containers",
        "navigation_manipulation",
    }
)
_LOCATION_ALIASES = {
    "shelf_b": "shelf",
    "counter": "table",
    "target_tray": "tray",
}
# Loop fillers: not a unique skill choice. Ignored on underspecified allowlists.
_FILLER_ACTIONS = frozenset({"look-at", "look_at", "navigate-to", "navigate_to"})
# Enrichment authoring invents fluent names; collapse synonyms to a canonical form.
_PREDICATE_ALIASES = {
    "poured": "poured",
    "transferred-liquid": "poured",
    "transferred_liquid": "poured",
    "cut-open": "cut-open",
    "cut": "cut-open",
    "tilted": "tilted",
    "stirred": "stirred",
    "painted": "painted",
    "coated": "painted",
    "drilled": "drilled",
    "drilled-hole": "drilled",
    "clamped": "clamped",
    "secured": "clamped",
}
_ACTION_HEAD = re.compile(r"\(:action\s+([\w-]+)", re.IGNORECASE)


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


_PDDL_COPY_NAMES = (
    "domain.pddl",
    "problem.pddl",
    "init.pddl",
    "goal.pddl",
    "fd_plan.json",
    "debug.json",
    "llm_plan.json",
    "llm_pddl.json",
    "scene_compact.json",
)


def latest_iter_dir(run_dir: Path | None) -> Path | None:
    """Newest ``iter_XX`` that has ``problem.pddl``, else newest iter dir."""
    if run_dir is None or not run_dir.is_dir():
        return None
    with_problem = sorted(run_dir.glob("iter_*/problem.pddl"))
    if with_problem:
        return with_problem[-1].parent
    dirs = sorted(p for p in run_dir.glob("iter_*") if p.is_dir())
    return dirs[-1] if dirs else None


def load_pddl_artifacts(
    summary: dict[str, Any],
    summary_path: Path | None,
) -> dict[str, Any]:
    """
    Collect PDDL problem / init / goal / domain from summary.json or iter artefacts.

    Old runs without these fields must not raise.
    """
    problem = summary.get("pddl_problem") or None
    init_block = summary.get("pddl_init") or None
    goal_block = summary.get("pddl_goal") or None
    domain = summary.get("pddl_domain") or None
    init_facts = summary.get("init_facts")
    parsed_goal: list[list[str]] = list(
        summary.get("pddl_goal_facts") or summary.get("goal_facts") or []
    )

    run_dir = summary_path.parent if summary_path is not None else None
    iter_dir = latest_iter_dir(run_dir)

    def _read(name: str) -> str | None:
        if iter_dir is None:
            return None
        path = iter_dir / name
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    if not problem:
        problem = _read("problem.pddl")
    if not init_block:
        init_block = _read("init.pddl")
    if not goal_block:
        goal_block = _read("goal.pddl")
    if not domain:
        domain = _read("domain.pddl")
    if not domain and iter_dir is not None:
        extras = sorted(iter_dir.glob("domain_*.pddl"))
        if extras:
            try:
                domain = extras[-1].read_text(encoding="utf-8")
            except OSError:
                domain = None
    if not domain:
        persisted = summary.get("domain_persisted")
        if persisted:
            path = Path(str(persisted))
            if not path.is_absolute():
                path = _REPO_ROOT / path
            if path.is_file():
                try:
                    domain = path.read_text(encoding="utf-8")
                except OSError:
                    domain = None
    if (
        not domain
        and not summary.get("enrichment_used")
        and summary.get("baseline") not in _BASELINE_ARMS
    ):
        template = summary.get("domain_template")
        if template:
            stock = _REPO_ROOT / "pddl" / "domains" / f"{template}.pddl"
            if stock.is_file():
                try:
                    domain = stock.read_text(encoding="utf-8")
                except OSError:
                    domain = None

    if problem and (
        not init_block
        or not goal_block
        or init_facts is None
        or not parsed_goal
    ):
        try:
            split = split_problem(problem)
        except Exception:
            split = {}
        init_block = init_block or split.get("pddl_init")
        goal_block = goal_block or split.get("pddl_goal")
        if init_facts is None:
            init_facts = split.get("init_facts") or []
        if not parsed_goal:
            parsed_goal = list(split.get("goal_facts") or [])

    if init_facts is None:
        init_facts = []

    return {
        "pddl_problem": problem,
        "pddl_init": init_block,
        "pddl_goal": goal_block,
        "pddl_domain": domain,
        "init_facts": [list(f) for f in init_facts],
        "pddl_goal_facts": [list(f) for f in parsed_goal],
        "iter_dir": str(iter_dir) if iter_dir is not None else None,
    }


def copy_case_pddl_artifacts(
    out_dir: Path,
    arm: str,
    case_id: str,
    *,
    summary_path: Path | None,
    artifacts: dict[str, Any],
) -> str | None:
    """Copy inspectable PDDL files into ``data/planning_eval/<ts>/cases/<arm>_<id>/``."""
    dest = out_dir / "cases" / f"{arm}_{case_id}"
    copied = False
    iter_dir: Path | None = None
    if artifacts.get("iter_dir"):
        iter_dir = Path(str(artifacts["iter_dir"]))
    elif summary_path is not None:
        iter_dir = latest_iter_dir(summary_path.parent)

    if iter_dir is not None and iter_dir.is_dir():
        for name in _PDDL_COPY_NAMES:
            src = iter_dir / name
            if src.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)
                copied = True
    elif summary_path is not None:
        run_root = summary_path.parent
        for name in _PDDL_COPY_NAMES:
            src = run_root / name
            if src.is_file():
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest / name)
                copied = True

    if artifacts.get("pddl_problem") and not (dest / "problem.pddl").exists():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "problem.pddl").write_text(
            str(artifacts["pddl_problem"]), encoding="utf-8"
        )
        copied = True
    if artifacts.get("pddl_init") and not (dest / "init.pddl").exists():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "init.pddl").write_text(str(artifacts["pddl_init"]), encoding="utf-8")
        copied = True
    if artifacts.get("pddl_goal") and not (dest / "goal.pddl").exists():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "goal.pddl").write_text(str(artifacts["pddl_goal"]), encoding="utf-8")
        copied = True
    if artifacts.get("pddl_domain") and not (dest / "domain.pddl").exists():
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "domain.pddl").write_text(
            str(artifacts["pddl_domain"]), encoding="utf-8"
        )
        copied = True

    if not copied:
        return None
    return _display(dest)


def load_suite(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_cases(
    suite: dict[str, Any],
    case_ids: list[str] | None,
) -> list[dict[str, Any]]:
    return enrich_live.select_cases(suite, case_ids)


def baseline_scene_file(world: str | None) -> Path:
    """World-faithful oracle_mock fixture for ``--mock-scene`` / ``--mock``."""
    name = str(world or "tabletop").strip() or "tabletop"
    path = _BASELINE_SCENE_DIR / f"{name}.json"
    if path.is_file():
        return path
    return _BASELINE_SCENE_DIR / "wood_cube_table.json"


def parse_arms(raw: str | None) -> list[str]:
    if not raw:
        return list(_ARMS)
    wanted = [a.strip() for a in raw.split(",") if a.strip()]
    unknown = [a for a in wanted if a not in _KNOWN_ARMS]
    if unknown:
        raise SystemExit(
            f"Unknown arm(s): {unknown} (use select,enrich,r1,llm_pddl,llm_plan)"
        )
    return wanted


# ── Outcome helpers ──────────────────────────────────────────────────────────


def _norm_skill(name: str) -> str:
    return str(name or "").strip().lower().replace("_", "-")


def primitive_names(row: dict[str, Any]) -> list[str]:
    """
    Action names present in the plan.

    Prefer FD action heads when available: ``stack`` is often normalised to the
    ROS primitive ``place``, which must not fail ``plan_covers(["stack"])``.
    """
    names: list[str] = []
    for act in row.get("fd_actions") or []:
        text = str(act).strip().strip("()")
        token = text.split()[0] if text else ""
        if token:
            names.append(_norm_skill(token))
    for prim in row.get("fd_primitives") or []:
        if isinstance(prim, dict) and prim.get("name"):
            names.append(_norm_skill(str(prim["name"])))
    return names


def plan_arg_tokens(row: dict[str, Any]) -> set[str]:
    """Object-like tokens from FD actions / primitive args (normalized)."""
    tokens: set[str] = set()
    for act in row.get("fd_actions") or []:
        text = str(act).strip().strip("()")
        parts = text.split()
        for part in parts[1:]:
            cleaned = part.strip("(),")
            if cleaned:
                tokens.add(_norm_skill(cleaned))
    for prim in row.get("fd_primitives") or []:
        if not isinstance(prim, dict):
            continue
        args = prim.get("args")
        if isinstance(args, dict):
            values = args.values()
        elif isinstance(args, (list, tuple)):
            values = args
        else:
            values = []
        for value in values:
            text = str(value).strip()
            if text:
                tokens.add(_norm_skill(text))
    return tokens


def authored_action_names(row: dict[str, Any]) -> list[str]:
    authored = row.get("enrichment_authored") or {}
    names: list[str] = []
    for act in authored.get("actions") or []:
        if isinstance(act, dict) and act.get("name"):
            names.append(str(act["name"]))
        elif act:
            names.append(str(act))
    return names


def authored_predicates(row: dict[str, Any]) -> list[str]:
    authored = row.get("enrichment_authored") or {}
    return [str(p) for p in (authored.get("new_predicates") or [])]


def underspecified_case_ok(row: dict[str, Any], expect: dict[str, Any]) -> bool:
    """
    Open-intent cases: refuse or a plan in the allowlist.

    Canonical gold / soft_golds still count (dry and the 'commonsense' completion).
    A skill outside allowed_actions is spurious. Camera/nav fillers are ignored.
    """
    accept = expect.get("accept") or {}
    if is_refused(row):
        return bool(accept.get("refuse", True))
    if is_invalid_pddl(row) or is_invalid_plan(row):
        return False
    if not plan_found(row):
        return False
    heads = {
        _norm_skill(n)
        for n in primitive_names(row)
        if n and _norm_skill(n) not in {_norm_skill(a) for a in _FILLER_ACTIONS}
    }
    allowed_actions = {
        _norm_skill(a) for a in (accept.get("allowed_actions") or []) if a
    }
    if allowed_actions and heads and not heads.issubset(allowed_actions):
        return False
    gold = expect.get("gold_goal")
    soft = list(expect.get("soft_golds") or []) + list(accept.get("soft_golds") or [])
    if gold or soft:
        if goal_matches(row.get("goal_facts"), gold, soft):
            return True
    allowed_objects = {
        _norm_skill(o) for o in (accept.get("allowed_objects") or []) if o
    }
    if allowed_actions and heads and heads.issubset(allowed_actions):
        if not allowed_objects:
            return True
        args = plan_arg_tokens(row)
        if args & allowed_objects:
            return True
        # Dry stubs use "(pick dry)" with no scene object.
        if args <= {"dry"} or not args:
            return True
    return False


def alias_facts(facts: list[list[str]] | None) -> list[list[str]]:
    out: list[list[str]] = []
    for fact in facts or []:
        out.append([_LOCATION_ALIASES.get(str(x), str(x)) for x in fact])
    return out


def _fact_key(fact: list[str]) -> tuple[str, ...]:
    return tuple(str(x) for x in fact)


def parse_raw_fact(text: str) -> list[str] | None:
    """Turn ``'(tilted can)'`` / ``tilted can`` into a fact token list."""
    body = str(text or "").strip()
    if not body:
        return None
    if body.startswith("(") and body.endswith(")"):
        body = body[1:-1].strip()
    body = body.split(";", 1)[0].strip()
    tokens = body.split()
    return tokens or None


def canonicalize_predicates(facts: list[list[str]] | None) -> list[list[str]]:
    out: list[list[str]] = []
    for fact in facts or []:
        if not fact:
            continue
        pred = _norm_skill(fact[0])
        canon = _PREDICATE_ALIASES.get(pred, pred)
        out.append([canon, *[str(x) for x in fact[1:]]])
    return out


def normalize_goal_facts(facts: list[list[str]] | None) -> list[list[str]]:
    """
    Expand ``_raw_fact`` rows, apply location + predicate aliases.

    Live summaries often store enrichment fluents as
    ``["_raw_fact", "(tilted can)"]`` instead of ``["tilted", "can"]``.
    """
    expanded: list[list[str]] = []
    for fact in facts or []:
        if not fact:
            continue
        if str(fact[0]) == "_raw_fact" and len(fact) >= 2:
            parsed = parse_raw_fact(str(fact[1]))
            if parsed:
                expanded.append(parsed)
            continue
        expanded.append([str(x) for x in fact])
    return canonicalize_predicates(alias_facts(expanded))


def facts_contain(
    got: list[list[str]] | None,
    required: list[list[str]] | None,
) -> bool:
    """True iff every required fact is present in ``got`` (alias-insensitive)."""
    if not required:
        return False
    have = {_fact_key(f) for f in alias_facts(got)}
    return all(_fact_key(f) in have for f in alias_facts(required))


def _unary_object_covered(need: list[str], have_facts: list[list[str]]) -> bool:
    """Gold ``(P x)`` is covered by ``(P … x …)`` after aliasing (tool args extra)."""
    if len(need) != 2:
        return False
    pred, obj = _norm_skill(need[0]), _norm_skill(need[1])
    for got in have_facts:
        if not got or _norm_skill(got[0]) != pred:
            continue
        if obj in {_norm_skill(x) for x in got[1:]}:
            return True
    return False


def goals_cover(
    got: list[list[str]] | None,
    required: list[list[str]] | None,
) -> bool:
    """True iff every required goal fact appears in ``got`` (extras allowed)."""
    need = normalize_goal_facts(required)
    if not need:
        return False
    have_facts = normalize_goal_facts(got)
    have = {_fact_key(f) for f in have_facts}
    for fact in need:
        if _fact_key(fact) in have:
            continue
        if _unary_object_covered(fact, have_facts):
            continue
        return False
    return True


def goal_matches(
    got: list[list[str]] | None,
    gold: list[list[str]] | None,
    soft_golds: list[list[list[str]]] | None = None,
) -> bool:
    """
    Goal OK if gold (or any soft_gold) is covered by ``got``.

    Coverage, not equality: extra fluents (e.g. ``camera-aimed-at`` beside
    ``tilted``) do not fail the case. Predicate synonyms apply.
    """
    candidates: list[list[list[str]]] = []
    if gold:
        candidates.append(list(gold))
    for alt in soft_golds or []:
        if alt:
            candidates.append(list(alt))
    if not candidates:
        return False
    if any(goals_cover(got, c) for c in candidates):
        return True
    # Keep legacy exact equality as a fallback for non-normalized callers.
    got_n = normalize_goal_facts(got)
    for c in candidates:
        if facts_equal(got_n, normalize_goal_facts(c)):
            return True
    return False


def domain_action_names(domain_text: str | None) -> set[str]:
    if not domain_text:
        return set()
    return {_norm_skill(name) for name in _ACTION_HEAD.findall(domain_text)}


def domain_has_skill(domain_text: str | None, skill: str | None) -> bool:
    if not skill:
        return False
    return _norm_skill(skill) in domain_action_names(domain_text)


def pddl_was_delivered(row: dict[str, Any]) -> bool:
    if row.get("pddl_problem") or row.get("pddl_init") or row.get("pddl_domain"):
        return True
    if row.get("init_facts") or row.get("goal_facts"):
        return True
    return False


def _not_false(flag: Any) -> bool:
    return flag is not False


def plan_covers(row: dict[str, Any], wanted: list[str] | None) -> bool:
    if not wanted:
        return False
    have = set(primitive_names(row))
    return all(_norm_skill(w) in have for w in wanted)


def is_refused(row: dict[str, Any]) -> bool:
    if int(row.get("exit_code") or 0) == 3:
        return True
    return str(row.get("exit_reason") or "") == "refused"


def is_invalid_pddl(row: dict[str, Any]) -> bool:
    return str(row.get("exit_reason") or "") == "invalid_pddl"


def is_invalid_plan(row: dict[str, Any]) -> bool:
    return str(row.get("exit_reason") or "") == "invalid_plan"


def plan_found(row: dict[str, Any]) -> bool:
    if is_refused(row) or is_invalid_pddl(row) or is_invalid_plan(row):
        return False
    n = int(row.get("n_plan_actions") or 0)
    if n > 0:
        return True
    if row.get("fd_actions") or row.get("fd_primitives"):
        return True
    return str(row.get("exit_reason") or "") == "planned"


def inferred_path(row: dict[str, Any]) -> str:
    if is_refused(row):
        return "refused"
    if row.get("enrichment_used") or row.get("enrichment_skills"):
        return "enriched"
    return "complete"


def outcome_label(row: dict[str, Any]) -> str:
    if is_refused(row):
        return "refused"
    if is_invalid_pddl(row):
        return "invalid_pddl"
    if is_invalid_plan(row):
        return "invalid_plan"
    if plan_found(row):
        return "planned"
    if str(row.get("exit_reason") or "") in {"fail", "abort"}:
        return "no_plan"
    return "error"


# ── Per-case scoring ─────────────────────────────────────────────────────────


def score_case(row: dict[str, Any]) -> dict[str, Any]:
    """Attach per-row scoring flags. Does not mix refuse into plan-success."""
    expect = row.get("expect") or {}
    arm_expect = expect.get(row.get("arm") or "select") or {}
    family = row.get("family") or ""
    phrasing = row.get("phrasing") or ""
    arm = row.get("arm") or "select"

    found = plan_found(row)
    refused = is_refused(row)
    template = row.get("domain_template")
    want_template = expect.get("template")
    domain_correct = (
        None if not want_template else template == want_template
    )
    want_complete = expect.get("completeness")
    completeness_correct = (
        None
        if not want_complete
        else row.get("domain_completeness") == want_complete
    )
    want_skill = expect.get("skill")
    want_skills = list(expect.get("skills") or [])
    skills = list(row.get("enrichment_skills") or [])
    have_skills = {_norm_skill(s) for s in skills}
    if want_skills:
        skill_correct = all(_norm_skill(s) in have_skills for s in want_skills)
    elif want_skill:
        skill_correct = _norm_skill(want_skill) in have_skills
    else:
        skill_correct = None
    gold = expect.get("gold_goal")
    soft_golds = list(expect.get("soft_golds") or [])
    required_init = expect.get("required_init")
    delivered = pddl_was_delivered(row)
    domain_text = row.get("pddl_domain")
    has_goal_spec = bool(gold) or bool(soft_golds)
    baseline = arm in _BASELINE_ARMS
    invalid_pddl = is_invalid_pddl(row)
    invalid_plan = is_invalid_plan(row)

    if baseline:
        domain_correct = None
        completeness_correct = None
        skill_correct = None

    if baseline:
        init_ok = None
        if arm == "llm_plan" or refused or invalid_pddl or invalid_plan or (
            not delivered and not found
        ):
            goal_ok = None
            domain_ok = None
            problem_ok = None
        else:
            goal_ok = (
                None
                if not has_goal_spec
                else goal_matches(row.get("goal_facts"), gold, soft_golds)
            )
            syntax_ok = bool(domain_text) and not pddl_syntax_errors(str(domain_text))
            fd_invoked = bool(domain_text) and str(
                row.get("exit_reason") or ""
            ) in {"planned", "fail"}
            domain_ok = bool(syntax_ok and fd_invoked)
            problem_ok = None if goal_ok is None else _not_false(goal_ok)
    elif refused or (not delivered and not found):
        init_ok = None
        goal_ok = None
        domain_ok = None
        problem_ok = None
    else:
        init_ok = (
            None if not required_init
            else facts_contain(row.get("init_facts"), required_init)
        )
        goal_ok = (
            None
            if not has_goal_spec
            else goal_matches(row.get("goal_facts"), gold, soft_golds)
        )
        template_ok = True if domain_correct is None else domain_correct
        if (
            arm in {"enrich", "r1"}
            and family in {"needs_enrichment", "multi_skill"}
            and want_skill
        ):
            domain_ok = bool(template_ok and domain_has_skill(domain_text, want_skill))
        elif arm == "r1" and want_skills:
            domain_ok = bool(
                template_ok
                and all(domain_has_skill(domain_text, s) for s in want_skills)
            )
        elif want_template or domain_text:
            domain_ok = bool(template_ok)
        else:
            domain_ok = None
        if init_ok is None and goal_ok is None:
            problem_ok = None
        else:
            problem_ok = _not_false(init_ok) and _not_false(goal_ok)

    covers = (
        plan_covers(row, expect.get("plan_actions"))
        if expect.get("plan_actions")
        else None
    )
    path = inferred_path(row)
    want_path = arm_expect.get("path")
    path_ok = None if not want_path else path == want_path

    plan_correct = False
    if found and not refused and family != "ungenerable":
        checks = [goal_ok, covers, domain_ok]
        if family == "needs_enrichment" and arm in {"enrich", "r1"}:
            checks.extend([skill_correct, path_ok])
        if family == "multi_skill" and arm == "r1":
            checks.extend([skill_correct, path_ok])
        plan_correct = all(_not_false(c) for c in checks)
    incorrect_plan = bool(found and not refused and not plan_correct)

    want_plan = arm_expect.get("plan")
    want_refuse = bool(arm_expect.get("refuse"))
    if baseline and not arm_expect:
        if family == "ungenerable":
            want_refuse = True
            want_plan = False
        else:
            want_plan = True
    if family == "ungenerable":
        case_ok = refused if want_refuse else (not found)
    elif family == "underspecified":
        case_ok = underspecified_case_ok(row, expect)
    elif want_plan is True:
        case_ok = plan_correct
        if path_ok is False:
            case_ok = False
    elif want_plan is False:
        case_ok = not found
    else:
        case_ok = True

    flags = {
        "outcome": outcome_label(row),
        "plan_found": found,
        "plan_correct": plan_correct,
        "incorrect_plan": incorrect_plan,
        "refused": refused,
        "false_plan": bool(family == "ungenerable" and found),
        "silent_fail": bool(
            family == "ungenerable"
            and (not found)
            and (not refused)
            and (not invalid_pddl)
            and (not invalid_plan)
        ),
        "invalid_pddl": invalid_pddl,
        "invalid_plan": invalid_plan,
        "domain_correct": domain_correct,
        "domain_ok": domain_ok,
        "init_ok": init_ok,
        "goal_ok": goal_ok,
        "goal_match": goal_ok,
        "problem_ok": problem_ok,
        "completeness_correct": completeness_correct,
        "skill_correct": skill_correct,
        "plan_covers": covers,
        "path": path,
        "path_ok": path_ok,
        "ok": case_ok,
        "family": family,
        "phrasing": phrasing,
        "affordances_declared": None,
        "affordances_in_precond": None,
        "init_assignment_ok": None,
    }
    if arm == "r1" and family in {"needs_enrichment", "multi_skill"} and not refused:
        authored = row.get("enrichment_authored") or {}
        pred_blob = " ".join(
            str(p) for p in (authored.get("new_predicates") or [])
        )
        pre_blob = " ".join(
            str(a.get("precondition") or "")
            for a in (authored.get("actions") or [])
            if isinstance(a, dict)
        )
        flags["affordances_declared"] = "can-" in pred_blob.lower()
        flags["affordances_in_precond"] = "can-" in pre_blob.lower()
        gold_aff = expect.get("affordances") or {}
        forbidden = expect.get("affordances_forbidden") or []
        if gold_aff or forbidden:
            init_set = {
                tuple(str(x) for x in fact)
                for fact in (row.get("init_facts") or [])
                if isinstance(fact, (list, tuple))
            }
            ok_ass = True
            for pred, objs in gold_aff.items():
                for obj in objs:
                    if (str(pred), str(obj)) not in init_set:
                        ok_ass = False
            for fact in forbidden:
                if tuple(str(x) for x in fact) in init_set:
                    ok_ass = False
            flags["init_assignment_ok"] = ok_ass
        else:
            flags["init_assignment_ok"] = None
    if family == "ungenerable":
        if refused:
            flags["verdict"] = f"refused as expected ({(row.get('refuse_reason') or '')[:80]})"
        elif found:
            flags["verdict"] = "FALSE PLAN on ungenerable task"
        elif invalid_pddl:
            flags["verdict"] = "invalid_pddl on ungenerable task"
        elif invalid_plan:
            flags["verdict"] = "invalid_plan on ungenerable task"
        else:
            flags["verdict"] = "no plan, but did not refuse (silent fail)"
    elif family == "underspecified":
        if refused:
            flags["verdict"] = "plausible refuse (underspecified)"
        elif case_ok and found:
            flags["verdict"] = "plausible plan (underspecified)"
        elif found:
            flags["verdict"] = "spurious plan on underspecified task"
        elif invalid_pddl:
            flags["verdict"] = "invalid_pddl"
        elif invalid_plan:
            flags["verdict"] = "invalid_plan"
        else:
            flags["verdict"] = "no plan, but did not refuse (silent fail)"
    elif found:
        extra = []
        if domain_ok is False:
            extra.append(f"domain {template} ≠ {want_template or 'gold'}")
        if init_ok is False:
            extra.append(":init missing required facts")
        if goal_ok is False:
            extra.append(":goal ≠ gold")
        if covers is False:
            extra.append("plan missing gold actions")
        if skill_correct is False:
            extra.append(f"skill {skills or 'none'} ≠ {want_skill}")
        flags["verdict"] = (
            ("incorrect plan" if incorrect_plan else "planned")
            + (f" — {'; '.join(extra)}" if extra else "")
        )
    elif invalid_pddl:
        flags["verdict"] = "invalid_pddl"
    elif invalid_plan:
        flags["verdict"] = "invalid_plan"
    else:
        flags["verdict"] = f"no plan ({row.get('exit_reason') or 'unknown'})"
    return flags


def attach_score(row: dict[str, Any]) -> dict[str, Any]:
    row.update(score_case(row))
    return row


# ── Aggregates (the three claims) ────────────────────────────────────────────


def _pct(ok: int, n: int) -> float | None:
    if n <= 0:
        return None
    return round(100.0 * ok / n, 1)


def _slice_rows(
    rows: list[dict[str, Any]],
    *,
    arm: str | None = None,
    family: str | None = None,
    families: tuple[str, ...] | None = None,
    phrasing: str | None = None,
) -> list[dict[str, Any]]:
    out = rows
    if arm is not None:
        out = [r for r in out if r.get("arm") == arm]
    if family is not None:
        out = [r for r in out if r.get("family") == family]
    if families is not None:
        out = [r for r in out if r.get("family") in families]
    if phrasing is not None:
        out = [r for r in out if r.get("phrasing") == phrasing]
    return out


def _count_true(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for r in rows if r.get(key) is True)


def _phrasing_block(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    """Rate ``key is True`` over rows where ``key`` is not None."""
    block: dict[str, Any] = {}
    for label, subset in (
        ("all", rows),
        ("explicit", _slice_rows(rows, phrasing="explicit")),
        ("implicit", _slice_rows(rows, phrasing="implicit")),
    ):
        defined = [r for r in subset if r.get(key) is not None]
        n = len(defined)
        ok = _count_true(defined, key)
        block[label] = {"n": n, "ok": ok, "pct": _pct(ok, n)}
    return block


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Claim 1: select arm, template_complete — plan_correct (not claim 2).
    Claim 2: enrich arm, needs_enrichment only — not the sum of claim 1.
    Claim 3: ungenerable refuse (both arms), never folded into 1/2.
    """
    claims: dict[str, Any] = {}

    select_complete = _slice_rows(
        rows, arm="select", family="template_complete"
    )
    enrich_complete = _slice_rows(
        rows, arm="enrich", family="template_complete"
    )
    enrich_gap = _slice_rows(
        rows, arm="enrich", family="needs_enrichment"
    )
    select_gap = _slice_rows(
        rows, arm="select", family="needs_enrichment"
    )
    claims["1_select_only"] = {
        "population": "select × template_complete",
        "plan_correct": _phrasing_block(select_complete, "plan_correct"),
        "plan_found": _phrasing_block(select_complete, "plan_found"),
        "incorrect_plan": _phrasing_block(select_complete, "incorrect_plan"),
        "domain_correct": _phrasing_block(select_complete, "domain_correct"),
        "domain_ok": _phrasing_block(select_complete, "domain_ok"),
        "init_ok": _phrasing_block(select_complete, "init_ok"),
        "goal_ok": _phrasing_block(select_complete, "goal_ok"),
        "problem_ok": _phrasing_block(select_complete, "problem_ok"),
        "completeness_correct": _phrasing_block(
            select_complete, "completeness_correct"
        ),
        "ablation_needs_enrichment_plan_found": _phrasing_block(
            select_gap, "plan_found"
        ),
    }

    claims["2_enrichment"] = {
        "population": "enrich × needs_enrichment",
        "plan_correct": _phrasing_block(enrich_gap, "plan_correct"),
        "plan_found": _phrasing_block(enrich_gap, "plan_found"),
        "incorrect_plan": _phrasing_block(enrich_gap, "incorrect_plan"),
        "path_ok": _phrasing_block(enrich_gap, "path_ok"),
        "skill_correct": _phrasing_block(enrich_gap, "skill_correct"),
        "domain_correct": _phrasing_block(enrich_gap, "domain_correct"),
        "domain_ok": _phrasing_block(enrich_gap, "domain_ok"),
        "init_ok": _phrasing_block(enrich_gap, "init_ok"),
        "goal_ok": _phrasing_block(enrich_gap, "goal_ok"),
        "problem_ok": _phrasing_block(enrich_gap, "problem_ok"),
        "side_template_complete_plan_correct": _phrasing_block(
            enrich_complete, "plan_correct"
        ),
        "side_template_complete_domain_correct": _phrasing_block(
            enrich_complete, "domain_correct"
        ),
    }

    r1_gap = _slice_rows(rows, arm="r1", family="needs_enrichment")
    r1_multi = _slice_rows(rows, arm="r1", family="multi_skill")
    r1_long = _slice_rows(rows, arm="r1", family="long_sequence")
    claims["r1"] = {
        "population": "r1 × needs_enrichment / long_sequence",
        "plan_correct": _phrasing_block(r1_gap, "plan_correct"),
        "plan_found": _phrasing_block(r1_gap, "plan_found"),
        "affordances_declared": _phrasing_block(r1_gap, "affordances_declared"),
        "affordances_in_precond": _phrasing_block(r1_gap, "affordances_in_precond"),
        "init_assignment_ok": _phrasing_block(r1_gap, "init_assignment_ok"),
        "multi_skill_plan_correct": _phrasing_block(r1_multi, "plan_correct"),
        "long_sequence_plan_correct": _phrasing_block(r1_long, "plan_correct"),
    }

    claim3: dict[str, Any] = {}
    for arm in _ARMS + (_R1_ARM,):
        ungen = _slice_rows(rows, arm=arm, family="ungenerable")
        claim3[arm] = {
            "refuse": _phrasing_block(ungen, "refused"),
            "false_plan": _phrasing_block(ungen, "false_plan"),
            "silent_fail": _phrasing_block(ungen, "silent_fail"),
            "domain_correct": _phrasing_block(ungen, "domain_correct"),
        }
    claims["3_ungenerable"] = claim3

    under = [r for r in rows if r.get("family") == "underspecified"]
    claims["underspecified"] = {
        "population": "underspecified (not Claim 1–3)",
        "ok": _phrasing_block(under, "ok"),
        "plan_found": _phrasing_block(under, "plan_found"),
        "refuse": _phrasing_block(under, "refused"),
    }

    # Template accuracy across every case that declares expect.template,
    # sliced by arm — independent of plan success.
    domain_select: dict[str, Any] = {}
    for arm in _ARMS:
        domain_select[arm] = _phrasing_block(
            _slice_rows(rows, arm=arm), "domain_correct"
        )
    domain_select["all"] = _phrasing_block(rows, "domain_correct")
    claims["domain_select"] = domain_select

    baseline: dict[str, Any] = {}
    for arm in _BASELINE_ARMS:
        arm_rows = _slice_rows(rows, arm=arm)
        if not arm_rows:
            continue
        complete = _slice_rows(arm_rows, family="template_complete")
        gap = _slice_rows(arm_rows, family="needs_enrichment")
        ungen = _slice_rows(arm_rows, family="ungenerable")
        plan_keys = {
            "plan_correct": _phrasing_block(complete, "plan_correct"),
            "plan_found": _phrasing_block(complete, "plan_found"),
            "incorrect_plan": _phrasing_block(complete, "incorrect_plan"),
            "invalid_pddl": _phrasing_block(complete, "invalid_pddl"),
            "invalid_plan": _phrasing_block(complete, "invalid_plan"),
            "refuse": _phrasing_block(complete, "refused"),
        }
        gap_keys = {
            "plan_correct": _phrasing_block(gap, "plan_correct"),
            "plan_found": _phrasing_block(gap, "plan_found"),
            "incorrect_plan": _phrasing_block(gap, "incorrect_plan"),
            "invalid_pddl": _phrasing_block(gap, "invalid_pddl"),
            "invalid_plan": _phrasing_block(gap, "invalid_plan"),
            "refuse": _phrasing_block(gap, "refused"),
        }
        baseline[arm] = {
            "template_complete": {
                "population": f"{arm} × template_complete",
                **plan_keys,
            },
            "needs_enrichment": {
                "population": f"{arm} × needs_enrichment",
                **gap_keys,
            },
            "ungenerable": {
                "refuse": _phrasing_block(ungen, "refused"),
                "false_plan": _phrasing_block(ungen, "false_plan"),
                "silent_fail": _phrasing_block(ungen, "silent_fail"),
                "invalid_pddl": _phrasing_block(ungen, "invalid_pddl"),
                "invalid_plan": _phrasing_block(ungen, "invalid_plan"),
            },
        }
    claims["baseline"] = baseline
    return claims


_FAMILY_ORDER = (
    "template_complete",
    "needs_enrichment",
    "ungenerable",
    "long_sequence",
    "underspecified",
)

_FAMILY_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "template_complete": (
        "plan_correct",
        "plan_found",
        "incorrect_plan",
        "domain_correct",
    ),
    "needs_enrichment": (
        "plan_correct",
        "plan_found",
        "incorrect_plan",
        "path_ok",
        "skill_correct",
    ),
    "ungenerable": ("refused", "false_plan", "silent_fail"),
    "long_sequence": ("plan_correct", "plan_found", "incorrect_plan"),
    "underspecified": ("ok", "plan_found", "refused"),
}


def _arm_order(rows: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in rows:
        arm = str(row.get("arm") or "")
        if arm and arm not in seen:
            seen.append(arm)
    return seen


def aggregate_families(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-family × arm × phrasing rates. Not pooled across settings."""
    present = {str(r.get("family") or "") for r in rows if r.get("family")}
    families = [f for f in _FAMILY_ORDER if f in present]
    families += sorted(present - set(_FAMILY_ORDER))
    arms = _arm_order(rows)
    by_family: dict[str, Any] = {}
    for family in families:
        keys = _FAMILY_METRIC_KEYS.get(family, ("ok", "plan_found"))
        block: dict[str, Any] = {}
        for arm in arms:
            subset = _slice_rows(rows, arm=arm, family=family)
            if not subset:
                continue
            metrics = {key: _phrasing_block(subset, key) for key in keys}
            if arm == _R1_ARM and family == "needs_enrichment":
                metrics["affordances_declared"] = _phrasing_block(
                    subset, "affordances_declared"
                )
                metrics["affordances_in_precond"] = _phrasing_block(
                    subset, "affordances_in_precond"
                )
                metrics["init_assignment_ok"] = _phrasing_block(
                    subset, "init_assignment_ok"
                )
            block[arm] = metrics
        by_family[family] = block
    return {
        "by_family": by_family,
        "arms": arms,
        "families": families,
    }


def _seconds_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "sum": 0.0,
        }
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "n": n,
        "mean": round(sum(ordered) / n, 3),
        "median": round(median, 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
        "sum": round(sum(ordered), 3),
    }


def _row_timings(row: dict[str, Any]) -> dict[str, Any]:
    return from_payload(row.get("timings"))


def _timing_row_fields(timings: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = from_payload(timings)
    return {
        "timings": payload,
        "llm_s": payload["llm_s"],
        "fd_s": payload["fd_s"],
        "llm_calls": payload["llm_calls"],
    }


def _timing_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    llm_totals = [float(_row_timings(r)["llm_s"] or 0.0) for r in rows]
    walls = [float(r.get("wall_s") or 0.0) for r in rows]
    fd_totals = [
        float(t["fd_s"])
        for r in rows
        if (t := _row_timings(r))["fd_s"] is not None
    ]
    by_name: dict[str, list[float]] = {}
    for row in rows:
        for call in _row_timings(row).get("llm_calls") or []:
            by_name.setdefault(str(call.get("name") or "llm"), []).append(
                float(call["s"])
            )
    return {
        "n": len(rows),
        "llm_s": _seconds_stats(llm_totals),
        "fd_s": _seconds_stats(fd_totals),
        "wall_s": _seconds_stats(walls),
        "by_name": {name: _seconds_stats(vals) for name, vals in sorted(by_name.items())},
    }


def aggregate_timings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-arm and family×arm LLM / Fast Downward seconds. Not pooled across settings."""
    arms = _arm_order(rows)
    present = {str(r.get("family") or "") for r in rows if r.get("family")}
    families = [f for f in _FAMILY_ORDER if f in present]
    families += sorted(present - set(_FAMILY_ORDER))
    by_arm = {arm: _timing_block(_slice_rows(rows, arm=arm)) for arm in arms}
    by_family: dict[str, Any] = {}
    for family in families:
        block: dict[str, Any] = {}
        for arm in arms:
            subset = _slice_rows(rows, arm=arm, family=family)
            if subset:
                block[arm] = _timing_block(subset)
        by_family[family] = block
    return {
        "all": _timing_block(rows),
        "by_arm": by_arm,
        "by_family": by_family,
        "arms": arms,
        "families": families,
    }


def format_pct(cell: dict[str, Any] | None) -> str:
    if not cell or cell.get("n") == 0:
        return "—"
    pct = cell.get("pct")
    if pct is None:
        return "—"
    return f"{pct:.1f}% ({cell['ok']}/{cell['n']})"


def format_seconds(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def format_llm_split(row: dict[str, Any]) -> str:
    calls = list((_row_timings(row).get("llm_calls") or []))
    if not calls:
        return "—"
    return "; ".join(
        f"{c.get('name') or 'llm'}={float(c['s']):.2f}" for c in calls
    )


def _flag_mark(value: Any) -> str:
    if value is True:
        return "OK"
    if value is False:
        return "FAIL"
    return "—"


def _plan_mark(row: dict[str, Any]) -> str:
    if row.get("incorrect_plan"):
        return "incorrect"
    if row.get("plan_correct"):
        return "correct"
    if row.get("plan_found"):
        return "found"
    if row.get("refused"):
        return "—"
    return "none"


def format_claims_tables(claims: dict[str, Any]) -> str:
    lines = [
        "## Claim 1 — domain select only (no enrichment)",
        "",
        "Population: **select × template_complete**. "
        "Claim 2 gap tasks are not included.",
        "",
        "### Plan",
        "",
        "| slice | plan correct | plan found | incorrect plan |",
        "|---|---|---|---|",
    ]
    c1 = claims["1_select_only"]
    for slice_id in ("all", "explicit", "implicit"):
        lines.append(
            f"| {slice_id} | "
            f"{format_pct(c1['plan_correct'][slice_id])} | "
            f"{format_pct(c1['plan_found'][slice_id])} | "
            f"{format_pct(c1['incorrect_plan'][slice_id])} |"
        )
    lines += [
        "",
        "### PDDL delivered to Fast Downward",
        "",
        "| slice | domain select | domain ok | :init ok | :goal ok | problem ok |",
        "|---|---|---|---|---|---|",
    ]
    for slice_id in ("all", "explicit", "implicit"):
        lines.append(
            f"| {slice_id} | "
            f"{format_pct(c1['domain_correct'][slice_id])} | "
            f"{format_pct(c1['domain_ok'][slice_id])} | "
            f"{format_pct(c1['init_ok'][slice_id])} | "
            f"{format_pct(c1['goal_ok'][slice_id])} | "
            f"{format_pct(c1['problem_ok'][slice_id])} |"
        )
    lines += [
        "",
        "Completeness (same population): "
        + format_pct(c1["completeness_correct"]["all"]),
        "",
        "Ablation (select × needs_enrichment — should stay low; not claim 1): "
        + format_pct(c1["ablation_needs_enrichment_plan_found"]["all"]),
        "",
        "## Claim 2 — catalog enrichment (gap tasks only)",
        "",
        "Population: **enrich × needs_enrichment**. "
        "Does **not** include claim 1 template-complete tasks.",
        "",
        "### Plan",
        "",
        "| slice | plan correct | plan found | incorrect plan | path ok | skill correct |",
        "|---|---|---|---|---|---|",
    ]
    c2 = claims["2_enrichment"]
    for slice_id in ("all", "explicit", "implicit"):
        lines.append(
            f"| {slice_id} | "
            f"{format_pct(c2['plan_correct'][slice_id])} | "
            f"{format_pct(c2['plan_found'][slice_id])} | "
            f"{format_pct(c2['incorrect_plan'][slice_id])} | "
            f"{format_pct(c2['path_ok'][slice_id])} | "
            f"{format_pct(c2['skill_correct'][slice_id])} |"
        )
    lines += [
        "",
        "### PDDL delivered to Fast Downward",
        "",
        "| slice | domain select | domain ok | :init ok | :goal ok | problem ok |",
        "|---|---|---|---|---|---|",
    ]
    for slice_id in ("all", "explicit", "implicit"):
        lines.append(
            f"| {slice_id} | "
            f"{format_pct(c2['domain_correct'][slice_id])} | "
            f"{format_pct(c2['domain_ok'][slice_id])} | "
            f"{format_pct(c2['init_ok'][slice_id])} | "
            f"{format_pct(c2['goal_ok'][slice_id])} | "
            f"{format_pct(c2['problem_ok'][slice_id])} |"
        )
    lines += [
        "",
        "Side check (enrich × template_complete — not claim 2): "
        + format_pct(c2["side_template_complete_plan_correct"]["all"])
        + " plan correct; "
        + format_pct(c2["side_template_complete_domain_correct"]["all"])
        + " domain select",
        "",
        "## Claim 3 — ungenerable recognition (not mixed into 1/2)",
        "",
        "| arm | slice | refuse | false plan | silent fail | domain select |",
        "|---|---|---|---|---|---|",
    ]
    c3 = claims["3_ungenerable"]
    for arm in _ARMS:
        if arm not in c3:
            continue
        for slice_id in ("all", "explicit", "implicit"):
            block = c3[arm]
            lines.append(
                f"| {arm} | {slice_id} | "
                f"{format_pct(block['refuse'][slice_id])} | "
                f"{format_pct(block['false_plan'][slice_id])} | "
                f"{format_pct(block['silent_fail'][slice_id])} | "
                f"{format_pct(block['domain_correct'][slice_id])} |"
            )
    ds = claims.get("domain_select") or {}
    lines += [
        "",
        "## Domain selection (template = gold, all families)",
        "",
        "Independent of plan success. "
        "`domain ok` above may also require the enrichment skill in the PDDL file; "
        "this table is **template identity only**.",
        "",
        "| arm | slice | domain select |",
        "|---|---|---|",
    ]
    for arm in ("all",) + _ARMS:
        block = ds.get(arm)
        if not block:
            continue
        for slice_id in ("all", "explicit", "implicit"):
            lines.append(
                f"| {arm} | {slice_id} | {format_pct(block[slice_id])} |"
            )
    under_md = format_underspecified_tables(claims.get("underspecified") or {})
    if under_md:
        lines += ["", under_md]
    baseline_md = format_baseline_tables(claims.get("baseline") or {})
    if baseline_md:
        lines += ["", baseline_md]
    r1_md = format_r1_tables(claims.get("r1") or {})
    if r1_md:
        lines += ["", r1_md]
    return "\n".join(lines)


def format_underspecified_tables(block: dict[str, Any]) -> str:
    """Open-intent family — never folded into Claim 1–3."""
    ok = (block.get("ok") or {}).get("all") or {}
    if not ok.get("n"):
        return ""
    lines = [
        "## Underspecified (not Claim 1–3)",
        "",
        "Population: **underspecified**. "
        "Pass = refuse or a plan in the case allowlist; fail = spurious skill.",
        "",
        "| slice | ok | plan found | refuse |",
        "|---|---|---|---|",
    ]
    for slice_id in ("all", "explicit", "implicit"):
        lines.append(
            f"| {slice_id} | "
            f"{format_pct((block.get('ok') or {}).get(slice_id))} | "
            f"{format_pct((block.get('plan_found') or {}).get(slice_id))} | "
            f"{format_pct((block.get('refuse') or {}).get(slice_id))} |"
        )
    return "\n".join(lines)


def format_r1_tables(r1: dict[str, Any]) -> str:
    """Suite v2 affordance metrics — never folded into Claim 1–3."""
    found = (r1.get("plan_found") or {}).get("all") or {}
    multi = (r1.get("multi_skill_plan_correct") or {}).get("all") or {}
    long = (r1.get("long_sequence_plan_correct") or {}).get("all") or {}
    if not found.get("n") and not multi.get("n") and not long.get("n"):
        return ""
    lines = [
        "## R1 — affordance enrichment (not Claim 1–3)",
        "",
        "Population: **r1 × needs_enrichment** (gap skills). "
        "Long-sequence is an extra slice. "
        "R0 is not scored here.",
        "",
        "| slice | plan correct | plan found | affordances declared | "
        "in precondition | init assignment |",
        "|---|---|---|---|---|---|",
    ]
    for slice_id in ("all", "explicit", "implicit"):
        lines.append(
            f"| {slice_id} | "
            f"{format_pct((r1.get('plan_correct') or {}).get(slice_id))} | "
            f"{format_pct((r1.get('plan_found') or {}).get(slice_id))} | "
            f"{format_pct((r1.get('affordances_declared') or {}).get(slice_id))} | "
            f"{format_pct((r1.get('affordances_in_precond') or {}).get(slice_id))} | "
            f"{format_pct((r1.get('init_assignment_ok') or {}).get(slice_id))} |"
        )
    extras = []
    if multi.get("n"):
        extras.append(
            "Multi-skill plan correct: "
            + format_pct(r1.get("multi_skill_plan_correct", {}).get("all"))
        )
    extras.append(
        "Long-sequence plan correct: "
        + format_pct(r1.get("long_sequence_plan_correct", {}).get("all"))
    )
    lines += ["", *extras]
    return "\n".join(lines)


def format_baseline_tables(baseline: dict[str, Any]) -> str:
    """Separate sections — never folded into Claim 1 / Claim 2."""
    if not baseline:
        return ""
    lines = [
        "## Baseline arms (not Claim 1 / Claim 2)",
        "",
        "`domain_correct` is null (no template identity). "
        "`init_ok` is null (LLM `:init` is not the hybrid gold). "
        "These populations are **not** added to select/enrich numerators.",
        "",
    ]
    for arm in _BASELINE_ARMS:
        block = baseline.get(arm)
        if not block:
            continue
        for family, title in (
            ("template_complete", f"{arm} × template_complete"),
            ("needs_enrichment", f"{arm} × needs_enrichment"),
        ):
            pop = block.get(family) or {}
            lines += [
                f"### {title}",
                "",
                "| slice | plan correct | plan found | incorrect plan | invalid_pddl | refuse |",
                "|---|---|---|---|---|---|",
            ]
            for slice_id in ("all", "explicit", "implicit"):
                lines.append(
                    f"| {slice_id} | "
                    f"{format_pct(pop['plan_correct'][slice_id])} | "
                    f"{format_pct(pop['plan_found'][slice_id])} | "
                    f"{format_pct(pop['incorrect_plan'][slice_id])} | "
                    f"{format_pct(pop['invalid_pddl'][slice_id])} | "
                    f"{format_pct(pop.get('refuse', {}).get(slice_id))} |"
                )
            lines.append("")
    lines += [
        "### Claim 3 — ungenerable (baseline arms)",
        "",
        "| arm | slice | refuse | false plan | silent fail | invalid_pddl | invalid_plan |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in _BASELINE_ARMS:
        block = (baseline.get(arm) or {}).get("ungenerable")
        if not block:
            continue
        for slice_id in ("all", "explicit", "implicit"):
            lines.append(
                f"| {arm} | {slice_id} | "
                f"{format_pct(block['refuse'][slice_id])} | "
                f"{format_pct(block['false_plan'][slice_id])} | "
                f"{format_pct(block['silent_fail'][slice_id])} | "
                f"{format_pct(block['invalid_pddl'][slice_id])} | "
                f"{format_pct(block.get('invalid_plan', {}).get(slice_id))} |"
            )
    return "\n".join(lines).rstrip()


def format_case_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| arm | case | phrasing | family | outcome | plan | tmpl | domain | init | goal | ok | llm_s | llm split | fd_s | verdict |",
        "|---|---|---|---|---|---|:---:|:---:|:---:|:---:|:---:|---:|---|---:|---|",
    ]
    for r in rows:
        mark = "OK" if r.get("ok") else "FAIL"
        timings = _row_timings(r)
        lines.append(
            f"| {r.get('arm')} | {r.get('id')} | {r.get('phrasing')} | "
            f"{r.get('family')} | {r.get('outcome')} | {_plan_mark(r)} | "
            f"{_flag_mark(r.get('domain_correct'))} | {_flag_mark(r.get('domain_ok'))} | "
            f"{_flag_mark(r.get('init_ok'))} | {_flag_mark(r.get('goal_ok'))} | {mark} | "
            f"{format_seconds(timings.get('llm_s'))} | {format_llm_split(r)} | "
            f"{format_seconds(timings.get('fd_s'))} | "
            f"{r.get('verdict', '')} |"
        )
    return "\n".join(lines)


def _md_cell(value: Any) -> str:
    text = "—" if value in (None, "", []) else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip() or "—"


def format_intermediate_report(rows: list[dict[str, Any]]) -> str:
    """Slim per-row table: authored actions + FD plan + verdict."""
    lines = [
        "| id | Task | arm | actions | new_predicates | Fd_action | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        actions = authored_action_names(r)
        preds = authored_predicates(r)
        fd = [str(x) for x in (r.get("fd_actions") or [])]
        lines.append(
            "| "
            + " | ".join(
                _md_cell(v)
                for v in (
                    r.get("id"),
                    r.get("task"),
                    r.get("arm"),
                    ", ".join(actions),
                    "; ".join(preds),
                    "; ".join(fd),
                    r.get("verdict"),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def format_enrichment_authored_table(rows: list[dict[str, Any]]) -> str:
    """Compact table of authored enrichment actions (claim 2 diagnostics)."""
    lines = [
        "| case | skills | action | effect | predicates | reused |",
        "|---|---|---|---|---|:---:|",
    ]
    any_row = False
    for r in rows:
        authored = r.get("enrichment_authored") or {}
        actions = list(authored.get("actions") or [])
        skills = ",".join(authored.get("skills") or r.get("enrichment_skills") or []) or "—"
        preds = "; ".join(authored.get("new_predicates") or []) or "—"
        reused = "Y" if authored.get("reused") else ("—" if not authored else "N")
        if not actions and not authored:
            lines.append(
                f"| {r.get('id')} | {skills} | — | — | {preds} | {reused} |"
            )
            any_row = True
            continue
        if not actions:
            lines.append(
                f"| {r.get('id')} | {skills} | (no action) | — | {preds} | {reused} |"
            )
            any_row = True
            continue
        for i, act in enumerate(actions):
            name = act.get("name") or "—"
            effect = act.get("effect") or "—"
            case_id = r.get("id") if i == 0 else ""
            skill_cell = skills if i == 0 else ""
            pred_cell = preds if i == 0 else ""
            reused_cell = reused if i == 0 else ""
            lines.append(
                f"| {case_id} | {skill_cell} | `{name}` | `{effect}` | "
                f"{pred_cell} | {reused_cell} |"
            )
            any_row = True
    if not any_row:
        return "_No enrichment-authored rows._"
    return "\n".join(lines)


def format_family_tables(stats: dict[str, Any]) -> str:
    """Markdown tables keyed by command family, then arm, then phrasing."""
    by_family = stats.get("by_family") or {}
    if not by_family:
        return ""
    titles = {
        "template_complete": "Template-complete (pick / place / look-at / stack)",
        "needs_enrichment": "Needs enrichment (catalog gap)",
        "ungenerable": "Ungenerable (refuse)",
        "long_sequence": "Long sequence (template only)",
        "underspecified": "Underspecified (open intent)",
    }
    lines: list[str] = ["## Families", ""]
    for family in stats.get("families") or by_family:
        block = by_family.get(family) or {}
        if not block:
            continue
        lines += [f"### {titles.get(family, family)}", ""]
        first_arm = next(iter(block), None)
        keys = list((block.get(first_arm) or {}).keys()) if first_arm else []
        header = "| arm | slice | " + " | ".join(k.replace("_", " ") for k in keys) + " |"
        sep = "|" + "|".join(["---"] * (2 + len(keys))) + "|"
        lines += [header, sep]
        for arm, metrics in block.items():
            for slice_id in ("all", "explicit", "implicit"):
                cells = [format_pct((metrics.get(k) or {}).get(slice_id)) for k in keys]
                lines.append(f"| {arm} | {slice_id} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_timing_tables(stats: dict[str, Any]) -> str:
    """Markdown: aggregated LLM (split + total) and Fast Downward seconds."""
    if not stats:
        return ""
    lines = [
        "## Timings",
        "",
        "Seconds of wall clock. **llm_s** is the sum of text-LLM calls on that "
        "case; **llm split** names each call. **fd_s** is Fast Downward search "
        "time on arms that invoke the planner (`llm_pddl`, `enrich`, `r1`). "
        "`llm_plan` never calls Fast Downward. Means are over cases; stage "
        "means are over individual calls.",
        "",
        "### By arm",
        "",
        "| arm | n | mean llm_s | median llm_s | mean fd_s | median fd_s | mean wall_s |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in stats.get("arms") or []:
        block = (stats.get("by_arm") or {}).get(arm) or {}
        llm = block.get("llm_s") or {}
        fd = block.get("fd_s") or {}
        wall = block.get("wall_s") or {}
        lines.append(
            f"| {arm} | {block.get('n', 0)} | "
            f"{format_seconds(llm.get('mean'))} | {format_seconds(llm.get('median'))} | "
            f"{format_seconds(fd.get('mean'))} | {format_seconds(fd.get('median'))} | "
            f"{format_seconds(wall.get('mean'))} |"
        )
    lines += [
        "",
        "### By family × arm",
        "",
        "| family | arm | n | mean llm_s | median llm_s | mean fd_s | median fd_s |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    by_family = stats.get("by_family") or {}
    for family in stats.get("families") or by_family:
        block = by_family.get(family) or {}
        for arm, metrics in block.items():
            llm = metrics.get("llm_s") or {}
            fd = metrics.get("fd_s") or {}
            lines.append(
                f"| {family} | {arm} | {metrics.get('n', 0)} | "
                f"{format_seconds(llm.get('mean'))} | {format_seconds(llm.get('median'))} | "
                f"{format_seconds(fd.get('mean'))} | {format_seconds(fd.get('median'))} |"
            )
    lines += [
        "",
        "### By LLM stage",
        "",
        "| arm | stage | n_calls | mean_s | median_s | sum_s |",
        "|---|---|---:|---:|---:|---:|",
    ]
    any_stage = False
    for arm in stats.get("arms") or []:
        by_name = ((stats.get("by_arm") or {}).get(arm) or {}).get("by_name") or {}
        for name, cell in by_name.items():
            any_stage = True
            lines.append(
                f"| {arm} | {name} | {cell.get('n', 0)} | "
                f"{format_seconds(cell.get('mean'))} | {format_seconds(cell.get('median'))} | "
                f"{format_seconds(cell.get('sum'))} |"
            )
    if not any_stage:
        lines.append("| — | — | 0 | — | — | — |")
    return "\n".join(lines).rstrip()


def format_report_md_families(
    context: str,
    stats: dict[str, Any],
    rows: list[dict[str, Any]],
    timings: dict[str, Any] | None = None,
) -> str:
    sections = [
        context,
        "",
        format_family_tables(stats),
        "",
    ]
    timing_md = format_timing_tables(timings or {})
    if timing_md:
        sections += [timing_md, ""]
    for family in stats.get("families") or []:
        subset = _slice_rows(rows, family=family)
        if not subset:
            continue
        sections += [
            f"## Cases — {family}",
            "",
            format_case_table(subset),
            "",
        ]
        if family == "needs_enrichment":
            authored = [
                r
                for r in subset
                if r.get("arm") in {"enrich", _R1_ARM}
            ]
            if authored:
                sections += [
                    "### Enrichment authored",
                    "",
                    format_enrichment_authored_table(authored),
                    "",
                ]
    return "\n".join(sections)


def format_report_md(
    context: str,
    claims: dict[str, Any],
    rows: list[dict[str, Any]],
    timings: dict[str, Any] | None = None,
) -> str:
    claim2_rows = _slice_rows(rows, arm="enrich", family="needs_enrichment")
    sections = [
        context,
        "",
        format_claims_tables(claims),
        "",
    ]
    timing_md = format_timing_tables(timings or {})
    if timing_md:
        sections += [timing_md, ""]
    sections += [
        "## Cases — Claim 1 (select × template_complete)",
        "",
        format_case_table(
            _slice_rows(rows, arm="select", family="template_complete")
        ),
        "",
        "## Cases — Claim 2 (enrich × needs_enrichment)",
        "",
        format_case_table(claim2_rows),
        "",
        "### Enrichment authored (claim 2)",
        "",
        format_enrichment_authored_table(claim2_rows),
        "",
        "## Cases — Claim 3 (ungenerable)",
        "",
        format_case_table(_slice_rows(rows, family="ungenerable")),
        "",
        "## Outside claims (ablation / enrich-on-complete)",
        "",
        format_case_table(
            [
                r
                for r in rows
                if (r.get("arm") == "select" and r.get("family") == "needs_enrichment")
                or (r.get("arm") == "enrich" and r.get("family") == "template_complete")
            ]
        ),
        "",
    ]
    baseline_rows = [r for r in rows if r.get("arm") in _BASELINE_ARMS]
    if baseline_rows:
        sections += [
            "## Cases — Baseline arms",
            "",
            "_Not Claim 1 / Claim 2. `domain_correct` and `init_ok` are null._",
            "",
        ]
        for arm in _BASELINE_ARMS:
            for family in ("template_complete", "needs_enrichment"):
                subset = _slice_rows(baseline_rows, arm=arm, family=family)
                if not subset:
                    continue
                sections += [
                    f"### {arm} × {family}",
                    "",
                    format_case_table(subset),
                    "",
                ]
        ungen = _slice_rows(baseline_rows, family="ungenerable")
        if ungen:
            sections += [
                "### Claim 3 — ungenerable (baseline)",
                "",
                format_case_table(ungen),
                "",
            ]
    under_rows = _slice_rows(rows, family="underspecified")
    if under_rows:
        sections += [
            "## Cases — Underspecified (not Claim 1–3)",
            "",
            format_case_table(under_rows),
            "",
        ]
    r1_rows = [
        r
        for r in _slice_rows(rows, arm="r1")
        if r.get("family") != "underspecified"
    ]
    if r1_rows:
        sections += [
            "## Cases — R1 (affordance enrichment, not Claim 1–3)",
            "",
            format_case_table(r1_rows),
            "",
            "### Enrichment authored (r1)",
            "",
            format_enrichment_authored_table(r1_rows),
            "",
        ]
    return "\n".join(sections)


# ── Dry / live rows ──────────────────────────────────────────────────────────


def _dry_domain_text(
    template: str | None,
    *,
    skill: str | None = None,
    skills: list[str] | None = None,
) -> str:
    actions = list(DOMAIN_PDDL_ACTIONS.get(template or "", frozenset()))
    extra = list(skills or [])
    if skill:
        extra.append(skill)
    for name in extra:
        pddl_name = catalog_to_pddl_action(name)
        if pddl_name not in actions:
            actions.append(pddl_name)
    body = "\n".join(f"  (:action {name})" for name in actions)
    return f"(define (domain dry)\n{body}\n)\n"


def _dry_baseline_domain_text(plan_actions: list[str]) -> str:
    """Minimal domain that passes ``pddl_syntax_errors`` (not a repo template)."""
    names: list[str] = []
    for a in plan_actions or ["pick"]:
        pddl_name = catalog_to_pddl_action(a)
        if pddl_name not in names:
            names.append(pddl_name)
    actions = "\n".join(
        "  (:action {name}\n"
        "    :parameters ()\n"
        "    :precondition (dummy)\n"
        "    :effect (dummy)\n"
        "  )".format(name=name)
        for name in names
    )
    return (
        "(define (domain dry-baseline)\n"
        "  (:requirements :strips)\n"
        "  (:predicates (dummy))\n"
        f"{actions}\n"
        ")\n"
    )


def _dry_baseline_row(case: dict[str, Any], arm: str) -> dict[str, Any]:
    """Ideal stub for llm_pddl / llm_plan — does not use suite expect.<arm> keys."""
    expect = case.get("expect") or {}
    family = case["family"]
    ungen = family == "ungenerable"
    want_plan = not ungen
    plan_actions = list(expect.get("plan_actions") or []) if want_plan else []
    gold = list(expect.get("gold_goal") or []) if want_plan else []
    prims = [{"name": a, "args": []} for a in plan_actions]
    is_pddl = arm == "llm_pddl"
    domain_text = (
        _dry_baseline_domain_text(plan_actions) if want_plan and is_pddl else None
    )
    row: dict[str, Any] = {
        "id": case["id"],
        "task": case["task"],
        "note": case.get("note", ""),
        "arm": arm,
        "phrasing": case["phrasing"],
        "family": family,
        "world": case.get("world"),
        "expect": expect,
        "exit_code": 3 if ungen else 0,
        "wall_s": 0.0,
        "run_dir": None,
        "exit_reason": "refused" if ungen else "planned",
        "success": False,
        "plan_only": True,
        "enrichment_used": False,
        "enrichment_skills": [],
        "enrichment_authored": None,
        "domain_persisted": None,
        "domain_reused": False,
        "domain_select_backend": None,
        "refuse_reason": "dry refuse" if ungen else None,
        "domain_template": None,
        "domain_completeness": None,
        "needed_skills": [],
        "fd_actions": [f"({a} dry)" for a in plan_actions],
        "fd_primitives": prims,
        "n_plan_actions": len(plan_actions),
        "pddl_problem": (
            "(define (problem dry) (:domain dry-baseline) (:init) (:goal (and)))"
            if want_plan and is_pddl
            else None
        ),
        "pddl_init": None,
        "pddl_goal": None,
        "pddl_domain": domain_text,
        "init_facts": [],
        "goal_facts": gold if is_pddl else [],
        "case_dir": None,
        "n_steps": 0,
        "replan_count": 0,
    }
    row.update(_timing_row_fields(empty_timings()))
    return attach_score(row)


def _dry_row(case: dict[str, Any], arm: str) -> dict[str, Any]:
    if arm in _BASELINE_ARMS:
        return _dry_baseline_row(case, arm)
    expect = case.get("expect") or {}
    arm_expect = expect.get(arm) or {}
    family = case["family"]
    want_plan = bool(arm_expect.get("plan"))
    want_refuse = bool(arm_expect.get("refuse"))
    path = arm_expect.get("path")
    if family == "ungenerable" or want_refuse:
        path = "refused"
        want_plan = False
    elif path is None:
        if arm in {"enrich", "r1"} and family in {"needs_enrichment", "multi_skill"}:
            path = "enriched"
        else:
            path = (
                "enriched"
                if arm == "enrich" and family == "needs_enrichment"
                else "complete"
            )

    skill = expect.get("skill") if path == "enriched" else None
    skills_list = list(expect.get("skills") or ([skill] if skill else []))
    if arm == "r1" and path == "enriched" and not skills_list and skill:
        skills_list = [skill]
    plan_actions = list(expect.get("plan_actions") or []) if want_plan else []
    prims = [{"name": a, "args": []} for a in plan_actions]
    gold = list(expect.get("gold_goal") or []) if want_plan else []
    required = list(expect.get("required_init") or []) if want_plan else []
    template = expect.get("template", "manipulation_base")
    domain_text = (
        None
        if path == "refused" or not want_plan
        else _dry_domain_text(template, skill=skill, skills=skills_list)
    )
    aff = dict(expect.get("affordances") or {})
    init_facts = list(required)
    if arm == "r1" and path == "enriched":
        for pred, objs in aff.items():
            for obj in objs:
                init_facts.append([pred, obj])
    authored = None
    if path == "enriched" and (skill or skills_list):
        names = skills_list or ([skill] if skill else [])
        new_preds = [f"({p} ?x - item)" for p in aff] or (
            ["(can-pour ?x - item)"] if arm == "r1" else []
        )
        authored = {
            "skills": names,
            "actions": [
                {
                    "name": n,
                    "parameters": "",
                    "precondition": "(and (can-x ?x))" if arm == "r1" else "",
                    "effect": "",
                }
                for n in names
            ],
            "new_predicates": new_preds,
            "new_types": [],
            "domain_path": None,
            "reused": False,
            "init_facts": [list(f) for f in init_facts if f and str(f[0]).startswith("can-")],
        }

    row: dict[str, Any] = {
        "id": case["id"],
        "task": case["task"],
        "note": case.get("note", ""),
        "arm": arm,
        "phrasing": case["phrasing"],
        "family": family,
        "world": case.get("world"),
        "expect": expect,
        "exit_code": 3 if path == "refused" else 0,
        "wall_s": 0.0,
        "run_dir": None,
        "exit_reason": "refused" if path == "refused" else ("planned" if want_plan else "fail"),
        "success": False,
        "plan_only": True,
        "enrichment_used": path == "enriched",
        "enrichment_skills": skills_list if path == "enriched" else [],
        "enrichment_authored": authored,
        "domain_persisted": None,
        "domain_reused": False,
        "domain_select_backend": "llm",
        "refuse_reason": "dry refuse" if path == "refused" else None,
        "domain_template": template,
        "domain_completeness": expect.get("completeness"),
        "needed_skills": skills_list if expect.get("completeness") == "incomplete" else [],
        "fd_actions": [f"({a} dry)" for a in plan_actions],
        "fd_primitives": prims,
        "n_plan_actions": len(plan_actions),
        "pddl_problem": "(define (problem dry))" if want_plan else None,
        "pddl_init": None,
        "pddl_goal": None,
        "pddl_domain": domain_text,
        "init_facts": init_facts,
        "goal_facts": gold,
        "case_dir": None,
        "n_steps": 0,
        "replan_count": 0,
    }
    row.update(_timing_row_fields(empty_timings()))
    return attach_score(row)


def build_loop_cmd(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    arm: str,
    python_bin: str,
    container: str,
    max_steps: int | None,
    scene_source: str | None,
    perception_only: bool | None,
    mock: bool = False,
    mock_init: bool = False,
    mock_llm: bool = False,
) -> list[str]:
    world = case.get("world") or defaults.get("world") or "tabletop"
    hybrid = case.get("hybrid") or defaults.get("hybrid") or "mvp"
    control = case.get("control") or defaults.get("control") or "fd"
    goal_backend = (
        case.get("goal_backend") or defaults.get("goal_backend") or "local_llm"
    )
    domain_select = (
        case.get("domain_select") or defaults.get("domain_select") or "llm"
    )
    steps = max_steps or int(case.get("max_steps") or defaults.get("max_steps") or 1)
    src = scene_source or case.get("scene_source") or defaults.get("scene_source") or "dino"
    perc = perception_only
    if perc is None:
        perc = bool(case.get("perception_only", defaults.get("perception_only", True)))

    if arm in _BASELINE_SCRIPTS:
        cmd = [
            python_bin,
            str(_REPO_ROOT / "scripts" / _BASELINE_SCRIPTS[arm]),
            "--task",
            case["task"],
            "--world",
            world,
            "--scene-source",
            src,
            "--plan-only",
            "--container",
            container,
        ]
        if perc:
            cmd.append("--perception-only")
        if mock or mock_init:
            cmd.extend(
                [
                    "--mock-scene",
                    "--scene-file",
                    str(baseline_scene_file(world)),
                ]
            )
            if mock or mock_llm:
                cmd.append("--mock-llm")
            if arm == "llm_pddl" and (mock or mock_llm):
                cmd.append("--mock-fd")
        return cmd

    cmd = [
        python_bin,
        str(_REPO_ROOT / "scripts" / "run_loop_host.py"),
        "--task",
        case["task"],
        "--world",
        world,
        "--hybrid",
        hybrid,
        "--control",
        control,
        "--goal-backend",
        goal_backend,
        "--domain-select",
        domain_select,
        "--scene-source",
        src,
        "--max-steps",
        str(steps),
        "--container",
        container,
        "--plan-only",
    ]
    if perc:
        cmd.append("--perception-only")
    if arm == "enrich":
        cmd.extend(["--online-enrichment", "1"])
    if arm == "r1":
        cmd.extend(["--online-enrichment", "1", "--enrichment-profile", "r1"])
    if mock_init:
        cmd.extend(
            [
                "--mock-scene",
                "--scene-file",
                str(baseline_scene_file(world)),
            ]
        )
        if mock_llm:
            cmd.extend(["--mock-llm", "--mock-fd"])
    return cmd


def resolve_run_settings(
    defaults: dict[str, Any],
    *,
    scene_source: str | None,
    no_perception_only: bool,
) -> dict[str, Any]:
    """CLI-effective loop settings (thesis default remains dino + perception-only)."""
    src = scene_source or defaults.get("scene_source") or "dino"
    perc = False if no_perception_only else bool(
        defaults.get("perception_only", True)
    )
    return {"scene_source": src, "perception_only": perc}


def arm_loads_llm_in_parent(arm: str, *, mock_llm: bool = False) -> bool:
    """Baselines keep one 7B in the battery process; host arms spawn children."""
    return arm in _BASELINE_ARMS and not mock_llm


def release_parent_text_llm(*, reason: str) -> None:
    """
    Free parent-process weights before a host arm (enrich/r1) loads its own.

    ``llm_plan``/``llm_pddl`` share one in-process client. Without this, the
    child ``run_loop_host.py`` hits ``device_map=auto`` CPU offload.
    """
    from planner.text_llm import reset_shared_text_client

    print(f"[EVAL] releasing parent text LLM ({reason})", flush=True)
    reset_shared_text_client()


def format_run_context(settings: dict[str, Any]) -> str:
    src = settings.get("scene_source") or "?"
    perc = bool(settings.get("perception_only"))
    lines = [
        f"Context: scene_source={src}, perception_only={str(perc).lower()}. "
        "Thesis default is dino + perception-only. "
        "Oracle `:init` ablation: `--scene-source oracle --no-perception-only`."
    ]
    extra = str(settings.get("live_notes") or "").strip()
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines)


def row_from_summary(
    case: dict[str, Any],
    arm: str,
    *,
    result: subprocess.CompletedProcess,
    wall: float,
    summary: dict[str, Any],
    summary_path: Path | None,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    hybrid = summary.get("hybrid_problem_gen") or {}
    artifacts = load_pddl_artifacts(summary, summary_path)
    goal_facts = hybrid.get("goal_facts") or hybrid.get("last_goal_facts") or []
    if not goal_facts:
        goal_facts = artifacts.get("pddl_goal_facts") or []
    case_dir = None
    if out_dir is not None:
        case_dir = copy_case_pddl_artifacts(
            out_dir,
            arm,
            str(case.get("id") or "unknown"),
            summary_path=summary_path,
            artifacts=artifacts,
        )
    row: dict[str, Any] = {
        "id": case["id"],
        "task": case["task"],
        "note": case.get("note", ""),
        "arm": arm,
        "phrasing": case["phrasing"],
        "family": case["family"],
        "world": case.get("world") or summary.get("world"),
        "expect": case.get("expect") or {},
        "exit_code": int(result.returncode),
        "wall_s": wall,
        "run_dir": (
            _display(summary_path.parent)
            if summary_path is not None
            else None
        ),
        "case_dir": case_dir,
        "exit_reason": summary.get("exit_reason"),
        "success": bool(summary.get("success")),
        "plan_only": bool(summary.get("plan_only")),
        "enrichment_used": bool(summary.get("enrichment_used")),
        "enrichment_skills": list(summary.get("enrichment_skills") or []),
        "enrichment_authored": summary.get("enrichment_authored"),
        "domain_persisted": summary.get("domain_persisted"),
        "domain_reused": summary.get("domain_reused"),
        "domain_select_backend": summary.get("domain_select_backend"),
        "refuse_reason": summary.get("refuse_reason"),
        "domain_template": summary.get("domain_template"),
        "domain_completeness": summary.get("domain_completeness"),
        "needed_skills": list(summary.get("needed_skills") or []),
        "fd_actions": list(summary.get("fd_actions") or []),
        "fd_primitives": list(summary.get("fd_primitives") or []),
        "n_plan_actions": int(summary.get("n_plan_actions") or 0),
        "pddl_problem": artifacts.get("pddl_problem"),
        "pddl_init": artifacts.get("pddl_init"),
        "pddl_goal": artifacts.get("pddl_goal"),
        "pddl_domain": artifacts.get("pddl_domain"),
        "init_facts": list(artifacts.get("init_facts") or []),
        "goal_facts": list(goal_facts),
        "n_steps": int(summary.get("n_steps") or 0),
        "replan_count": int(summary.get("replan_count") or 0),
    }
    row.update(_timing_row_fields(summary.get("timings")))
    return attach_score(row)


_BASELINE_MODS: dict[str, Any] = {}


def _load_baseline_script(arm: str) -> Any:
    if arm in _BASELINE_MODS:
        return _BASELINE_MODS[arm]
    script = _BASELINE_SCRIPTS[arm]
    path = _REPO_ROOT / "scripts" / script
    spec = importlib.util.spec_from_file_location(f"eval_mock_{arm}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _BASELINE_MODS[arm] = mod
    return mod


def format_live_baseline_notes(model: str | None) -> str:
    mid = model or "default (1.5B)"
    return (
        "## Live notes (not a paper default)\n"
        "\n"
        f"Text LLM: `{mid}` (live weights, no NL decision cache).\n"
        "Compact scene: live DINO via ``planner.live_scene.acquire_live_scene`` "
        "(same pre-scan / capture / sweep as ``run_loop_host``). "
        "`--mock-scene` remains CI/offline only.\n"
        "`--scene-source dino --perception-only` are the thesis flags. "
        "These tables are **not** a product default.\n"
        "Fast Downward (`llm_pddl`): host `FastDownwardPlanner` if "
        "`fast-downward` is on PATH, else `docker exec` of "
        "`_fd_solve_problem.py` with the **LLM domain override** "
        "(empty domain is refused — no template file).\n"
    )


def ensure_fd_container(container: str) -> str | None:
    """Start the ROS image if needed so Fast Downward is available."""
    inspect = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
    )
    if inspect.returncode != 0:
        return (
            f"container {container!r} not found. "
            "Build/start once: (cd docker && docker compose up -d)"
        )
    if inspect.stdout.strip() == "true":
        return None
    start = subprocess.run(
        ["docker", "start", container],
        capture_output=True,
        text=True,
    )
    if start.returncode != 0:
        return start.stderr.strip() or f"docker start {container} failed"
    return None


def audit_baseline_template_fallback(rows: list[dict[str, Any]]) -> list[str]:
    """E5 done-when: enrichment_used false; no official template as FD domain."""
    issues: list[str] = []
    for row in rows:
        if row.get("arm") not in _BASELINE_ARMS:
            continue
        cid = f"{row.get('arm')}/{row.get('id')}"
        if row.get("enrichment_used"):
            issues.append(f"{cid}: enrichment_used is true")
        tmpl = row.get("domain_template")
        if tmpl in _OFFICIAL_TEMPLATES:
            issues.append(f"{cid}: domain_template={tmpl!r} (official template)")
    return issues


def run_baseline_inprocess(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    arm: str,
    scene_source: str | None,
    perception_only: bool | None,
    out_dir: Path,
    generate_fn: Any,
    fd_planner: Any | None = None,
    mock_scene: bool = False,
    container: str | None = None,
) -> dict[str, Any]:
    """
    Run llm_pddl / llm_plan in-process (shared generate_fn; no NL cache).

    Suite task strings are passed verbatim. Live DINO unless ``mock_scene``.
    """
    world = case.get("world") or defaults.get("world") or "tabletop"
    src = scene_source or case.get("scene_source") or defaults.get("scene_source") or "dino"
    perc = perception_only
    if perc is None:
        perc = bool(case.get("perception_only", defaults.get("perception_only", True)))
    run_dir = out_dir / "runs" / f"{arm}_{case['id']}"
    argv = [
        "--task",
        case["task"],
        "--world",
        world,
        "--scene-source",
        src,
        "--plan-only",
        "--run-dir",
        str(run_dir),
    ]
    if perc:
        argv.append("--perception-only")
    ctr = container or str(defaults.get("container") or "vlm_ros2")
    argv.extend(["--container", ctr])
    if mock_scene:
        argv.extend(
            [
                "--mock-scene",
                "--scene-file",
                str(baseline_scene_file(world)),
            ]
        )
    mod = _load_baseline_script(arm)
    t0 = time.time()
    if arm == "llm_plan":
        code = mod.main(argv, generate_fn=generate_fn)
    else:
        kwargs: dict[str, Any] = {"generate_fn": generate_fn}
        if fd_planner is not None:
            kwargs["fd_planner"] = fd_planner
        code = mod.main(argv, **kwargs)
    wall = round(time.time() - t0, 1)
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    result = subprocess.CompletedProcess(args=argv, returncode=int(code))
    return row_from_summary(
        case,
        arm,
        result=result,
        wall=wall,
        summary=summary,
        summary_path=summary_path if summary_path.is_file() else None,
        out_dir=out_dir,
    )


def run_mock_case(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    arm: str,
    scene_source: str | None,
    perception_only: bool | None,
    out_dir: Path,
) -> dict[str, Any]:
    """
    Run llm_pddl / llm_plan through the real scripts with mock generate_fn.

    Suite task strings are passed verbatim. No GPU, no NL decision cache.
    """
    from planner.baselines.mock_llm import (
        MockFastDownward,
        generate_pddl_mock,
        generate_plan_mock,
    )

    gen = generate_plan_mock if arm == "llm_plan" else generate_pddl_mock
    fd = MockFastDownward() if arm == "llm_pddl" else None
    return run_baseline_inprocess(
        case,
        defaults,
        arm=arm,
        scene_source=scene_source,
        perception_only=perception_only,
        out_dir=out_dir,
        generate_fn=gen,
        fd_planner=fd,
        mock_scene=True,
    )


def run_case(
    case: dict[str, Any],
    defaults: dict[str, Any],
    *,
    arm: str,
    python_bin: str,
    container: str,
    max_steps: int | None,
    scene_source: str | None,
    perception_only: bool | None,
    env: dict[str, str],
    out_dir: Path | None = None,
    mock_init: bool = False,
    mock_llm: bool = False,
) -> dict[str, Any]:
    cmd = build_loop_cmd(
        case,
        defaults,
        arm=arm,
        python_bin=python_bin,
        container=container,
        max_steps=max_steps,
        scene_source=scene_source,
        perception_only=perception_only,
        mock_init=mock_init,
        mock_llm=mock_llm,
    )
    run_dir_override: Path | None = None
    if arm in _BASELINE_ARMS and out_dir is not None:
        run_dir_override = out_dir / "runs" / f"{arm}_{case.get('id') or 'unknown'}"
        cmd.extend(["--run-dir", str(run_dir_override)])
    print()
    print(f"[RUN]  arm={arm}  {' '.join(shlex.quote(c) for c in cmd)}")
    print("─" * 72)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env)
    wall = round(time.time() - t0, 1)
    print("─" * 72)

    summary_path: Path | None
    if run_dir_override is not None:
        candidate = run_dir_override / "summary.json"
        summary_path = candidate if candidate.is_file() else None
    else:
        summary_path = enrich_live.newest_summary_after(t0)
    summary: dict[str, Any] = {}
    if summary_path is not None:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] could not read {summary_path}: {exc}")

    return row_from_summary(
        case,
        arm,
        result=result,
        wall=wall,
        summary=summary,
        summary_path=summary_path,
        out_dir=out_dir,
    )


def banner(case: dict[str, Any], arm: str, index: int, total: int) -> None:
    expect = case.get("expect") or {}
    print()
    print("╔" + "═" * 70 + "╗")
    print(f"║  {index}/{total}  arm={arm:<8} {case['id']:<48} ║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  task     : {case['task'][:56]:<56} ║")
    print(
        f"║  phrasing : {case.get('phrasing', '?'):<10} "
        f"family={case.get('family', '?'):<20} ║"
    )
    print(
        f"║  expect   : template={str(expect.get('template') or '—'):<24} "
        f"skill={str(expect.get('skill') or '—'):<12} ║"
    )
    print("╚" + "═" * 70 + "╝")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Planning-only battery: select vs enrich, explicit vs implicit, "
            "ungenerable refuse scored separately. Claim 2 is not the sum of claim 1."
        )
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=_DEFAULT_SUITE,
        help=f"Suite JSON (default {_DEFAULT_SUITE.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids (default: all)",
    )
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "Comma-separated arms: select,enrich,r1,llm_pddl,llm_plan "
            "(default: select,enrich — or suite defaults.arms / v2 mock arms)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Text LLM id (sets {_TEXT_LLM_ENV}). Local HuggingFace id or, "
            "with VLMRP_TEXT_LLM_BASE_URL, the provider model name."
        ),
    )
    parser.add_argument("--container", default="vlm_ros2")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--scene-source",
        default=None,
        choices=["fused", "dino", "oracle"],
        help="Override suite default (dino).",
    )
    parser.add_argument(
        "--no-perception-only",
        action="store_true",
        help="Allow NameMatch / oracle snap (default is perception-only).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds between cases (0=off)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Wait for Enter between cases",
    )
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Skip /reset_world between cases",
    )
    parser.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Do not require the orchestrator topic before starting",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Score ideal summaries only — no Docker / Gazebo",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help=(
            "Offline baseline mock: run llm_pddl/llm_plan with canned "
            "generate_fn (suite v1 texts verbatim, no GPU). Default arms "
            "become llm_pddl,llm_plan."
        ),
    )
    parser.add_argument(
        "--mock-init",
        action="store_true",
        help=(
            "Perception off: oracle_mock fixture as SceneState for every arm "
            "(select/enrich/r1 and baselines). No Gazebo. Combine with "
            "--mock-llm for CI; without it uses the live text LLM + host FD."
        ),
    )
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help=(
            "Canned generate_fn on host / baseline loops. Use with --mock-init "
            "for a GPU-free v2 battery."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Report directory (default {_DEFAULT_OUT.relative_to(_REPO_ROOT)}/<ts>)",
    )
    args = parser.parse_args()

    suite = load_suite(args.suite)
    defaults = dict(suite.get("defaults") or {})
    case_ids = (
        [c.strip() for c in args.cases.split(",") if c.strip()]
        if args.cases
        else None
    )
    cases = select_cases(suite, case_ids)
    if args.mock and args.dry:
        print("[FAIL] use either --mock or --dry, not both")
        return 2
    if args.mock_init and args.dry:
        print("[FAIL] use either --mock-init or --dry, not both")
        return 2
    if args.mock and args.mock_init:
        print("[FAIL] use either --mock (baselines canned) or --mock-init")
        return 2

    suite_arms = defaults.get("arms")
    if args.arms is not None:
        arms = parse_arms(args.arms)
    elif args.mock:
        arms = list(_BASELINE_ARMS)
    elif isinstance(suite_arms, (list, tuple)) and suite_arms:
        arms = parse_arms(",".join(str(a) for a in suite_arms))
    elif args.mock_init or any(
        tag in Path(args.suite).name for tag in ("v2", "v3", "v4")
    ):
        arms = list(_V2_DEFAULT_ARMS)
    else:
        arms = parse_arms(None)

    if args.mock:
        extra = [a for a in arms if a not in _BASELINE_ARMS]
        if extra:
            print(
                f"[FAIL] --mock only supports llm_pddl,llm_plan (not {extra}; "
                "use --mock-init for enrich/r1)"
            )
            return 2
    if not cases:
        print("[FAIL] no cases selected")
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out_dir or (_DEFAULT_OUT / ts)
    out.mkdir(parents=True, exist_ok=True)

    model = args.model or os.environ.get(_TEXT_LLM_ENV)
    python_bin = str(_REPO_ROOT / ".venv" / "bin" / "python")
    if not Path(python_bin).exists():
        python_bin = sys.executable
    perception_only = False if args.no_perception_only else None
    run_settings = resolve_run_settings(
        defaults,
        scene_source=args.scene_source,
        no_perception_only=bool(args.no_perception_only),
    )
    total = len(cases) * len(arms)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  Planning battery — perception + FD, no execution            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  suite   : {_display(args.suite)}")
    print(f"  cases   : {', '.join(c['id'] for c in cases)}")
    print(f"  arms    : {', '.join(arms)}")
    print(f"  model   : {model or 'default (1.5B — weak for PDDL authoring)'}")
    print(f"  out     : {_display(out)}")
    print(
        f"  scene   : {run_settings['scene_source']}  "
        f"perception_only={str(run_settings['perception_only']).lower()}"
    )
    print("  scoring : plan_correct ≠ plan_found; families stay separate; split explicit / implicit")
    print()

    rows: list[dict[str, Any]] = []
    if args.dry:
        n = 0
        for arm in arms:
            for case in cases:
                n += 1
                banner(case, arm, n, total)
                row = _dry_row(case, arm)
                rows.append(row)
                mark = "OK  " if row["ok"] else "FAIL"
                print(f"[{mark}] {row['verdict']}")
    elif args.mock:
        n = 0
        for arm in arms:
            for case in cases:
                n += 1
                banner(case, arm, n, total)
                row = run_mock_case(
                    case,
                    defaults,
                    arm=arm,
                    scene_source=args.scene_source,
                    perception_only=perception_only,
                    out_dir=out,
                )
                rows.append(row)
                mark = "OK  " if row["ok"] else "FAIL"
                print(f"[{mark}] {arm}/{row['id']}: {row['verdict']}")
                if row.get("run_dir"):
                    print(f"       run → {row['run_dir']}/summary.json")
    elif args.mock_init:
        n = 0
        env = dict(os.environ)
        if model:
            env[_TEXT_LLM_ENV] = model
            os.environ[_TEXT_LLM_ENV] = model
        run_settings["scene_mode"] = "mock_init"
        run_settings["paper_default"] = False
        run_settings["live_notes"] = (
            "mock-init: oracle_mock fixture, no Gazebo/DINO. "
            + ("canned LLM+FD. " if args.mock_llm else "live text LLM. ")
        )
        if not args.mock_llm:
            err = ensure_fd_container(args.container)
            if err:
                print(f"[FAIL] Fast Downward container: {err}")
                print(f"       Start it: docker start {args.container}")
                return 2
            print(
                f"[EVAL] FD: host `fast-downward` if on PATH, else "
                f"docker exec {args.container} (no Gazebo)"
            )
            run_settings["live_notes"] += (
                f"FD via host PATH or docker exec {args.container}. "
            )
        parent_llm = False
        try:
            for arm in arms:
                uses_parent = arm_loads_llm_in_parent(
                    arm, mock_llm=bool(args.mock_llm)
                )
                if parent_llm and not uses_parent:
                    release_parent_text_llm(
                        reason=f"{arm} runs in a child process"
                    )
                parent_llm = uses_parent
                for case in cases:
                    n += 1
                    banner(case, arm, n, total)
                    if arm in _BASELINE_ARMS and args.mock_llm:
                        row = run_mock_case(
                            case,
                            defaults,
                            arm=arm,
                            scene_source=args.scene_source,
                            perception_only=perception_only,
                            out_dir=out,
                        )
                    elif arm in _BASELINE_ARMS:
                        from planner.text_llm import get_shared_text_client

                        client = get_shared_text_client(
                            model, max_new_tokens=_LIVE_MAX_NEW_TOKENS
                        )
                        row = run_baseline_inprocess(
                            case,
                            defaults,
                            arm=arm,
                            scene_source=args.scene_source,
                            perception_only=perception_only,
                            out_dir=out,
                            generate_fn=client.complete,
                            mock_scene=True,
                            container=args.container,
                        )
                    else:
                        row = run_case(
                            case,
                            defaults,
                            arm=arm,
                            python_bin=python_bin,
                            container=args.container,
                            max_steps=args.max_steps,
                            scene_source=args.scene_source,
                            perception_only=perception_only,
                            env=env,
                            out_dir=out,
                            mock_init=True,
                            mock_llm=bool(args.mock_llm),
                        )
                    rows.append(row)
                    mark = "OK  " if row["ok"] else "FAIL"
                    print(f"[{mark}] {arm}/{row['id']}: {row['verdict']}")
                    if row.get("run_dir"):
                        print(f"       run → {row['run_dir']}/summary.json")
        finally:
            if parent_llm:
                release_parent_text_llm(reason="end of mock-init run")
    else:
        env = dict(os.environ)
        if model:
            env[_TEXT_LLM_ENV] = model
            os.environ[_TEXT_LLM_ENV] = model

        if not args.skip_ready_check and not enrich_live.orchestrator_ready(
            args.container
        ):
            print(
                f"[FAIL] Orchestrator not ready in container {args.container!r}.\n"
                "       Start the sim first, e.g.:\n"
                "         bin/eval_planning_battery.sh\n"
                "       or:\n"
                "         bin/reset_and_test_fd.sh --world tabletop\n"
                "         python scripts/eval_planning_battery.py …"
            )
            return 2

        if any(a in _BASELINE_ARMS for a in arms):
            run_settings["live_notes"] = format_live_baseline_notes(model)
            run_settings["scene_mode"] = "live_dino"
            run_settings["paper_default"] = False
            if "llm_pddl" in arms:
                err = ensure_fd_container(args.container)
                if err:
                    print(f"[FAIL] Fast Downward container: {err}")
                    return 2
                print(
                    f"[EVAL] FD via docker exec {args.container} "
                    "(LLM domain override, no template fallback)"
                )

        generate_fn = None
        parent_llm = False
        current_world: str | None = defaults.get("world")
        n = 0
        try:
            for arm in arms:
                uses_parent = arm_loads_llm_in_parent(arm)
                if parent_llm and not uses_parent:
                    generate_fn = None
                    release_parent_text_llm(
                        reason=f"{arm} runs in a child process"
                    )
                if uses_parent and generate_fn is None:
                    from planner.text_llm import get_shared_text_client

                    print(
                        f"[EVAL] loading live text LLM once for {arm} "
                        f"({model or 'default'}, "
                        f"max_new_tokens={_LIVE_MAX_NEW_TOKENS})…"
                    )
                    client = get_shared_text_client(
                        model, max_new_tokens=_LIVE_MAX_NEW_TOKENS
                    )
                    generate_fn = client.complete
                parent_llm = uses_parent
                for case in cases:
                    n += 1
                    banner(case, arm, n, total)
                    wanted_world = (
                        case.get("world") or defaults.get("world") or "tabletop"
                    )
                    current_world = enrich_live.ensure_world(
                        wanted_world, args.container, current_world
                    )
                    if not args.no_reset:
                        enrich_live.reset_world(args.container)
                    if arm in _BASELINE_ARMS:
                        row = run_baseline_inprocess(
                            case,
                            defaults,
                            arm=arm,
                            scene_source=args.scene_source,
                            perception_only=perception_only,
                            out_dir=out,
                            generate_fn=generate_fn,
                            mock_scene=False,
                            container=args.container,
                        )
                    else:
                        row = run_case(
                            case,
                            defaults,
                            arm=arm,
                            python_bin=python_bin,
                            container=args.container,
                            max_steps=args.max_steps,
                            scene_source=args.scene_source,
                            perception_only=perception_only,
                            env=env,
                            out_dir=out,
                        )
                    rows.append(row)
                    mark = "OK  " if row["ok"] else "FAIL"
                    print(f"\n[{mark}] {arm}/{row['id']}: {row['verdict']}")
                    if row.get("run_dir"):
                        print(f"       run → {row['run_dir']}/summary.json")
                    remaining = total - n
                    enrich_live.pause_between(args, remaining=remaining)
        except KeyboardInterrupt:
            print("\n[EVAL] interrupted — writing partial report")
        finally:
            if parent_llm:
                release_parent_text_llm(reason="end of live run")

    families = aggregate_families(rows)
    timings = aggregate_timings(rows)
    report_style = str(defaults.get("report_style") or "claims")
    claims = None if report_style == "families" else aggregate(rows)
    passed = sum(1 for r in rows if r.get("ok"))
    fallback_issues = audit_baseline_template_fallback(rows)
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dry": bool(args.dry),
        "mock": bool(args.mock),
        "mock_init": bool(getattr(args, "mock_init", False)),
        "model": model,
        "text_llm_base_url": os.environ.get("VLMRP_TEXT_LLM_BASE_URL") or None,
        "text_llm_timeout_s": os.environ.get("VLMRP_TEXT_LLM_TIMEOUT_S") or None,
        "campaign_profile": os.environ.get("VLMRP_CAMPAIGN_PROFILE") or None,
        "suite": _display(args.suite),
        "arms": arms,
        "defaults": defaults,
        "scene_source": run_settings["scene_source"],
        "perception_only": run_settings["perception_only"],
        "scene_mode": run_settings.get("scene_mode"),
        "paper_default": run_settings.get("paper_default"),
        "template_fallback_issues": fallback_issues,
        "passed": passed,
        "total": len(rows),
        "families": families,
        "timings": timings,
        "cases": rows,
    }
    if claims is not None:
        report["claims"] = claims
    report_path = out / "report.json"
    context = format_run_context(run_settings)
    timing_tables = format_timing_tables(timings)
    if report_style == "families":
        tables = format_family_tables(families)
        report_md = format_report_md_families(context, families, rows, timings)
    else:
        assert claims is not None
        tables = format_claims_tables(claims)
        report_md = format_report_md(context, claims, rows, timings)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "report.md").write_text(
        report_md,
        encoding="utf-8",
    )
    (out / "report_cases.md").write_text(
        format_intermediate_report(rows) + "\n",
        encoding="utf-8",
    )

    print()
    print(tables)
    if timing_tables:
        print()
        print(timing_tables)
    print()
    print(format_case_table(rows))
    print(f"\n[EVAL] {passed}/{len(rows)} cases matched gold expect")
    if fallback_issues:
        print("[EVAL] TEMPLATE FALLBACK detected (not allowed on baselines):")
        for issue in fallback_issues:
            print(f"       {issue}")
    else:
        baseline_n = sum(1 for r in rows if r.get("arm") in _BASELINE_ARMS)
        if baseline_n:
            print(
                f"[EVAL] template-fallback audit clean "
                f"({baseline_n} baseline summaries)"
            )
    print(f"[EVAL] report → {_display(report_path)}")
    print(f"[EVAL] tables → {_display(out / 'report.md')}")
    print(f"[EVAL] cases  → {_display(out / 'report_cases.md')}")
    if fallback_issues:
        return 2
    return 0 if rows and passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
