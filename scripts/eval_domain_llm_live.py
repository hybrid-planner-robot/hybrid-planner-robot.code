#!/usr/bin/env python3
"""
Probe the single text LLM on the real closed-catalog enrichment path.

Sessions 29–30 are covered offline by a mocked ``generate_fn``; those tests
prove the plumbing, not that a real model can do the job. This script runs the
same selector / enricher against live weights and reports what actually came
back: which template, complete or incomplete, which catalog skill, and whether
the authored PDDL survived validation, merge and persistence.

No Gazebo, no VLM, no motion — just the text model, so prompts can be iterated
in seconds instead of sim runs. The full loop is
``run_loop_host.py --online-enrichment 1 --domain-select llm``.

Examples
--------
Default text LLM (``Qwen/Qwen2.5-1.5B-Instruct``) on GPU::

    python scripts/eval_domain_llm_live.py

A larger model, one env var — as designed in §3.3.1::

    python scripts/eval_domain_llm_live.py --model Qwen/Qwen2.5-7B-Instruct

Harness check without weights (canned replies, always passes)::

    python scripts/eval_domain_llm_live.py --mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from planner.enrichment_effects import (  # noqa: E402
    EnrichmentContext,
    ground_enrichment_goal,
)
from planner.online_enrichment import (  # noqa: E402
    DomainCompleteness,
    EnrichmentStatus,
    SelectionBackend,
    make_domain_selector,
    make_goal_binder,
    make_online_enricher,
    resolve_domain_for_task,
)
from planner.text_llm import resolve_text_llm_model_id  # noqa: E402

_OUT_DIR = _REPO_ROOT / "data" / "domain_llm_live"

# Scene symbols mirror the kitchen world (`can`, `cup`, `glass`, `spoon`, …),
# which is the only shipped world with pourable vessels.
_KITCHEN = ("can", "cup", "glass", "spoon", "knife", "plate", "counter", "table")
_TABLETOP = ("red_cup", "blue_box", "wood_cube", "table", "shelf_b")

# expect: what a correct answer looks like, so the report scores itself.
CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "paraphrase_drink",
        "task": "get me something to drink",
        "symbols": _KITCHEN,
        "expect": {"completeness": "incomplete", "skill": "pour"},
        "note": "The word 'pour' never appears and no vessel is named.",
    },
    {
        "id": "paraphrase_thirsty",
        "task": "i am thirsty, can you help",
        "symbols": _KITCHEN,
        "expect": {"completeness": "incomplete", "skill": "pour"},
        "note": "Same gap, stated as a condition rather than a request.",
    },
    {
        "id": "explicit_pour",
        "task": "pour the can into the glass",
        "symbols": _KITCHEN,
        "expect": {"completeness": "incomplete", "skill": "pour"},
        "note": "Literal phrasing — the keyword fallback also gets this one.",
    },
    {
        "id": "paraphrase_stir",
        "task": "the sugar has settled at the bottom of the cup",
        "symbols": _KITCHEN,
        "expect": {"completeness": "incomplete", "skill": "stir"},
        "note": "Second catalog skill, so the report is not pour-only.",
    },
    {
        "id": "complete_place",
        "task": "place the wood_cube on the shelf_b",
        "symbols": _TABLETOP,
        "expect": {"completeness": "complete"},
        "note": "Fixed domain suffices — the enricher must not be called. Prefer wood_cube for live motion.",
    },
    {
        "id": "complete_stack",
        "task": "stack the wood_cube on the blue_box",
        "symbols": _TABLETOP,
        "expect": {"completeness": "complete", "template": "manipulation_stacking"},
        "note": "Template choice as well as completeness. Grasp target is wood_cube.",
    },
    {
        "id": "refuse_solder",
        "task": "solder the broken wire on the board",
        "symbols": _KITCHEN,
        "expect": {"refuse": True},
        "note": "Real gap, no catalog skill can close it → refuse, never invent.",
    },
    {
        "id": "refuse_levitate",
        "task": "make the cup float in the air",
        "symbols": _KITCHEN,
        "expect": {"refuse": True},
        "note": "Physically impossible for this robot.",
    },
)


# Canned PDDL per catalog skill, mirroring the Session 29 fixtures.
_MOCK_ACTIONS: dict[str, dict[str, Any]] = {
    "pour": {
        "action": {
            "name": "pour",
            "parameters": "(?src - item ?dst - item)",
            "precondition": "(and (holding ?src) (camera-aimed-at ?dst))",
            "effect": "(and (poured ?src ?dst) (clear ?src))",
        },
        "new_predicates": ["(poured ?src - item ?dst - item)"],
    },
    "stir": {
        "action": {
            "name": "stir",
            "parameters": "(?c - item)",
            "precondition": "(and (camera-aimed-at ?c))",
            "effect": "(and (stirred ?c))",
        },
        "new_predicates": ["(stirred ?c - item)"],
    },
    "cut": {
        "action": {
            "name": "cut",
            "parameters": "(?i - item)",
            "precondition": "(and (camera-aimed-at ?i))",
            "effect": "(and (cut-open ?i))",
        },
        "new_predicates": ["(cut-open ?i - item)"],
    },
}


def _display(path: Path) -> str:
    """Repo-relative when possible, absolute for a custom --out-dir."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _prompt_field(user: str, key: str) -> str:
    for line in user.splitlines():
        if line.startswith(f"{key}:"):
            return line[len(key) + 1 :].strip()
    return ""


def _mock_generate(system: str, user: str) -> str:
    """
    Ideal replies derived from each case's expectation.

    This checks the harness, not a model: ``--mock`` must score 8/8, so a
    failure there means the plumbing broke rather than the LLM being weak.
    """
    task = _prompt_field(user, "task").lower()
    case = next((c for c in CASES if c["task"].lower() == task), None)

    if "current_domain_pddl:" in user:
        skill = (case or {}).get("expect", {}).get("skill", "")
        payload = _MOCK_ACTIONS.get(skill)
        if payload is None:
            return json.dumps({"refuse": True, "reason": "mock has no payload"})
        return json.dumps(
            {
                "skill": skill,
                "action": payload["action"],
                "new_predicates": payload["new_predicates"],
                "new_types": [],
                "reason": "mock",
            }
        )

    if _prompt_field(user, "parameters"):  # binding prompt
        arity = _prompt_field(user, "parameters").count("?")
        symbols = [
            s.strip()
            for s in _prompt_field(user, "scene_symbols").split(",")
            if s.strip()
        ]
        if len(symbols) < arity:
            return json.dumps({"refuse": True, "reason": "not enough symbols"})
        return json.dumps({"bindings": symbols[:arity]})

    expect = (case or {}).get("expect", {})
    if expect.get("refuse"):
        # A real gap the catalog cannot close: incomplete, no skill named.
        return json.dumps(
            {
                "template": "manipulation_base",
                "completeness": "incomplete",
                "needed_skills": [],
                "reason": "mock: no catalog skill fits",
            }
        )
    skill = expect.get("skill")
    return json.dumps(
        {
            "template": expect.get("template", "manipulation_base"),
            "completeness": expect.get("completeness", "complete"),
            "needed_skills": [skill] if skill else [],
            "reason": "mock",
        }
    )


def _score(case: dict[str, Any], row: dict[str, Any]) -> tuple[bool, str]:
    """Compare one result against the case's expectation."""
    expect = case["expect"]

    if expect.get("refuse"):
        if row["status"] == "refused":
            return True, "refused as expected"
        return False, f"expected refuse, got status={row['status']}"

    if row["completeness"] != expect["completeness"]:
        return False, (
            f"expected {expect['completeness']}, got {row['completeness']}"
        )

    if expect.get("template") and row["template"] != expect["template"]:
        return False, f"expected template {expect['template']}, got {row['template']}"

    if expect["completeness"] == "complete":
        if row["status"] not in {"skipped", "not_called"}:
            return False, f"complete task still ran the enricher ({row['status']})"
        return True, "complete, enricher not called"

    skill = expect.get("skill")
    if skill and skill not in row["skills_grounded"]:
        return False, (
            f"expected skill {skill}, got {row['skills_grounded'] or 'none'} "
            f"(status={row['status']})"
        )
    if row["status"] != "enriched":
        return False, f"expected enriched domain, got {row['status']}"
    if not row["goal_facts"]:
        return False, "enriched but the goal could not be grounded"
    return True, f"enriched {','.join(row['skills_grounded'])} → {row['goal_facts']}"


def run_case(
    case: dict[str, Any],
    *,
    generate_fn,
    enriched_dir: Path,
) -> dict[str, Any]:
    """Selection → enrichment → goal grounding for one task."""
    t0 = time.monotonic()
    resolution = resolve_domain_for_task(
        case["task"],
        enrichment_enabled=True,
        selector=make_domain_selector(SelectionBackend.LLM, generate_fn=generate_fn),
        enricher=make_online_enricher(
            SelectionBackend.LLM,
            generate_fn=generate_fn,
            enriched_dir=enriched_dir,
        ),
        scene_symbols=tuple(case["symbols"]),
    )
    selection = resolution.selection
    enrichment = resolution.enrichment

    row: dict[str, Any] = {
        "id": case["id"],
        "task": case["task"],
        "note": case["note"],
        "template": selection.template,
        "completeness": selection.completeness.value,
        "needed_skills": list(selection.needed_skills),
        "select_backend": selection.backend,
        "select_reason": selection.reason,
        "select_error": selection.error,
        "status": "not_called" if enrichment is None else enrichment.status.value,
        "skills_grounded": (
            [] if enrichment is None else list(enrichment.skills_grounded)
        ),
        "ros_primitives": (
            [] if enrichment is None else list(enrichment.ros_primitives)
        ),
        "refuse_message": None if enrichment is None else enrichment.refuse_message,
        "domain_path": None if enrichment is None else enrichment.domain_path,
        "reused": bool(enrichment is not None and enrichment.reused),
        "action_pddl": None,
        "goal_facts": [],
    }

    if enrichment is not None and enrichment.status == EnrichmentStatus.ENRICHED:
        additions = enrichment.domain_additions or {}
        row["action_pddl"] = additions.get("new_actions") or []
        context = EnrichmentContext.from_domain_additions(additions)
        row["catalog_only"] = context.catalog_only
        # Same binding the loop performs when there are no VLM steps.
        row["goal_facts"] = [
            fact
            for _, fact in ground_enrichment_goal(
                context,
                case["task"],
                list(case["symbols"]),
                binder=make_goal_binder(SelectionBackend.LLM, generate_fn=generate_fn),
            )
        ]

    row["wall_s"] = round(time.monotonic() - t0, 2)
    ok, verdict = _score(case, row)
    row["ok"] = ok
    row["verdict"] = verdict
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live probe of the single text LLM on select / enrich / bind"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HF model id; default resolves VLMRP_TEXT_LLM_MODEL then the 1.5B.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Canned replies instead of weights (harness check only).",
    )
    parser.add_argument(
        "--cases",
        default=None,
        help="Comma-separated case ids (default: all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Generation budget; PDDL authoring needs more room than selection.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=f"Report directory (default {_OUT_DIR.relative_to(_REPO_ROOT)}).",
    )
    args = parser.parse_args()

    wanted = (
        {c.strip() for c in args.cases.split(",") if c.strip()}
        if args.cases
        else None
    )
    cases = [c for c in CASES if wanted is None or c["id"] in wanted]
    if not cases:
        print(f"[FAIL] no cases match {args.cases!r}")
        return 2

    model_id = "(mock)" if args.mock else resolve_text_llm_model_id(args.model)
    out = Path(args.out_dir).resolve() if args.out_dir else _OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    # Authored domains go next to the report, never into the shared cache, so a
    # probe run cannot make a later loop reuse prompt-experiment output.
    enriched_dir = out / "enriched"
    enriched_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PROBE] model   : {model_id}")
    print(f"[PROBE] cases   : {', '.join(c['id'] for c in cases)}")
    print(f"[PROBE] enriched: {_display(enriched_dir)}\n")

    if args.mock:
        generate_fn = _mock_generate
    else:
        from planner.text_llm import get_shared_text_client

        client = get_shared_text_client(
            args.model, max_new_tokens=args.max_new_tokens
        )
        print("[PROBE] loading weights (first call may take a while)…")
        generate_fn = client.complete

    rows: list[dict[str, Any]] = []
    for case in cases:
        row = run_case(case, generate_fn=generate_fn, enriched_dir=enriched_dir)
        rows.append(row)
        mark = "OK  " if row["ok"] else "FAIL"
        print(f"[{mark}] {row['id']:<18} {row['task']!r}")
        print(
            f"        template={row['template']} "
            f"completeness={row['completeness']} "
            f"backend={row['select_backend']} ({row['wall_s']}s)"
        )
        if row["select_error"]:
            print(f"        select fell back: {row['select_error']}")
        print(f"        {row['verdict']}")
        if row["action_pddl"]:
            for act in row["action_pddl"]:
                print(
                    f"        action: ({act.get('name')} {act.get('parameters')}) "
                    f"effect {act.get('effect')}"
                )
        if row["refuse_message"]:
            print(f"        refuse: {row['refuse_message']}")
        print()

    passed = sum(1 for r in rows if r["ok"])
    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "mock": bool(args.mock),
        "max_new_tokens": args.max_new_tokens,
        "passed": passed,
        "total": len(rows),
        "cases": rows,
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"[PROBE] {passed}/{len(rows)} cases as expected")
    print(f"[PROBE] report → {_display(report_path)}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
