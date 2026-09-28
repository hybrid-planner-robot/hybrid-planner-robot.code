#!/usr/bin/env python3
"""
Run the incremental goal-LLM difficulty battery.

Loads ``tests/fixtures/goal_llm_eval/battery_v1.json``, generates goals with the local
text LLM (and rule-based for comparison), scores each level, writes a report.

Supports single-model runs and multi-model sweeps (Session 9c).

Examples
--------
Live local LLM (GPU)::

    python scripts/eval_goal_llm_battery.py

Rule-based only (no model download)::

    python scripts/eval_goal_llm_battery.py --rule-only

Subset of levels / dry mock LLM::

    python scripts/eval_goal_llm_battery.py --levels L0,L1 --mock-llm
    python scripts/eval_goal_llm_battery.py --levels L5 --no-fallback

Multi-model sweep + prompt A/B::

    python scripts/eval_goal_llm_battery.py --models Qwen/Qwen2.5-0.5B-Instruct,Qwen/Qwen2.5-1.5B-Instruct
    python scripts/eval_goal_llm_battery.py --model-cards tests/fixtures/goal_llm_eval/model_cards_v1.json --prompt-id v2
    python scripts/eval_goal_llm_battery.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt-id v1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from planner.problem_generator.goal_generator.backends.local_llm import (  # noqa: E402
    DEFAULT_GOAL_LLM_MODEL_ID,
    LocalLLMGoalGenerator,
    TransformersLocalClient,
)
from planner.problem_generator.goal_generator.backends.rule_based import (  # noqa: E402
    RuleBasedGoalGenerator,
)
from planner.problem_generator.goal_generator.battery_scoring import (  # noqa: E402
    score_case,
    summarize_by_level,
)
from planner.problem_generator.goal_generator.predicates import (  # noqa: E402
    resolve_allowed_predicates,
)

_DEFAULT_BATTERY = _REPO_ROOT / "tests" / "fixtures" / "goal_llm_eval" / "battery_v1.json"
_DEFAULT_MODEL_CARDS = (
    _REPO_ROOT / "tests" / "fixtures" / "goal_llm_eval" / "model_cards_v1.json"
)


def _load_test_goal_llm():
    path = _REPO_ROOT / "scripts" / "test_goal_llm.py"
    spec = spec_from_file_location("test_goal_llm", path)
    assert spec and spec.loader
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _print_summary(summary: dict[str, Any], *, title: str = "OVERALL") -> None:
    print("\n" + "═" * 64)
    print(f"{title}  n={summary['n']}  pass_rate={summary['pass_rate']:.1%}")
    print("─" * 64)
    for lid, stats in summary["levels"].items():
        failed = ", ".join(stats["failed_ids"]) if stats["failed_ids"] else "—"
        print(
            f"  {lid}: pass={stats['pass_rate']:.0%}  "
            f"exact={stats['exact_rate']:.0%}  "
            f"valid={stats['valid_rate']:.0%}  "
            f"fallback={stats['fallback_rate']:.0%}  "
            f"beats_rule={stats['llm_beats_rule']}  "
            f"fail=[{failed}]"
        )
    print("═" * 64)


def _mean_latency(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return round(sum(r["seconds"] for r in rows) / len(rows), 3)


def _print_comparison(runs: list[dict[str, Any]]) -> None:
    """Print pass/exact/valid/fallback/latency by level across models."""
    if len(runs) < 1:
        return
    level_ids: list[str] = []
    for run in runs:
        for lid in run["summary"]["levels"]:
            if lid not in level_ids:
                level_ids.append(lid)

    print("\n" + "═" * 88)
    print("COMPARISON (pass / exact / valid / fallback / mean_s)")
    print("═" * 88)
    header = f"{'level':<6}"
    for run in runs:
        label = run.get("short_label") or run.get("model") or "?"
        header += f" | {label[:18]:^18}"
    print(header)
    print("─" * 88)

    for lid in level_ids:
        line = f"{lid:<6}"
        for run in runs:
            st = run["summary"]["levels"].get(lid)
            if not st:
                line += f" | {'—':^18}"
                continue
            mean_s = st.get("mean_latency_s")
            mean_txt = f"{mean_s:.2f}s" if isinstance(mean_s, (int, float)) else "?"
            cell = (
                f"{st['pass_rate']:.0%}/{st['exact_rate']:.0%}/"
                f"{st['valid_rate']:.0%}/{st['fallback_rate']:.0%} {mean_txt}"
            )
            line += f" | {cell:^18}"
        print(line)

    line = f"{'ALL':<6}"
    for run in runs:
        s = run["summary"]
        mean_s = s.get("mean_latency_s")
        mean_txt = f"{mean_s:.2f}s" if isinstance(mean_s, (int, float)) else "?"
        cell = f"{s['pass_rate']:.0%} pass  {mean_txt}"
        line += f" | {cell:^18}"
    print(line)

    print("─" * 88)
    for run in runs:
        vram = run.get("peak_vram_mb")
        vram_s = f"{vram:.0f} MiB" if isinstance(vram, (int, float)) else "n/a (CPU or unknown)"
        print(
            f"  {run.get('short_label') or run.get('model')}: "
            f"prompt_id={run.get('prompt_id')!r}  peak_vram={vram_s}  "
            f"elapsed={run.get('elapsed_s')}s  "
            f"notes={run.get('model_notes') or '—'}"
        )
    print("═" * 88)


def _mock_generate_fn(gold_by_command: dict[str, list[list[str]]]):
    """Deterministic generate_fn for CI / --mock-llm (returns gold JSON)."""

    def generate(system: str, user: str) -> str:
        # Extract command line from user prompt
        cmd = ""
        for line in user.splitlines():
            if line.startswith("command:"):
                cmd = line[len("command:") :].strip()
                break
        gold = gold_by_command.get(cmd)
        if gold is None:
            return "not json"
        return json.dumps({"facts": gold})

    return generate


def _peak_vram_mb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001
        return None


def _reset_cuda_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def _resolve_model_list(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return list of {id, tier?, notes?, params_b?} specs to evaluate."""
    if args.rule_only:
        return [{"id": "rule_based", "tier": "baseline"}]

    if args.mock_llm:
        # Allow --models mock-a,mock-b to exercise the sweep CLI without GPU.
        if args.models:
            return [
                {"id": m.strip(), "tier": "mock"}
                for m in args.models.split(",")
                if m.strip()
            ]
        return [{"id": "mock", "tier": "baseline"}]

    if args.models:
        return [{"id": m.strip()} for m in args.models.split(",") if m.strip()]

    if args.model_cards:
        cards_path = args.model_cards
        if not cards_path.is_file():
            raise SystemExit(f"Model cards file not found: {cards_path}")
        cards = json.loads(cards_path.read_text(encoding="utf-8"))
        models = cards.get("models") or []
        if args.tiers:
            want = {t.strip() for t in args.tiers.split(",")}
            models = [m for m in models if m.get("tier") in want]
        if not models:
            raise SystemExit("No models selected from model cards")
        return models

    import os

    model_id = (
        args.model
        or os.environ.get("VLMRP_GOAL_LLM_MODEL")
        or DEFAULT_GOAL_LLM_MODEL_ID
    )
    return [{"id": model_id, "tier": "default"}]


def _short_label(spec: dict[str, Any]) -> str:
    mid = str(spec.get("id") or "?")
    name = mid.rsplit("/", 1)[-1]
    # Qwen2.5-1.5B-Instruct → 1.5B
    for token in name.replace("_", "-").split("-"):
        if token.lower().endswith("b") and token[:-1].replace(".", "", 1).isdigit():
            return token
    return name[:16]


def _enrich_summary_latency(
    summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_level: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_level.setdefault(row["level_id"], []).append(row)
    for lid, items in by_level.items():
        if lid in summary["levels"]:
            summary["levels"][lid]["mean_latency_s"] = _mean_latency(items)
    summary["mean_latency_s"] = _mean_latency(rows)
    return summary


def _run_battery(
    *,
    battery: dict[str, Any],
    want_levels: set[str] | None,
    local_gen: LocalLLMGoalGenerator | None,
    rule_gen: RuleBasedGoalGenerator,
    tgl: Any,
    backend_label: str,
    model_id: str | None,
    prompt_id: str | None,
    quiet: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for level in battery["levels"]:
        lid = level["id"]
        if want_levels is not None and lid not in want_levels:
            continue
        if not quiet:
            print(f"\n▶ {lid} — {level['name']}: {level['description']}")

        for case in level["cases"]:
            scene = tgl.load_scenario(case["scenario"])
            domain = case.get("domain_template") or scene.domain_template or "manipulation_base"
            scene.domain_template = domain
            objs = [o.name for o in scene.objects]
            locs = [loc.name for loc in scene.locations]
            known = set(objs) | set(locs)
            allowed = resolve_allowed_predicates(domain)

            command = case["command"]
            gold = case.get("gold")
            soft_golds = case.get("soft_golds")
            expect_fail = bool(case.get("expect_fail", False))

            rule = rule_gen.generate(
                command,
                objs,
                locations=locs,
                domain_template=domain,
                scene_state=scene,
            )

            if local_gen is not None:
                t0 = time.perf_counter()
                result = local_gen.generate(
                    command,
                    objs,
                    locations=locs,
                    domain_template=domain,
                    scene_state=scene,
                )
                elapsed = round(time.perf_counter() - t0, 3)
                label = backend_label
            else:
                t0 = time.perf_counter()
                result = rule
                elapsed = round(time.perf_counter() - t0, 3)
                label = "rule_based"

            score = score_case(
                gold=gold,
                expect_fail=expect_fail,
                result_ok=result.ok,
                result_facts=[list(f) for f in result.facts],
                result_error=result.error,
                allowed_predicates=set(allowed),
                known_symbols=known,
                rule_facts=[list(f) for f in rule.facts],
                rule_ok=rule.ok,
                soft_golds=soft_golds,
            )

            row = {
                "id": case["id"],
                "level_id": lid,
                "level_name": level["name"],
                "scenario": case["scenario"],
                "command": command,
                "domain_template": domain,
                "gold": gold,
                "soft_golds": soft_golds,
                "expect_fail": expect_fail,
                "backend": label,
                "model": model_id,
                "prompt_id": prompt_id,
                "seconds": elapsed,
                "result": {
                    "ok": result.ok,
                    "facts": [list(f) for f in result.facts],
                    "error": result.error,
                    "raw": (result.raw[:500] if result.raw else None),
                },
                "rule_based": {
                    "ok": rule.ok,
                    "facts": [list(f) for f in rule.facts],
                    "error": rule.error,
                },
                "score": score,
                "notes": case.get("notes"),
            }
            rows.append(row)

            if not quiet:
                mark = "PASS" if score["passed"] else "FAIL"
                facts_s = " ".join(
                    "(" + " ".join(map(str, f)) + ")" for f in result.facts
                ) or "(none)"
                objs_s = ", ".join(objs) if objs else "(none)"
                print(f"  [{mark}] {case['id']}")
                print(f"         command: {command}")
                print(f"         objects: {objs_s}")
                print(
                    f"         result:  {facts_s}"
                    f"  (rule_exact={score.get('rule_exact')}, {elapsed}s)"
                )

    summary = _enrich_summary_latency(summarize_by_level(rows), rows)
    return rows, summary, round(time.perf_counter() - t_start, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Goal LLM difficulty battery")
    parser.add_argument(
        "--battery",
        type=Path,
        default=_DEFAULT_BATTERY,
        help="Path to battery JSON",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated level ids (e.g. L0,L2,L5). Default: all",
    )
    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="Fake local LLM that returns gold JSON (CI / no GPU)",
    )
    parser.add_argument("--model", default=None, help="Single HuggingFace model id")
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids for a sweep (writes per-model + comparison)",
    )
    parser.add_argument(
        "--model-cards",
        type=Path,
        nargs="?",
        const=_DEFAULT_MODEL_CARDS,
        default=None,
        help=(
            "JSON model card list for a sweep. Pass flag alone to use "
            f"{_DEFAULT_MODEL_CARDS.name}; or pass an explicit path."
        ),
    )
    parser.add_argument(
        "--tiers",
        default=None,
        help="With --model-cards: comma-separated tiers (small,default,mid,upper)",
    )
    parser.add_argument(
        "--prompt-id",
        default=None,
        help="Prompt version pin (v1 / v2) or explicit stem; default = active .md",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full results JSON (default: data/goal_llm_eval/runs/<ts>.json)",
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=None,
        help="Write multi-model comparison JSON (default alongside per-model reports)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    battery_path = args.battery
    if not battery_path.is_file():
        raise SystemExit(f"Battery file not found: {battery_path}")
    battery = json.loads(battery_path.read_text(encoding="utf-8"))
    want_levels = (
        {x.strip() for x in args.levels.split(",")} if args.levels else None
    )

    tgl = _load_test_goal_llm()
    rule_gen = RuleBasedGoalGenerator()

    gold_by_command: dict[str, list[list[str]]] = {}
    for level in battery["levels"]:
        for case in level["cases"]:
            if case.get("gold"):
                gold_by_command[case["command"]] = case["gold"]
            elif case.get("soft_golds"):
                gold_by_command[case["command"]] = case["soft_golds"][0]

    model_specs = _resolve_model_list(args)
    runs_dir = _REPO_ROOT / "data" / "goal_llm_eval" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prompt_tag = args.prompt_id or "default"

    run_payloads: list[dict[str, Any]] = []
    exit_code = 0

    for spec in model_specs:
        model_id = spec["id"]
        short = _short_label(spec)
        if not args.quiet:
            print(f"\n{'━' * 64}\nMODEL {model_id}  (prompt_id={args.prompt_id!r})\n{'━' * 64}")

        local_gen: LocalLLMGoalGenerator | None = None
        backend_label = "local_llm"
        mode = "local_llm"
        client: TransformersLocalClient | None = None

        if args.rule_only:
            mode = "rule_only"
            backend_label = "rule_based"
            model_id = "rule_based"
        elif args.mock_llm:
            local_gen = LocalLLMGoalGenerator(
                generate_fn=_mock_generate_fn(gold_by_command),
                fallback_rule_based=False,
                prompt_id=args.prompt_id,
            )
            mode = "mock_llm"
        else:
            _reset_cuda_peak()
            device_map = None if args.device == "cpu" else "auto"
            client = TransformersLocalClient(
                model_id=model_id,
                max_new_tokens=args.max_new_tokens,
                device_map=device_map,
                device=args.device,
            )
            client._ensure_loaded()
            local_gen = LocalLLMGoalGenerator(
                client=client,
                model_id=model_id,
                fallback_rule_based=not args.no_fallback,
                prompt_id=args.prompt_id,
            )

        rows, summary, elapsed_s = _run_battery(
            battery=battery,
            want_levels=want_levels,
            local_gen=local_gen,
            rule_gen=rule_gen,
            tgl=tgl,
            backend_label=backend_label,
            model_id=model_id,
            prompt_id=args.prompt_id,
            quiet=args.quiet,
        )
        _print_summary(summary, title=f"OVERALL [{short}]")

        peak_vram = _peak_vram_mb() if mode == "local_llm" else None

        payload = {
            "battery": battery.get("name"),
            "model": model_id,
            "short_label": short,
            "tier": spec.get("tier"),
            "params_b": spec.get("params_b"),
            "model_notes": spec.get("notes"),
            "prompt_id": args.prompt_id,
            "mode": mode,
            "device": args.device,
            "peak_vram_mb": peak_vram,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": elapsed_s,
            "summary": summary,
            "results": rows,
        }
        run_payloads.append(payload)

        if args.json_out is not None and len(model_specs) == 1:
            out_path = args.json_out
        else:
            safe = short.replace("/", "_")
            tag = "rule" if args.rule_only else ("mock" if args.mock_llm else "llm")
            out_path = runs_dir / f"battery_{tag}_{safe}_{prompt_tag}_{ts}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[wrote] {out_path}")

        # Free GPU memory between sweep models when possible.
        if client is not None:
            try:
                import torch

                del client._model
                del client._tokenizer
                client._model = None
                client._tokenizer = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

        if args.mock_llm and summary["pass_rate"] < 0.99:
            exit_code = 1

    if len(run_payloads) > 1:
        _print_comparison(run_payloads)
        comparison = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_id": args.prompt_id,
            "levels": args.levels,
            "runs": [
                {
                    "model": r["model"],
                    "short_label": r["short_label"],
                    "tier": r.get("tier"),
                    "params_b": r.get("params_b"),
                    "peak_vram_mb": r.get("peak_vram_mb"),
                    "elapsed_s": r.get("elapsed_s"),
                    "summary": r["summary"],
                    "model_notes": r.get("model_notes"),
                }
                for r in run_payloads
            ],
        }
        cmp_path = args.comparison_out
        if cmp_path is None:
            cmp_path = runs_dir / f"comparison_{prompt_tag}_{ts}.json"
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        cmp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        print(f"[wrote comparison] {cmp_path}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
