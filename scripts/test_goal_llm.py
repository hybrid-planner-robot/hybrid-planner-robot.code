#!/usr/bin/env python3
"""
Interactive / batch test pipeline for local goal LLM.

Given a mock SceneState scenario and a natural-language command, produce PDDL
``:goal`` facts via the local text model (and optionally compare to rule-based).

Examples
--------
List scenarios::

    python scripts/test_goal_llm.py --list-scenarios

Single command (downloads Qwen2.5-1.5B on first run)::

    python scripts/test_goal_llm.py \\
        --scenario table \\
        --command "place the red cup on the shelf"

Compare rule-based vs local LLM::

    python scripts/test_goal_llm.py -s table -c "pick up the red cup" --compare

Interactive REPL (model loaded once)::

    python scripts/test_goal_llm.py --scenario table --interactive

Batch file of commands::

    python scripts/test_goal_llm.py --scenario table --batch tests/fixtures/goal_llm_eval/commands_example.json

CPU-only / custom model::

    python scripts/test_goal_llm.py -s table -c "look at blue_box" \\
        --model Qwen/Qwen2.5-1.5B-Instruct --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from planner.problem_generator.goal_generator.renderer import GoalRenderer  # noqa: E402
from planner.problem_generator.goal_generator.scene_compact import (  # noqa: E402
    compact_scene_dict,
)
from planner.problem_generator.init_generator.adapters.mock import (  # noqa: E402
    DinoMockAdapter,
    OracleMockAdapter,
    TrackerMockAdapter,
)
from planner.problem_generator.init_generator.builder import build_scene  # noqa: E402
from planner.problem_generator.init_generator.schema import SceneState  # noqa: E402
from planner.state_tracker import StateTracker  # noqa: E402

_MOCK_DIR = (
    _REPO_ROOT
    / "planner"
    / "problem_generator"
    / "init_generator"
    / "mock"
)

SCENARIOS: dict[str, str] = {
    "table": "Oracle tabletop (red_cup + blue_box on table, shelf empty)",
    "fused": "Oracle + noisy DINO fused (sim-like)",
    "holding": "Mid-task: gripper holding red_cup (tracker mock)",
    "holding_fused": "Oracle table + tracker holding red_cup after pick",
    "office": "Office desk (OFFICE_SUITE objects: pen, keyboard, laptop, …)",
}

_OFFICE_SCENE = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "goal_llm_eval"
    / "scenes"
    / "office_desk.json"
)


def list_scenarios() -> None:
    print("Available mock scenarios:\n")
    for name, desc in SCENARIOS.items():
        print(f"  {name:16s}  {desc}")
    print(f"\nMock fixtures dir: {_MOCK_DIR}")


def load_scenario(name: str) -> SceneState:
    key = name.strip().lower()
    if key == "table":
        return OracleMockAdapter.load()
    if key == "fused":
        return build_scene(
            oracle_scene=OracleMockAdapter.load(),
            dino_scene=DinoMockAdapter.load(),
            sim_like=True,
        )
    if key == "holding":
        return TrackerMockAdapter.load()
    if key == "holding_fused":
        tracker = StateTracker()
        # Seed tracker as if pick(red_cup) already succeeded
        from planner.primitive_transitions import CompletedAction

        tracker.apply(CompletedAction("pick", {"object": "red_cup"}))
        return build_scene(
            oracle_scene=OracleMockAdapter.load(),
            tracker=tracker,
            sim_like=True,
        )
    if key == "office":
        return OracleMockAdapter.load(_OFFICE_SCENE)
    # Custom path to a SceneState JSON or mock fixture
    path = Path(name)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        fmt = data.get("format")
        if fmt == OracleMockAdapter.FORMAT:
            return OracleMockAdapter.load(path)
        if fmt == DinoMockAdapter.FORMAT:
            return DinoMockAdapter.load(path)
        if fmt == TrackerMockAdapter.FORMAT:
            return TrackerMockAdapter.load(path)
        return SceneState.from_dict(data)
    raise SystemExit(
        f"Unknown scenario {name!r}. Use --list-scenarios or a JSON path."
    )


def _symbols(scene: SceneState) -> tuple[list[str], list[str]]:
    return [o.name for o in scene.objects], [loc.name for loc in scene.locations]


def _format_facts(facts: list[tuple]) -> str:
    if not facts:
        return "(none)"
    return ", ".join("(" + " ".join(map(str, f)) + ")" for f in facts)


def run_once(
    *,
    scene: SceneState,
    command: str,
    local_gen: LocalLLMGoalGenerator | None,
    rule_gen: RuleBasedGoalGenerator,
    compare: bool,
    show_prompt_scene: bool,
    show_raw: bool,
) -> dict[str, Any]:
    objs, locs = _symbols(scene)
    domain = scene.domain_template or "manipulation_base"
    out: dict[str, Any] = {
        "command": command,
        "domain_template": domain,
        "objects": objs,
        "locations": locs,
    }

    if show_prompt_scene:
        print("\n── compact SceneState (prompt input) ──")
        print(json.dumps(compact_scene_dict(scene), indent=2))

    if compare or local_gen is None:
        t0 = time.perf_counter()
        rule = rule_gen.generate(
            command, objs, locations=locs, domain_template=domain, scene_state=scene
        )
        out["rule_based"] = {
            "ok": rule.ok,
            "facts": [list(f) for f in rule.facts],
            "error": rule.error,
            "seconds": round(time.perf_counter() - t0, 3),
        }
        print(f"\n[rule_based] ok={rule.ok}  {_format_facts(rule.facts)}")
        if rule.error:
            print(f"             error: {rule.error}")

    if local_gen is not None:
        t0 = time.perf_counter()
        local = local_gen.generate(
            command,
            objs,
            locations=locs,
            domain_template=domain,
            scene_state=scene,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        out["local_llm"] = {
            "ok": local.ok,
            "facts": [list(f) for f in local.facts],
            "error": local.error,
            "seconds": elapsed,
            "backend": local.backend,
        }
        print(f"\n[local_llm]  ok={local.ok}  {_format_facts(local.facts)}  ({elapsed}s)")
        if local.error:
            print(f"             note: {local.error}")
        if show_raw and local.raw:
            print("── raw model output ──")
            print(local.raw.strip())
        if local.ok and local.facts:
            print("── (:goal …) ──")
            print(GoalRenderer().render_section(local.facts))

        if compare and "rule_based" in out:
            match = out["rule_based"]["facts"] == out["local_llm"]["facts"]
            out["exact_match"] = match
            print(f"\n[compare] exact_match={match}")

    return out


def load_batch(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "commands" in data:
        items = data["commands"]
    elif isinstance(data, list):
        items = data
    else:
        raise SystemExit("Batch JSON must be a list or {\"commands\": [...]}")
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, str):
            out.append({"command": item})
        elif isinstance(item, dict) and "command" in item:
            out.append(item)
        else:
            raise SystemExit(f"Invalid batch item: {item!r}")
    return out


def build_local_generator(
    *,
    model_id: str,
    device: str,
    no_fallback: bool,
    max_new_tokens: int,
) -> LocalLLMGoalGenerator:
    device_map: str | None
    if device == "cpu":
        device_map = None
    elif device == "cuda":
        device_map = "auto"
    else:  # auto
        device_map = "auto"

    client = TransformersLocalClient(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        device_map=device_map,
    )
    # Force load now so REPL / batch pay the cost once up front
    client._ensure_loaded()
    return LocalLLMGoalGenerator(
        client=client,
        model_id=model_id,
        fallback_rule_based=not no_fallback,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test local goal LLM on mock SceneState scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument(
        "-s",
        "--scenario",
        default="table",
        help="Mock scenario name or path to SceneState/mock JSON (default: table)",
    )
    parser.add_argument("-c", "--command", default=None, help="Natural-language task")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="REPL: type commands against the loaded scenario",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=None,
        help="JSON file with a list of commands (or {\"commands\": [...]})",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also run rule-based and print exact-match",
    )
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="Skip local LLM (no model download)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"HF model id (default: {DEFAULT_GOAL_LLM_MODEL_ID} or VLMRP_GOAL_LLM_MODEL)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Do not fall back to rule-based on invalid LLM output",
    )
    parser.add_argument(
        "--show-scene",
        action="store_true",
        help="Print compact SceneState JSON used in the prompt",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print raw model text",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write results JSON to this path",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        list_scenarios()
        return 0

    if not args.command and not args.interactive and args.batch is None:
        parser.error("Provide --command, --batch, --interactive, or --list-scenarios")

    scene = load_scenario(args.scenario)
    print(f"[scenario] {args.scenario}")
    print(f"           objects={[o.name for o in scene.objects]}")
    print(f"           locations={[loc.name for loc in scene.locations]}")
    print(
        f"           robot: empty={scene.robot.gripper_empty} "
        f"holding={scene.robot.holding!r} aimed={scene.robot.camera_aimed_at!r}"
    )

    rule_gen = RuleBasedGoalGenerator()
    local_gen: LocalLLMGoalGenerator | None = None
    if not args.rule_only:
        import os

        model_id = (
            args.model
            or os.environ.get("VLMRP_GOAL_LLM_MODEL")
            or DEFAULT_GOAL_LLM_MODEL_ID
        )
        print(f"[model]    {model_id} (device={args.device})")
        try:
            local_gen = build_local_generator(
                model_id=model_id,
                device=args.device,
                no_fallback=args.no_fallback,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] Could not load local LLM: {exc}", file=sys.stderr)
            print(
                "Hint: pip install transformers torch; "
                "or use --rule-only / check network for HF download.",
                file=sys.stderr,
            )
            return 1

    results: list[dict[str, Any]] = []

    def _run(cmd: str) -> None:
        results.append(
            run_once(
                scene=scene,
                command=cmd,
                local_gen=local_gen,
                rule_gen=rule_gen,
                compare=args.compare or args.rule_only,
                show_prompt_scene=args.show_scene,
                show_raw=args.show_raw,
            )
        )

    if args.batch is not None:
        for item in load_batch(args.batch):
            cmd = item["command"]
            print(f"\n{'═'*60}\ncommand: {cmd}")
            _run(cmd)
    elif args.interactive:
        print("\nInteractive mode — type a command, empty line / quit to exit.")
        while True:
            try:
                line = input("\ngoal> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in {"q", "quit", "exit"}:
                break
            if line.startswith("!"):
                # meta: !scene
                if line == "!scene":
                    print(json.dumps(compact_scene_dict(scene), indent=2))
                elif line == "!help":
                    print("Enter an NL command. Meta: !scene  !help  quit")
                else:
                    print(f"Unknown meta command: {line}")
                continue
            _run(line)
    elif args.command:
        _run(args.command)
    else:
        parser.error("Provide --command, --batch, --interactive, or --list-scenarios")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scenario": args.scenario,
            "results": results,
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n[wrote] {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
