#!/usr/bin/env python3
"""
Demo narrata del hybrid problem generator (solo mock — no Gazebo / GPU / ROS).

Mostra passo passo:
  1. Oracle + DINO mock (+ perché blu_box ≠ blue_box non si fondono)
  2. Fusion → SceneState
  3. InitRenderer: SceneState → sezione :init (esplicita)
  4. Comando NL → GoalGenerator (rule-based + mock local_llm) → :goal
  5. Assemblaggio problem + pick mid-task
  6. Verifier GREEN / YELLOW / RED

Uso (presentazione):
  python scripts/demo_hybrid_pipeline.py
  python scripts/demo_hybrid_pipeline.py --pause
  python scripts/demo_hybrid_pipeline.py --pause --command "put the red cup onto the shelf"

Exit 0 on success.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from planner.hybrid_runtime import (  # noqa: E402
    GoalBackend,
    HybridMode,
    HybridProblemSession,
)
from planner.problem_generator.goal_generator.backends.local_llm import (  # noqa: E402
    LocalLLMGoalGenerator,
)
from planner.problem_generator.goal_generator.backends.rule_based import (  # noqa: E402
    RuleBasedGoalGenerator,
)
from planner.problem_generator.goal_generator.renderer import GoalRenderer  # noqa: E402
from planner.problem_generator.init_generator.adapters.mock import (  # noqa: E402
    DinoMockAdapter,
    OracleMockAdapter,
)
from planner.problem_generator.init_generator.builder import build_scene  # noqa: E402
from planner.problem_generator.init_generator.renderer import InitRenderer  # noqa: E402
from planner.problem_generator.init_generator.schema import SceneState  # noqa: E402
from planner.problem_generator.init_generator.vlm_fusion import (  # noqa: E402
    MockVlmFusionClient,
    patch_to_scene,
)
from planner.state_verifier import StateVerifier  # noqa: E402
from vlm.planner import PlanStep, VLMPlan  # noqa: E402

_WIDTH = 72
_DEFAULT_COMMAND = "place the red cup on the shelf"


class DemoUI:
    def __init__(self, *, pause: bool = False, slow: bool = False) -> None:
        self.pause = pause
        self.slow = slow
        self.step_n = 0

    def banner(self, title: str) -> None:
        self.step_n += 1
        line = "═" * _WIDTH
        print()
        print(line)
        print(f"  PASSO {self.step_n}: {title}")
        print(line)

    def say(self, text: str = "") -> None:
        print(text)

    def code(self, text: str, *, indent: int = 2) -> None:
        pad = " " * indent
        for line in text.strip("\n").splitlines():
            print(f"{pad}{line}")

    def wait(self) -> None:
        if self.slow:
            time.sleep(0.6)
        if self.pause:
            try:
                input("\n  [Invio per continuare] ")
            except EOFError:
                pass


def _plan() -> VLMPlan:
    return VLMPlan(
        goal="place red_cup on shelf",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": "shelf"},
            ),
        ],
        raw_output="",
        domain_template="manipulation_base",
    )


def _rel_summary(scene: SceneState) -> str:
    rels = [
        f"({r.predicate} {' '.join(r.args)})@{r.source}"
        for r in scene.relations
        if r.predicate in {"on", "stacked-on", "in-container"}
    ]
    objs = [f"{o.name}@{o.source}" for o in scene.objects]
    robot = (
        f"holding={scene.robot.holding!r} "
        f"gripper_empty={scene.robot.gripper_empty} "
        f"src={scene.robot.source}"
    )
    return (
        f"objects:   {objs}\n"
        f"  robot:     {robot}\n"
        f"  relations: {rels or '(none)'}"
    )


def run_demo(
    *,
    pause: bool = False,
    slow: bool = False,
    command: str = _DEFAULT_COMMAND,
) -> int:
    ui = DemoUI(pause=pause, slow=slow)
    print("=" * _WIDTH)
    print("  DEMO — Hybrid PDDL problem generator (MOCK ONLY)")
    print("  Nessun Gazebo / ROS / GPU richiesto.")
    print(f"  Comando NL (goal): {command!r}")
    print("=" * _WIDTH)

    # ── 1. Perception mocks ──────────────────────────────────────────────
    ui.banner("Carico oracle + DINO mock (fixture offline)")
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()
    ui.say(f"  Oracle objects: {sorted(o.name for o in oracle.objects)}")
    ui.say(f"  DINO objects:   {sorted(o.name for o in dino.objects)}")
    ui.say(
        "  DINO normalizza label → snake_case:  'blu box' → blu_box\n"
        "  Oracle usa il nome Gazebo:           blue_box"
    )
    ui.wait()

    # ── 2. Fusion + name mismatch ────────────────────────────────────────
    ui.banner("Fusion deterministica → SceneState (+ name mismatch)")
    fused = build_scene(oracle_scene=oracle, dino_scene=dino)
    ui.say(f"  sources_used: {fused.meta.sources_used}")
    ui.say("  Stato fuso:")
    ui.code(_rel_summary(fused))
    ui.say()
    ui.say("  Perché blu_box e blue_box restano SDOPPIATI?")
    ui.say("  1) Merge oggetti: solo se il NOME ESATTO coincide (red_cup sì).")
    ui.say("  2) Alias fuzzy attuale: confronta i nomi togliendo solo '_'")
    ui.say("     → 'blu_box'→'blubox'  vs  'blue_box'→'bluebox'  → NON uguali")
    ui.say("       (manca una lettera: blu ≠ blue). Quindi nemmeno il warning")
    ui.say("       'alias not merged' scatta: risultano dino-only + oracle-only.")
    ui.say("  3) Scelta MVP (Session 7): niente auto-rename aggressivo —")
    ui.say("     un false merge è peggio di un duplicato in :init.")
    mismatch_notes = [n for n in fused.meta.fusion_notes if "name mismatch" in n]
    ui.say("  fusion_notes:")
    for n in mismatch_notes:
        ui.say(f"    • {n}")
    ui.say(
        "  Evoluzione possibile: alias table / edit-distance / VLM-on-YELLOW\n"
        "  per allineare label rumorose ai simboli PDDL oracle."
    )
    ui.wait()

    # ── 3. Explicit :init ────────────────────────────────────────────────
    ui.banner("Creazione esplicita di :init  (InitRenderer)")
    ui.say("  Pipeline:  SceneState  →  InitRenderer.render_facts / render_section")
    init_facts = InitRenderer().render_facts(fused)
    ui.say(f"  Fatti :init ({len(init_facts)}):")
    for fact in init_facts:
        ui.say(f"    {fact}")
    ui.say("\n  Sezione PDDL prodotta:")
    ui.code(InitRenderer().render_section(fused))
    ui.wait()

    # ── 4. Explicit command → :goal ──────────────────────────────────────
    ui.banner("Creazione esplicita di :goal  (comando NL → GoalGenerator)")
    ui.say(f"  COMANDO (input testuale, non visione):")
    ui.say(f"    >>> {command}")
    objects = [o.name for o in fused.objects]
    locations = [loc.name for loc in fused.locations]
    ui.say(f"  symbols objects={objects}")
    ui.say(f"  symbols locations={locations}")

    ui.say("\n  [4a] Backend rule_based (default hybrid MVP)")
    rule = RuleBasedGoalGenerator()
    rule_result = rule.generate(
        command,
        objects,
        locations=locations,
        domain_template="manipulation_base",
    )
    ui.say(f"    ok={rule_result.ok}  backend={rule_result.backend}")
    ui.say(f"    facts={list(rule_result.facts)}")
    ui.say("    Sezione PDDL :goal:")
    ui.code(GoalRenderer().render_section(rule_result.facts))

    ui.say("\n  [4b] Backend local_llm (opt-in) — generate_fn MOCK, niente GPU")
    ui.say("    Il modello riceverebbe: system prompt + comando NL + compact SceneState")

    def mock_local_llm(system: str, user: str) -> str:
        ui.say("    --- chiamata mock local_llm ---")
        ui.say(f"    system[:120]= {system[:120]!r}…")
        # Show that the user prompt contains the command
        if "place" in user.lower() or "cup" in user.lower() or "shelf" in user.lower():
            # find a short excerpt with the command
            for line in user.splitlines():
                if command[:20].lower() in line.lower() or "command" in line.lower():
                    ui.say(f"    user line: {line.strip()[:100]}")
                    break
        ui.say('    mock response → {"facts": [["on", "red_cup", "shelf"]]}')
        return json.dumps({"facts": [["on", "red_cup", "shelf"]]})

    local = LocalLLMGoalGenerator(
        generate_fn=mock_local_llm,
        fallback_rule_based=True,
    )
    local_result = local.generate(
        command,
        objects,
        locations=locations,
        domain_template="manipulation_base",
        scene_state=fused,
    )
    ui.say(f"    ok={local_result.ok}  backend={local_result.backend}")
    ui.say(f"    facts={list(local_result.facts)}")
    ui.say("    Sezione PDDL :goal (da local_llm mock):")
    ui.code(GoalRenderer().render_section(local_result.facts))
    ui.say(
        "  Nota: in produzione local_llm resta OPT-IN (Session 9c non chiusa);\n"
        "  qui mostriamo solo il contratto comando → facts con generate_fn mock."
    )
    ui.wait()

    # ── 5. Session assembles full problem ────────────────────────────────
    ui.banner("Sessione hybrid: assembla problem (goal locked una volta)")
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command=command,
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
    )
    plan = _plan()
    pddl0, scene0 = session.generate_hybrid_problem(
        plan,
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="demo_iter_0",
    )
    ui.say(f"  session.command = {session.command!r}")
    ui.say(f"  session.goal_facts (locked) = {session.goal_facts}")
    ui.say("  Problem PDDL completo (iter 0):")
    ui.code(pddl0)
    ui.wait()

    # ── 6. Pick → tracker ────────────────────────────────────────────────
    ui.banner("Dopo pick: :init cambia, :goal resta lo stesso")
    session.note_completed(PlanStep(primitive="pick", args={"object": "red_cup"}))
    ui.say(f"  tracker: holding={session.holding_object()!r}")
    place_plan = VLMPlan(
        goal=command,
        steps=[
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": "shelf"},
            )
        ],
        raw_output="",
        domain_template="manipulation_base",
    )
    pddl1, scene1 = session.generate_hybrid_problem(
        place_plan,
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="demo_iter_1",
    )
    ui.say("  Nuovo :init (mid-task):")
    ui.code(InitRenderer().render_section(scene1))
    ui.say("  :goal (invariato):")
    ui.code(GoalRenderer().render_section(session.goal_facts or []))
    assert "(holding red_cup)" in pddl1
    ui.wait()

    # ── 7. Verifier FULL ─────────────────────────────────────────────────
    ui.banner("Mode FULL — StateVerifier GREEN / YELLOW / RED (mock VLM)")

    def _fresh_full(client: MockVlmFusionClient) -> HybridProblemSession:
        s = HybridProblemSession(
            mode=HybridMode.FULL,
            command=command,
            vlm_fusion_client=client,
        )
        s.generate_hybrid_problem(
            plan, oracle_scene=oracle, dino_scene=dino, problem_name="demo_full"
        )
        return s

    pick = PlanStep(primitive="pick", args={"object": "red_cup"})

    client_g = MockVlmFusionClient()
    s_g = _fresh_full(client_g)
    pre = s_g.last_fused_scene
    assert pre is not None
    expected = StateVerifier().expect(pre, pick)
    out_g = s_g.process_completed_step(
        pick, observed=expected, pre_scene=pre, success_flag=True
    )
    ui.say("  [A] GREEN — osservato == atteso")
    ui.say(
        f"      verdict={out_g.verdict}  vlm_calls={out_g.vlm_calls}  "
        f"(mock calls={client_g.call_count})"
    )

    client_y = MockVlmFusionClient()
    s_y = _fresh_full(client_y)
    pre_y = s_y.last_fused_scene
    assert pre_y is not None
    expected_y = StateVerifier().expect(pre_y, pick)
    stale = SceneState(
        objects=list(expected_y.objects),
        locations=list(expected_y.locations),
        relations=list(pre_y.relations),
        robot=expected_y.robot,
    )
    out_y = s_y.process_completed_step(
        pick, observed=stale, pre_scene=pre_y, success_flag=True
    )
    ui.say("  [B] YELLOW → mock VLM patch → risolto")
    ui.say(
        f"      verdict={out_y.verdict}  vlm_calls={out_y.vlm_calls}  "
        f"(mock calls={client_y.call_count})"
    )

    client_r = MockVlmFusionClient()
    s_r = _fresh_full(client_r)
    pre_r = s_r.last_fused_scene
    assert pre_r is not None
    out_r = s_r.process_completed_step(
        pick, observed=pre_r, pre_scene=pre_r, success_flag=False
    )
    ui.say("  [C] RED — success_flag=False → replan, 0 VLM")
    ui.say(
        f"      verdict={out_r.verdict}  replan={out_r.replan}  "
        f"vlm_calls={out_r.vlm_calls}"
    )

    client_u = MockVlmFusionClient(patch=patch_to_scene(relations=[]))
    s_u = _fresh_full(client_u)
    pre_u = s_u.last_fused_scene
    assert pre_u is not None
    expected_u = StateVerifier().expect(pre_u, pick)
    stale_u = SceneState(
        objects=list(expected_u.objects),
        locations=list(expected_u.locations),
        relations=list(pre_u.relations),
        robot=expected_u.robot,
    )
    out_u = s_u.process_completed_step(
        pick, observed=stale_u, pre_scene=pre_u, success_flag=True
    )
    ui.say(
        f"  [D] YELLOW non risolto → tracker held  "
        f"(verdict={out_u.verdict}, updated={out_u.tracker_updated})"
    )

    print()
    print("═" * _WIDTH)
    print("  DEMO OK")
    print(f"  Comando usato: {command!r}")
    print("  Smoke CI: python scripts/smoke_hybrid_pipeline.py --mock --full")
    print("═" * _WIDTH)
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Demo narrata hybrid problem generator (mock only)"
    )
    parser.add_argument(
        "--pause",
        action="store_true",
        help="Aspetta Invio tra un passo e l'altro",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Piccola pausa automatica tra i passi",
    )
    parser.add_argument(
        "--command",
        default=_DEFAULT_COMMAND,
        help="Comando NL esplicito per GoalGenerator (default: %(default)r)",
    )
    args = parser.parse_args()
    return run_demo(pause=args.pause, slow=args.slow, command=args.command)


if __name__ == "__main__":
    raise SystemExit(main())
