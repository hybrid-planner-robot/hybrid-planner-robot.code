#!/usr/bin/env python3
"""
Hybrid problem-generation smoke (Sessions 11–16).

Phase A (MVP): fused init + rule-based / mocked local_llm goal.
Phase B (full): StateVerifier GREEN / YELLOW / RED with Mock VLM.
Session 13: opt-in live Gazebo oracle → SceneState → :init (rule_based only).
Session 14: oracle + real-shaped / live DINO into fusion (still rule_based).
Session 15: opt-in local_llm goal against real/oracle SceneState (once at start).
Session 16: live verifier + VLM-on-YELLOW (``--live-verifier`` / ``--hybrid full``).

  python scripts/smoke_hybrid_pipeline.py --mock
  python scripts/smoke_hybrid_pipeline.py --mock --goal-backend local_llm
  python scripts/smoke_hybrid_pipeline.py --mock --full
  python scripts/smoke_hybrid_pipeline.py --live-oracle
  python scripts/smoke_hybrid_pipeline.py --live-oracle --dry
  python scripts/smoke_hybrid_pipeline.py --live-dino --dry
  python scripts/smoke_hybrid_pipeline.py --live-dino   # needs GPU + scene image
  python scripts/smoke_hybrid_pipeline.py --live-goal --dry
  VLMRP_GOAL_LLM_LIVE=1 python scripts/smoke_hybrid_pipeline.py --live-goal --dry
  python scripts/smoke_hybrid_pipeline.py --live-verifier --dry
  VLMRP_VLM_FUSION_LIVE=1 python scripts/smoke_hybrid_pipeline.py --live-verifier --dry

Exit 0 on success or clear live skip. Exit 2 if no mode flag given.

:init vs oracle checklist (Session 13)
--------------------------------------
Pass when, for a place/pick MVP task on manipulation_base:
  1. Oracle SceneState.objects names appear under PDDL (:objects) as items.
  2. Each oracle RelationFact ``on(obj, loc)`` appears as ``(on obj loc)`` in :init.
  3. ``robot.gripper_empty`` → ``(gripper-empty)`` in :init (when not holding).
  4. Rule-based :goal matches the command (e.g. ``(on red_cup shelf_b)``).
  5. Session 13 path may omit DINO; Session 14 adds DINO into fusion.

Oracle vs DINO vs fused (Session 14)
------------------------------------
Pass when debug/metrics ``scene_compare`` shows all three sides and fusion
keeps oracle ``on()`` over noisy DINO names (e.g. blu_box vs blue_box) without
dropping oracle-only objects. Known failure modes: name mismatch, false
negatives (missing detections), pose drift — see README / design doc.

Live local_llm goal (Session 15)
--------------------------------
Pass when ``--live-goal`` populates ``goal_backend`` / ``goal_backend_used`` /
``goal_fallback_count``; invalid JSON falls back to rule_based; CI keeps
mocked ``local_generate_fn`` unless ``VLMRP_GOAL_LLM_LIVE=1``. Default hybrid
goal backend remains rule_based.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from planner.hybrid_runtime import (  # noqa: E402
    GoalBackend,
    HybridMode,
    HybridProblemSession,
    plan_mvp_compatible,
    partition_gazebo_for_hybrid,
    scene_from_dino_payload,
    scene_from_xyz_poses,
)
from planner.problem_generator import write_problem  # noqa: E402
from planner.problem_generator.init_generator.adapters.dino import (  # noqa: E402
    DinoAdapter,
)
from planner.problem_generator.init_generator.adapters.mock import (  # noqa: E402
    DinoMockAdapter,
    OracleMockAdapter,
)
from planner.problem_generator.init_generator.adapters.oracle import (  # noqa: E402
    OracleAdapter,
)
from planner.problem_generator.init_generator.renderer import InitRenderer  # noqa: E402
from planner.problem_generator.init_generator.schema import SceneState  # noqa: E402
from planner.problem_generator.init_generator.vlm_fusion import (  # noqa: E402
    MockVlmFusionClient,
    patch_to_scene,
)
from planner.state_verifier import StateVerifier  # noqa: E402
from vlm.planner import PlanStep, VLMPlan  # noqa: E402

# Match run_loop_host infra filter (sim-only models that are not graspable items).
_INFRA_MODELS = frozenset(
    {
        "floor",
        "room",
        "ground_plane",
        "sun",
        "robot_pedestal",
        "overview_camera",
        "table",
        "workbench",
        "panda",
        "world",
        "sky",
        "default",
    }
)
_DEFAULT_CONTAINER = "vlm_ros2"
_LIVE_DINO_OUT = _REPO_ROOT / "data" / "hybrid_live_dino"
_LIVE_GOAL_OUT = _REPO_ROOT / "data" / "hybrid_live_goal"
_SIDE_BY_SIDE_FIXTURE = (
    _REPO_ROOT
    / "tests"
    / "fixtures"
    / "hybrid_live_dino_side_by_side.json"
)
_LIVE_GOAL_CHECKLIST_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "hybrid_live_goal_checklist.json"
)

# Session 15 mini checklist: pass / fallback / invalid JSON (MVP commands).
_LIVE_GOAL_CHECKLIST: list[dict[str, Any]] = [
    {
        "id": "place",
        "command": "place the red cup on the shelf",
        "plan": "place",
        "case": "pass",
        "expect_substr": "(on red_cup shelf)",
    },
    {
        "id": "pick",
        "command": "pick the red cup",
        "plan": "pick",
        "case": "pass",
        "expect_substr": "(holding red_cup)",
    },
    {
        "id": "look",
        "command": "look at the blue box",
        "plan": "place",
        "case": "pass",
        "expect_substr": "(camera-aimed-at blue_box)",
    },
    {
        "id": "gibberish",
        "command": "make coffee with the red_cup",
        "plan": "place",
        "case": "fallback_or_empty",
        "expect_substr": None,
    },
    {
        "id": "invalid_json",
        "command": "place the red cup on the shelf",
        "plan": "place",
        "case": "invalid_json",
        "expect_substr": "(on red_cup shelf)",  # rule-based fallback
        "force_bad_json": True,
    },
]


def _plan_place() -> VLMPlan:
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


def _plan_pick_only() -> VLMPlan:
    return VLMPlan(
        goal="pick up the red cup",
        steps=[PlanStep(primitive="pick", args={"object": "red_cup"})],
        raw_output="",
        domain_template="manipulation_base",
    )


def _plan_place_on(location: str) -> VLMPlan:
    return VLMPlan(
        goal=f"place red_cup on {location}",
        steps=[
            PlanStep(primitive="pick", args={"object": "red_cup"}),
            PlanStep(
                primitive="place",
                args={"object": "red_cup", "location": location},
            ),
        ],
        raw_output="",
        domain_template="manipulation_base",
    )


def _docker_exec(container: str, bash_cmd: str, *, sudo_docker: bool) -> subprocess.CompletedProcess:
    prefix = ["sudo", "docker"] if sudo_docker else ["docker"]
    return subprocess.run(
        prefix + ["exec", "-i", container, "bash", "-c", bash_cmd],
        capture_output=True,
        timeout=20,
        check=False,
    )


def _fetch_live_gazebo_models(
    *,
    container: str = _DEFAULT_CONTAINER,
    sudo_docker: bool = False,
) -> dict[str, dict[str, float]] | None:
    """Return Gazebo model xyz map via ``_get_model_states.py``, or None if unavailable."""
    bash_cmd = (
        "source /opt/ros/humble/setup.bash && "
        "source /workspace/ros2_ws/install/setup.bash && "
        "python3 /workspace/scripts/_get_model_states.py"
    )
    try:
        result = _docker_exec(container, bash_cmd, sudo_docker=sudo_docker)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        print(f"[SKIP] live oracle: docker/exec failed ({exc})", file=sys.stderr)
        return None
    if result.returncode != 0:
        err = (result.stderr or b"").decode(errors="replace").strip()
        print(
            f"[SKIP] live oracle: container {container!r} / Gazebo unavailable"
            + (f" — {err}" if err else ""),
            file=sys.stderr,
        )
        return None
    try:
        payload = json.loads((result.stdout or b"").decode().strip())
        models = payload.get("models") or {}
    except (json.JSONDecodeError, AttributeError):
        print("[SKIP] live oracle: could not parse _get_model_states.py JSON", file=sys.stderr)
        return None
    if not models:
        print("[SKIP] live oracle: empty model list from Gazebo", file=sys.stderr)
        return None
    return {str(k): dict(v) for k, v in models.items()}


def _oracle_scene_from_gazebo_models(
    models: dict[str, dict[str, float]],
) -> tuple[SceneState, list[str], str]:
    """
    Map live Gazebo poses → oracle SceneState (no DINO).

    Location models (e.g. ``shelf_b``) become known_locations only; graspables
    default to ``on`` table. Returns ``(scene, known_locations, place_target)``.
    """
    filtered = {
        name: xyz
        for name, xyz in models.items()
        if name not in _INFRA_MODELS and not name.startswith("_")
    }
    item_poses, known_locations = partition_gazebo_for_hybrid(
        filtered,
        base_locations=["table", "shelf"],
    )
    on_surface = {name: "table" for name in item_poses}
    place_target = "shelf_b" if "shelf_b" in known_locations else "shelf"
    scene = scene_from_xyz_poses(
        item_poses,
        known_locations=known_locations,
        on_surface=on_surface,
        gripper_empty=True,
        holding=None,
        domain_template="manipulation_base",
    )
    return scene, known_locations, place_target


def _print_init_vs_oracle_checklist(
    oracle: SceneState,
    fused: SceneState,
    pddl: str,
    *,
    source_label: str,
) -> None:
    """Short oracle objects/relations vs ``:init`` facts report."""
    renderer = InitRenderer()
    init_facts = renderer.render_facts(fused)
    # Prefer facts from the fused scene (authoritative :init); fall back to PDDL
    # :init section only (exclude :goal) when parsing the string.
    init_section = pddl
    if "(:init" in pddl:
        start = pddl.index("(:init")
        end = pddl.find("(:goal", start)
        init_section = pddl[start:end] if end > start else pddl[start:]
    pddl_on = set(re.findall(r"\(on\s+(\S+)\s+(\S+)\)", init_section))
    oracle_on = {
        (r.args[0], r.args[1])
        for r in oracle.relations
        if r.predicate == "on" and len(r.args) >= 2
    }
    missing_on = oracle_on - pddl_on
    print(f"     source={source_label}")
    print(f"     oracle objects={[o.name for o in oracle.objects]}")
    print(f"     oracle locations={[loc.name for loc in oracle.locations]}")
    print(f"     oracle on-relations={sorted(oracle_on)}")
    print(f"     :init facts (fused)={init_facts}")
    print(f"     :init on-facts in PDDL={sorted(pddl_on)}")
    if missing_on:
        print(f"     [WARN] oracle on() missing from PDDL :init: {sorted(missing_on)}")
    else:
        print("     checklist: all oracle on() relations present in :init")
    if "(gripper-empty)" in init_section and oracle.robot.gripper_empty:
        print("     checklist: gripper-empty matches oracle")
    # Oracle-only path: fused should still carry oracle-sourced objects.
    oracle_names = {o.name for o in oracle.objects}
    fused_names = {o.name for o in fused.objects}
    if not oracle_names <= fused_names:
        print(
            f"     [WARN] fused objects missing oracle names: "
            f"{sorted(oracle_names - fused_names)}"
        )
    else:
        print("     checklist: oracle objects ⊆ fused :objects")


def run_live_oracle_smoke(
    *,
    dry: bool = False,
    container: str = _DEFAULT_CONTAINER,
    sudo_docker: bool = False,
    out_dir: Path | None = None,
) -> int:
    """
    Session 13: real (or dry-fixture) oracle → HybridProblemSession → :init.

    No DINO fusion trust, no local_llm, no --hybrid full.
    """
    place_target = "shelf"
    known_locations = ["table", "shelf"]
    source_label = "dry-fixture"

    if dry:
        oracle = OracleAdapter.load()
        known_locations = [loc.name for loc in oracle.locations] or known_locations
        if "shelf" in known_locations:
            place_target = "shelf"
        elif "shelf_b" in known_locations:
            place_target = "shelf_b"
    else:
        models = _fetch_live_gazebo_models(container=container, sudo_docker=sudo_docker)
        if models is None:
            print(
                "[SKIP] live oracle path — fix Docker/Gazebo, or re-run with "
                "--live-oracle --dry to prove OracleAdapter wiring offline.",
                file=sys.stderr,
            )
            return 0
        oracle, known_locations, place_target = _oracle_scene_from_gazebo_models(models)
        source_label = f"gazebo:{container}"
        print(f"[OK]   live Gazebo models: {sorted(models.keys())}")

    command = f"place red_cup on {place_target}"
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command=command,
        domain_template="manipulation_base",
        known_locations=known_locations,
    )
    plan = _plan_place_on(place_target)
    assert plan_mvp_compatible(plan), "live oracle smoke plan must be MVP-compatible"

    # Oracle only — DINO left None (Session 14 democks DINO).
    pddl, fused = session.generate_hybrid_problem(
        plan,
        oracle_scene=oracle,
        dino_scene=None,
        problem_name="live_oracle_iter_0",
    )

    on_goal = f"(on red_cup {place_target})"
    assert on_goal in pddl, f"expected goal {on_goal} in PDDL"
    assert "(gripper-empty)" in pddl
    assert session.goal_backend_used == "rule_based"
    assert "red_cup" in {o.name for o in fused.objects}

    out = out_dir or (_REPO_ROOT / "data" / "hybrid_live_oracle")
    out.mkdir(parents=True, exist_ok=True)
    problem_path = out / "problem.pddl"
    write_problem(
        plan,
        problem_path,
        scene_state=fused,
        goal_facts=session.goal_facts,
        use_hybrid=True,
    )
    debug_path = out / "debug.json"
    debug_path.write_text(
        json.dumps(
            {
                "source": source_label,
                "command": command,
                "place_target": place_target,
                "known_locations": known_locations,
                "oracle_objects": [o.name for o in oracle.objects],
                "oracle_relations": [
                    {"predicate": r.predicate, "args": list(r.args)}
                    for r in oracle.relations
                ],
                "goal_facts": [list(f) for f in (session.goal_facts or [])],
                "metrics": session.metrics_snapshot(),
                "pddl_problem": pddl,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("[OK] smoke_hybrid_pipeline --live-oracle" + (" --dry" if dry else ""))
    print(
        f"     mode={session.mode.value} goal_backend={session.goal_backend_used} "
        f"(DINO=None, verifier=off)"
    )
    print(f"     wrote {problem_path.relative_to(_REPO_ROOT)}")
    print(f"     wrote {debug_path.relative_to(_REPO_ROOT)}")
    _print_init_vs_oracle_checklist(oracle, fused, pddl, source_label=source_label)
    print(f"     metrics={session.metrics_snapshot()}")
    return 0


def _print_scene_compare(compare: dict) -> None:
    for side in ("oracle", "dino", "fused"):
        block = compare.get(side)
        if block is None:
            print(f"     {side}: None")
            continue
        print(
            f"     {side}: objects={block.get('objects')} "
            f"on={block.get('on')} notes={block.get('fusion_notes')}"
        )


def _try_live_dino_detections(
    *,
    image_path: Path,
    object_names: list[str],
) -> list[dict] | None:
    """
    Run PerceptionModule.get_pose on a host image for each name.

    Returns detection payloads or None when GPU/weights/image unavailable.
    """
    if not image_path.is_file():
        print(
            f"[SKIP] live DINO: no scene image at {image_path} "
            "(capture via loop / _capture_scene, or use --dry).",
            file=sys.stderr,
        )
        return None
    try:
        import numpy as np
        import torch
        from PIL import Image

        from vlm.perception import PerceptionModule
    except Exception as exc:  # noqa: BLE001 — smoke skip path
        print(f"[SKIP] live DINO: perception stack import failed ({exc})", file=sys.stderr)
        return None

    if not torch.cuda.is_available():
        print("[SKIP] live DINO: CUDA not available (need host GPU)", file=sys.stderr)
        return None

    try:
        perception = PerceptionModule()
        perception.load()
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] live DINO: PerceptionModule.load failed ({exc})", file=sys.stderr)
        return None

    image = Image.open(image_path).convert("RGB")
    # Identity cam→base + default K: enough to populate _last_detection boxes/scores;
    # Session 14 cares about detections into fusion, not metric pose accuracy.
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    cam_to_base = np.eye(4)
    detections: list[dict] = []
    for name in object_names:
        pose = perception.get_pose(name, image, K, cam_to_base)
        if perception._last_detection is not None:
            det = dict(perception._last_detection)
            if pose is not None:
                det["pose"] = pose
            detections.append(det)
            print(
                f"[OK]   DINO '{name}' score={det.get('score')} "
                f"pose={'yes' if pose else 'no'}"
            )
        else:
            print(f"[WARN] DINO miss for '{name}'")
    if not detections:
        print("[SKIP] live DINO: zero detections on image", file=sys.stderr)
        return None
    return detections


def run_live_dino_smoke(
    *,
    dry: bool = False,
    container: str = _DEFAULT_CONTAINER,
    sudo_docker: bool = False,
    image_path: Path | None = None,
    out_dir: Path | None = None,
    write_fixture: bool = True,
) -> int:
    """
    Session 14: oracle + DINO → fusion → :init (rule_based; no VLM-YELLOW).

    ``--dry`` uses production-shaped fixtures (CI-safe). Live mode needs a
    scene image + GPU + GroundingDINO weights; otherwise skips clearly.
    """
    place_target = "shelf"
    known_locations = ["table", "shelf"]
    source_label = "dry-fixtures"
    dino_source = "fixture:dino_detection_payload.json"
    detections_raw: list[dict] | None = None

    # Oracle: prefer live Gazebo when not dry; fall back to fixture.
    if dry:
        oracle = OracleAdapter.load()
        known_locations = [loc.name for loc in oracle.locations] or known_locations
        place_target = "shelf" if "shelf" in known_locations else place_target
        dino = DinoAdapter.load()
    else:
        models = _fetch_live_gazebo_models(container=container, sudo_docker=sudo_docker)
        if models is None:
            print(
                "[SKIP] live DINO: Gazebo unavailable — use --live-dino --dry "
                "for fixture fusion wiring.",
                file=sys.stderr,
            )
            return 0
        oracle, known_locations, place_target = _oracle_scene_from_gazebo_models(models)
        source_label = f"gazebo:{container}"
        print(f"[OK]   live Gazebo models: {sorted(models.keys())}")

        img = image_path or (_REPO_ROOT / "data" / "scene_overview.png")
        if not img.is_file():
            img = _REPO_ROOT / "data" / "scene.png"
        item_names = [o.name for o in oracle.objects]
        detections_raw = _try_live_dino_detections(image_path=img, object_names=item_names)
        if detections_raw is None:
            return 0
        on_surface = {o.name: (o.location or "table") for o in oracle.objects}
        poses = {
            d["name"]: d["pose"]
            for d in detections_raw
            if isinstance(d.get("pose"), dict) and "x" in d["pose"]
        }
        dino = scene_from_dino_payload(
            detections_raw,
            poses=poses or None,
            on_surface=on_surface,
            known_locations=known_locations,
            domain_template="manipulation_base",
        )
        if dino is None:
            print("[SKIP] live DINO: empty DinoAdapter scene", file=sys.stderr)
            return 0
        dino_source = f"perception:{img.name}"

    command = f"place red_cup on {place_target}"
    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.RULE_BASED,
        command=command,
        domain_template="manipulation_base",
        known_locations=known_locations,
    )
    plan = _plan_place_on(place_target)
    assert plan_mvp_compatible(plan)

    pddl, fused = session.generate_hybrid_problem(
        plan,
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="live_dino_iter_0",
    )
    compare = session.last_scene_compare or {}
    assert session.goal_backend_used == "rule_based"
    assert f"(on red_cup {place_target})" in pddl
    assert "dino" in (fused.meta.sources_used if fused.meta else [])
    assert "oracle" in (fused.meta.sources_used if fused.meta else [])
    # Oracle on(red_cup, table) must survive noisy DINO (sim_like fusion).
    assert ("on", "red_cup", "table") in {
        (r.predicate, r.args[0], r.args[1])
        for r in fused.relations
        if r.predicate == "on" and len(r.args) >= 2
    }

    out = out_dir or _LIVE_DINO_OUT
    out.mkdir(parents=True, exist_ok=True)
    problem_path = out / "problem.pddl"
    write_problem(
        plan,
        problem_path,
        scene_state=fused,
        goal_facts=session.goal_facts,
        use_hybrid=True,
    )
    payload = {
        "source": source_label,
        "dino_source": dino_source,
        "command": command,
        "place_target": place_target,
        "known_locations": known_locations,
        "detections": detections_raw,
        "scene_compare": compare,
        "goal_facts": [list(f) for f in (session.goal_facts or [])],
        "metrics": session.metrics_snapshot(),
        "pddl_problem": pddl,
        "known_failure_modes": [
            "name mismatch (e.g. blu_box vs blue_box) — fusion keeps both + notes",
            "false negatives / missing detections — oracle-only objects retained",
            "pose drift — oracle preferred for on/pose when sim_like=True",
            "camera topics down — live path skips; use --dry or fix Gazebo cameras",
        ],
    }
    debug_path = out / "debug.json"
    debug_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if write_fixture and dry:
        # Checked-in sample for Session 14 done criterion (no GPU required).
        _SIDE_BY_SIDE_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        sample = {
            "description": (
                "Session 14 sample: oracle + DINO fixture → fused SceneState "
                "side-by-side (offline; mirrors live --live-dino --dry)."
            ),
            "scene_compare": compare,
            "goal_facts": payload["goal_facts"],
            "pddl_excerpt_init_on": sorted(
                re.findall(r"\(on\s+(\S+)\s+(\S+)\)", pddl.split("(:goal")[0])
            ),
        }
        _SIDE_BY_SIDE_FIXTURE.write_text(
            json.dumps(sample, indent=2) + "\n", encoding="utf-8"
        )

    print("[OK] smoke_hybrid_pipeline --live-dino" + (" --dry" if dry else ""))
    print(
        f"     mode={session.mode.value} goal_backend={session.goal_backend_used} "
        f"dino_source={dino_source}"
    )
    print(f"     wrote {problem_path.relative_to(_REPO_ROOT)}")
    print(f"     wrote {debug_path.relative_to(_REPO_ROOT)}")
    if write_fixture and dry:
        print(f"     wrote {_SIDE_BY_SIDE_FIXTURE.relative_to(_REPO_ROOT)}")
    _print_scene_compare(compare)
    notes = (compare.get("fused") or {}).get("fusion_notes") or []
    if any("name mismatch" in n for n in notes):
        print("     checklist: name mismatch recorded in fusion_notes")
    print("     checklist: oracle+dino sources in fused scene")
    return 0


def _checklist_mock_generate_fn(row: dict[str, Any]) -> Callable[[str, str], str]:
    """Deterministic generate_fn for CI / --live-goal --dry without LIVE=1."""

    def _gen(system: str, user: str) -> str:
        if row.get("force_bad_json"):
            return "not-json{"
        case_id = row["id"]
        if case_id == "place":
            return json.dumps({"facts": [["on", "red_cup", "shelf"]]})
        if case_id == "pick":
            return json.dumps({"facts": [["holding", "red_cup"]]})
        if case_id == "look":
            return json.dumps({"facts": [["camera-aimed-at", "blue_box"]]})
        if case_id == "gibberish":
            # Valid empty refusal JSON → LocalLLM falls back to rule-based.
            return json.dumps({"facts": []})
        return json.dumps({"facts": [["on", "red_cup", "shelf"]]})

    return _gen


def _plan_for_checklist(row: dict[str, Any], place_target: str) -> VLMPlan:
    if row.get("plan") == "pick":
        return _plan_pick_only()
    return _plan_place_on(place_target)


def run_live_goal_smoke(
    *,
    dry: bool = False,
    container: str = _DEFAULT_CONTAINER,
    sudo_docker: bool = False,
    out_dir: Path | None = None,
    write_fixture: bool = True,
) -> int:
    """
    Session 15: local_llm goal against real/oracle SceneState (once at task start).

    CI-safe default: fixtures + mocked generate_fn.
    Live weights: set ``VLMRP_GOAL_LLM_LIVE=1`` (uses Qwen2.5-1.5B + prompt v2).
    ``invalid_json`` row always uses a bad generate_fn to prove fallback metrics.
    """
    live_weights = os.environ.get("VLMRP_GOAL_LLM_LIVE") == "1"
    known_locations = ["table", "shelf"]
    place_target = "shelf"
    source_label = "dry-fixture"
    dino_scene: SceneState | None = None

    if dry:
        oracle = OracleAdapter.load()
        known_locations = [loc.name for loc in oracle.locations] or known_locations
        if "shelf" in known_locations:
            place_target = "shelf"
        elif "shelf_b" in known_locations:
            place_target = "shelf_b"
        # Prefer fused SceneState from Session 14 dry path when available.
        try:
            dino = DinoAdapter.load()
            dino_scene = dino
            source_label = "dry-fixture+dino"
        except Exception:  # noqa: BLE001 — oracle-only is fine
            dino_scene = None
    else:
        models = _fetch_live_gazebo_models(container=container, sudo_docker=sudo_docker)
        if models is None:
            print(
                "[SKIP] live goal: Gazebo unavailable — re-run with "
                "--live-goal --dry (fixtures ± VLMRP_GOAL_LLM_LIVE=1).",
                file=sys.stderr,
            )
            return 0
        oracle, known_locations, place_target = _oracle_scene_from_gazebo_models(models)
        source_label = f"gazebo:{container}"
        print(f"[OK]   live Gazebo models: {sorted(models.keys())}")

    rows_out: list[dict[str, Any]] = []
    fallback_total = 0
    pass_total = 0

    # One shared client for live rows (avoid reloading weights per command).
    live_generate_fn: Callable[[str, str], str] | None = None
    if live_weights:
        from planner.problem_generator.goal_generator.backends.local_llm import (
            TransformersLocalClient,
        )

        _client = TransformersLocalClient()
        live_generate_fn = lambda system, user: _client.complete(system, user)

    for row in _LIVE_GOAL_CHECKLIST:
        force_bad = bool(row.get("force_bad_json"))
        use_live = live_weights and not force_bad
        local_fn: Callable[[str, str], str] | None
        if use_live:
            local_fn = live_generate_fn
        else:
            local_fn = _checklist_mock_generate_fn(row)

        # Adjust expect substr when place_target is shelf_b.
        expect = row.get("expect_substr")
        if expect and place_target != "shelf" and "shelf)" in expect:
            expect = expect.replace("shelf)", f"{place_target})")

        session = HybridProblemSession(
            mode=HybridMode.MVP,
            goal_backend=GoalBackend.LOCAL_LLM,
            command=row["command"],
            domain_template="manipulation_base",
            known_locations=known_locations,
            local_generate_fn=local_fn,
        )
        plan = _plan_for_checklist(row, place_target)
        assert plan_mvp_compatible(plan)

        try:
            pddl, _fused = session.generate_hybrid_problem(
                plan,
                oracle_scene=oracle,
                dino_scene=dino_scene,
                problem_name=f"live_goal_{row['id']}",
            )
        except ValueError as exc:
            # Unsupported NL after local_llm + rule_based both fail (gibberish).
            if row["case"] != "fallback_or_empty":
                raise
            rows_out.append(
                {
                    "id": row["id"],
                    "command": row["command"],
                    "case": row["case"],
                    "weights": "live" if use_live else "mock",
                    "ok": True,
                    "note": f"both backends refused: {exc}",
                    "goal_facts": [],
                    "goal_backend": "local_llm",
                    "goal_backend_used": session.goal_backend_used,
                    "goal_fallback_count": session.goal_fallback_count,
                    "goal_error": str(exc),
                    "expect_substr": expect,
                }
            )
            pass_total += 1
            continue

        metrics = session.metrics_snapshot()
        fb = int(metrics.get("goal_fallback_count") or 0)
        fallback_total += fb
        ok_case = True
        note = ""

        if row["case"] == "pass":
            if expect and expect not in pddl:
                ok_case = False
                note = f"missing {expect}"
            elif fb:
                # Live model may fall back; still record as soft fail for table.
                ok_case = False
                note = "unexpected fallback on pass case"
            elif metrics.get("goal_backend_used") != "local_llm":
                ok_case = False
                note = f"backend_used={metrics.get('goal_backend_used')}"
        elif row["case"] == "invalid_json":
            if fb < 1:
                ok_case = False
                note = "expected goal_fallback_count>=1"
            elif expect and expect not in pddl:
                ok_case = False
                note = f"fallback goal missing {expect}"
        elif row["case"] == "fallback_or_empty":
            # Accept: fallback to rule-based, empty refusal, or any valid facts.
            if not session.goal_facts and not fb:
                note = "empty goal (refusal)"
            elif fb:
                note = "fallback"
            else:
                note = f"facts={metrics.get('goal_facts')}"

        if ok_case:
            pass_total += 1

        rows_out.append(
            {
                "id": row["id"],
                "command": row["command"],
                "case": row["case"],
                "weights": "live" if use_live else "mock",
                "ok": ok_case,
                "note": note,
                "goal_facts": [list(f) for f in (session.goal_facts or [])],
                "goal_backend": metrics.get("goal_backend"),
                "goal_backend_used": metrics.get("goal_backend_used"),
                "goal_fallback_count": fb,
                "goal_error": metrics.get("goal_error"),
                "expect_substr": expect,
            }
        )

    n = len(rows_out)
    fallback_rate = fallback_total / n if n else 0.0
    payload = {
        "session": 15,
        "source": source_label,
        "live_weights": live_weights,
        "place_target": place_target,
        "known_locations": known_locations,
        "n": n,
        "pass_count": pass_total,
        "fallback_total": fallback_total,
        "fallback_rate": round(fallback_rate, 3),
        "checklist": rows_out,
        "note": (
            "invalid_json always uses mocked bad generate_fn to prove "
            "validate-then-accept + rule_based fallback metrics. "
            "Set VLMRP_GOAL_LLM_LIVE=1 for real Qwen2.5-1.5B on other rows. "
            "rule_based remains hybrid default; this path is opt-in only."
        ),
    }

    out = out_dir or _LIVE_GOAL_OUT
    out.mkdir(parents=True, exist_ok=True)
    debug_path = out / "debug.json"
    debug_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # Last successful place PDDL for inspection (reuse oracle scene).
    place_row = next(r for r in rows_out if r["id"] == "place")
    session_last = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=GoalBackend.LOCAL_LLM,
        command=place_row["command"],
        domain_template="manipulation_base",
        known_locations=known_locations,
        local_generate_fn=(
            live_generate_fn
            if live_weights
            else _checklist_mock_generate_fn(
                next(r for r in _LIVE_GOAL_CHECKLIST if r["id"] == "place")
            )
        ),
    )
    pddl_last, fused_last = session_last.generate_hybrid_problem(
        _plan_place_on(place_target),
        oracle_scene=oracle,
        dino_scene=dino_scene,
        problem_name="live_goal_place",
    )
    problem_path = out / "problem.pddl"
    write_problem(
        _plan_place_on(place_target),
        problem_path,
        scene_state=fused_last,
        goal_facts=session_last.goal_facts,
        use_hybrid=True,
    )
    payload["place_metrics"] = session_last.metrics_snapshot()
    payload["pddl_problem"] = pddl_last
    debug_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if write_fixture and dry and not live_weights:
        _LIVE_GOAL_CHECKLIST_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _LIVE_GOAL_CHECKLIST_FIXTURE.write_text(
            json.dumps(
                {
                    "description": (
                        "Session 15 mini checklist (mock weights; CI-safe). "
                        "Live: VLMRP_GOAL_LLM_LIVE=1 python scripts/"
                        "smoke_hybrid_pipeline.py --live-goal --dry"
                    ),
                    **{k: payload[k] for k in (
                        "session",
                        "source",
                        "live_weights",
                        "n",
                        "pass_count",
                        "fallback_total",
                        "fallback_rate",
                        "checklist",
                        "note",
                    )},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    # Hard asserts for CI dry/mock path (and live when pass cases succeed).
    hard_ids = {"place", "pick", "look", "invalid_json"}
    for r in rows_out:
        if r["id"] in hard_ids and not r["ok"]:
            raise AssertionError(
                f"live-goal checklist {r['id']} failed: {r.get('note')} ({r})"
            )

    print(
        "[OK] smoke_hybrid_pipeline --live-goal"
        + (" --dry" if dry else "")
        + (" LIVE" if live_weights else " (mocked weights)")
    )
    print(
        f"     goal_backend=local_llm  pass={pass_total}/{n}  "
        f"fallback_total={fallback_total}  fallback_rate={fallback_rate:.0%}"
    )
    for r in rows_out:
        status = "PASS" if r["ok"] else "FAIL"
        print(
            f"     [{status}] {r['id']:12} used={r['goal_backend_used']} "
            f"fb={r['goal_fallback_count']} facts={r['goal_facts']}  {r['note']}"
        )
    print(f"     wrote {problem_path.relative_to(_REPO_ROOT)}")
    print(f"     wrote {debug_path.relative_to(_REPO_ROOT)}")
    if write_fixture and dry and not live_weights:
        print(f"     wrote {_LIVE_GOAL_CHECKLIST_FIXTURE.relative_to(_REPO_ROOT)}")
    return 0


def run_mock_smoke(*, goal_backend: str = "rule_based") -> int:
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()

    backend = (
        GoalBackend.LOCAL_LLM
        if goal_backend == "local_llm"
        else GoalBackend.RULE_BASED
    )
    local_fn = None
    if backend == GoalBackend.LOCAL_LLM:
        # Deterministic mock — no GPU / weights; inspect command in the user prompt.
        def local_fn(system: str, user: str) -> str:
            command_line = ""
            for line in user.splitlines():
                if line.lower().startswith("command:"):
                    command_line = line.lower()
                    break
            haystack = command_line or user.lower()
            if "pick" in haystack:
                return json.dumps({"facts": [["holding", "red_cup"]]})
            if "look" in haystack:
                return json.dumps({"facts": [["camera-aimed-at", "red_cup"]]})
            return json.dumps({"facts": [["on", "red_cup", "shelf"]]})

    session = HybridProblemSession(
        mode=HybridMode.MVP,
        goal_backend=backend,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
        local_generate_fn=local_fn,
    )

    plan = _plan_place()
    assert plan_mvp_compatible(plan), "smoke plan must be MVP-compatible"

    # Iteration 0 — initial fused scene + goal once
    pddl0, scene0 = session.generate_hybrid_problem(
        plan,
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="smoke_iter_0",
    )
    assert "(on red_cup table)" in pddl0
    assert "(on red_cup shelf)" in pddl0
    assert "(gripper-empty)" in pddl0
    assert session.goal_facts == [("on", "red_cup", "shelf")]
    goal_locked = list(session.goal_facts)

    # Simulate successful pick → tracker holding
    session.note_completed(PlanStep(primitive="pick", args={"object": "red_cup"}))
    assert session.is_holding
    assert session.holding_object() == "red_cup"

    # Iteration 1 — mid-task: holding in :init, same :goal
    place_plan = VLMPlan(
        goal="place red_cup on shelf",
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
        problem_name="smoke_iter_1",
    )
    assert session.goal_facts == goal_locked, "goal must stay fixed across iterations"
    assert "(holding red_cup)" in pddl1
    assert "(on red_cup shelf)" in pddl1  # goal
    init_facts = InitRenderer().render_facts(scene1)
    assert ("holding", "red_cup") in init_facts

    # Place → tracker empty + moved relation
    session.note_completed(
        PlanStep(primitive="place", args={"object": "red_cup", "location": "shelf"})
    )
    assert not session.is_holding
    assert session.tracker.moved_objects().get("red_cup") == "shelf"

    pddl2, _scene2 = session.generate_hybrid_problem(
        _plan_pick_only(),
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="smoke_iter_2",
    )
    # Goal still locked to original place command (invalidate only on replan)
    assert "(on red_cup shelf)" in pddl2
    assert session.goal_facts == goal_locked

    # Replan regenerates goal from (possibly new) command
    session.invalidate_goal()
    session.command = "pick up the red cup"
    pddl_replan, _ = session.generate_hybrid_problem(
        _plan_pick_only(),
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="smoke_replan",
    )
    assert "(holding red_cup)" in pddl_replan
    assert session.goal_facts == [("holding", "red_cup")]

    print("[OK] smoke_hybrid_pipeline --mock")
    print(f"     mode={session.mode.value} goal_backend={session.goal_backend_used}")
    print(f"     metrics={session.metrics_snapshot()}")
    print(f"     iter0 init has on(red_cup,table); goal on(red_cup,shelf)")
    print(f"     iter1 holding after pick; goal unchanged")
    print(f"     replan regenerates goal → holding")
    print(f"     sample PDDL lines:\n       " + "\n       ".join(pddl0.splitlines()[:8]))
    return 0


def run_full_verifier_smoke() -> int:
    """Phase B: exercise GREEN / YELLOW→GREEN / RED with mocks (no Gazebo)."""
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()
    client = MockVlmFusionClient()

    session = HybridProblemSession(
        mode=HybridMode.FULL,
        goal_backend=GoalBackend.RULE_BASED,
        command="place red_cup on shelf",
        domain_template="manipulation_base",
        known_locations=["table", "shelf"],
        vlm_fusion_client=client,
    )
    assert session.verifier_enabled

    plan = _plan_place()
    pddl0, pre = session.generate_hybrid_problem(
        plan,
        oracle_scene=oracle,
        dino_scene=dino,
        problem_name="smoke_full_0",
    )
    assert "(on red_cup table)" in pddl0

    pick = PlanStep(primitive="pick", args={"object": "red_cup"})
    expected = StateVerifier().expect(pre, pick)

    # --- GREEN: observed matches expectation ---
    green = session.process_completed_step(
        pick,
        observed=expected,
        pre_scene=pre,
        success_flag=True,
        oracle_facts=oracle,
        dino_facts=dino,
        images=["mock_frame.png"],
    )
    assert green.verdict == "GREEN"
    assert green.tracker_updated is True
    assert green.replan is False
    assert green.vlm_calls == 0
    assert session.is_holding
    assert client.call_count == 0

    # Reset for YELLOW / RED scenarios on a fresh session branch
    session_y = HybridProblemSession(
        mode=HybridMode.FULL,
        command="place red_cup on shelf",
        vlm_fusion_client=MockVlmFusionClient(),
    )
    _, pre_y = session_y.generate_hybrid_problem(
        plan, oracle_scene=oracle, dino_scene=dino, problem_name="smoke_full_y"
    )
    expected_y = StateVerifier().expect(pre_y, pick)
    # Stale relation → YELLOW; mock VLM resolves → GREEN
    stale = SceneState(
        objects=list(expected_y.objects),
        locations=list(expected_y.locations),
        relations=list(pre_y.relations),  # still (on red_cup table)
        robot=expected_y.robot,
    )
    yellow = session_y.process_completed_step(
        pick,
        observed=stale,
        pre_scene=pre_y,
        success_flag=True,
        images=["mock_yellow.png"],
        oracle_facts=oracle,
        dino_facts=dino,
    )
    assert yellow.verdict == "GREEN", yellow.mismatches  # resolved by mock VLM
    assert yellow.tracker_updated is True
    assert yellow.vlm_calls == 1
    assert session_y.vlm_fusion_client.call_count == 1  # type: ignore[union-attr]
    assert session_y.is_holding

    # Unresolved YELLOW: noop patch → no tracker update
    session_u = HybridProblemSession(
        mode=HybridMode.FULL,
        command="place red_cup on shelf",
        vlm_fusion_client=MockVlmFusionClient(patch=patch_to_scene(relations=[])),
    )
    _, pre_u = session_u.generate_hybrid_problem(
        plan, oracle_scene=oracle, dino_scene=dino, problem_name="smoke_full_u"
    )
    expected_u = StateVerifier().expect(pre_u, pick)
    stale_u = SceneState(
        objects=list(expected_u.objects),
        locations=list(expected_u.locations),
        relations=list(pre_u.relations),
        robot=expected_u.robot,
    )
    unresolved = session_u.process_completed_step(
        pick, observed=stale_u, pre_scene=pre_u, success_flag=True
    )
    assert unresolved.verdict == "YELLOW"
    assert unresolved.tracker_updated is False
    assert unresolved.replan is False
    assert unresolved.vlm_calls == 1
    assert not session_u.is_holding

    # RED: executor failure → replan, 0 VLM calls
    session_r = HybridProblemSession(
        mode=HybridMode.FULL,
        command="place red_cup on shelf",
        vlm_fusion_client=MockVlmFusionClient(),
    )
    _, pre_r = session_r.generate_hybrid_problem(
        plan, oracle_scene=oracle, dino_scene=dino, problem_name="smoke_full_r"
    )
    red = session_r.process_completed_step(
        pick,
        observed=pre_r,
        pre_scene=pre_r,
        success_flag=False,
    )
    assert red.verdict == "RED"
    assert red.tracker_updated is False
    assert red.replan is True
    assert red.vlm_calls == 0
    assert session_r.vlm_fusion_client.call_count == 0  # type: ignore[union-attr]
    assert session_r.goal_facts is None  # invalidate_goal on RED

    print("[OK] smoke_hybrid_pipeline --mock --full")
    print("     GREEN → 0 VLM, tracker updated")
    print("     YELLOW (resolvable) → 1 VLM, tracker updated")
    print("     YELLOW (unresolved) → 1 VLM, tracker held")
    print("     RED → 0 VLM, replan signal")
    # Aggregate metrics from the sessions exercised above
    metrics = {
        "green_session": session.metrics_snapshot(),
        "yellow_resolved": session_y.metrics_snapshot(),
        "yellow_unresolved": session_u.metrics_snapshot(),
        "red_session": session_r.metrics_snapshot(),
    }
    print(
        "     metrics sample (yellow_resolved): "
        f"vlm_calls={metrics['yellow_resolved']['vlm_call_count']} "
        f"verdicts={metrics['yellow_resolved']['verdict_counts']}"
    )
    return 0


_LIVE_VERIFIER_OUT = _REPO_ROOT / "data" / "hybrid_live_verifier"
_LIVE_VERIFIER_FIXTURE = (
    _REPO_ROOT / "tests" / "fixtures" / "hybrid_live_verifier_verdicts.json"
)


def _stale_pick_observation(expected: SceneState, pre: SceneState) -> SceneState:
    """Robot matches expect after pick, but stale on() remains → YELLOW."""
    from planner.problem_generator.init_generator.schema import Meta

    return SceneState(
        objects=list(expected.objects) or list(pre.objects),
        locations=list(expected.locations) or list(pre.locations),
        relations=list(pre.relations),
        robot=expected.robot,
        domain_template=expected.domain_template or pre.domain_template,
        frame_id=expected.frame_id or pre.frame_id,
        meta=Meta(
            sources_used=["oracle", "fusion"],
            fusion_notes=["stale on() after pick (Session 16 YELLOW probe)"],
        ),
    )


def run_live_verifier_smoke(
    *,
    dry: bool = True,
    write_fixture: bool = True,
) -> int:
    """
    Session 16: controlled GREEN / YELLOW / RED with call-count asserts.

    Default (dry): MockVlmFusionClient — CI-safe, writes verdict fixture.
    ``VLMRP_VLM_FUSION_LIVE=1``: also exercise ``LiveVlmFusionClient`` with an
    injectable complete_fn (no GPU) *or* a real planner if available.
    """
    from planner.problem_generator.init_generator.vlm_fusion import (
        LiveVlmFusionClient,
        MockVlmFusionClient,
    )

    live_env = os.environ.get("VLMRP_VLM_FUSION_LIVE") == "1"
    oracle = OracleMockAdapter.load()
    dino = DinoMockAdapter.load()
    rows: list[dict[str, Any]] = []

    def _run_matrix(client, *, label: str) -> dict[str, Any]:
        session = HybridProblemSession(
            mode=HybridMode.FULL,
            goal_backend=GoalBackend.RULE_BASED,
            command="place red_cup on shelf",
            domain_template="manipulation_base",
            known_locations=["table", "shelf"],
            vlm_fusion_client=client,
        )
        plan = _plan_place()
        _, pre = session.generate_hybrid_problem(
            plan,
            oracle_scene=oracle,
            dino_scene=dino,
            problem_name=f"live_verifier_{label}_0",
        )
        pick = PlanStep(primitive="pick", args={"object": "red_cup"})
        expected = StateVerifier().expect(pre, pick)

        # GREEN
        g = session.process_completed_step(
            pick, observed=expected, pre_scene=pre, success_flag=True
        )
        assert g.verdict == "GREEN" and g.vlm_calls == 0

        # Fresh session for YELLOW (avoid tracker pollution)
        session_y = HybridProblemSession(
            mode=HybridMode.FULL,
            goal_backend=GoalBackend.RULE_BASED,
            command="place red_cup on shelf",
            domain_template="manipulation_base",
            known_locations=["table", "shelf"],
            vlm_fusion_client=client,
        )
        _, pre_y = session_y.generate_hybrid_problem(
            plan, oracle_scene=oracle, dino_scene=dino, problem_name="yv0"
        )
        expected_y = StateVerifier().expect(pre_y, pick)
        stale = _stale_pick_observation(expected_y, pre_y)
        y = session_y.process_completed_step(
            pick, observed=stale, pre_scene=pre_y, success_flag=True
        )
        assert y.vlm_calls == 1, f"{label}: YELLOW must call VLM once, got {y.vlm_calls}"
        assert int(getattr(client, "call_count", 0)) >= 1

        # RED
        session_r = HybridProblemSession(
            mode=HybridMode.FULL,
            goal_backend=GoalBackend.RULE_BASED,
            command="place red_cup on shelf",
            domain_template="manipulation_base",
            known_locations=["table", "shelf"],
            vlm_fusion_client=client,
        )
        _, pre_r = session_r.generate_hybrid_problem(
            plan, oracle_scene=oracle, dino_scene=dino, problem_name="rv0"
        )
        r = session_r.process_completed_step(
            pick, observed=pre_r, pre_scene=pre_r, success_flag=False
        )
        assert r.verdict == "RED" and r.vlm_calls == 0

        return {
            "label": label,
            "green": {
                "verdict": g.verdict,
                "vlm_calls": g.vlm_calls,
                "tracker_updated": g.tracker_updated,
            },
            "yellow": {
                "verdict": y.verdict,
                "vlm_calls": y.vlm_calls,
                "tracker_updated": y.tracker_updated,
                "client_call_count": int(getattr(client, "call_count", 0)),
            },
            "red": {
                "verdict": r.verdict,
                "vlm_calls": r.vlm_calls,
                "replan": r.replan,
            },
            "metrics_yellow_session": session_y.metrics_snapshot(),
        }

    # --- Mock path (always) ---
    mock_client = MockVlmFusionClient()
    mock_row = _run_matrix(mock_client, label="mock")
    rows.append(mock_row)

    # --- Live client with injectable complete_fn (no GPU) ---
    def _resolve_complete(system: str, user: str, images: Any) -> str:
        del system, images
        # Prefer expected holding fact from mismatches / expected block.
        if "holding" in user.lower() or "pick" in user.lower():
            return json.dumps(
                {
                    "relations": [],
                    "robot": {"holding": "red_cup", "gripper_empty": False},
                    "remove_relations": [["on", "red_cup", "table"]],
                }
            )
        return json.dumps({"relations": [["on", "red_cup", "shelf"]]})

    live_client = LiveVlmFusionClient(complete_fn=_resolve_complete)
    live_row = _run_matrix(live_client, label="live_client_injected")
    rows.append(live_row)

    # Optional: real planner weights (heavy). Only when env set AND dry=False
    # or explicitly LIVE — keep CI free.
    real_note = "skipped (set VLMRP_VLM_FUSION_LIVE=1 to exercise LiveVlmFusionClient+planner)"
    if live_env:
        try:
            from vlm.planner import VLMPlanner

            planner = VLMPlanner()
            planner.load()
            real_client = LiveVlmFusionClient.from_planner(planner)
            # Only YELLOW probe with real weights (GREEN/RED already covered).
            session_y = HybridProblemSession(
                mode=HybridMode.FULL,
                goal_backend=GoalBackend.RULE_BASED,
                command="place red_cup on shelf",
                vlm_fusion_client=real_client,
            )
            _, pre_y = session_y.generate_hybrid_problem(
                _plan_place(),
                oracle_scene=oracle,
                dino_scene=dino,
                problem_name="live_vlm_yellow",
            )
            pick = PlanStep(primitive="pick", args={"object": "red_cup"})
            expected_y = StateVerifier().expect(pre_y, pick)
            stale = _stale_pick_observation(expected_y, pre_y)
            y = session_y.process_completed_step(
                pick, observed=stale, pre_scene=pre_y, success_flag=True
            )
            assert y.vlm_calls == 1
            real_note = (
                f"live planner YELLOW ok: verdict={y.verdict} "
                f"vlm_calls={y.vlm_calls} raw_len={len(real_client.last_raw or '')}"
            )
            rows.append(
                {
                    "label": "live_planner",
                    "yellow": {
                        "verdict": y.verdict,
                        "vlm_calls": y.vlm_calls,
                        "client_call_count": real_client.call_count,
                        "last_error": real_client.last_error,
                    },
                    "metrics": session_y.metrics_snapshot(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            real_note = f"live planner skipped/failed: {exc}"
            print(f"[WARN] {real_note}", file=sys.stderr)

    payload = {
        "session": 16,
        "dry": dry,
        "live_env": live_env,
        "real_note": real_note,
        "asserts": {
            "GREEN_vlm_calls": 0,
            "YELLOW_vlm_calls": 1,
            "RED_vlm_calls": 0,
        },
        "rows": rows,
        "note": (
            "Mock path is CI default. LiveVlmFusionClient is wired in "
            "run_loop_host --hybrid full (reuses planning VLM). "
            "VLM never authors :init alone — only YELLOW relation patches."
        ),
    }

    out = _LIVE_VERIFIER_OUT
    out.mkdir(parents=True, exist_ok=True)
    debug_path = out / "debug.json"
    debug_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if write_fixture and dry and not live_env:
        _LIVE_VERIFIER_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _LIVE_VERIFIER_FIXTURE.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    elif write_fixture and dry and live_env:
        live_path = (
            _REPO_ROOT / "tests" / "fixtures" / "hybrid_live_verifier_live_yellow.json"
        )
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        "[OK] smoke_hybrid_pipeline --live-verifier"
        + (" --dry" if dry else "")
        + (" LIVE" if live_env else "")
    )
    for row in rows:
        lab = row["label"]
        if "green" in row:
            print(
                f"     [{lab}] GREEN vlm={row['green']['vlm_calls']}  "
                f"YELLOW vlm={row['yellow']['vlm_calls']} "
                f"verdict={row['yellow']['verdict']}  "
                f"RED vlm={row['red']['vlm_calls']}"
            )
        else:
            print(f"     [{lab}] {row.get('yellow')}")
    print(f"     {real_note}")
    print(f"     wrote {debug_path.relative_to(_REPO_ROOT)}")
    if write_fixture and dry and not live_env:
        print(f"     wrote {_LIVE_VERIFIER_FIXTURE.relative_to(_REPO_ROOT)}")
    elif write_fixture and dry and live_env:
        print("     wrote tests/fixtures/hybrid_live_verifier_live_yellow.json")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid problem-gen smoke (mock + live oracle/DINO/goal/verifier)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline mock pipeline (required for CI / no Gazebo)",
    )
    parser.add_argument(
        "--live-oracle",
        action="store_true",
        help="Session 13: build :init from live Gazebo poses (or --dry fixture)",
    )
    parser.add_argument(
        "--live-dino",
        action="store_true",
        help="Session 14: oracle + DINO into fusion (use --dry without GPU/cameras)",
    )
    parser.add_argument(
        "--live-goal",
        action="store_true",
        help=(
            "Session 15: local_llm goal vs oracle SceneState "
            "(mocked weights unless VLMRP_GOAL_LLM_LIVE=1)"
        ),
    )
    parser.add_argument(
        "--live-verifier",
        action="store_true",
        help=(
            "Session 16: GREEN/YELLOW/RED call-count matrix "
            "(LiveVlmFusionClient; VLMRP_VLM_FUSION_LIVE=1 for real planner)"
        ),
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help=(
            "With --live-oracle/--live-dino/--live-goal/--live-verifier: "
            "use fixtures (no Docker/Gazebo)"
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Scene image for --live-dino (default data/scene_overview.png or scene.png)",
    )
    parser.add_argument(
        "--container",
        default=_DEFAULT_CONTAINER,
        help=f"Docker container for live paths (default {_DEFAULT_CONTAINER})",
    )
    parser.add_argument(
        "--sudo-docker",
        action="store_true",
        help="Prefix docker with sudo (same as run_loop_host)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run Phase B verifier smoke (GREEN/YELLOW/RED with mocks)",
    )
    parser.add_argument(
        "--goal-backend",
        default="rule_based",
        choices=["rule_based", "local_llm"],
        help="Goal backend for the MVP smoke (local_llm uses a mocked generate_fn)",
    )
    args = parser.parse_args()
    if (
        not args.mock
        and not args.live_oracle
        and not args.live_dino
        and not args.live_goal
        and not args.live_verifier
    ):
        print(
            "Specify --mock and/or --live-oracle / --live-dino / --live-goal "
            "/ --live-verifier "
            "(live loop: run_loop_host --hybrid full).",
            file=sys.stderr,
        )
        return 2

    if args.live_oracle:
        rc = run_live_oracle_smoke(
            dry=args.dry,
            container=args.container,
            sudo_docker=args.sudo_docker,
        )
        if rc != 0:
            return rc

    if args.live_dino:
        rc = run_live_dino_smoke(
            dry=args.dry,
            container=args.container,
            sudo_docker=args.sudo_docker,
            image_path=args.image,
        )
        if rc != 0:
            return rc

    if args.live_goal:
        rc = run_live_goal_smoke(
            dry=args.dry,
            container=args.container,
            sudo_docker=args.sudo_docker,
        )
        if rc != 0:
            return rc

    if args.live_verifier:
        rc = run_live_verifier_smoke(dry=args.dry if args.dry else True)
        if rc != 0:
            return rc

    if args.live_oracle or args.live_dino or args.live_goal or args.live_verifier:
        if not args.mock:
            return 0

    rc = run_mock_smoke(goal_backend=args.goal_backend)
    if rc != 0:
        return rc
    if args.full:
        return run_full_verifier_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
