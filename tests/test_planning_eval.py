"""Offline tests for the planning-only battery (no Gazebo / GPU)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "eval_planning_battery.py"
_SUITE = _REPO / "tests" / "fixtures" / "planning_eval" / "suite_v1.json"

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

import eval_planning_battery as battery  # noqa: E402


def test_suite_covers_the_three_claims():
    suite = battery.load_suite(_SUITE)
    cases = suite["cases"]
    phrasings = {c["phrasing"] for c in cases}
    families = {c["family"] for c in cases}
    assert phrasings == {"explicit", "implicit"}
    assert families == {"template_complete", "needs_enrichment", "ungenerable"}
    complete = [c for c in cases if c["family"] == "template_complete"]
    gap = [c for c in cases if c["family"] == "needs_enrichment"]
    ungen = [c for c in cases if c["family"] == "ungenerable"]
    assert len(cases) == 24
    assert len(complete) == 10 and len(gap) == 10 and len(ungen) == 4
    assert sum(1 for c in complete if c["phrasing"] == "explicit") == 5
    assert sum(1 for c in gap if c["phrasing"] == "explicit") == 5
    assert {c["expect"]["skill"] for c in gap} == {"pour", "stir", "cut", "tilt"}
    assert all(c["expect"].get("required_init") for c in complete + gap)
    assert any(c["family"] == "ungenerable" and c["phrasing"] == "implicit" for c in cases)
    tasks = {c["id"]: c["task"] for c in cases}
    assert "pick" in tasks["explicit_pick"]
    assert "thirsty" in tasks["implicit_thirsty"]
    assert "red cup" in tasks["explicit_place_cup"]
    assert "tea box" in tasks["explicit_cut"]


def test_dry_rows_pass_and_split_refuse_from_plan_success():
    suite = battery.load_suite(_SUITE)
    rows = [
        battery._dry_row(case, arm)
        for arm in ("select", "enrich")
        for case in suite["cases"]
    ]
    assert all(r["ok"] for r in rows)
    claims = battery.aggregate(rows)

    c1 = claims["1_select_only"]["plan_correct"]
    c1_all = c1["all"]
    assert c1_all["n"] == 10 and c1_all["ok"] == 10 and c1_all["pct"] == 100.0
    assert claims["1_select_only"]["incorrect_plan"]["all"]["ok"] == 0
    assert claims["1_select_only"]["init_ok"]["all"]["ok"] == 10
    assert claims["1_select_only"]["goal_ok"]["all"]["ok"] == 10
    assert claims["1_select_only"]["domain_ok"]["all"]["ok"] == 10
    assert claims["1_select_only"]["domain_correct"]["all"]["ok"] == 10
    assert claims["domain_select"]["all"]["all"]["n"] == 48
    assert claims["domain_select"]["all"]["all"]["ok"] == 48

    n_ungen_select = sum(
        1 for r in rows if r["arm"] == "select" and r["family"] == "ungenerable"
    )
    assert n_ungen_select == 4
    assert c1_all["n"] == sum(
        1
        for r in rows
        if r["arm"] == "select" and r["family"] == "template_complete"
    )

    c2 = claims["2_enrichment"]["plan_correct"]["all"]
    assert c2["n"] == 10 and c2["ok"] == 10
    # Claim 2 must not include claim 1's template-complete enrich rows.
    assert claims["2_enrichment"]["side_template_complete_plan_correct"]["all"]["n"] == 10
    assert claims["2_enrichment"]["incorrect_plan"]["all"]["ok"] == 0

    c3_enrich = claims["3_ungenerable"]["enrich"]["refuse"]["all"]
    assert c3_enrich["n"] == 4 and c3_enrich["ok"] == 4
    assert claims["3_ungenerable"]["enrich"]["false_plan"]["all"]["ok"] == 0

    assert c1["explicit"]["n"] == 5 and c1["implicit"]["n"] == 5
    table = battery.format_claims_tables(claims)
    assert "Claim 1" in table and "Claim 3" in table
    assert "Does **not** include claim 1" in table
    assert "implicit" in table and "explicit" in table
    md = battery.format_report_md("ctx", claims, rows)
    assert "## Cases — Claim 1" in md
    assert "## Cases — Claim 2" in md
    assert "### Enrichment authored (claim 2)" in md
    assert "explicit_place" in md
    assert "explicit_pour" in md


def test_false_plan_on_ungenerable_does_not_count_as_plan_success():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "explicit_solder")
    row = battery._dry_row(case, "enrich")
    row["exit_code"] = 0
    row["exit_reason"] = "planned"
    row["fd_actions"] = ["(pick wood_cube table)"]
    row["fd_primitives"] = [{"name": "pick", "args": ["wood_cube"]}]
    row["n_plan_actions"] = 1
    row["refuse_reason"] = None
    scored = battery.attach_score(dict(row))
    assert scored["false_plan"] is True
    assert scored["refused"] is False
    assert scored["ok"] is False

    claims = battery.aggregate([scored])
    # Not present in claim 1/2 numerators (those arms/families don't match).
    assert claims["1_select_only"]["plan_correct"]["all"]["n"] == 0
    assert claims["2_enrichment"]["plan_correct"]["all"]["n"] == 0
    assert claims["3_ungenerable"]["enrich"]["false_plan"]["all"]["ok"] == 1
    assert scored["plan_correct"] is False
    assert scored["incorrect_plan"] is True


def test_select_arm_needs_enrichment_ablation_is_not_claim1_success():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "implicit_thirsty")
    row = battery._dry_row(case, "select")
    assert row["plan_found"] is False
    claims = battery.aggregate([row])
    assert claims["1_select_only"]["plan_correct"]["all"]["n"] == 0
    ablation = claims["1_select_only"]["ablation_needs_enrichment_plan_found"]["all"]
    assert ablation["n"] == 1 and ablation["ok"] == 0


def test_goal_match_accepts_shelf_alias():
    assert battery.goal_matches(
        [["on", "wood_cube", "shelf_b"]],
        [["on", "wood_cube", "shelf"]],
    )


def test_goal_match_parses_raw_fact_and_allows_extras():
    assert battery.goal_matches(
        [
            ["camera-aimed-at", "can"],
            ["_raw_fact", "(tilted can)"],
        ],
        [["tilted", "can"]],
    )


def test_goal_match_accepts_enrichment_fluent_synonym():
    assert battery.goal_matches(
        [["_raw_fact", "(transferred-liquid mug glass)"]],
        [["poured", "mug", "glass"]],
    )


def test_goal_match_workshop_unary_synonyms_and_extra_tool_arg():
    assert battery.goal_matches(
        [["_raw_fact", "(coated wood_board)"]],
        [["painted", "wood_board"]],
    )
    assert battery.goal_matches(
        [["painted", "paint_can", "wood_board"]],
        [["painted", "wood_board"]],
    )
    assert battery.goal_matches(
        [["cut", "scissors", "wood_board"]],
        [["cut-open", "wood_board"]],
    )
    assert battery.goal_matches(
        [["_raw_fact", "(drilled-hole metal_plate)"]],
        [["drilled", "metal_plate"]],
    )
    assert battery.goal_matches(
        [["secured", "clamp", "wood_board"]],
        [["clamped", "wood_board"]],
    )
    # Binary golds stay strict: extra/missing tool args are not a match.
    assert not battery.goal_matches(
        [["poured", "glass"]],
        [["poured", "can", "glass"]],
    )


def test_goal_match_soft_golds_for_implicit_pour():
    soft = [
        [["poured", "can", "glass"]],
        [["poured", "mug", "glass"]],
    ]
    assert battery.goal_matches(
        [
            ["holding", "cup"],
            ["_raw_fact", "(transferred-liquid mug glass)"],
        ],
        [["poured", "can", "glass"]],
        soft,
    )
    # holding-only must not pass even with soft_golds listed.
    assert not battery.goal_matches(
        [["holding", "cup"]],
        [["poured", "can", "glass"]],
        soft,
    )


def test_plan_covers_uses_fd_action_head_for_stack():
    row = {
        "fd_actions": [
            "(pick wood_cube table)",
            "(stack wood_cube blue_box table)",
        ],
        "fd_primitives": [
            {"name": "pick", "args": ["wood_cube", "table"]},
            {"name": "place", "args": ["wood_cube", "blue_box", "table"]},
        ],
    }
    assert battery.plan_covers(row, ["stack"]) is True
    assert battery.plan_covers(row, ["pick", "stack"]) is True


def test_live_style_tilt_and_thirsty_score_as_plan_correct():
    suite = battery.load_suite(_SUITE)
    tilt = next(c for c in suite["cases"] if c["id"] == "explicit_tilt")
    row = battery._dry_row(tilt, "enrich")
    row["goal_facts"] = [
        ["camera-aimed-at", "can"],
        ["_raw_fact", "(tilted can)"],
    ]
    scored = battery.attach_score(dict(row))
    assert scored["goal_ok"] is True
    assert scored["plan_correct"] is True
    assert scored["ok"] is True

    thirsty = next(c for c in suite["cases"] if c["id"] == "implicit_thirsty")
    row = battery._dry_row(thirsty, "enrich")
    row["goal_facts"] = [
        ["holding", "cup"],
        ["_raw_fact", "(transferred-liquid mug glass)"],
    ]
    row["fd_actions"] = [
        "(look-at mug)",
        "(pick mug table)",
        "(pour mug glass)",
    ]
    row["fd_primitives"] = [
        {"name": "pour", "args": ["mug", "glass"]},
    ]
    scored = battery.attach_score(dict(row))
    assert scored["goal_ok"] is True
    assert scored["plan_covers"] is True
    assert scored["plan_correct"] is True
    assert scored["ok"] is True


def test_holding_only_drink_is_still_incorrect():
    suite = battery.load_suite(_SUITE)
    drink = next(c for c in suite["cases"] if c["id"] == "implicit_drink")
    row = battery._dry_row(drink, "enrich")
    row["enrichment_skills"] = []
    row["enrichment_used"] = False
    row["domain_template"] = "containers_manipulation"
    row["pddl_domain"] = battery._dry_domain_text("containers_manipulation")
    row["goal_facts"] = [["holding", "cup"]]
    row["fd_actions"] = ["(pick cup table)"]
    row["fd_primitives"] = [{"name": "pick", "args": ["cup", "table"]}]
    row["n_plan_actions"] = 1
    row["exit_reason"] = "planned"
    scored = battery.attach_score(dict(row))
    assert scored["goal_ok"] is False
    assert scored["plan_correct"] is False
    assert scored["ok"] is False


def test_dry_cli_writes_report(tmp_path: Path):
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry",
            "--out-dir",
            str(out),
            "--cases",
            "explicit_place,implicit_thirsty,explicit_solder",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["dry"] is True
    assert report["passed"] == report["total"] == 6  # 3 cases × 2 arms
    assert "1_select_only" in report["claims"]
    assert (out / "report.md").exists()
    assert (out / "report_cases.md").exists()
    dry_row = report["cases"][0]
    assert dry_row.get("init_facts")
    assert dry_row.get("plan_correct") is True
    assert dry_row.get("incorrect_plan") is False
    assert dry_row.get("init_ok") is True
    assert dry_row.get("goal_ok") is True
    assert dry_row.get("domain_ok") is True
    assert report["scene_source"] == "dino"
    assert report["perception_only"] is True
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "scene_source=dino" in md
    assert "perception_only=true" in md


def _mini_case(**overrides) -> dict:
    case = {
        "id": "explicit_look",
        "task": "look at the red cup",
        "note": "",
        "phrasing": "explicit",
        "family": "template_complete",
        "world": "tabletop",
        "expect": {
            "template": "manipulation_base",
            "select": {"plan": True},
            "enrich": {"plan": True},
        },
    }
    case.update(overrides)
    return case


def _proc(code: int = 1) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["run_loop_host.py"], returncode=code)


def test_row_from_summary_includes_pddl_artifacts(tmp_path: Path):
    problem = """(define (problem loop_fd_iter_1)
  (:domain look-at)
  (:objects
    red_cup - item
    table - location
  )
  (:init
    (on red_cup table)
    (clear red_cup)
  )
  (:goal (and (camera-aimed-at red_cup)))
)
"""
    run_dir = tmp_path / "run"
    iter_dir = run_dir / "iter_01"
    iter_dir.mkdir(parents=True)
    (iter_dir / "problem.pddl").write_text(problem, encoding="utf-8")
    (iter_dir / "init.pddl").write_text("  (:init\n    (on red_cup table)\n  )\n")
    (iter_dir / "goal.pddl").write_text(
        "  (:goal (and (camera-aimed-at red_cup)))\n"
    )
    (iter_dir / "fd_plan.json").write_text(
        json.dumps({"success": False, "error": "search failed"}),
        encoding="utf-8",
    )
    (iter_dir / "domain.pddl").write_text(
        "(define (domain look-at)\n  (:action look-at)\n)\n",
        encoding="utf-8",
    )
    summary = {
        "exit_reason": "fail",
        "success": False,
        "plan_only": True,
        "pddl_problem": problem,
        "pddl_init": "  (:init\n    (on red_cup table)\n    (clear red_cup)\n  )",
        "pddl_goal": "  (:goal (and (camera-aimed-at red_cup)))",
        "pddl_domain": "(define (domain look-at)\n  (:action look-at)\n)\n",
        "init_facts": [["on", "red_cup", "table"], ["clear", "red_cup"]],
        "hybrid_problem_gen": {
            "goal_facts": [["camera-aimed-at", "red_cup"]],
        },
        "fd_actions": [],
        "n_plan_actions": 0,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "eval"

    row = battery.row_from_summary(
        _mini_case(),
        "enrich",
        result=_proc(1),
        wall=1.2,
        summary=summary,
        summary_path=summary_path,
        out_dir=out,
    )
    assert row["pddl_problem"] == problem
    assert ["on", "red_cup", "table"] in row["init_facts"]
    assert row["goal_facts"] == [["camera-aimed-at", "red_cup"]]
    assert "(:goal" in (row["pddl_goal"] or "")
    case_dir = out / "cases" / "enrich_explicit_look"
    assert case_dir.is_dir()
    assert (case_dir / "problem.pddl").is_file()
    assert (case_dir / "init.pddl").is_file()
    assert (case_dir / "goal.pddl").is_file()
    assert (case_dir / "fd_plan.json").is_file()
    assert (case_dir / "domain.pddl").is_file()
    assert row["case_dir"]
    assert "look-at" in (row["pddl_domain"] or "")


def test_row_from_old_summary_without_pddl_fields_does_not_crash():
    summary = {
        "exit_reason": "fail",
        "success": False,
        "hybrid_problem_gen": {"goal_facts": [["on", "wood_cube", "shelf"]]},
    }
    row = battery.row_from_summary(
        _mini_case(id="explicit_place", task="place the cube on the shelf"),
        "enrich",
        result=_proc(1),
        wall=0.5,
        summary=summary,
        summary_path=None,
    )
    assert row["init_facts"] == []
    assert row["pddl_problem"] is None
    assert row["pddl_init"] is None
    assert row["pddl_goal"] is None
    assert row["pddl_domain"] is None
    assert row["goal_facts"] == [["on", "wood_cube", "shelf"]]
    assert row["case_dir"] is None


def test_load_pddl_artifacts_falls_back_to_problem_file(tmp_path: Path):
    problem = """(define (problem old_success)
  (:domain manipulation-base)
  (:objects cup - item table - location)
  (:init (on cup table) (gripper-empty))
  (:goal (and (holding cup)))
)
"""
    run_dir = tmp_path / "old_run"
    iter_dir = run_dir / "iter_01"
    iter_dir.mkdir(parents=True)
    (iter_dir / "problem.pddl").write_text(problem, encoding="utf-8")
    summary_path = run_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")

    art = battery.load_pddl_artifacts({}, summary_path)
    assert art["pddl_problem"] == problem
    assert ["gripper-empty"] in art["init_facts"]
    assert art["pddl_goal_facts"] == [["holding", "cup"]]


def test_resolve_run_settings_keeps_thesis_default():
    defaults = {"scene_source": "dino", "perception_only": True}
    got = battery.resolve_run_settings(
        defaults, scene_source=None, no_perception_only=False
    )
    assert got == {"scene_source": "dino", "perception_only": True}


def test_oracle_ablation_flags_reach_the_loop():
    suite = battery.load_suite(_SUITE)
    case = suite["cases"][0]
    defaults = suite["defaults"]
    thesis = battery.build_loop_cmd(
        case,
        defaults,
        arm="enrich",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source=None,
        perception_only=None,
    )
    assert "--scene-source" in thesis
    assert thesis[thesis.index("--scene-source") + 1] == "dino"
    assert "--perception-only" in thesis

    oracle = battery.build_loop_cmd(
        case,
        defaults,
        arm="enrich",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source="oracle",
        perception_only=False,
    )
    assert oracle[oracle.index("--scene-source") + 1] == "oracle"
    assert "--perception-only" not in oracle
    settings = battery.resolve_run_settings(
        defaults, scene_source="oracle", no_perception_only=True
    )
    assert settings == {"scene_source": "oracle", "perception_only": False}


def test_init_ok_is_required_subset_with_aliases():
    assert battery.facts_contain(
        [
            ["on", "can", "counter"],
            ["on", "glass", "counter"],
            ["clear", "can"],
            ["clear", "glass"],
            ["gripper-empty"],
            ["on", "wall_left", "table"],
        ],
        [
            ["on", "can", "table"],
            ["on", "glass", "table"],
            ["gripper-empty"],
        ],
    )
    assert not battery.facts_contain(
        [["on", "can", "table"], ["gripper-empty"]],
        [["on", "can", "table"], ["on", "glass", "table"]],
    )


def test_incorrect_plan_does_not_count_as_claim2_success():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "implicit_stir")
    row = battery._dry_row(case, "enrich")
    row["fd_actions"] = ["(pour can glass)"]
    row["fd_primitives"] = [{"name": "pour", "args": ["can", "glass"]}]
    row["enrichment_skills"] = ["pour"]
    row["goal_facts"] = [["poured", "can", "glass"]]
    row["pddl_domain"] = battery._dry_domain_text(
        "manipulation_base", skill="pour"
    )
    scored = battery.attach_score(dict(row))
    assert scored["plan_found"] is True
    assert scored["plan_correct"] is False
    assert scored["incorrect_plan"] is True
    assert scored["ok"] is False
    claims = battery.aggregate([scored])
    c2 = claims["2_enrichment"]
    assert c2["plan_found"]["all"]["ok"] == 1
    assert c2["plan_correct"]["all"]["ok"] == 0
    assert c2["incorrect_plan"]["all"]["ok"] == 1


def test_claim2_excludes_template_complete_enrich_rows():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "explicit_place")
    row = battery._dry_row(case, "enrich")
    claims = battery.aggregate([row])
    assert claims["1_select_only"]["plan_correct"]["all"]["n"] == 0
    assert claims["2_enrichment"]["plan_correct"]["all"]["n"] == 0
    assert claims["2_enrichment"]["side_template_complete_plan_correct"]["all"]["n"] == 1
    assert claims["2_enrichment"]["side_template_complete_plan_correct"]["all"]["ok"] == 1


def test_domain_ok_requires_gap_skill_in_domain_file():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "explicit_pour")
    row = battery._dry_row(case, "enrich")
    row["pddl_domain"] = battery._dry_domain_text("manipulation_base", skill=None)
    scored = battery.attach_score(dict(row))
    assert scored["domain_ok"] is False
    assert scored["plan_correct"] is False
    assert scored["incorrect_plan"] is True


def test_parse_arms_default_is_select_enrich():
    assert battery.parse_arms(None) == ["select", "enrich"]
    assert battery.parse_arms("select,enrich") == ["select", "enrich"]
    assert battery.parse_arms("llm_pddl,llm_plan") == ["llm_pddl", "llm_plan"]
    assert battery.parse_arms("select,llm_pddl") == ["select", "llm_pddl"]


def test_build_loop_cmd_baseline_scripts_keep_suite_flags():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "explicit_place")
    defaults = suite["defaults"]
    pddl = battery.build_loop_cmd(
        case,
        defaults,
        arm="llm_pddl",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source=None,
        perception_only=None,
    )
    assert pddl[1].endswith("run_loop_llm_pddl.py")
    assert pddl[pddl.index("--task") + 1] == case["task"]
    assert pddl[pddl.index("--world") + 1] == "tabletop"
    assert pddl[pddl.index("--scene-source") + 1] == "dino"
    assert "--perception-only" in pddl
    assert "--plan-only" in pddl
    assert "--container" in pddl
    assert pddl[pddl.index("--container") + 1] == "vlm_ros2"
    assert "--mock-scene" not in pddl
    assert "--scene-file" not in pddl
    assert "--mock-llm" not in pddl
    assert "--online-enrichment" not in pddl
    assert "--domain-select" not in pddl
    assert "run_loop_host.py" not in " ".join(pddl)

    plan = battery.build_loop_cmd(
        case,
        defaults,
        arm="llm_plan",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source="oracle",
        perception_only=False,
    )
    assert plan[1].endswith("run_loop_llm_plan.py")
    assert plan[plan.index("--scene-source") + 1] == "oracle"
    assert "--perception-only" not in plan
    assert "--control" not in plan
    assert "--online-enrichment" not in plan
    assert "--container" in plan
    assert "--mock-scene" not in plan


def test_baseline_dry_nulls_domain_correct_and_init_ok():
    suite = battery.load_suite(_SUITE)
    place = next(c for c in suite["cases"] if c["id"] == "explicit_place")
    pour = next(c for c in suite["cases"] if c["id"] == "explicit_pour")
    solder = next(c for c in suite["cases"] if c["id"] == "explicit_solder")

    pddl_place = battery._dry_row(place, "llm_pddl")
    assert pddl_place["domain_correct"] is None
    assert pddl_place["init_ok"] is None
    assert pddl_place["plan_correct"] is True
    assert pddl_place["goal_ok"] is True
    assert pddl_place["domain_ok"] is True
    assert pddl_place["ok"] is True

    plan_place = battery._dry_row(place, "llm_plan")
    assert plan_place["domain_correct"] is None
    assert plan_place["init_ok"] is None
    assert plan_place["goal_ok"] is None
    assert plan_place["domain_ok"] is None
    assert plan_place["problem_ok"] is None
    assert plan_place["plan_correct"] is True
    assert plan_place["ok"] is True

    pddl_pour = battery._dry_row(pour, "llm_pddl")
    assert pddl_pour["domain_correct"] is None
    assert pddl_pour["init_ok"] is None
    assert pddl_pour["plan_correct"] is True

    pddl_solder = battery._dry_row(solder, "llm_pddl")
    assert pddl_solder["refused"] is True
    assert pddl_solder["plan_found"] is False
    assert pddl_solder["domain_correct"] is None
    assert pddl_solder["ok"] is True


def test_baseline_rows_do_not_change_claim1_or_claim2_numerators():
    suite = battery.load_suite(_SUITE)
    r0 = [
        battery._dry_row(case, arm)
        for arm in ("select", "enrich")
        for case in suite["cases"]
    ]
    baseline = [
        battery._dry_row(case, arm)
        for arm in ("llm_pddl", "llm_plan")
        for case in suite["cases"]
    ]
    claims_r0 = battery.aggregate(r0)
    claims_mix = battery.aggregate(r0 + baseline)
    assert claims_r0["1_select_only"] == claims_mix["1_select_only"]
    assert claims_r0["2_enrichment"] == claims_mix["2_enrichment"]
    assert claims_r0["3_ungenerable"]["select"] == claims_mix["3_ungenerable"]["select"]
    assert claims_r0["3_ungenerable"]["enrich"] == claims_mix["3_ungenerable"]["enrich"]
    assert claims_r0["domain_select"] == claims_mix["domain_select"]
    assert "llm_pddl" not in claims_mix["3_ungenerable"]
    assert "llm_plan" not in claims_mix["3_ungenerable"]
    b = claims_mix["baseline"]
    assert b["llm_pddl"]["template_complete"]["plan_correct"]["all"]["n"] == 10
    assert b["llm_pddl"]["template_complete"]["plan_correct"]["all"]["ok"] == 10
    assert b["llm_pddl"]["needs_enrichment"]["plan_correct"]["all"]["n"] == 10
    assert b["llm_plan"]["template_complete"]["plan_correct"]["all"]["ok"] == 10
    assert b["llm_pddl"]["ungenerable"]["refuse"]["all"]["n"] == 4
    assert b["llm_pddl"]["ungenerable"]["refuse"]["all"]["ok"] == 4
    table = battery.format_claims_tables(claims_mix)
    assert "Claim 1" in table
    assert "llm_pddl × template_complete" in table
    assert "not Claim 1" in table
    r0_table = battery.format_claims_tables(claims_r0)
    assert "llm_pddl × template_complete" not in r0_table
    md = battery.format_report_md("ctx", claims_mix, r0 + baseline)
    assert "## Cases — Baseline arms" in md
    assert "### llm_pddl × template_complete" in md


def test_dry_cli_baseline_arms_classifies_suite(tmp_path: Path):
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry",
            "--arms",
            "llm_pddl,llm_plan",
            "--out-dir",
            str(out),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["arms"] == ["llm_pddl", "llm_plan"]
    assert report["total"] == 48  # 24 × 2
    assert report["passed"] == 48
    assert all(row.get("domain_correct") is None for row in report["cases"])
    assert all(row.get("init_ok") is None for row in report["cases"])
    plan_rows = [r for r in report["cases"] if r["arm"] == "llm_plan"]
    assert all(r.get("goal_ok") is None for r in plan_rows)
    assert all(r.get("domain_ok") is None for r in plan_rows)
    assert all(r.get("problem_ok") is None for r in plan_rows)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "llm_pddl × template_complete" in md
    assert "llm_plan × needs_enrichment" in md
    assert "Claim 3 — ungenerable (baseline" in md


def test_default_dry_cli_report_has_no_baseline_sections(tmp_path: Path):
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--dry",
            "--out-dir",
            str(out),
            "--cases",
            "explicit_place,implicit_thirsty,explicit_solder",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["arms"] == ["select", "enrich"]
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "llm_pddl × template_complete" not in md
    assert "## Cases — Baseline arms" not in md
    assert report["claims"]["baseline"] == {}
    assert report["claims"]["1_select_only"]["plan_correct"]["all"]["n"] == 1


def test_mock_llm_uses_verbatim_suite_tasks():
    from planner.baselines.mock_llm import case_for_task, suite_task_texts
    from planner.domain_llm import pddl_syntax_errors
    from planner.baselines import mock_llm as mock

    suite = battery.load_suite(_SUITE)
    texts = suite_task_texts()
    assert texts == tuple(c["task"] for c in suite["cases"])
    assert len(texts) == 24
    for case in suite["cases"]:
        assert case_for_task(case["task"])["id"] == case["id"]
        user = f"scene: {{}}\n\ntask: {case['task']}\n"
        if case["family"] == "ungenerable":
            plan = json.loads(mock.generate_plan_mock("", user))
            pddl = json.loads(mock.generate_pddl_mock("", user))
            assert plan["refuse"] is True
            assert pddl["refuse"] is True
        else:
            domain, problem = mock.domain_and_problem_for_case(case)
            assert pddl_syntax_errors(domain) == []
            assert "(:init" in problem and "(:goal" in problem

    place = mock.generate_plan_mock(
        "", "task: place the wood cube on the shelf\n"
    )
    actions = json.loads(place)["actions"]
    assert actions[0]["name"] == "pick"
    assert actions[0]["args"] == ["wood_cube"]


def test_mock_cli_classifies_suite_and_exercises_plan_correct(tmp_path: Path):
    out = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--mock",
            "--out-dir",
            str(out),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    suite = battery.load_suite(_SUITE)
    assert report["mock"] is True
    assert report["dry"] is False
    assert report["arms"] == ["llm_pddl", "llm_plan"]
    assert report["total"] == 48
    assert report["passed"] == 48
    tasks_in_report = {(r["arm"], r["id"], r["task"]) for r in report["cases"]}
    expected = {
        (arm, c["id"], c["task"])
        for arm in ("llm_pddl", "llm_plan")
        for c in suite["cases"]
    }
    assert tasks_in_report == expected

    ungen = [r for r in report["cases"] if r["family"] == "ungenerable"]
    assert len(ungen) == 8
    assert all(r["refused"] for r in ungen)
    assert all(r["false_plan"] is False for r in ungen)

    place = [
        r
        for r in report["cases"]
        if r["id"] == "explicit_place"
    ]
    assert len(place) == 2
    assert all(r["plan_correct"] is True for r in place)
    assert all(r["domain_correct"] is None for r in report["cases"])

    md = (out / "report.md").read_text(encoding="utf-8")
    assert "llm_pddl × template_complete" in md
    assert "llm_plan × needs_enrichment" in md
    assert (out / "runs" / "llm_pddl_explicit_place" / "domain.pddl").is_file()
    assert (out / "runs" / "llm_plan_explicit_place" / "llm_plan.json").is_file()


def test_build_loop_cmd_mock_flags():
    suite = battery.load_suite(_SUITE)
    case = next(c for c in suite["cases"] if c["id"] == "explicit_place")
    cmd = battery.build_loop_cmd(
        case,
        suite["defaults"],
        arm="llm_pddl",
        python_bin="python",
        container="vlm_ros2",
        max_steps=1,
        scene_source=None,
        perception_only=None,
        mock=True,
    )
    assert "--mock-scene" in cmd
    assert "--mock-llm" in cmd
    assert "--mock-fd" in cmd
    assert "--scene-file" in cmd
    assert cmd[cmd.index("--task") + 1] == case["task"]


def test_format_live_baseline_notes_mentions_live_dino():
    notes = battery.format_live_baseline_notes("Qwen/Qwen2.5-7B-Instruct")
    assert "acquire_live_scene" in notes
    assert "not a paper default" in notes.lower() or "not a product default" in notes
    assert "world-faithful" not in notes
    assert "not wired" not in notes


def test_baseline_scene_file_follows_world():
    table = battery.baseline_scene_file("tabletop")
    kitchen = battery.baseline_scene_file("kitchen")
    workshop = battery.baseline_scene_file("workshop")
    assert table.name == "tabletop.json"
    assert kitchen.name == "kitchen.json"
    assert workshop.name == "workshop.json"
    assert table.is_file() and kitchen.is_file() and workshop.is_file()
    fallback = battery.baseline_scene_file("no_such_world")
    assert fallback.name == "wood_cube_table.json"


def test_baseline_tables_include_refuse_split():
    suite = battery.load_suite(_SUITE)
    rows = [
        battery._dry_row(case, arm)
        for arm in ("llm_pddl", "llm_plan")
        for case in suite["cases"]
    ]
    claims = battery.aggregate(rows)
    block = claims["baseline"]["llm_pddl"]["template_complete"]
    assert "refuse" in block
    assert block["refuse"]["all"]["ok"] == 0
    md = battery.format_baseline_tables(claims["baseline"])
    assert "invalid_pddl | refuse" in md
    assert block["refuse"]["explicit"]["n"] == 5


def test_audit_baseline_template_fallback_clean_on_dry():
    suite = battery.load_suite(_SUITE)
    rows = [
        battery._dry_row(case, "llm_pddl")
        for case in suite["cases"]
    ]
    assert battery.audit_baseline_template_fallback(rows) == []
    dirty = dict(rows[0])
    dirty["enrichment_used"] = True
    dirty["domain_template"] = "manipulation_base"
    issues = battery.audit_baseline_template_fallback([dirty])
    assert any("enrichment_used" in i for i in issues)
    assert any("domain_template" in i for i in issues)


