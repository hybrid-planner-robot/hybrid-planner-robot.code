"""L-2 campaign ladder + launcher (no GPU, no Tesla)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_LADDER = _REPO / "tests" / "fixtures" / "planning_eval" / "model_ladder_l2.json"
_SCRIPT = _REPO / "scripts" / "eval_planning_campaign.py"


def test_ladder_profiles_are_complete():
    data = json.loads(_LADDER.read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["profiles"]]
    assert ids == [
        "local-qwen15b",
        "local-qwen7b",
        "local-qwen14b",
        "local-qwen32b",
        "tesla-qwen7b",
        "tesla-qwen14b",
        "tesla-qwen32b",
        "tesla-qwen72b",
        "tesla-llama3-8b",
    ]
    for item in data["profiles"]:
        assert item["model"]
        assert item["arms"]
        if item["kind"] in {"tesla", "api"}:
            assert int(item["port"]) > 0
        if item["kind"] == "tesla":
            assert item["systemd"]
            assert item.get("skip_default") is True
    assert data["profiles"][-2]["model_name_unconfirmed"] is True
    assert data["defaults"]["temperature"] == 0
    assert [p["id"] for p in data["profiles"] if not p.get("skip_default")] == [
        "local-qwen7b",
        "local-qwen14b",
        "local-qwen32b",
    ]


def test_campaign_list_and_dry_print():
    listed = subprocess.run(
        [sys.executable, str(_SCRIPT), "--ladder", str(_LADDER), "--list"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0, listed.stdout + listed.stderr
    assert "tesla-qwen7b" in listed.stdout
    assert "local-qwen32b" in listed.stdout

    tesla = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_LADDER),
            "--profile",
            "tesla-qwen7b",
            "--dry-print",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert tesla.returncode == 0, tesla.stdout + tesla.stderr
    assert "8080/v1" in tesla.stdout
    assert "qwen2.5-7b-instruct-q4_k_m.gguf" in tesla.stdout
    assert "export VLMRP_TEXT_LLM_TEMPERATURE=0" in tesla.stdout

    local = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_LADDER),
            "--profile",
            "local-qwen15b",
            "--dry-print",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "VLMRP_TEXT_LLM_BASE_URL": "http://127.0.0.1:9999/v1",
        },
    )
    assert local.returncode == 0, local.stdout + local.stderr
    assert "Qwen/Qwen2.5-1.5B-Instruct" in local.stdout
    # local profile must drop a leftover Tesla tunnel URL
    assert "export VLMRP_TEXT_LLM_BASE_URL=http" not in local.stdout


def test_unconfirmed_gguf_refuses_run_without_override():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_LADDER),
            "--profile",
            "tesla-qwen72b",
            "--skip-docker",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 2
    assert "unconfirmed" in r.stdout.lower() or "unconfirmed" in r.stderr.lower()


def test_print_tesla_mentions_systemctl():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_LADDER),
            "--profile",
            "tesla-qwen32b",
            "--print-tesla",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "sudo systemctl start llama-qwen32b" in r.stdout
    assert "-L 8068:localhost:8068" in r.stdout


def test_print_api_mentions_ollama_pull():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_LADDER),
            "--profile",
            "local-qwen32b",
            "--print-tesla",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ollama pull qwen2.5:32b-instruct-q4_K_M" in r.stdout
    assert "11434" in r.stdout


def test_api_dry_print_sets_ollama_base_url():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_LADDER),
            "--profile",
            "local-qwen32b",
            "--dry-print",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env=os.environ.copy(),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "11434/v1" in r.stdout
    assert "qwen2.5:32b-instruct-q4_K_M" in r.stdout
    assert "export VLMRP_TEXT_LLM_TEMPERATURE=0" in r.stdout


_V3_LADDER = _REPO / "tests" / "fixtures" / "planning_eval" / "model_ladder_v3.json"


def test_v3_ladder_is_7_14_35_glm():
    data = json.loads(_V3_LADDER.read_text(encoding="utf-8"))
    ids = [p["id"] for p in data["profiles"]]
    assert ids == [
        "local-qwen7b",
        "local-qwen14b",
        "lab-qwen36",
        "lab-glm53",
    ]
    assert data["suite"].endswith("suite_v3_mock.json")
    assert data["defaults"]["api_base_url"] == "http://127.0.0.1:8000/v1"
    assert [p["id"] for p in data["profiles"] if not p.get("skip_default")] == ids
    glm = next(p for p in data["profiles"] if p["id"] == "lab-glm53")
    assert glm["timeout_s"] == 900
    assert glm["model"] == "lab-glm53"
    qwen36 = next(p for p in data["profiles"] if p["id"] == "lab-qwen36")
    assert qwen36.get("enable_thinking") is False
    for item in data["profiles"]:
        if item["kind"] == "remote":
            assert "port" not in item
            assert item["model"].startswith("lab-")
        if item["kind"] == "local":
            assert item["model"].startswith("Qwen/")


def test_v3_remote_dry_print_uses_lab_url_not_the_key():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_V3_LADDER),
            "--profile",
            "lab-qwen36",
            "--dry-print",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "VLMRP_TEXT_LLM_API_KEY": "sk-test-not-real"},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "http://127.0.0.1:8000/v1" in r.stdout
    assert "lab-qwen36" in r.stdout
    assert "suite_v3_mock.json" in r.stdout
    assert "sk-test-not-real" not in r.stdout
    assert "# VLMRP_TEXT_LLM_API_KEY=set" in r.stdout
    assert "export VLMRP_TEXT_LLM_ENABLE_THINKING=0" in r.stdout
    assert "export VLMRP_TEXT_LLM_ENABLE_THINKING=1" not in r.stdout


def test_v3_remote_run_without_key_fails():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_V3_LADDER),
            "--profile",
            "lab-glm53",
            "--skip-docker",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "VLMRP_TEXT_LLM_API_KEY": ""},
    )
    assert r.returncode == 2
    blob = r.stdout + r.stderr
    assert "VLMRP_TEXT_LLM_API_KEY" in blob


def test_v3_print_remote_setup():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_V3_LADDER),
            "--profile",
            "lab-glm53",
            "--print-tesla",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "VLMRP_TEXT_LLM_API_KEY": ""},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "lab-glm53" in r.stdout
    assert "127.0.0.1:8000" in r.stdout
    assert "sk-" not in r.stdout


_V4_LADDER = _REPO / "tests" / "fixtures" / "planning_eval" / "model_ladder_v4.json"


def test_v4_ladder_has_three_setting_suites():
    data = json.loads(_V4_LADDER.read_text(encoding="utf-8"))
    assert data["out_prefix"] == "v4"
    assert [p["id"] for p in data["profiles"]] == [
        "local-qwen7b",
        "local-qwen14b",
        "lab-qwen36",
        "lab-glm53",
    ]
    suites = data["suites"]
    assert len(suites) == 3
    assert suites[0].endswith("suite_v4_tabletop_mock.json")
    assert suites[1].endswith("suite_v4_kitchen_mock.json")
    assert suites[2].endswith("suite_v4_workshop_mock.json")
    assert data["defaults"]["smoke_cases"] == "explicit_place"
    assert "suite" not in data
    qwen36 = next(p for p in data["profiles"] if p["id"] == "lab-qwen36")
    assert qwen36.get("enable_thinking") is False
    glm = next(p for p in data["profiles"] if p["id"] == "lab-glm53")
    assert "enable_thinking" not in glm


def test_v4_dry_print_emits_three_setting_commands():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_V4_LADDER),
            "--profile",
            "lab-qwen36",
            "--dry-print",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "VLMRP_TEXT_LLM_API_KEY": "sk-test-not-real"},
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("# setting ") == 3
    assert "setting tabletop" in r.stdout
    assert "setting kitchen" in r.stdout
    assert "setting workshop" in r.stdout
    assert "suite_v4_tabletop_mock.json" in r.stdout
    assert "suite_v4_kitchen_mock.json" in r.stdout
    assert "suite_v4_workshop_mock.json" in r.stdout
    assert "sk-test-not-real" not in r.stdout
    assert "export VLMRP_TEXT_LLM_ENABLE_THINKING=0" in r.stdout


def test_v4_smoke_dry_print_uses_explicit_place_on_each_setting():
    r = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--ladder",
            str(_V4_LADDER),
            "--profile",
            "local-qwen7b",
            "--smoke",
            "--dry-print",
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("--cases explicit_place") == 3


def test_all_worlds_note_is_operational_only(tmp_path: Path):
    sys.path.insert(0, str(_REPO / "scripts"))
    import eval_planning_campaign as campaign  # noqa: E402

    campaign.write_all_worlds_note(
        tmp_path,
        [
            {
                "setting": "tabletop",
                "passed": 80,
                "total": 80,
                "exit_code": 0,
                "suite": "suite_v4_tabletop_mock.json",
            },
            {
                "setting": "kitchen",
                "passed": 120,
                "total": 120,
                "exit_code": 0,
                "suite": "suite_v4_kitchen_mock.json",
            },
            {
                "setting": "workshop",
                "passed": 119,
                "total": 120,
                "exit_code": 1,
                "suite": "suite_v4_workshop_mock.json",
            },
        ],
    )
    data = json.loads((tmp_path / "all_worlds.json").read_text())
    assert data["passed"] == 319
    assert data["total"] == 320
    assert "Not a pooled paper statistic" in data["note"]
    md = (tmp_path / "all_worlds.md").read_text()
    assert "operational total" in md.lower()
    assert "tabletop" in md and "kitchen" in md and "workshop" in md
    assert "mean llm_s" in md
    assert "—" in md
