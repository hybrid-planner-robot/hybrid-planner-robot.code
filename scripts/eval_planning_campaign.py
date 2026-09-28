#!/usr/bin/env python3
"""Model campaign: one profile at a time, mock-init.

Default ladder is V4 (three setting suites: tabletop / kitchen / workshop):

  python scripts/eval_planning_campaign.py \\
    --ladder tests/fixtures/planning_eval/model_ladder_v4.json --list

Local profiles unset VLMRP_TEXT_LLM_BASE_URL (transformers).
SSH-tunnel / Ollama ``api`` profiles use a localhost port.
Remote profiles (``kind=remote``) use an OpenAI-compatible HTTP API
and ``VLMRP_TEXT_LLM_API_KEY``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_LADDER = (
    _REPO / "tests" / "fixtures" / "planning_eval" / "model_ladder_v4.json"
)
_BATTERY = _REPO / "scripts" / "eval_planning_battery.py"
_TESLA_USER_ENV = "TESLA_USER"
_API_KEY_ENV = "VLMRP_TEXT_LLM_API_KEY"
_REMOTE_KIND = "remote"
_HTTP_KINDS = frozenset({"tesla", "api", _REMOTE_KIND})


def load_ladder(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def suite_paths(ladder: Mapping[str, Any]) -> list[Path]:
    raw_list = ladder.get("suites")
    if isinstance(raw_list, list) and raw_list:
        return [_REPO / p if not Path(str(p)).is_absolute() else Path(str(p)) for p in raw_list]
    one = ladder.get("suite")
    if one:
        path = Path(str(one))
        return [path if path.is_absolute() else _REPO / path]
    return [_REPO / "tests/fixtures/planning_eval/suite_v2_mock.json"]


def setting_id(suite_path: Path) -> str:
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    world = str((data.get("defaults") or {}).get("world") or "").strip()
    if world:
        return world
    for case in data.get("cases") or []:
        if case.get("world"):
            return str(case["world"])
    return suite_path.stem


def suite_case_ids(suite_path: Path) -> set[str]:
    data = json.loads(suite_path.read_text(encoding="utf-8"))
    return {str(c.get("id") or "") for c in data.get("cases") or [] if c.get("id")}


def cases_for_suite(wanted: str | None, suite_path: Path) -> str | None:
    """Return --cases for this suite, or '' to skip, or None for all cases."""
    if not wanted:
        return None
    ids = [c.strip() for c in wanted.split(",") if c.strip()]
    have = suite_case_ids(suite_path)
    keep = [i for i in ids if i in have]
    return ",".join(keep) if keep else ""


def load_repo_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE from repo ``.env`` without overriding the process env."""
    env_path = path if path is not None else _REPO / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def api_base_url(ladder: Mapping[str, Any], profile: Mapping[str, Any]) -> str:
    kind = str(profile.get("kind") or "")
    if kind == _REMOTE_KIND:
        defaults = ladder.get("defaults") or {}
        url = str(profile.get("base_url") or defaults.get("api_base_url") or "").strip()
        if not url:
            raise SystemExit(
                f"remote profile {profile.get('id')!r} needs base_url "
                "or defaults.api_base_url"
            )
        return url.rstrip("/")
    port = int(profile["port"])
    return f"http://127.0.0.1:{port}/v1"


def require_api_key(env: Mapping[str, str], *, profile_id: str) -> str:
    key = str(env.get(_API_KEY_ENV) or "").strip()
    if key:
        return key
    print(
        f"[FAIL] {profile_id}: missing {_API_KEY_ENV}. "
        "Put it in .env (gitignored) or export it. See .env.example."
    )
    raise SystemExit(2)


def profile_by_id(ladder: Mapping[str, Any], profile_id: str) -> dict[str, Any]:
    for item in ladder.get("profiles") or []:
        if str(item.get("id")) == profile_id:
            return dict(item)
    known = ", ".join(str(p.get("id")) for p in (ladder.get("profiles") or []))
    raise SystemExit(f"unknown profile {profile_id!r}. Known: {known}")


def format_list(ladder: Mapping[str, Any]) -> str:
    lines = [
        f"{ladder.get('name')}: {ladder.get('description', '')}".rstrip(),
        "",
        f"{'id':<20} {'kind':<8} {'arms':<32} model",
        "-" * 88,
    ]
    for item in ladder.get("profiles") or []:
        flag = " (skip)" if item.get("skip_default") else ""
        unconf = " [?GGUF]" if item.get("model_name_unconfirmed") else ""
        lines.append(
            f"{item.get('id', ''):<20} {item.get('kind', ''):<8} "
            f"{str(item.get('arms', '')):<32} {item.get('model', '')}{flag}{unconf}"
        )
    return "\n".join(lines)


def tesla_user() -> str:
    return str(os.environ.get(_TESLA_USER_ENV, "") or "").strip() or "YOUR_USER"


def format_tesla_instructions(
    ladder: Mapping[str, Any], profile: Mapping[str, Any]
) -> str:
    host = str(ladder.get("tesla_host") or "gpu-host.example.com")
    user = tesla_user()
    port = int(profile["port"])
    systemd = str(profile["systemd"])
    alt = str(profile.get("systemd_alt") or "").strip()
    alt_line = f"# if that unit is missing: sudo systemctl start {alt}\n" if alt else ""
    pid = str(profile["id"])
    return f"""# Remote GPU host — profile {pid}
# Set TESLA_USER if you have not already.

export TESLA_USER={user}
TESLA_HOST={host}

# --- on Tesla (ssh) ---
ssh -l "$TESLA_USER" "$TESLA_HOST"
systemctl list-units --type=service | grep llama
nvidia-smi
sudo systemctl start {systemd}
{alt_line}journalctl -u {systemd} -f
# wait for: server listening on 0.0.0.0
# HTTP 503 = still loading. Then Ctrl-C the journal follow.

# --- tunnel on this workstation (leave running) ---
ssh -N -L {port}:localhost:{port} "$TESLA_USER"@$TESLA_HOST

# --- then here ---
python scripts/eval_planning_campaign.py --profile {pid} --probe
python scripts/eval_planning_campaign.py --profile {pid} --smoke
python scripts/eval_planning_campaign.py --profile {pid}

# --- STOP as soon as you pause or finish (other users need the GPUs) ---
sudo systemctl stop {systemd}
"""


def timeout_s(ladder: Mapping[str, Any], profile: Mapping[str, Any]) -> int:
    if profile.get("timeout_s") is not None:
        return int(profile["timeout_s"])
    defaults = ladder.get("defaults") or {}
    return int(defaults.get("timeout_s") or 600)


def temperature(ladder: Mapping[str, Any], profile: Mapping[str, Any]) -> float:
    if profile.get("temperature") is not None:
        return float(profile["temperature"])
    defaults = ladder.get("defaults") or {}
    if defaults.get("temperature") is not None:
        return float(defaults["temperature"])
    return 0.0


def enable_thinking_env(ladder: Mapping[str, Any], profile: Mapping[str, Any]) -> str | None:
    defaults = ladder.get("defaults") or {}
    raw = profile.get("enable_thinking", defaults.get("enable_thinking"))
    if raw is False:
        return "0"
    if raw is True:
        return "1"
    return None


def child_env(
    ladder: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, str]:
    env = dict(os.environ)
    env["VLMRP_TEXT_LLM_MODEL"] = str(profile["model"])
    env["VLMRP_TEXT_LLM_TIMEOUT_S"] = str(timeout_s(ladder, profile))
    env["VLMRP_TEXT_LLM_TEMPERATURE"] = str(temperature(ladder, profile))
    thinking = enable_thinking_env(ladder, profile)
    kind = str(profile.get("kind") or "")
    if kind == _REMOTE_KIND:
        env["VLMRP_TEXT_LLM_BASE_URL"] = api_base_url(ladder, profile)
        if thinking is not None:
            env["VLMRP_TEXT_LLM_ENABLE_THINKING"] = thinking
        else:
            env.pop("VLMRP_TEXT_LLM_ENABLE_THINKING", None)
    elif kind in {"tesla", "api"}:
        port = int(profile["port"])
        env["VLMRP_TEXT_LLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        env.setdefault("VLMRP_TEXT_LLM_API_KEY", "dummy")
        if thinking is not None:
            env["VLMRP_TEXT_LLM_ENABLE_THINKING"] = thinking
    else:
        env.pop("VLMRP_TEXT_LLM_BASE_URL", None)
        env.pop("VLMRP_TEXT_LLM_ENABLE_THINKING", None)
    env["VLMRP_CAMPAIGN_PROFILE"] = str(profile["id"])
    return env


def python_bin() -> str:
    venv = _REPO / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def battery_cmd(
    ladder: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    suite: str,
    arms: str | None,
    cases: str | None,
    out_dir: Path,
    container: str,
    extra: list[str],
) -> list[str]:
    cmd = [
        python_bin(),
        str(_BATTERY),
        "--mock-init",
        "--suite",
        suite,
        "--model",
        str(profile["model"]),
        "--arms",
        arms or str(profile.get("arms") or "r1,enrich"),
        "--container",
        container,
        "--out-dir",
        str(out_dir),
    ]
    if cases:
        cmd.extend(["--cases", cases])
    cmd.extend(extra)
    return cmd


def ensure_fd_container(container: str) -> None:
    probe = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip() == "true":
        print(f"[CAMPAIGN] docker {container} already running")
        return
    print(f"[CAMPAIGN] docker start {container}")
    started = subprocess.run(
        ["docker", "start", container], capture_output=True, text=True, check=False
    )
    if started.returncode != 0:
        raise SystemExit(
            f"cannot start docker container {container}: "
            f"{started.stderr.strip() or started.stdout.strip()}"
        )


def format_api_instructions(profile: Mapping[str, Any]) -> str:
    port = int(profile.get("port") or 11434)
    model = str(profile["model"])
    pid = str(profile["id"])
    return f"""# Local Ollama — profile {pid}

# 1. Install once (needs sudo):
curl -fsSL https://ollama.com/install.sh | sh

# 2. Service (usually starts by itself):
sudo systemctl start ollama
# or: ollama serve

# 3. Pull Q4 (~20 GB). Leave this running until it finishes:
ollama pull {model}

# 4. Check:
ollama list
curl -s http://127.0.0.1:{port}/v1/models

# 5. Battery (this repo):
python scripts/eval_planning_campaign.py --profile {pid} --probe
python scripts/eval_planning_campaign.py --profile {pid} --smoke
python scripts/eval_planning_campaign.py --profile {pid}

# Stop using GPU when done (optional):
# ollama stop {model}
"""


def format_remote_instructions(
    ladder: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    ladder_path: Path | None = None,
) -> str:
    pid = str(profile["id"])
    model = str(profile["model"])
    base = api_base_url(ladder, profile)
    hf = str(profile.get("hf_id") or "").strip()
    hf_line = f"# weights: {hf}\n" if hf else ""
    flag = "tests/fixtures/planning_eval/model_ladder_v3.json"
    if ladder_path is not None:
        flag = _repo_rel(ladder_path)
    return f"""# Remote OpenAI-compatible API — profile {pid}
{hf_line}# Key lives in .env (gitignored) as {_API_KEY_ENV}, never in the ladder.

export {_API_KEY_ENV}=  # already loaded from .env if present

# Probe then smoke then full suite (suite(s) from the ladder):
python scripts/eval_planning_campaign.py \\
  --ladder {flag} \\
  --profile {pid} --probe
python scripts/eval_planning_campaign.py \\
  --ladder {flag} \\
  --profile {pid} --smoke
python scripts/eval_planning_campaign.py \\
  --ladder {flag} \\
  --profile {pid}

# Endpoint: {base}/chat/completions  model={model}
"""


def probe_http(
    ladder: Mapping[str, Any],
    profile: Mapping[str, Any],
    timeout: float,
    *,
    temperature: float = 0.0,
    env: Mapping[str, str] | None = None,
) -> int:
    kind = str(profile.get("kind") or "")
    if kind not in _HTTP_KINDS:
        print("[CAMPAIGN] --probe is for tesla / api / remote (HTTP OpenAI-compatible).")
        return 2
    source = env if env is not None else os.environ
    base = api_base_url(ladder, profile)
    model = str(profile["model"])
    headers = {"Content-Type": "application/json"}
    if kind == _REMOTE_KIND:
        key = require_api_key(source, profile_id=str(profile.get("id") or "remote"))
        headers["Authorization"] = f"Bearer {key}"
    if kind != _REMOTE_KIND:
        print(f"[CAMPAIGN] GET {base}/models  (expect {model})")
        try:
            req = urllib.request.Request(
                f"{base}/models", headers=headers, method="GET"
            )
            with urllib.request.urlopen(req, timeout=min(timeout, 30)) as resp:
                listing = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            print(f"[FAIL] HTTP {exc.code} listing models: {detail}")
            if exc.code == 503:
                print("       model still loading")
            return 2
        except urllib.error.URLError as exc:
            port = int(profile.get("port") or 0)
            print(f"[FAIL] nothing listening on port {port}: {exc}")
            if kind == "tesla":
                print(
                    f"       ssh -N -L {port}:localhost:{port} "
                    '"$TESLA_USER"@gpu-host.example.com'
                )
            else:
                print("       start Ollama: sudo systemctl start ollama")
            return 2
        ids = []
        for row in listing.get("data") or []:
            if isinstance(row, dict) and row.get("id"):
                ids.append(str(row["id"]))
        print(f"       advertised: {ids or listing}")
        if ids and model not in ids:
            print(
                f"[WARN] ladder model {model!r} is not in /v1/models. "
                "Pass the advertised id with --model-override after confirming."
            )
            if profile.get("model_name_unconfirmed"):
                print(
                    "       this profile is flagged model_name_unconfirmed — "
                    "fix the ladder."
                )
    print(f"[CAMPAIGN] POST {base}/chat/completions  model={model}")
    payload_obj: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
        "temperature": float(temperature),
    }
    thinking = enable_thinking_env(ladder, profile)
    if thinking is not None:
        flag = thinking == "1"
        payload_obj["enable_thinking"] = flag
        payload_obj["chat_template_kwargs"] = {"enable_thinking": flag}
    elif kind == "tesla":
        payload_obj["enable_thinking"] = False
        payload_obj["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload_obj).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        print(f"[FAIL] HTTP {exc.code} chat: {detail}")
        return 2
    except urllib.error.URLError as exc:
        print(f"[FAIL] chat connect/timeout: {exc}")
        return 2
    msg = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
    print(f"       reply: {msg!r}"[:400])
    print("[CAMPAIGN] probe ok")
    return 0


def write_all_worlds_note(out: Path, setting_rows: list[dict[str, Any]]) -> None:
    payload = {
        "note": (
            "Operational total across settings. Not a pooled paper statistic — "
            "keep tabletop / kitchen / workshop separate."
        ),
        "settings": setting_rows,
        "passed": sum(int(r.get("passed") or 0) for r in setting_rows),
        "total": sum(int(r.get("total") or 0) for r in setting_rows),
    }
    (out / "all_worlds.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# All worlds (operational note)",
        "",
        payload["note"],
        "",
        "| setting | passed | total | mean llm_s | mean fd_s | exit | suite |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in setting_rows:
        lines.append(
            f"| {row.get('setting')} | {row.get('passed')} | {row.get('total')} | "
            f"{_fmt_mean_s(row.get('mean_llm_s'))} | {_fmt_mean_s(row.get('mean_fd_s'))} | "
            f"{row.get('exit_code')} | {row.get('suite')} |"
        )
    lines += [
        "",
        f"**operational total:** {payload['passed']}/{payload['total']}",
        "",
        "mean llm_s / mean fd_s are per-setting (not pooled).",
        "",
    ]
    (out / "all_worlds.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt_mean_s(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "—"


def _timing_means_from_report(data: Mapping[str, Any]) -> dict[str, Any]:
    timings = data.get("timings") or {}
    all_block = timings.get("all") or {}
    llm = all_block.get("llm_s") or {}
    fd = all_block.get("fd_s") or {}
    by_arm: dict[str, Any] = {}
    for arm, block in (timings.get("by_arm") or {}).items():
        by_arm[str(arm)] = {
            "mean_llm_s": (block.get("llm_s") or {}).get("mean"),
            "median_llm_s": (block.get("llm_s") or {}).get("median"),
            "mean_fd_s": (block.get("fd_s") or {}).get("mean"),
            "median_fd_s": (block.get("fd_s") or {}).get("median"),
        }
    return {
        "mean_llm_s": llm.get("mean"),
        "median_llm_s": llm.get("median"),
        "mean_fd_s": fd.get("mean"),
        "median_fd_s": fd.get("median"),
        "by_arm": by_arm,
    }


def run_battery(cmd: list[str], env: Mapping[str, str]) -> int:
    print("[CAMPAIGN] " + " ".join(cmd))
    print(
        f"[CAMPAIGN] BASE_URL={env.get('VLMRP_TEXT_LLM_BASE_URL', '(unset)')}  "
        f"MODEL={env.get('VLMRP_TEXT_LLM_MODEL')}  "
        f"TIMEOUT={env.get('VLMRP_TEXT_LLM_TIMEOUT_S')}s  "
        f"TEMPERATURE={env.get('VLMRP_TEXT_LLM_TEMPERATURE', '0')}"
    )
    return subprocess.call(cmd, cwd=str(_REPO), env=dict(env))


def default_out_dir(profile_id: str, tag: str, prefix: str = "l2") -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return _REPO / "data" / "planning_eval" / f"{prefix}_{profile_id}_{tag}_{ts}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Planning campaign (one model profile, mock-init)."
    )
    parser.add_argument(
        "--ladder",
        type=Path,
        default=_DEFAULT_LADDER,
        help="Profile JSON (default tests/fixtures/planning_eval/model_ladder_v4.json).",
    )
    parser.add_argument("--list", action="store_true", help="Print profiles and exit")
    parser.add_argument("--profile", default=None, help="Profile id from the ladder")
    parser.add_argument(
        "--print-tesla",
        action="store_true",
        help="Print ssh / systemctl / tunnel commands, or remote API setup",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Hit tunneled /v1/models and a tiny chat completion",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One r1 case (ladder smoke_cases, default explicit_place on v4) before the full run",
    )
    parser.add_argument(
        "--dry-print",
        action="store_true",
        help="Print env + battery command, do not run",
    )
    parser.add_argument("--arms", default=None, help="Override profile arms")
    parser.add_argument("--cases", default=None, help="Override case ids")
    parser.add_argument(
        "--model-override",
        default=None,
        help="Replace the GGUF / HF id from the ladder (after --probe)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--container", default=None)
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Do not docker start the Fast Downward container",
    )
    args, extra = parser.parse_known_args()
    load_repo_dotenv()
    ladder = load_ladder(args.ladder)

    if args.list:
        print(format_list(ladder))
        return 0
    if not args.profile:
        print(format_list(ladder))
        print("\nPass --profile ID. Use --print-tesla on tesla-* before the tunnel.")
        return 2

    profile = profile_by_id(ladder, args.profile)
    if args.model_override:
        profile["model"] = args.model_override.strip()
    defaults = ladder.get("defaults") or {}
    container = args.container or str(defaults.get("container") or "vlm_ros2")

    if args.print_tesla:
        kind = str(profile.get("kind") or "")
        if kind == "tesla":
            print(format_tesla_instructions(ladder, profile))
            return 0
        if kind == "api":
            print(format_api_instructions(profile))
            return 0
        if kind == _REMOTE_KIND:
            print(format_remote_instructions(ladder, profile, ladder_path=args.ladder))
            return 0
        print(f"[FAIL] {profile['id']} is transformers-local — no HTTP server to start")
        return 2

    env = child_env(ladder, profile)
    if args.probe:
        return probe_http(
            ladder,
            profile,
            float(timeout_s(ladder, profile)),
            temperature=temperature(ladder, profile),
            env=env,
        )

    if args.smoke:
        arms = args.arms or str(defaults.get("smoke_arms") or "r1")
        cases = args.cases or str(defaults.get("smoke_cases") or "explicit_pour")
        tag = "smoke"
    else:
        arms = args.arms or str(profile.get("arms") or "r1,enrich")
        cases = args.cases
        tag = "run"
    prefix = str(ladder.get("out_prefix") or "l2")
    out = args.out_dir or default_out_dir(str(profile["id"]), tag, prefix=prefix)
    paths = suite_paths(ladder)
    jobs: list[tuple[str, Path, Path, list[str]]] = []
    for suite_path in paths:
        suite_cases = cases_for_suite(cases, suite_path)
        if suite_cases == "":
            print(
                f"[CAMPAIGN] skip {suite_path.name}: none of {cases} in this setting"
            )
            continue
        world = setting_id(suite_path)
        setting_out = out if len(paths) == 1 else out / world
        cmd = battery_cmd(
            ladder,
            profile,
            suite=str(suite_path),
            arms=arms,
            cases=suite_cases,
            out_dir=setting_out,
            container=container,
            extra=extra,
        )
        jobs.append((world, suite_path, setting_out, cmd))
    if not jobs:
        print("[FAIL] no suites matched --cases")
        return 2
    if args.dry_print:
        for key in (
            "VLMRP_TEXT_LLM_MODEL",
            "VLMRP_TEXT_LLM_BASE_URL",
            "VLMRP_TEXT_LLM_TIMEOUT_S",
            "VLMRP_TEXT_LLM_TEMPERATURE",
            "VLMRP_TEXT_LLM_ENABLE_THINKING",
            "VLMRP_CAMPAIGN_PROFILE",
        ):
            print(f"export {key}={env.get(key, '')}")
        key_set = "set" if str(env.get(_API_KEY_ENV) or "").strip() else "unset"
        print(f"# {_API_KEY_ENV}={key_set}")
        for world, suite_path, setting_out, cmd in jobs:
            print(f"# setting {world} → {setting_out}")
            print(" ".join(cmd))
        return 0
    if str(profile.get("kind") or "") == _REMOTE_KIND:
        require_api_key(env, profile_id=str(profile["id"]))
    if profile.get("model_name_unconfirmed") and not args.model_override:
        print(
            f"[FAIL] {profile['id']} has an unconfirmed GGUF filename. "
            "Run --probe, then --model-override <id from /v1/models>."
        )
        return 2
    if not args.skip_docker:
        ensure_fd_container(container)
    setting_rows: list[dict[str, Any]] = []
    codes: list[int] = []
    for world, suite_path, setting_out, cmd in jobs:
        rc = run_battery(cmd, env)
        codes.append(rc)
        report = setting_out / "report.json"
        passed = total = None
        timing: dict[str, Any] = {}
        if report.is_file():
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                data = {}
            passed = data.get("passed")
            total = data.get("total")
            timing = _timing_means_from_report(data)
        setting_rows.append(
            {
                "setting": world,
                "suite": _repo_rel(suite_path),
                "out_dir": _repo_rel(setting_out),
                "passed": passed,
                "total": total,
                "exit_code": rc,
                **timing,
            }
        )
    if len(jobs) > 1:
        write_all_worlds_note(out, setting_rows)
        print(f"[CAMPAIGN] all-worlds note → {_repo_rel(out / 'all_worlds.md')}")
    if any(rc != 0 for rc in codes):
        return max(codes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
