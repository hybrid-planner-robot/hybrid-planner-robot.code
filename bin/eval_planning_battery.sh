#!/bin/bash
# eval_planning_battery.sh — Reset Gazebo, then run the planning-only battery.
#
# Perception (scan + DINO) + plan-only. No arm execution.
# Default arms: select,enrich (R0). Baselines are opt-in via --arms.
#
# Arms:
#   select,enrich     official hybrid (template ± enrich + Fast Downward)
#   llm_plan          text LLM emits a grounded action list
#   llm_pddl          text LLM authors domain+problem, then Fast Downward
#   r1                host with --enrichment-profile r1 (suite v2 / --mock-init)
#
# Artefacts under data/planning_eval/<ts>/
#   report.md / report.json
#   runs/<arm>_<case_id>/
#     llm_pddl: domain.pddl, problem.pddl, fd_plan.json, llm_pddl.json
#     llm_plan: llm_plan.json (always, including refuse / invalid)
#
# Uso:
#   bin/eval_planning_battery.sh --dry
#   bin/eval_planning_battery.sh --mock
#   bin/eval_planning_battery.sh --mock-init --mock-llm --suite tests/fixtures/planning_eval/suite_v2_mock.json
#   bin/eval_planning_battery.sh --arms select
#   bin/eval_planning_battery.sh --arms enrich
#   bin/eval_planning_battery.sh --arms llm_plan
#   bin/eval_planning_battery.sh --arms llm_pddl
#   bin/eval_planning_battery.sh --arms select,enrich
#   bin/eval_planning_battery.sh --arms llm_plan,llm_pddl
#   bin/eval_planning_battery.sh --cases explicit_place,implicit_thirsty,explicit_solder
#   bin/eval_planning_battery.sh --model Qwen/Qwen2.5-7B-Instruct
#   bin/eval_planning_battery.sh --no-reset-sim
#   bin/eval_planning_battery.sh --arms enrich --scene-source oracle --no-perception-only
#
# Extra args after the options above are forwarded to
# scripts/eval_planning_battery.py unchanged.
# Thesis default remains --scene-source dino --perception-only (suite defaults).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORLD="tabletop"
MODEL="${VLMRP_TEXT_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RESET_SIM=1
DRY=0
MOCK=0
ARMS=""
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world) WORLD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --arms)
      ARMS="$2"
      FORWARD+=("$1" "$2")
      shift 2
      ;;
    --no-reset-sim) RESET_SIM=0; shift ;;
    --dry) DRY=1; FORWARD+=("$1"); shift ;;
    --mock) MOCK=1; FORWARD+=("$1"); shift ;;
    --mock-init) MOCK=1; FORWARD+=("$1"); shift ;;
    --mock-llm) FORWARD+=("$1"); shift ;;
    -h|--help)
      sed -n '2,34p' "$0"
      exit 0
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

ARMS_LABEL="${ARMS:-select,enrich (default)}"

if [[ "$DRY" -eq 1 || "$MOCK" -eq 1 ]]; then
  if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
  fi
  echo "[EVAL] Offline battery  arms=$ARMS_LABEL  model=$MODEL"
  exec python "$REPO_ROOT/scripts/eval_planning_battery.py" \
    --model "$MODEL" \
    "${FORWARD[@]}"
fi

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local: >/dev/null 2>&1 || true
fi

if [[ "$RESET_SIM" -eq 1 ]]; then
  echo "[EVAL] Resetting simulation stack (world=$WORLD)…"
  echo "       Plan-only: DINO still runs; the arm is not injected a plan."
  bin/reset_and_test_fd.sh --world "$WORLD"
else
  echo "[EVAL] Reusing the running simulation (--no-reset-sim)."
fi

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export VLMRP_TEXT_LLM_MODEL="$MODEL"

echo ""
echo "[EVAL] Starting planning-only battery…"
echo "       arms=$ARMS_LABEL  model=$MODEL  world=$WORLD"
echo "       plan-only + perception-only (thesis flags unless overridden)"
echo ""

exec python "$REPO_ROOT/scripts/eval_planning_battery.py" \
  --model "$MODEL" \
  "${FORWARD[@]}"
