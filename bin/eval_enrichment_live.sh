#!/bin/bash
# eval_enrichment_live.sh — Reset Gazebo, then run the enrichment live demo suite.
#
# For each case the Python harness resets the world, runs the full host loop
# (select → enrich/refuse → FD → execute) with stdout visible, and scores the
# neurosymbolic path. Gazebo shows the arm for every non-refuse case.
#
# Uso:
#   bin/eval_enrichment_live.sh
#   bin/eval_enrichment_live.sh --cases paraphrase_drink,refuse_solder
#   bin/eval_enrichment_live.sh --interactive
#   bin/eval_enrichment_live.sh --model Qwen/Qwen2.5-7B-Instruct --pause 8
#   bin/eval_enrichment_live.sh --no-reset-sim   # sim already up; only run suite
#
# Extra args after the options above are forwarded to
# scripts/eval_enrichment_live.py unchanged.
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORLD="tabletop"
MODEL="${VLMRP_TEXT_LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
RESET_SIM=1
FORWARD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world) WORLD="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --no-reset-sim) RESET_SIM=0; shift ;;
    -h|--help)
      sed -n '2,22p' "$0"
      exit 0
      ;;
    *)
      FORWARD+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local: >/dev/null 2>&1 || true
fi

if [[ "$RESET_SIM" -eq 1 ]]; then
  echo "[DEMO] Resetting simulation stack (world=$WORLD)…"
  echo "       Tip: tabletop keeps the Gazebo GUI stable under software GL."
  echo "       kitchen is needed for pour/stir but gzclient may crash without"
  echo "       nvidia-container-toolkit — the terminal path still runs."
  bin/reset_and_test_fd.sh --world "$WORLD"
else
  echo "[DEMO] Reusing the running simulation (--no-reset-sim)."
fi

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export VLMRP_TEXT_LLM_MODEL="$MODEL"

echo ""
echo "[DEMO] Starting enrichment live suite…"
echo "       model=$MODEL  world=$WORLD"
echo "       Gazebo stays open — watch each case, then the next starts after a pause."
echo ""

exec python "$REPO_ROOT/scripts/eval_enrichment_live.py" \
  --model "$MODEL" \
  "${FORWARD[@]}"
