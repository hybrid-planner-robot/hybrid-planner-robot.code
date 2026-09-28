#!/bin/bash
# eval_planning_campaign.sh — mock-init campaign (one model profile).
#
#   bin/eval_planning_campaign.sh --list
#   bin/eval_planning_campaign.sh --profile local-qwen7b
#
# V4 (tabletop + kitchen + workshop, separate reports) is the default ladder:
#   bin/eval_planning_campaign.sh --ladder tests/fixtures/planning_eval/model_ladder_v4.json --list
#   bin/eval_planning_campaign.sh --ladder tests/fixtures/planning_eval/model_ladder_v4.json --profile local-qwen7b --smoke
#
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi
exec python "$REPO_ROOT/scripts/eval_planning_campaign.py" "$@"
