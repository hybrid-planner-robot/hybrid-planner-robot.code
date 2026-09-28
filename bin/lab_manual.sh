#!/bin/bash
# lab_manual.sh — Manual experimentation cockpit.
#
# Starts (or reuses) the Gazebo/MoveIt/orchestrator stack, opens one terminal
# window per monitoring channel, and a REPL where you type NL tasks.
#
# Uso:
#   bin/lab_manual.sh
#   bin/lab_manual.sh --world tabletop --hybrid mvp
#   bin/lab_manual.sh --world tabletop --hybrid mvp --scene-source dino
#   bin/lab_manual.sh --world kitchen --hybrid mvp --control fd \
#       --online-enrichment --domain-select llm \
#       --text-llm-model Qwen/Qwen2.5-7B-Instruct
#   bin/lab_manual.sh --no-start-sim          # only open monitors + REPL
#   bin/lab_manual.sh --restart-sim           # kill and relaunch sim
#   bin/lab_manual.sh --no-gui                # headless Gazebo (cameras may die)
#
# Canali aperti (una finestra ciascuno):
#   [SIM]     launch log (gzserver / move_group / orchestrator)
#   [STATUS]  /vlm_planner/status
#   [STEPS]   /vlm_planner/step_complete
#   [CAM]     camera topic presence + hz
#   [MODELS]  /gazebo/model_states (names only, throttled)
#   [CMD]     NL task REPL → run_loop_host

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

WORLD="tabletop"
HYBRID="mvp"
GOAL_BACKEND="rule_based"
SCENE_SOURCE="fused"
CONTROL=""
ONLINE_ENRICHMENT=0
DOMAIN_SELECT=""
TEXT_LLM_MODEL=""
CONTAINER="vlm_ros2"
START_SIM=1
RESTART_SIM=0
GUI=true
RVIZ=false
MAX_STEPS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world) WORLD="$2"; shift 2 ;;
    --hybrid) HYBRID="$2"; shift 2 ;;
    --goal-backend) GOAL_BACKEND="$2"; shift 2 ;;
    --scene-source) SCENE_SOURCE="$2"; shift 2 ;;
    --control) CONTROL="$2"; shift 2 ;;
    --online-enrichment) ONLINE_ENRICHMENT=1; shift ;;
    --domain-select) DOMAIN_SELECT="$2"; shift 2 ;;
    --text-llm-model) TEXT_LLM_MODEL="$2"; shift 2 ;;
    --container) CONTAINER="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --no-start-sim) START_SIM=0; shift ;;
    --restart-sim) RESTART_SIM=1; shift ;;
    --no-gui) GUI=false; shift ;;
    --rviz) RVIZ=true; shift ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1 (try --help)" >&2
      exit 2
      ;;
  esac
done

SIM_LOG_HOST="$REPO_ROOT/data/sim_lab.log"
SIM_LOG_CTR="/workspace/data/sim_lab.log"
ROS_SETUP="source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash"

# ── Terminal opener (ptyxis / x-terminal-emulator / gnome-terminal) ─────────
_pick_term() {
  if command -v ptyxis >/dev/null 2>&1; then
    echo ptyxis
  elif command -v gnome-terminal >/dev/null 2>&1; then
    echo gnome-terminal
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    echo xterm-emu
  else
    echo none
  fi
}

TERM_KIND="$(_pick_term)"
if [[ "$TERM_KIND" == "none" ]]; then
  echo "[FAIL] No GUI terminal found (need ptyxis / gnome-terminal / x-terminal-emulator)." >&2
  echo "       On a headless Cursor shell, run this from a desktop terminal instead." >&2
  exit 1
fi

open_term() {
  # open_term TITLE BASH_BODY
  local title="$1"
  local body="$2"
  local runner
  runner=$(cat <<EOF
cd $(printf %q "$REPO_ROOT")
printf '\\033]0;%s\\007' $(printf %q "$title")
set +e
$body
echo
read -r -p '[Enter to close] ' _
EOF
)

  case "$TERM_KIND" in
    ptyxis)
      ptyxis --new-window -s -T "$title" -d "$REPO_ROOT" -- bash -lc "$runner" >/dev/null 2>&1 &
      ;;
    gnome-terminal)
      gnome-terminal --title="$title" --working-directory="$REPO_ROOT" -- bash -lc "$runner" &
      ;;
    xterm-emu)
      x-terminal-emulator -T "$title" -e bash -lc "$runner" &
      ;;
  esac
  sleep 0.2
}

docker_bash() {
  docker exec -i "$CONTAINER" bash -lc "$ROS_SETUP && $*"
}

# ── 1. X11 + container ─────────────────────────────────────────────────────
echo "=== VLM-RobotPlanner manual lab ==="
echo "    world=$WORLD hybrid=$HYBRID control=${CONTROL:-vlm_steps} goal=$GOAL_BACKEND scene=$SCENE_SOURCE"
echo "    enrichment=$ONLINE_ENRICHMENT domain_select=${DOMAIN_SELECT:-rule_based} gui=$GUI"
echo ""

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local: >/dev/null 2>&1 || true
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "[LAB] Starting container..."
  (cd "$REPO_ROOT/docker" && docker compose up -d)
  sleep 2
fi

# ── 2. Simulation ──────────────────────────────────────────────────────────
sim_running() {
  docker exec "$CONTAINER" bash -c 'ps aux | grep -E "[g]zserver|[o]rchestrator" | grep -v grep' >/dev/null 2>&1
}

cameras_ok() {
  docker_bash "timeout 4 ros2 topic list 2>/dev/null | grep -q '/overview_camera/image_raw'" \
    || docker_bash "timeout 4 ros2 topic list 2>/dev/null | grep -q '/wrist_camera/image_raw'"
}

if [[ "$RESTART_SIM" -eq 1 ]]; then
  echo "[LAB] Restarting simulation processes..."
  # Fed through stdin (`bash -s`): with `bash -c '<script>'` the script text is
  # part of the wrapper shell's own cmdline, so the first `pkill -f gzserver`
  # SIGKILLs that shell and the remaining kills never run.
  docker exec -i "$CONTAINER" bash -s <<'CLEANUP' || true
    set +e
    pkill -9 -f gzserver 2>/dev/null
    pkill -9 -f gzclient 2>/dev/null
    pkill -9 -f "ros2 launch" 2>/dev/null
    pkill -9 -f orchestrator 2>/dev/null
    pkill -9 -f move_group 2>/dev/null
    pkill -9 -f spawner 2>/dev/null
    sleep 1
CLEANUP

  survivors="$(docker exec -i "$CONTAINER" bash -s <<'SURVIVORS'
    ps -eo pid,cmd --no-headers 2>/dev/null \
      | grep -E 'gzserver|gzclient|/move_group|orchestrator|ros2 launch' \
      | grep -v grep
SURVIVORS
)" || true
  if [[ -n "$survivors" ]]; then
    echo "[LAB] WARN: sim processes survived the restart — a second stack would" >&2
    echo "      make every pick fail with CONTROL_FAILED (-4):" >&2
    echo "$survivors" >&2
  fi
fi

if [[ "$START_SIM" -eq 1 ]]; then
  if sim_running && [[ "$RESTART_SIM" -eq 0 ]]; then
    echo "[LAB] Simulation already running — reusing."
  else
    echo "[LAB] Launching simulation (log → data/sim_lab.log)..."
    mkdir -p "$REPO_ROOT/data"
    : > "$SIM_LOG_HOST"
    # Headless launch; GUI attaches after the stack is ready (kitchen crash
    # when gzclient opens 4K while panda/controllers spawn).
    docker exec -d "$CONTAINER" bash -lc "
      $ROS_SETUP
      export DISPLAY=\${DISPLAY:-:0}
      export LD_LIBRARY_PATH=/usr/local/nvidia/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      unset LIBGL_ALWAYS_SOFTWARE
      if [[ ! -e /dev/nvidia0 || ! -d /usr/local/nvidia/lib ]]; then
        echo '[FAIL] NVIDIA GPU not visible in the container.' >&2
        echo '       Run: bin/sync_nvidia_gl_libs.sh && (cd docker && docker compose up -d --force-recreate)' >&2
        exit 1
      fi
      ros2 launch vlm_robot_planner_bringup simulation.launch.py \
        world_name:=${WORLD} rviz:=${RVIZ} gui:=false \
        > ${SIM_LOG_CTR} 2>&1
    "
  fi

  echo "[LAB] Waiting for orchestrator..."
  ready=0
  for i in $(seq 1 60); do
    if docker_bash "timeout 3 ros2 topic list 2>/dev/null | grep -q '/vlm_planner/inject_plan'"; then
      echo "[OK]   Orchestrator ready."
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "[WARN] Orchestrator not seen after ~120s — monitors will still open."
  fi

  if cameras_ok; then
    echo "[OK]   Camera topics present."
  else
    echo "[WARN] Wrist/overview camera topics missing."
    echo "       If capture fails, relaunch with GUI/GPU: bin/lab_manual.sh --restart-sim"
  fi

  if [[ "$GUI" == "true" ]] && ! docker_bash 'pgrep -x gzclient >/dev/null'; then
    echo "[LAB] Starting gzclient on NVIDIA GPU…"
    docker exec -d "$CONTAINER" bash -lc \
      "$ROS_SETUP; bash /workspace/scripts/_start_gzclient_gpu.sh > /workspace/data/gzclient_gpu.log 2>&1"
    sleep 4
    if docker_bash 'pgrep -x gzclient >/dev/null'; then
      echo "[OK]   gzclient running."
    else
      echo "[WARN] gzclient exited — see data/gzclient_gpu.log" >&2
    fi
  fi
fi

# ── 3. Monitor windows ─────────────────────────────────────────────────────
echo "[LAB] Opening monitor terminals ($TERM_KIND)..."

open_term "[SIM] launch log" \
  "echo '=== data/sim_lab.log ==='; touch data/sim_lab.log; tail -n 80 -F data/sim_lab.log"

open_term "[STATUS] /vlm_planner/status" \
  "docker exec -it $(printf %q "$CONTAINER") bash -lc $(printf %q "$ROS_SETUP; echo Listening /vlm_planner/status; ros2 topic echo /vlm_planner/status")"

open_term "[STEPS] /vlm_planner/step_complete" \
  "docker exec -it $(printf %q "$CONTAINER") bash -lc $(printf %q "$ROS_SETUP; echo Listening /vlm_planner/step_complete; ros2 topic echo /vlm_planner/step_complete")"

open_term "[CAM] image topics" \
  "docker exec -it $(printf %q "$CONTAINER") bash -lc $(printf %q "$ROS_SETUP; while true; do clear; date; echo; ros2 topic list 2>/dev/null | grep -E \"image_raw|camera\" || echo \"(no camera topics)\"; echo; echo \"--- hz overview (5s) ---\"; timeout 5 ros2 topic hz /overview_camera/image_raw 2>&1 | head -8 || true; echo; echo \"--- hz wrist (5s) ---\"; timeout 5 ros2 topic hz /wrist_camera/image_raw 2>&1 | head -8 || true; sleep 2; done")"

open_term "[MODELS] gazebo models" \
  "docker exec -it $(printf %q "$CONTAINER") bash -lc $(printf %q "$ROS_SETUP; while true; do clear; date; echo; python3 /workspace/scripts/_get_model_states.py 2>/dev/null || echo \"(unavailable)\"; sleep 3; done")"

# ── 4. Command REPL ────────────────────────────────────────────────────────
REPL_EXTRA=""
[[ -n "$CONTROL" ]] && REPL_EXTRA+=" --control $(printf %q "$CONTROL")"
[[ "$ONLINE_ENRICHMENT" -eq 1 ]] && REPL_EXTRA+=" --online-enrichment"
[[ -n "$DOMAIN_SELECT" ]] && REPL_EXTRA+=" --domain-select $(printf %q "$DOMAIN_SELECT")"
[[ -n "$TEXT_LLM_MODEL" ]] && REPL_EXTRA+=" --text-llm-model $(printf %q "$TEXT_LLM_MODEL")"

open_term "[CMD] NL task REPL" \
  "source .venv/bin/activate && python scripts/lab_manual_repl.py --world $(printf %q "$WORLD") --hybrid $(printf %q "$HYBRID") --goal-backend $(printf %q "$GOAL_BACKEND") --scene-source $(printf %q "$SCENE_SOURCE") --max-steps $(printf %q "$MAX_STEPS") --container $(printf %q "$CONTAINER")${REPL_EXTRA}"

echo ""
echo "[OK]   Lab up."
echo "      Type a natural-language task in the [CMD] window, e.g.:"
echo "        place the wood cube on the shelf"
echo "        get me something to drink"
echo "      Watch [STATUS]/[STEPS]/[SIM] for pipeline / motion feedback."
echo "      Host loop stdout/stderr also appears in [CMD]."
echo ""
echo "Tips:"
echo "  /hybrid mvp|full|off   inside REPL"
echo "  /control fd|vlm_steps"
echo "  /enrich on|off   /select llm|rule_based   /model <HF id>"
echo "  /scene fused|dino|oracle   (dino = real-world-like, no Gazebo GT)"
echo "  /world tabletop|kitchen|office inside REPL"
echo "  bin/lab_manual.sh --restart-sim   if cameras died"
echo ""
