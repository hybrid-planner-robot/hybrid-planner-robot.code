#!/bin/bash
# reset_and_test_fd.sh — Hard-reset Docker/ROS/Gazebo, then (optionally) run FD smoke.
#
# Follows the same cleanup rules as bin/start_sim.sh:
#   - do NOT docker compose down (ipc:host + orphan DDS semaphores)
#   - kill sim processes inside the container
#   - wipe FastRTPS / Gazebo locks on host /dev/shm
#   - relaunch simulation in background
#   - wait until orchestrator + camera frames are alive
#
# Uso:
#   bin/reset_and_test_fd.sh
#       # reset + start sim only (no loop)
#   bin/reset_and_test_fd.sh --run
#       # reset + start sim + run Session 22 FD smoke
#   bin/reset_and_test_fd.sh --run --world tabletop --task "place the wood cube on the shelf"
#   bin/reset_and_test_fd.sh --no-gui          # headless (cameras may fail — prefer GUI)
#   bin/reset_and_test_fd.sh --reset-only      # kill/clean only, do not relaunch sim
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONTAINER="${VLMRP_CONTAINER:-vlm_ros2}"
WORLD="tabletop"
# Prefer wood_cube (5 cm wooden cube): top_down grasp is OK; blue_box path is flaky.
# red_cup often wants side grasp; grasp_mode under control=fd is still top_down until Session 24.
TASK="place the wood cube on the shelf"
HYBRID="mvp"
CONTROL="fd"
MAX_STEPS=10
RUN_TEST=0
RESET_ONLY=0
GUI=true
RVIZ=false
WAIT_SECS=180
# Per-run log name. A fixed shared name is unsafe: two `docker exec -d … > file`
# writers keep independent offsets and inflate it into a sparse multi-GB file.
SIM_LOG_NAME="sim_reset_fd_$(date +%Y-%m-%d_%H-%M-%S).log"
SIM_LOG_HOST="$REPO_ROOT/data/$SIM_LOG_NAME"
SIM_LOG_CTR="/workspace/data/$SIM_LOG_NAME"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --world)       WORLD="$2"; shift 2 ;;
    --task)        TASK="$2"; shift 2 ;;
    --hybrid)      HYBRID="$2"; shift 2 ;;
    --control)     CONTROL="$2"; shift 2 ;;
    --max-steps)   MAX_STEPS="$2"; shift 2 ;;
    --container)   CONTAINER="$2"; shift 2 ;;
    --run)         RUN_TEST=1; shift ;;
    --reset-only)  RESET_ONLY=1; shift ;;
    --no-gui)      GUI=false; shift ;;
    --rviz)        RVIZ=true; shift ;;
    --wait-secs)   WAIT_SECS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *)
      echo "[ERR] Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ROS_SETUP='source /opt/ros/humble/setup.bash && source /workspace/ros2_ws/install/setup.bash'

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  RESET SIM + optional FD smoke                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "    container=$CONTAINER  world=$WORLD  gui=$GUI"
echo "    run_test=$RUN_TEST  control=$CONTROL  hybrid=$HYBRID"
echo ""

# ── 1. X11 ─────────────────────────────────────────────────────────────────
if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local: >/dev/null 2>&1 || true
fi

# ── 2. Host DDS / Gazebo lock cleanup (BEFORE touching the container) ──────
echo "[RESET] Cleaning host FastRTPS / Gazebo shared memory…"
rm -f /dev/shm/fastrtps_* 2>/dev/null || true
rm -f /dev/shm/sem.fastrtps_* 2>/dev/null || true
rm -f /dev/shm/ros_* /dev/shm/*.shm 2>/dev/null || true
rm -f /tmp/.gazebo_master.lock /tmp/gazebo_*.lock 2>/dev/null || true

# ── 3. Ensure container is up ──────────────────────────────────────────────
if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "[RESET] Starting container $CONTAINER…"
  (cd "$REPO_ROOT/docker" && docker compose up -d)
  sleep 2
else
  echo "[RESET] Container $CONTAINER already running."
fi

# ── 4. Kill everything sim-related inside the container ────────────────────
echo "[RESET] Killing Gazebo / ROS / MoveIt / orchestrator inside container…"
# The script is fed through stdin (`bash -s`), NOT as a `bash -c '<script>'`
# argument.  With `-c` the whole script text lands in the wrapper shell's own
# /proc/PID/cmdline, so the first `pkill -f gzserver` matches that shell and
# SIGKILLs it — every later line silently never runs, leaving move_group /
# orchestrator / robot_state_publisher alive and producing two competing stacks.
docker exec -i "$CONTAINER" bash -s <<'CLEANUP' || true
  set +e
  pkill -9 -f gzserver 2>/dev/null
  pkill -9 -f gzclient 2>/dev/null
  pkill -9 -f "ros2 launch" 2>/dev/null
  pkill -9 -f orchestrator 2>/dev/null
  pkill -9 -f move_group 2>/dev/null
  pkill -9 -f spawner 2>/dev/null
  pkill -9 -f controller_manager 2>/dev/null
  pkill -9 -f robot_state_pub 2>/dev/null
  pkill -9 -f static_transform 2>/dev/null
  pkill -9 -f component_container 2>/dev/null
  pkill -9 -f robot_state_publisher 2>/dev/null
  pkill -9 -f joint_state 2>/dev/null
  pkill -9 -f gazebo_ros 2>/dev/null
  sleep 1
  # leftover children
  pkill -9 -f gzserver 2>/dev/null
  pkill -9 -f orchestrator 2>/dev/null
  sleep 0.5
  rm -f /tmp/ros_* /tmp/fastdds_* /tmp/.ros_* 2>/dev/null
  source /opt/ros/humble/setup.bash 2>/dev/null
  ros2 daemon stop 2>/dev/null
  sleep 0.3
  ros2 daemon start 2>/dev/null
  echo "[RESET] inside-container cleanup done"
CLEANUP

# ── 4b. Verify the cleanup actually happened ───────────────────────────────
# A survivor here silently turns into a second stack: two move_group nodes both
# send FollowJointTrajectory to the single controller, the second goal preempts
# the first, and every pick fails with error_code=-4 (CONTROL_FAILED).
survivors="$(docker exec -i "$CONTAINER" bash -s <<'SURVIVORS'
  ps -eo pid,cmd --no-headers 2>/dev/null \
    | grep -E 'gzserver|gzclient|/move_group|orchestrator|robot_state_publisher|static_transform_publisher|ros2 launch' \
    | grep -v grep
SURVIVORS
)" || true

if [[ -n "$survivors" ]]; then
  echo "[FAIL] Sim processes survived the cleanup — refusing to launch a second stack." >&2
  echo "$survivors" >&2
  echo "       Kill them manually, then re-run this script." >&2
  exit 1
fi
echo "[OK]   No sim processes left over."

if [[ "$RESET_ONLY" -eq 1 ]]; then
  echo "[OK]   Reset-only complete (sim not relaunched)."
  exit 0
fi

# ── 5. Relaunch simulation in background ───────────────────────────────────
echo "[RESET] Launching simulation (log → data/$SIM_LOG_NAME)…"
mkdir -p "$REPO_ROOT/data"
: > "$SIM_LOG_HOST"

# Always headless in ros2 launch; gzclient is started AFTER the stack is
# ready (see below). Kitchen + concurrent GUI spawn was SIGSEGVing gzclient
# on a 4K secondary display.
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

# ── 6. Wait for orchestrator + camera frame ────────────────────────────────
echo "[RESET] Waiting for orchestrator + camera (up to ${WAIT_SECS}s)…"

ready_orch=0
ready_cam=0
deadline=$((SECONDS + WAIT_SECS))
while (( SECONDS < deadline )); do
  if [[ "$ready_orch" -eq 0 ]]; then
    if docker exec "$CONTAINER" bash -lc \
      "$ROS_SETUP && timeout 3 ros2 topic list 2>/dev/null | grep -q '/vlm_planner/inject_plan'"; then
      echo "[OK]   Orchestrator topic present."
      ready_orch=1
    fi
  fi

  if [[ "$ready_cam" -eq 0 ]]; then
    # Prefer an actual frame, not just topic list (avoids the 8s capture timeout).
    if docker exec "$CONTAINER" bash -lc \
      "$ROS_SETUP && timeout 5 ros2 topic echo /wrist_camera/image_raw --once >/dev/null 2>&1"; then
      echo "[OK]   Wrist camera frame received."
      ready_cam=1
    elif docker exec "$CONTAINER" bash -lc \
      "$ROS_SETUP && timeout 5 ros2 topic echo /overview_camera/image_raw --once >/dev/null 2>&1"; then
      echo "[OK]   Overview camera frame received."
      ready_cam=1
    fi
  fi

  if [[ "$ready_orch" -eq 1 && "$ready_cam" -eq 1 ]]; then
    break
  fi
  sleep 2
done

if [[ "$ready_orch" -ne 1 ]]; then
  echo "[FAIL] Orchestrator not ready. See $SIM_LOG_HOST" >&2
  tail -n 40 "$SIM_LOG_HOST" 2>/dev/null || true
  exit 1
fi
if [[ "$ready_cam" -ne 1 ]]; then
  echo "[FAIL] No camera frames. Capture will fail." >&2
  echo "       Try without --no-gui, or check GPU/DISPLAY. Log: $SIM_LOG_HOST" >&2
  tail -n 40 "$SIM_LOG_HOST" 2>/dev/null || true
  exit 1
fi

# ── 6b. Assert a single stack ──────────────────────────────────────────────
# Count real binaries only. Never bare-grep `ps` for "gzserver"/"move_group":
# the long-lived `docker exec -d bash -lc '…'` launcher keeps those strings in
# its own cmdline and would count as a duplicate of a healthy stack.
n_mg="$(docker exec "$CONTAINER" bash -lc 'pgrep -c -x move_group 2>/dev/null || echo 0')"
n_gz="$(docker exec "$CONTAINER" bash -lc 'pgrep -c -x gzserver 2>/dev/null || echo 0')"
n_or="$(docker exec "$CONTAINER" bash -lc \
  "ps -eo args --no-headers | grep -c '^/usr/bin/python3 .*/vlm_robot_planner/orchestrator' || echo 0")"
n_mg="${n_mg//[^0-9]/}"; n_gz="${n_gz//[^0-9]/}"; n_or="${n_or//[^0-9]/}"
for label_n in "move_group:$n_mg" "gzserver:$n_gz" "orchestrator:$n_or"; do
  label="${label_n%%:*}"; n="${label_n##*:}"
  if [[ "${n:-0}" -ne 1 ]]; then
    echo "[FAIL] Expected exactly 1 '$label' process, found ${n:-0}." >&2
    echo "       Duplicate stacks make every pick fail with CONTROL_FAILED (-4)." >&2
    docker exec "$CONTAINER" bash -lc "ps -eo pid,args | grep -E '$label|orchestrator' | grep -v grep" >&2 || true
    exit 1
  fi
done
echo "[OK]   Single simulation stack confirmed."

# Soft world reset (objects to spawn poses) once services are up.
echo "[RESET] Calling /reset_world…"
docker exec "$CONTAINER" bash -lc \
  "$ROS_SETUP && ros2 service call /reset_world std_srvs/srv/Empty" \
  >/dev/null 2>&1 || echo "[WARN] /reset_world unavailable — continuing"
sleep 2

# Attach Gazebo GUI after spawn/controllers settle (GPU, capped Ogre window).
if [[ "$GUI" == "true" ]]; then
  echo "[RESET] Starting gzclient on NVIDIA GPU…"
  docker exec -d "$CONTAINER" bash -lc \
    "$ROS_SETUP; bash /workspace/scripts/_start_gzclient_gpu.sh > /workspace/data/gzclient_gpu.log 2>&1"
  # Give Ogre a moment; fail soft if it dies (stack still usable headless).
  sleep 4
  if docker exec "$CONTAINER" bash -lc 'pgrep -x gzclient >/dev/null'; then
    echo "[OK]   gzclient running (see data/gzclient_gpu.log)."
  else
    echo "[WARN] gzclient exited — check data/gzclient_gpu.log / data/gazebo/ogre.log" >&2
    tail -n 30 "$REPO_ROOT/data/gzclient_gpu.log" 2>/dev/null || true
  fi
fi

echo "[OK]   Stack ready."

if [[ "$RUN_TEST" -ne 1 ]]; then
  echo ""
  echo "Sim is up. Run the FD smoke with:"
  echo "  source .venv/bin/activate"
  echo "  python scripts/run_loop_host.py \\"
  echo "    --task $(printf %q "$TASK") \\"
  echo "    --world $(printf %q "$WORLD") --hybrid $(printf %q "$HYBRID") --control $(printf %q "$CONTROL")"
  exit 0
fi

# ── 7. Run FD smoke test ───────────────────────────────────────────────────
echo ""
echo "[TEST] Starting FD control smoke…"
if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

python "$REPO_ROOT/scripts/run_loop_host.py" \
  --task "$TASK" \
  --world "$WORLD" \
  --hybrid "$HYBRID" \
  --control "$CONTROL" \
  --max-steps "$MAX_STEPS" \
  --container "$CONTAINER"

echo ""
echo "[OK]   FD smoke finished. Inspect latest data/runs/*/summary.json"
