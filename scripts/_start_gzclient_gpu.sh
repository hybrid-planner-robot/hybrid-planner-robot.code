#!/bin/bash
# _start_gzclient_gpu.sh — Start gzclient on the host NVIDIA GPU (runs IN container).
#
# Called by bin/reset_and_test_fd.sh / bin/lab_manual.sh AFTER gzserver is up.
# Starting the GUI later (not in the same ros2 launch as spawn/controllers)
# avoids the kitchen SIGSEGV: a huge window + heavy meshes + panda spawn at once.
#
# Uso (inside vlm_ros2):
#   bash /workspace/scripts/_start_gzclient_gpu.sh
#
set -euo pipefail

export DISPLAY="${DISPLAY:-:0}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
# Gazebo Classic hardcodes FSAA=4 in RenderEngine; ask the NVIDIA driver to
# keep multisample off anyway (helps kitchen on RTX 50-class GPUs).
export __GL_FSAA_MODE=0
export __GL_ALLOW_FXAA_USAGE=0
# Prevent Qt from opening a 4K-maximized window on the secondary HDMI screen.
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1
export QT_SCREEN_SCALE_FACTORS=1
unset LIBGL_ALWAYS_SOFTWARE

if [[ ! -e /dev/nvidia0 || ! -d /usr/local/nvidia/lib ]]; then
  echo "[FAIL] NVIDIA GPU not visible in the container." >&2
  echo "       Run on the host: bin/sync_nvidia_gl_libs.sh" >&2
  echo "       then: (cd docker && docker compose up -d --force-recreate)" >&2
  exit 1
fi

mkdir -p /root/.gazebo
# Primary-monitor corner + capped size (Gazebo Classic honors width/height).
cat > /root/.gazebo/gui.ini <<'EOF'
[geometry]
x=80
y=60
width=1024
height=576
EOF

# Best-effort Ogre prefs (Gazebo overwrites FSAA=4 at init; size still helps).
cat > /root/.gazebo/ogre.cfg <<'EOF'
Render System=OpenGL Rendering Subsystem

[OpenGL Rendering Subsystem]
FSAA=0
Full Screen=No
RTT Preferred Mode=FBO
VSync=No
Video Mode=1024 x  576
sRGB Gamma Conversion=No
EOF

pkill -9 -x gzclient 2>/dev/null || true
sleep 0.3

echo "[GUI]  Starting gzclient on NVIDIA GPU (1024x576, FSAA driver-off)…"
exec gzclient --gui-client-plugin=libgazebo_ros_eol_gui.so
