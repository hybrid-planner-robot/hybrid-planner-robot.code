#!/bin/bash
# sync_nvidia_gl_libs.sh — Stage host NVIDIA driver userspace into docker/
#
# Gazebo's GUI needs the NVIDIA OpenGL/EGL libraries *inside* the container.
# nvidia-container-toolkit does this automatically; this script is the
# toolkit-free path used by docker-compose.yml (bind-mount the staged libs).
#
# Re-run after every NVIDIA driver upgrade on the host.
#
# Uso:
#   bin/sync_nvidia_gl_libs.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$REPO_ROOT/docker/nvidia-driver-libs"

if ! ls /dev/nvidia0 >/dev/null 2>&1; then
  echo "[FAIL] /dev/nvidia0 not found — install the NVIDIA proprietary driver first." >&2
  exit 1
fi

mkdir -p "$STAGE/egl_vendor.d"
# Drop previous staging so removed SONAMEs cannot linger after a driver bump.
find "$STAGE" -mindepth 1 -maxdepth 1 ! -name egl_vendor.d -exec rm -rf {} +
rm -f "$STAGE/egl_vendor.d"/*

shopt -s nullglob
copied=0
for f in \
  /usr/lib/x86_64-linux-gnu/libcuda.so* \
  /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
  /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
  /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* \
  /usr/lib/x86_64-linux-gnu/libGLESv1_CM_nvidia.so* \
  /usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so*
do
  cp -a "$f" "$STAGE/"
  copied=$((copied + 1))
done

if [[ ! -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
  echo "[FAIL] missing /usr/share/glvnd/egl_vendor.d/10_nvidia.json" >&2
  exit 1
fi
cp -a /usr/share/glvnd/egl_vendor.d/10_nvidia.json "$STAGE/egl_vendor.d/"

if [[ -x /usr/bin/nvidia-smi ]]; then
  cp -a /usr/bin/nvidia-smi "$STAGE/"
fi

echo "[OK]   Staged $copied NVIDIA libraries → docker/nvidia-driver-libs/"
echo "       Driver probe:"
"$STAGE/nvidia-smi" -L 2>/dev/null || nvidia-smi -L
echo ""
echo "Recreate the container so the new libs are mounted:"
echo "  cd docker && docker compose up -d --force-recreate"
