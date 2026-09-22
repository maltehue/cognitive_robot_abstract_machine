#!/usr/bin/env bash
# Start a coraplex demo with the live cramera bridge (port 8765).
#
# Usage:  ./start_demo.sh [demo]
#   demo can be:
#     - a path to a .py file            e.g. ./start_demo.sh coraplex/demos/foo/demo.py
#     - a Plan-Builder demo name        e.g. ./start_demo.sh my_demo
#       (resolved to coraplex/demos/coraplex_generated/my_demo.py)
#   with no argument, runs the default PR2 bullet-world breakfast demo.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

DEFAULT="coraplex/demos/coraplex_bullet_world_demo/demo.py"
GENERATED="coraplex/demos/coraplex_generated"

ARG="${1:-$DEFAULT}"
# Resolve the demo file: an existing path as-is, else a saved Plan-Builder demo by name
# (with or without .py), else give a clear error listing what's available.
if [ -f "$ARG" ]; then
  DEMO="$ARG"
elif [ -f "$GENERATED/$ARG" ]; then
  DEMO="$GENERATED/$ARG"
elif [ -f "$GENERATED/$ARG.py" ]; then
  DEMO="$GENERATED/$ARG.py"
else
  echo "demo not found: '$ARG'"
  echo "give a path to a .py file, or a saved demo name from $GENERATED/:"
  ls "$GENERATED"/*.py 2>/dev/null | sed 's#.*/#  - #;s/\.py$//' || echo "  (none saved yet — use the Plan Builder's 'Save to demos')"
  exit 1
fi

# ROS workspace: sets AMENT_PREFIX_PATH + rclpy so robot descriptions (URDF/meshes) resolve
source ~/ros2_ws/install/setup.bash 2>/dev/null || echo "warn: ~/ros2_ws/install/setup.bash not found"

export CORAPLEX_VISUALIZATION=cramera   # browser backend -> starts the live bridge on :8765
# Resolve the cramera-live CLI without hardcoding a venv (honor $VENV, else repo .venv,
# else PATH) — so a fresh checkout with a different venv works without editing this file.
LIVE="${VENV:+$VENV/bin/cramera-live}"
[ -n "$LIVE" ] && [ -x "$LIVE" ] || LIVE="$REPO/.venv/bin/cramera-live"
[ -x "$LIVE" ] || LIVE="$(command -v cramera-live || true)"
[ -n "$LIVE" ] || { echo "cramera-live not found — activate your venv or set VENV=/path/to/venv"; exit 1; }
echo "▶ running demo: $DEMO"
echo "  (first run caches meshes ~1 min; bridge stays up after — Ctrl-C to quit)"
exec "$LIVE" "$DEMO"
