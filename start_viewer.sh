#!/usr/bin/env bash
# Start the cramera web viewer (http://localhost:8711). Open it, click "◉ Live",
# then the Plan tab to see the running demo. Run alongside ./start_demo.sh.
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ~/ros2_ws/install/setup.bash 2>/dev/null || true
cd "$REPO"
# Resolve the cramera CLI without hardcoding a venv: honor $VENV, else the repo .venv,
# else whatever's on PATH (an activated environment). So a fresh checkout with a
# differently placed/named venv works without editing this file.
CRAMERA="${VENV:+$VENV/bin/cramera}"
[ -n "$CRAMERA" ] && [ -x "$CRAMERA" ] || CRAMERA="$REPO/.venv/bin/cramera"
[ -x "$CRAMERA" ] || CRAMERA="$(command -v cramera || true)"
[ -n "$CRAMERA" ] || { echo "cramera not found — activate your venv or set VENV=/path/to/venv"; exit 1; }
echo "▶ viewer on http://localhost:8711/  (Ctrl-C to quit)"
exec "$CRAMERA" --no-browser
