#!/usr/bin/env bash
set -euo pipefail

cramera_repository="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cramera_bundle="${CRAMERA_OFFLINE_BUNDLE:-$HOME/.local/share/cramera/offline-demo}"
exec "$cramera_repository/.venv/bin/python" -m cramera.offline \
    "$cramera_bundle" --isolated "$@"
