"""
Keep the real recorded geometry available until exit-time bundling completes.
"""

import json
import os
from pathlib import Path
import subprocess
import sys


def test_recording_finalizes_before_lazy_mesh_storage_cleanup(tmp_path: Path) -> None:
    """
    A process with lazily generated supporting geometry leaves a usable bundle.

    :param tmp_path: Isolated recording destination for the child process.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "dataset" / "recording_at_exit.py"),
        ],
        env={**os.environ, "CRAMERA_DATA": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    scene_path = tmp_path / "scenes" / "__recording__" / "scene.json"
    assert scene_path.is_file(), result.stderr
    scene = json.loads(scene_path.read_text())
    assert len(scene["objects"]) == 1
    assert scene["missingAssets"] == []
    trajectory = json.loads((scene_path.parent / scene["trajectory"]).read_text())
    assert trajectory["frames"]
    assert "could not finalize" not in result.stderr
