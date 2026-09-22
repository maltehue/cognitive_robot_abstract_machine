"""
Renderer failure recovery leaves authored plans available.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% browser recovery


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_renderer_unavailable() -> None:
    """
    Exercise the renderer failure boundary and scene-only retry controls.
    """
    result = subprocess.run(
        [
            "node",
            "--test",
            str(Path(__file__).parent / "js" / "test_scene_renderer_unavailable.js"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
