"""
Keep the active robot stable while its new scene is starting.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% startup selection


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_robot_startup_selection() -> None:
    """
    A delayed live startup cannot publish a plan for a newly selected robot.
    """
    result = subprocess.run(
        [
            "node",
            str(
                Path(__file__).parent / "js" / "test_builder_robot_startup_selection.js"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
