"""
A completed robot keeps its live pose when another robot's plan starts.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% actual page launch and selection
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_robot_restart_preserves_completed_instances() -> None:
    """
    Exercise scene replacement after motion and preserve later unapplied edits.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_builder_robot_restart.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
