"""
Viewer animation keeps articulated robot instances independent.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% independent model animation
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_model_poses() -> None:
    """
    Exercise native model routing and the actual page's replay callback.
    """
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).parent / "js" / "test_model_poses.js")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
