"""
Verify independent robot authoring against the real browser state module.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% authoring state


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_robot_instances() -> None:
    """
    Repeated robot models retain independent poses, plans, and capability choices.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_builder_robot_instances.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
