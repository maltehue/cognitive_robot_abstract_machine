"""
Preserve independent robot poses across live scene replacements.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% live poses


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_robot_pose_capture() -> None:
    """
    A live capture retains every robot and respects pending authored pose edits.
    """
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).parent / "js" / "test_builder_robot_pose_capture.js"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
