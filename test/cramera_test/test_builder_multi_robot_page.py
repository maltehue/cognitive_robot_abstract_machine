"""
Exercise actual multi-robot Builder callbacks without a live browser.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% independent authoring and live selection


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_multi_robot_page() -> None:
    """
    Robot selection retains plans and all live poses before a native run.
    """
    result = subprocess.run(
        [
            "node",
            str(Path(__file__).parent / "js" / "test_builder_multi_robot_page.js"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
