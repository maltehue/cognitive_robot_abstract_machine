"""
Run browser state regressions without a live browser.
"""

from pathlib import Path
import shutil
import subprocess

import pytest

# %% browser model


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_state() -> None:
    """
    Robot capabilities and captured poses stay consistent during authoring.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_builder_state.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_scene_drag_messages() -> None:
    """
    Releasing a drag sends the authored pose back to the embedded builder.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_scene_drag_messages.js")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
