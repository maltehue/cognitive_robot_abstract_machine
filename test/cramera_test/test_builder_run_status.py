"""
Exercise Builder run status and asynchronous lifecycle in the real page script.
"""

from pathlib import Path
import shutil
import subprocess

import pytest


# %% plan result monitoring
@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_builder_run_status() -> None:
    """
    Finished plans stay visible and stale callbacks cannot overwrite newer runs.
    """
    result = subprocess.run(
        ["node", str(Path(__file__).parent / "js" / "test_builder_run_status.js")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
