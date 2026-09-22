"""
The offline storyboard uses local assets and the existing viewer lifecycle.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

from cramera.paths import WEB_ROOT

# %% packaged tour


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_offline_tour_frontend() -> None:
    """
    Exercise chapter progression, readiness and the existing viewer adapter.
    """
    result = subprocess.run(
        ["node", "--test", "test/cramera_test/js/test_offline_tour.js"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tour_assets_are_local() -> None:
    """
    Every script and stylesheet required by the tour ships in the web package.
    """
    html = (WEB_ROOT / "tour.html").read_text()
    references = re.findall(r'(?:src|href)="([^"]+)"', html)
    assert references
    for reference in references:
        assert not reference.startswith(("http:", "https:", "//"))
        assert (WEB_ROOT / reference).is_file(), reference
    stylesheet = (WEB_ROOT / "tour.css").read_text()
    assert "@import" not in stylesheet
    assert "http:" not in stylesheet and "https:" not in stylesheet
