"""
Browser review regressions execute through the repository's pytest suite.
"""

import shutil

import pytest

from .test_web_assets import TestJsUnits as BrowserUnits


# %% asynchronous panel and scene regressions
@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.parametrize(
    "source",
    (
        "test_eql_panel.js",
        "test_eql_suggestions.js",
        "test_graph_navigation.js",
        "test_robot_scene_markers.js",
        "test_scene_picker_rendering.js",
        "test_scene_playback.js",
    ),
)
def test_browser_review_regressions(source: str) -> None:
    """
    Exercise delayed requests, scene markers, and playback through production scripts.

    :param source: Browser regression module executed by Node.
    """
    BrowserUnits().run_node(source)
