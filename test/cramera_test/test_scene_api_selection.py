"""
The viewer's scene and its query API resolve the same available recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from .test_server import get_json, server


# %% recorded scene selection
class TestAvailableSceneSelection:
    """
    Stale index entries cannot separate the scene and knowledge panels.
    """

    def test_api_default_matches_the_available_scene(
        self, server: str, fixture_scene: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        An omitted scene selects the same existing bundle as the browser.
        """
        monkeypatch.delenv("CRAMERA_SCENE", raising=False)
        (fixture_scene / "scenes" / "index.json").write_text(
            json.dumps({"default": "removed", "scenes": [{"name": "removed"}]})
        )
        index = get_json(server + "/scenes/index.json")
        selected = index["scenes"][0]["name"]
        assert get_json(server + "/api/knowledge") == get_json(
            server + "/api/knowledge?scene=" + selected
        )
