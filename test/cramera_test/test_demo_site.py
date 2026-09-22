"""
The demo site's scenes index: a subset of the collection, with its own default.
"""

import json
from pathlib import Path

import pytest

from cramera.demo_site import IndexedScene, SceneMissing, ScenesIndex, main

COLLECTION = ScenesIndex(
    default="pr2_kitchen",
    scenes=[
        IndexedScene(
            name="pr2_pouring", robot="pr2", environment="kitchen", task="pour"
        ),
        IndexedScene(
            name="g1_warehouse",
            robot="unitreeg1",
            environment="warehouse",
            task="pick up",
        ),
        IndexedScene(
            name="garmi_pick_place",
            robot="garmi",
            environment="lab",
            task="pick and place",
        ),
    ],
)


# %% restricting the collection to the site's scenes
class TestRestriction:
    def test_keeps_only_the_named_scenes_in_the_given_order(self):
        site = COLLECTION.restricted_to("g1_warehouse", ["g1_warehouse", "pr2_pouring"])
        assert site.scenes == [COLLECTION.scenes[1], COLLECTION.scenes[0]]

    def test_opens_the_given_default(self):
        site = COLLECTION.restricted_to("g1_warehouse", ["g1_warehouse", "pr2_pouring"])
        assert site.default == "g1_warehouse"

    def test_a_scene_the_collection_lacks_is_an_error(self):
        with pytest.raises(SceneMissing):
            COLLECTION.restricted_to("g1_warehouse", ["g1_warehouse", "hsr_lunch"])

    def test_a_default_the_collection_lacks_is_an_error(self):
        with pytest.raises(SceneMissing):
            COLLECTION.restricted_to("hsr_lunch", ["g1_warehouse"])


# %% the file round trip the workflow performs
class TestScript:
    def test_writes_the_restricted_index_the_viewer_reads(self, tmp_path: Path):
        collection = tmp_path / "collection.json"
        site = tmp_path / "site.json"
        COLLECTION.write(collection)

        main([str(collection), str(site), "pr2_pouring", "pr2_pouring", "g1_warehouse"])

        expected = COLLECTION.restricted_to(
            "pr2_pouring", ["pr2_pouring", "g1_warehouse"]
        )
        assert ScenesIndex.read(site) == expected
        assert json.loads(site.read_text(encoding="utf-8")) == expected.to_json()
