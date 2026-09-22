"""
Semantic answers retain frozen geometry for live and recorded highlighting.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import Mock

import pytest

from cramera.knowledge.eql_session import EqlSession
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase
from cramera.knowledge.query_runner import EqlQueryRunner
from cramera.live.world_query import WorldQuerySource
from cramera.spatial_annotations import (
    SpatialAnnotations,
    PlacementAnswer,
    UnknownPlacementQuery,
)

from cramera.live.bridge import Bridge
from cramera.live.recording_bundle import write_recording_bundle
from semantic_digital_twin.semantic_annotations.semantic_annotations import Handle
from semantic_digital_twin.api import Connection6DoFSpecification

from .test_world_queries import annotated_bridge
from .test_recording_bundle import frame_with_milk
from .test_semantic_placement import PlacementScene, placement_scene
from .test_placement_surface import SurfaceSamples, surface_scene


# %% query-time geometry
@dataclass(init=False, eq=False)
class TestSemanticAnswerGeometry:
    """
    A handle answer carries geometry independently of graph node identities.
    """

    def test_handle_answer_has_a_marker_on_its_body(
        self, annotated_bridge: Bridge
    ) -> None:
        """
        Every annotated handle is represented by a frozen world-space marker.
        """
        result = annotated_bridge.run_query("an(entity(handle))").to_payload()
        [marker] = result["spatial"]
        [handle] = annotated_bridge.world.get_semantic_annotations_by_type(Handle)
        bounds = handle.root.visual.as_bounding_box_collection_in_frame(
            annotated_bridge.world.root
        ).bounding_box()
        assert marker["position"] == list(bounds.center.to_np()[:3])
        assert marker["kind"] == "cube"

    def test_saved_recording_retains_queryable_handles(
        self, annotated_bridge: Bridge, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        A finalized bundle answers the same handle query without a live source.
        """
        output = tmp_path / "scenes" / "handles"
        write_recording_bundle(
            annotated_bridge, [frame_with_milk(objects={})], 20.0, output, "handles"
        )
        monkeypatch.setenv("CRAMERA_DATA", str(tmp_path))
        monkeypatch.setenv("CRAMERA_SCENES", str(tmp_path / "scenes"))
        EpisodeKnowledgeBase.reset()
        expected = annotated_bridge.run_query("an(entity(handle))").to_payload()[
            "spatial"
        ]
        actual = (
            EqlSession.of_scene("handles")
            .run("an(entity(handle))")
            .to_payload()["spatial"]
        )
        assert actual == expected
        assert json.loads((output / "scene.json").read_text())["semanticAnnotations"]


# %% object-specific placement answers


@dataclass(init=False, eq=False)
class TestPlacementAnswerGeometry:
    """
    A saved geometric answer uses the same object and surface as a live query.
    """

    def test_live_callable_draws_the_free_area(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        An ordinary world source offers a lazy object-specific area query.

        :param surface_scene: Actual table and candidate object with support geometry.
        """
        source = WorldQuerySource(surface_scene.world)
        knowledge = source.knowledge()[0]
        runner = EqlQueryRunner(knowledge.domains, extra_names=knowledge.extra_names)
        code = f"placement_region({str(surface_scene.object.root.name)!r}, {str(surface_scene.surface.name)!r})"
        result = runner.run(code).to_payload()
        assert result["spatial"][0]["ns"] == "free_placement"
        assert result["rows"][0]["object_name"] == str(surface_scene.object.root.name)

    def test_saved_areas_are_independent_of_later_world_changes(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Recorded placement answers remain fixed after obstacles are added.

        :param surface_scene: Actual table and candidate whose free area is recorded.
        """
        snapshot = SpatialAnnotations.of_world(
            surface_scene.world, [surface_scene.object.root]
        )
        decoded = SpatialAnnotations.of_scene(snapshot.scene_fields())
        names = str(surface_scene.object.root.name), str(surface_scene.surface.name)
        expected = snapshot.placement_region(*names)
        SurfaceSamples(surface_scene).obstruct(SurfaceSamples(surface_scene).point())
        assert decoded.placement_region(*names).markers == expected.markers
        assert expected.markers

    def test_listing_questions_does_not_compute_regions(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Building question labels must leave geometry computation for Run query.

        :param surface_scene: Candidate and tabletop supplying one placement preset.
        :param monkeypatch: Fixture replacing geometry evaluation with a counted call.
        """
        source = WorldQuerySource(
            surface_scene.world, lambda: [surface_scene.object.root]
        )
        evaluate = Mock(return_value=None)
        monkeypatch.setattr(source, "placement_region", evaluate)
        bridge = Bridge()
        bridge.attach(surface_scene.world)
        bridge.register_query_source(source)
        bridge.query_presets()
        assert evaluate.call_count == 0

    def test_missing_live_pair_has_an_actionable_error(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Unknown names cannot accidentally produce another object's clearance.

        :param surface_scene: Existing tabletop and candidate.
        """
        with pytest.raises(UnknownPlacementQuery):
            PlacementAnswer.of_world(
                surface_scene.world, "missing", str(surface_scene.surface.name)
            )

    def test_missing_recorded_pair_has_an_actionable_error(self) -> None:
        """
        Older recordings explicitly report that no placement answer was captured.
        """
        with pytest.raises(UnknownPlacementQuery) as caught:
            SpatialAnnotations.of_scene({}).placement_region("milk", "table")
        assert "milk" in caught.value.error_message()
        assert "presets" in caught.value.suggest_correction()

    def test_attached_world_offers_non_mesh_objects(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        A plain named, free-floating body receives placement presets automatically.

        :param surface_scene: Existing world whose milk body is free-floating.
        """
        body, world = surface_scene.object.root, surface_scene.world
        with world.modify_world():
            transform = body.global_transform
            world.remove_connection(body.parent_connection)
            Connection6DoFSpecification().connect(
                world, body, parent_T_connection=transform
            )
        bridge = Bridge()
        bridge.attach(surface_scene.world)
        expected = PlacementAnswer.code_for(
            str(surface_scene.object.root.name), str(surface_scene.surface.name)
        )
        assert expected in [preset.code for preset in bridge.query_presets()]

    def test_markers_use_the_existing_renderer_pose_contract(
        self, annotated_bridge: Bridge
    ) -> None:
        """
        The browser receives the existing seven-number world pose format.

        :param annotated_bridge: Actual handle world attached without custom query code.
        """
        [marker] = annotated_bridge.run_query("an(entity(handle))").to_payload()[
            "spatial"
        ]
        assert marker["pose"] == marker["position"] + marker["quaternion"]
