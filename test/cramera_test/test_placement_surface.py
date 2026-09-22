"""
Runtime semantic placement search and geometry validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

from cramera.live.placement_surface import (
    PlacementGeometryMissing,
    PlacementSpaceUnavailable,
    PlacementSurface,
    PlacementSurfaceMissing,
)
from krrood.entity_query_language.factories import variable
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    Milk,
    Table,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.world_description.geometry import Scale

from .test_semantic_placement import PlacementScene, placement_scene

# %% supporting geometry


@pytest.fixture
def surface_scene(placement_scene: PlacementScene) -> PlacementScene:
    """
    Prepare the real table's supporting region for deterministic candidate tests.

    :param placement_scene: World containing an annotated table and movable object.
    """
    with placement_scene.world.modify_world():
        placement_scene.surface.calculate_supporting_surface()
    return placement_scene


@dataclass
class SurfaceSamples:
    """
    Deterministic points expressed in a real supporting surface's frame.
    """

    scene: PlacementScene
    """Scene that defines the region and object dimensions."""

    def point(self, x: float = 0.0, y: float = 0.0) -> Point3:
        """
        Choose a candidate center at the height used by the SDT sampler.

        :param x: Candidate x-coordinate in the supporting region.
        :param y: Candidate y-coordinate in the supporting region.
        """
        region = self.scene.surface.supporting_surface
        return Point3(
            x,
            y,
            region.area.max_point.z
            + self.scene.object.root.combined_mesh.extents[2] / 2,
            reference_frame=region,
        )

    def obstruct(self, point: Point3) -> None:
        """
        Place an unannotated object at a candidate pose.

        :param point: Center occupied by the obstacle.
        """
        world = self.scene.world
        world_P_obstacle = world.transform(point, world.root)
        with world.modify_world():
            obstacle = Milk.create_with_new_body_in_world(
                name="obstacle",
                world=world,
                scale=Scale(0.3, 0.3, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                    world_P_obstacle.x, world_P_obstacle.y, world_P_obstacle.z
                ),
            )
            world.remove_semantic_annotation(obstacle)


# %% candidate filtering


@dataclass(init=False, eq=False)
class TestPlacementCandidates:
    """
    Sampling respects object dimensions, support boundaries, and live occupancy.
    """

    def test_adapts_a_raw_body_to_the_existing_sampler(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Supply the semantic root interface expected by SDT's sampler.

        :param surface_scene: Scene whose table already has a supporting region.
        :param monkeypatch: Pytest attribute replacement fixture.
        """
        point = SurfaceSamples(surface_scene).point()
        sampler = Mock(return_value=[point])
        monkeypatch.setattr(Table, "sample_points_from_surface", sampler)
        location = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table
        )
        poses = list(location)
        supplied = sampler.call_args.kwargs["body_to_sample_for"]
        assert isinstance(supplied, HasRootBody)
        assert supplied.root is surface_scene.object.root
        assert sampler.call_args.kwargs["amount"] == location.sample_count
        assert len(poses) == 1

    def test_rejects_points_whose_object_overhangs_the_edge(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Skip a sampled center inside the region when its full footprint protrudes.

        :param surface_scene: Scene whose table already has a supporting region.
        :param monkeypatch: Pytest attribute replacement fixture.
        """
        samples = SurfaceSamples(surface_scene)
        center = samples.point()
        edge = samples.point(surface_scene.surface.supporting_surface.area.max_point.x)
        monkeypatch.setattr(
            Table, "sample_points_from_surface", Mock(return_value=[edge, center])
        )
        poses = list(
            PlacementSurface(surface_scene.world, surface_scene.object.root, Table)
        )
        assert len(poses) == 1
        np.testing.assert_array_equal(
            poses[0].to_position().to_np()[:2], center.to_np()[:2]
        )

    def test_skips_unannotated_obstacles(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Find a free candidate even when the blocking object has no annotation.

        :param surface_scene: Scene whose table already has a supporting region.
        :param monkeypatch: Pytest attribute replacement fixture.
        """
        samples = SurfaceSamples(surface_scene)
        occupied, free = samples.point(), samples.point(x=0.7)
        samples.obstruct(occupied)
        monkeypatch.setattr(
            Table, "sample_points_from_surface", Mock(return_value=[occupied, free])
        )
        poses = list(
            PlacementSurface(surface_scene.world, surface_scene.object.root, Table)
        )
        assert len(poses) == 1
        np.testing.assert_array_equal(
            poses[0].to_position().to_np()[:2], free.to_np()[:2]
        )

    def test_samples_only_when_the_action_domain_is_consumed(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        EQL construction must not sample or freeze the world before execution.

        :param surface_scene: Scene whose table already has a supporting region.
        :param monkeypatch: Pytest attribute replacement fixture.
        """
        samples = SurfaceSamples(surface_scene)
        sampler = Mock(return_value=[samples.point()])
        monkeypatch.setattr(Table, "sample_points_from_surface", sampler)
        location = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table
        )
        variable(Pose, domain=location)
        sampler.assert_not_called()
        samples.obstruct(samples.point())
        with pytest.raises(PlacementSpaceUnavailable):
            next(iter(location))

    def test_corrects_an_object_origin_away_from_its_geometric_center(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Mesh origins at the base must produce the same supported bottom height.

        :param surface_scene: Scene whose table already has a supporting region.
        """
        location = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table
        )
        bounds = (
            surface_scene.object.root.collision.as_bounding_box_collection_in_frame(
                surface_scene.object.root
            ).bounding_box()
        )
        offset = bounds.height
        bounds.min_z += offset
        bounds.max_z += offset
        point = SurfaceSamples(surface_scene).point()
        pose = location.placement_pose(point, bounds)
        assert float(pose.to_position().z) == pytest.approx(
            float(point.z - bounds.center.z)
        )


# %% surface selection and failures


@dataclass(init=False, eq=False)
class TestPlacementTargets:
    """
    Surface selection searches available instances and explains missing geometry.
    """

    def test_searches_the_next_surface_if_the_first_has_no_samples(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        An unspecified instance must not fail merely because the first is full.

        :param surface_scene: Scene whose table already has a supporting region.
        :param monkeypatch: Pytest attribute replacement fixture.
        """
        world = surface_scene.world
        with world.modify_world():
            second = Table.create_with_new_body_in_world(
                name="second_table",
                world=world,
                scale=Scale(2, 2, 0.1),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(y=3),
            )
            second.calculate_supporting_surface()
        second_scene = PlacementScene(world, second, surface_scene.object)
        point = SurfaceSamples(second_scene).point()
        sampler = Mock(side_effect=[[], [point]])
        monkeypatch.setattr(Table, "sample_points_from_surface", sampler)
        pose = next(iter(PlacementSurface(world, surface_scene.object.root, Table)))
        assert pose.reference_frame is second.supporting_surface
        assert sampler.call_count == 2

    @pytest.mark.parametrize("annotation_name", [False, True])
    def test_selects_an_exact_root_or_annotation_name(
        self, surface_scene: PlacementScene, annotation_name: bool
    ) -> None:
        """
        The selection accepts names exposed by either semantic or body queries.

        :param surface_scene: Scene whose table already has a supporting region.
        :param annotation_name: Whether to select the annotation rather than its root.
        """
        surface_scene.surface.name = PrefixedName("dining_table")
        name = (
            surface_scene.surface.name
            if annotation_name
            else surface_scene.surface.root.name
        )
        location = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table, str(name)
        )
        assert location.matching_surfaces() == [surface_scene.surface]

    def test_reports_a_missing_named_surface(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        A named target must not silently fall back to another instance.

        :param surface_scene: Scene whose table already has a supporting region.
        """
        location = PlacementSurface(
            surface_scene.world, surface_scene.object.root, Table, "missing"
        )
        with pytest.raises(PlacementSurfaceMissing) as failure:
            next(iter(location))
        assert failure.value.surface_name == location.surface_name
        assert failure.value.surface_type is Table
        assert Table.__name__ in str(failure.value)

    def test_reports_empty_matching_surfaces(
        self, surface_scene: PlacementScene, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        An exhausted sampler produces a placement error carrying object identity.

        :param surface_scene: Scene whose table already has a supporting region.
        :param monkeypatch: Pytest attribute replacement fixture.
        """
        monkeypatch.setattr(Table, "sample_points_from_surface", Mock(return_value=[]))
        with pytest.raises(PlacementSpaceUnavailable) as failure:
            next(
                iter(
                    PlacementSurface(
                        surface_scene.world, surface_scene.object.root, Table
                    )
                )
            )
        assert failure.value.body is surface_scene.object.root
        assert str(surface_scene.object.root.name) in str(failure.value)

    def test_reports_an_object_without_geometry(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        A body without a mesh cannot receive guessed placement clearance.

        :param surface_scene: Scene whose table already has a supporting region.
        """
        with pytest.raises(PlacementGeometryMissing) as failure:
            next(
                iter(
                    PlacementSurface(
                        surface_scene.world, surface_scene.world.root, Table
                    )
                )
            )
        assert failure.value.body is surface_scene.world.root
        assert str(surface_scene.world.root.name) in str(failure.value)

    def test_samples_an_apartment_surface(self, apartment_world_copy: World) -> None:
        """
        A real apartment annotation yields a finite semantic placement pose.

        :param apartment_world_copy: Existing apartment fixture with annotated
            furniture.
        """
        with apartment_world_copy.modify_world():
            WorldReasoner(apartment_world_copy).reason()
        body = apartment_world_copy.get_body_by_name("milk.stl")
        pose = next(iter(PlacementSurface(apartment_world_copy, body, Cabinet)))
        assert np.isfinite(pose.to_np()).all()
        assert pose.reference_frame in apartment_world_copy.regions
