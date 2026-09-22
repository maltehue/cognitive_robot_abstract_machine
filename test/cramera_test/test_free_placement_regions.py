"""
Frozen free-area answers from real tabletop geometry and live obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from cramera.body_geometry import rounded_pose
from cramera.live.free_placement_regions import FreePlacementRegions
from semantic_digital_twin.semantic_annotations.mixins import HasSupportingSurface
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk, Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Scale, Cylinder
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from .test_placement_surface import SurfaceSamples, surface_scene
from .test_semantic_placement import PlacementScene, placement_scene


# %% supported footprint and physical height
@dataclass(init=False, eq=False)
class TestFreePlacementRegions:
    """
    Area answers fit the object and stay tied to the physical supporting surface.
    """

    def test_erodes_the_table_by_the_object_footprint(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Only display center positions whose complete footprint lies on the table.

        :param surface_scene: Actual table and movable object with supporting geometry.
        """
        query = FreePlacementRegions(surface_scene.world)
        markers = query.for_body(surface_scene.object.root, surface_scene.surface)
        surface_extent = surface_scene.surface.root.combined_mesh.extents
        object_extent = surface_scene.object.root.combined_mesh.extents
        assert len(markers) == 1
        assert markers[0].scale == pytest.approx(
            [
                surface_extent[0] - object_extent[0],
                surface_extent[1] - object_extent[1],
                query.thickness,
            ]
        )

    def test_uses_physical_height_when_semantic_region_height_is_wrong(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Do not draw an imported supporting region floating above the actual mesh.

        :param surface_scene: Table whose semantic region can be offset independently.
        """
        surface = surface_scene.surface
        with surface_scene.world.modify_world():
            connection = surface.supporting_surface.parent_connection
            surface_scene.world.remove_connection(connection)
            surface_scene.world.add_connection(
                FixedConnection(
                    parent=connection.parent,
                    child=connection.child,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=2, reference_frame=surface.root
                    ),
                )
            )
        query = FreePlacementRegions(surface_scene.world)
        markers = query.for_body(surface_scene.object.root, surface)
        expected_height = (
            float(surface.root.combined_mesh.bounds[1, 2]) + query.thickness / 2
        )
        assert [marker.position[2] for marker in markers] == pytest.approx(
            [expected_height]
        )

    def test_subtracts_unannotated_obstacles(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        A collidable body excludes center positions even without a semantic label.

        :param surface_scene: Scene with an object-sized candidate and a broad table.
        """
        query = FreePlacementRegions(surface_scene.world)
        unobstructed = query.for_body(surface_scene.object.root, surface_scene.surface)
        samples = SurfaceSamples(surface_scene)
        samples.obstruct(samples.point())
        markers = query.for_body(surface_scene.object.root, surface_scene.surface)
        assert markers
        assert sum(marker.scale[0] * marker.scale[1] for marker in markers) < sum(
            marker.scale[0] * marker.scale[1] for marker in unobstructed
        )
        assert all(
            abs(marker.position[0]) > marker.scale[0] / 2
            or abs(marker.position[1]) > marker.scale[1] / 2
            for marker in markers
        )

    def test_distant_height_does_not_obstruct(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        An obstacle above the candidate's volume does not remove tabletop space.

        :param surface_scene: Existing table and candidate dimensions.
        """
        query = FreePlacementRegions(surface_scene.world)
        expected = query.for_body(surface_scene.object.root, surface_scene.surface)
        with surface_scene.world.modify_world():
            Milk.create_with_new_body_in_world(
                name="overhead",
                world=surface_scene.world,
                scale=Scale(0.3, 0.3, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=3),
            )
        assert (
            query.for_body(surface_scene.object.root, surface_scene.surface) == expected
        )

    def test_draws_world_coordinates_and_freezes_them(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        A translated and yawed table produces markers that later movement cannot alter.

        :param surface_scene: Scene whose table root can move in world coordinates.
        """
        world, surface = surface_scene.world, surface_scene.surface
        with world.modify_world():
            connection = surface.root.parent_connection
            world.remove_connection(connection)
            world.add_connection(
                FixedConnection(
                    parent=connection.parent,
                    child=connection.child,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=2, y=3, z=1, yaw=np.pi / 2, reference_frame=world.root
                    ),
                )
            )
        query = FreePlacementRegions(world)
        markers = query.for_body(surface_scene.object.root, surface)
        expected_height = (
            1 + float(surface.root.combined_mesh.bounds[1, 2]) + query.thickness / 2
        )
        assert len(markers) == 1
        assert markers[0].position == pytest.approx([2, 3, expected_height])
        assert markers[0].quaternion == rounded_pose(surface.root)[3:]
        with world.modify_world():
            connection = surface.root.parent_connection
            world.remove_connection(connection)
            world.add_connection(
                FixedConnection(
                    parent=connection.parent,
                    child=connection.child,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=5, reference_frame=world.root
                    ),
                )
            )
        assert markers[0].position == pytest.approx([2, 3, expected_height])

    def test_initializes_missing_support_geometry(
        self, placement_scene: PlacementScene
    ) -> None:
        """
        Use SDT's own surface construction when the annotation has no region yet.

        :param placement_scene: Existing table fixture before supporting-region
            creation.
        """
        markers = FreePlacementRegions(placement_scene.world).for_body(
            placement_scene.object.root, placement_scene.surface
        )
        assert len(markers) == 1
        assert (
            placement_scene.surface.supporting_surface in placement_scene.world.regions
        )

    def test_rejects_an_object_larger_than_the_table(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Empty area is reported when the candidate footprint cannot fit.

        :param surface_scene: Table whose dimensions bound the accepted footprint.
        """
        world = surface_scene.world
        with world.modify_world():
            large = Milk.create_with_new_body_in_world(
                name="oversize",
                world=world,
                scale=Scale(3, 3, 0.2),
                world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=5),
            )
        assert (
            FreePlacementRegions(world).for_body(large.root, surface_scene.surface)
            == []
        )

    def test_rejects_tilted_tabletops(self, surface_scene: PlacementScene) -> None:
        """
        A sloped top is outside the documented placement-area contract.

        :param surface_scene: Table whose root orientation can be changed.
        """
        world, surface = surface_scene.world, surface_scene.surface
        with world.modify_world():
            connection = surface.root.parent_connection
            world.remove_connection(connection)
            world.add_connection(
                FixedConnection(
                    parent=connection.parent,
                    child=connection.child,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        roll=0.2, reference_frame=world.root
                    ),
                )
            )
        assert (
            FreePlacementRegions(world).for_body(surface_scene.object.root, surface)
            == []
        )

    def test_rejects_internal_or_unspecified_support_types(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        Only the explicitly supported outer-tabletop annotation types are offered.

        :param surface_scene: Scene supplying a real root for an unspecified surface.
        """
        unspecified = HasSupportingSurface(root=surface_scene.surface.root)
        assert (
            FreePlacementRegions(surface_scene.world).for_body(
                surface_scene.object.root, unspecified
            )
            == []
        )

    def test_rejects_a_candidate_without_geometry(
        self, surface_scene: PlacementScene
    ) -> None:
        """
        A shapeless candidate receives no invented footprint.

        :param surface_scene: Scene whose world root has no object geometry.
        """
        assert (
            FreePlacementRegions(surface_scene.world).for_body(
                surface_scene.world.root, surface_scene.surface
            )
            == []
        )

    def test_rejects_a_round_tabletop(self, surface_scene: PlacementScene) -> None:
        """
        A circular top must not expose its unsupported bounding-box corners.

        :param surface_scene: Existing world and candidate object for the query.
        """
        world = surface_scene.world
        body = Body(
            name=PrefixedName("round_table"),
            collision=ShapeCollection(shapes=[Cylinder(width=2, height=0.1)]),
        )
        surface = Table(root=body)
        with world.modify_world():
            world.add_body(body)
            world.add_connection(
                FixedConnection(
                    parent=world.root,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=5, reference_frame=world.root
                    ),
                )
            )
            world.add_semantic_annotation(surface)
        assert (
            FreePlacementRegions(world).for_body(surface_scene.object.root, surface)
            == []
        )
