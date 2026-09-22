"""
Resolve semantic placement targets against the current digital twin.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from typing_extensions import TYPE_CHECKING

from coraplex.locations.base import PoseGeneratorBackend
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.reasoning.predicates import is_place_occupied
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body


# %% target failures


@dataclass
class PlacementSurfaceMissing(ValueError):
    """
    The requested semantic surface is absent from the world.
    """

    surface_type: type[HasSupportingSurface]
    """Required annotation type."""

    surface_name: str | None
    """
    Optional annotation or root-body name.
    """

    def __str__(self) -> str:
        """
        Describe the missing target and the required scene annotation.
        """
        return (
            f"No {self.surface_type.__name__} {self.surface_name or ''} is annotated "
            "in this world. Choose an available surface or an exact pose."
        )


@dataclass
class PlacementSpaceUnavailable(ValueError):
    """
    Matching surfaces provide no supported, unoccupied placement pose.
    """

    body: Body
    """Object that needs a free placement pose."""

    surface_type: type[HasSupportingSurface]
    """
    Requested supporting surface type.
    """

    def __str__(self) -> str:
        """
        Describe the object and surface whose free space was exhausted.
        """
        return (
            f"No free placement for {self.body.name} on "
            f"{self.surface_type.__name__}. The surface may be occupied or too small."
        )


@dataclass
class PlacementGeometryMissing(ValueError):
    """
    The transported body has no geometry from which to determine clearance.
    """

    body: Body
    """Object lacking placement geometry."""

    def __str__(self) -> str:
        """
        Identify the object whose dimensions are unavailable.
        """
        return (
            f"Cannot determine placement clearance for {self.body.name}: no geometry."
        )


# %% runtime pose generation


@dataclass
class PlacementCandidate:
    surface: HasSupportingSurface
    """
    Supporting surface from which the candidate was sampled.
    """

    pose: Pose
    """Object origin expressed in the supporting surface's frame."""


@dataclass
class PlacementSurface(PoseGeneratorBackend):
    """
    Generate supported, free object poses when a semantic action is grounded.
    """

    world: World
    """Current world searched when iteration begins."""

    body: Body
    """
    Object whose geometry determines support and collision clearance.
    """

    surface_type: type[HasSupportingSurface]
    """Semantic annotation type requested for placement."""

    surface_name: str | None = None
    """
    Optional exact annotation or root-body name restricting the search.
    """

    sample_count: int = 100
    """
    Maximum candidate points requested from each supporting surface.
    """

    support_tolerance: float = 0.005
    """
    Maximum height variation across the object's support footprint, in metres.
    """

    def __iter__(self) -> Iterator[Pose]:
        """
        Prioritize nearby samples across matching surfaces when iteration begins.
        """
        surfaces = self.matching_surfaces()
        mesh = self.body.combined_mesh
        if mesh is None or mesh.is_empty:
            raise PlacementGeometryMissing(self.body)
        bounds = BoundingBox.from_mesh(
            mesh, HomogeneousTransformationMatrix(reference_frame=self.body)
        )
        object_annotation = HasRootBody(root=self.body)
        object_position = self.body.global_pose.to_position()
        candidates = (
            PlacementCandidate(surface, self.placement_pose(point, bounds))
            for surface in surfaces
            for point in surface.sample_points_from_surface(
                body_to_sample_for=object_annotation, amount=self.sample_count
            )
        )
        found_pose = False
        for candidate in sorted(
            candidates,
            key=lambda candidate: float(
                object_position.euclidean_distance(
                    self.world.transform(candidate.pose, self.world.root).to_position()
                )
            ),
        ):
            if not self.supports_pose(candidate.surface, candidate.pose, bounds):
                continue
            pose = self.supported_pose(candidate.surface, candidate.pose, bounds)
            if pose is None:
                continue
            if is_place_occupied(
                bounds,
                pose,
                self.world,
                allowed_bodies=[self.body, candidate.surface.root],
            ):
                continue
            found_pose = True
            yield pose
        if not found_pose:
            raise PlacementSpaceUnavailable(self.body, self.surface_type)

    def matching_surfaces(self) -> list[HasSupportingSurface]:
        """
        Find annotations matching the requested type and optional exact name.
        """
        surfaces = [
            surface
            for surface in self.world.get_semantic_annotations_by_type(
                self.surface_type
            )
            if not self.surface_name
            or self.surface_name in (str(surface.name), str(surface.root.name))
        ]
        if not surfaces:
            raise PlacementSurfaceMissing(self.surface_type, self.surface_name)
        return surfaces

    def placement_pose(self, point: Point3, bounds: BoundingBox) -> Pose:
        """
        Convert a sampled object-center point to the object's origin pose.

        :param point: Sampled center in the supporting surface frame.
        :param bounds: Object bounds expressed in the object's frame.
        """
        center = bounds.center
        return Pose.from_xyz_rpy(
            point.x - center.x,
            point.y - center.y,
            point.z - center.z,
            reference_frame=point.reference_frame,
        )

    def supports_pose(
        self, surface: HasSupportingSurface, pose: Pose, bounds: BoundingBox
    ) -> bool:
        """
        Require the object's complete footprint to lie within the surface region.

        :param surface: Annotation whose sampler produced the candidate.
        :param pose: Candidate object pose in the supporting region's frame.
        :param bounds: Object bounds expressed in the object's frame.
        """
        area = BoundingBoxCollection.from_shapes(surface.supporting_surface.area)
        area.transform_all_shapes_to_own_frame()
        footprint = (
            replace(bounds, origin=pose.to_homogeneous_matrix())
            .simple_event.as_composite_set()
            .marginal(SpatialVariables.xy)
        )
        return (footprint - area.event.marginal(SpatialVariables.xy)).is_empty()

    def supported_pose(
        self, surface: HasSupportingSurface, pose: Pose, bounds: BoundingBox
    ) -> Pose | None:
        """
        Project an object's bottom onto a level patch of the actual surface mesh.

        :param surface: Annotation defining the physical supporting mesh.
        :param pose: Candidate object pose from the semantic sampler.
        :param bounds: Object bounds in the object's local frame.
        :return: A resting pose, or None if the footprint lacks level support.
        """
        mesh = surface.root.combined_mesh
        surface_T_object = self.world.transform(pose, surface.root)
        position = surface_T_object.to_position()
        ray_height = mesh.bounds[1, 2] + bounds.height
        corners = list(
            product((bounds.min_x, bounds.max_x), (bounds.min_y, bounds.max_y))
        )
        origins = np.array(
            [
                [float(position.x) + x, float(position.y) + y, ray_height]
                for x, y in corners
            ]
            + [
                [
                    float(position.x + bounds.center.x),
                    float(position.y + bounds.center.y),
                    ray_height,
                ]
            ]
        )
        directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))
        intersections, ray_indices, _ = mesh.ray.intersects_location(
            origins, directions
        )
        heights = np.full(len(origins), -np.inf)
        np.maximum.at(heights, ray_indices, intersections[:, 2])
        if not np.isfinite(heights).all() or np.ptp(heights) > self.support_tolerance:
            return None
        resting_pose = Pose.from_xyz_rpy(
            position.x,
            position.y,
            float(heights.max()) - bounds.min_z,
            reference_frame=surface.root,
        )
        return self.world.transform(resting_pose, pose.reference_frame)
