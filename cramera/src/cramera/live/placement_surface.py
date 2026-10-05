"""
Resolve semantic placement targets against the current digital twin.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import product

import numpy as np

from typing_extensions import TYPE_CHECKING

from coraplex.datastructures.enums import ApproachDirection
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.rotations import Rotations
from coraplex.locations.base import PoseGeneratorBackend
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.reasoning.predicates import is_place_occupied
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootBody,
    HasSupportingSurface,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    Quaternion,
    RotationMatrix,
)
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
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
        bounds = VolumetricBoundingBox.from_mesh(
            mesh, HomogeneousTransformationMatrix(reference_frame=self.body)
        )
        object_annotation = HasRootBody(root=self.body)
        object_position = self.body.global_pose.to_position()
        yaw = self.placed_yaw()
        candidates = (
            PlacementCandidate(surface, self.placement_pose(point, bounds, yaw))
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

    def holder(self) -> EndEffector | None:
        """
        :return: The end effector the object hangs from, if a robot holds it.
        """
        parent = self.body.parent_connection.parent if self.body.parent_connection else None
        for end_effector in self.world.get_semantic_annotations_by_type(EndEffector):
            if end_effector.tool_frame is parent:
                return end_effector
        return None

    def placed_yaw(self) -> float:
        """
        :return: How far about the vertical the object is turned when put down, in the
            world, in radians.

        A held object is put down turned the way the robot would pick it up from where
        it stands: the robot's heading runs the same way through the object as it did
        when the robot took it, whichever way the robot has turned since and however
        the parked arm has turned the object meanwhile. A hand on top of an object
        cannot turn it about the vertical without swinging the whole arm around. An
        object nobody holds keeps its yaw.
        """
        held_by = self.holder()
        if held_by is None:
            return self._yaw_of(self.body.global_pose.to_np())
        heading_in_body = self._rotation(
            Quaternion(*Rotations.SIDE_ROTATIONS[self.approached_from(held_by)])
        ) @ np.array([1.0, 0.0, 0.0])
        robot_yaw = self._yaw_of(held_by._robot.root.global_pose.to_np())
        return robot_yaw - float(np.arctan2(heading_in_body[1], heading_in_body[0]))

    def approached_from(self, held_by: EndEffector) -> ApproachDirection:
        """
        :param held_by: The end effector holding the object.
        :return: The side of the object the robot's heading went through when it took
            the object: the one whose grasp, aligned as the hand holds the object and
            turned about the approach as the hand prefers, matches how the object is
            held best.

        Turning the hand a quarter turn about the approach is the same hand orientation
        as approaching from the next side over, so the side alone is ambiguous; the
        hand's preference for the turn settles it, since the grasp that took the object
        followed that preference.
        """
        holding = GraspDescription.from_attachment(held_by, self.body)
        preference = getattr(held_by, "preferred_grasp_alignment", None)
        rotated = (
            preference.with_rotated_gripper
            if preference is not None
            else holding.rotate_gripper
        )
        measured = GraspDescription._measured_grasp_orientation(held_by, self.body)
        return min(
            Rotations.SIDE_ROTATIONS,
            key=lambda side: GraspDescription(
                side, holding.vertical_alignment, held_by, rotated
            )
            .grasp_orientation()
            .rotational_distance(measured),
        )

    @staticmethod
    def _rotation(quaternion) -> np.ndarray:
        """
        :param quaternion: An orientation.
        :return: Its 3 by 3 rotation matrix.
        """
        return RotationMatrix.from_quaternion(quaternion).to_np()[:3, :3]

    @staticmethod
    def _yaw_of(transform: np.ndarray) -> float:
        """
        :param transform: A homogeneous transformation matrix.
        :return: Its rotation about the vertical, in radians.
        """
        return float(np.arctan2(transform[1, 0], transform[0, 0]))

    def placement_pose(
        self,
        point: Point3,
        bounds: VolumetricBoundingBox,
        world_yaw: float | None = None,
    ) -> Pose:
        """
        Convert a sampled object-center point to the object's origin pose.

        The object is turned as :meth:`placed_yaw` says rather than as the surface
        faces.

        :param point: Sampled center in the supporting surface frame.
        :param bounds: Object bounds expressed in the object's frame.
        :param world_yaw: How far the object is turned about the vertical in the world,
            in radians; as it is turned now when left out.
        """
        if world_yaw is None:
            world_yaw = self.placed_yaw()
        frame = point.reference_frame
        yaw = world_yaw - self._yaw_of(
            self.world.compute_forward_kinematics_np(self.world.root, frame)
        )
        center = bounds.center
        turned = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        offset = turned @ np.array([float(center.x), float(center.y)])
        return Pose.from_xyz_rpy(
            float(point.x) - offset[0],
            float(point.y) - offset[1],
            float(point.z) - float(center.z),
            yaw=yaw,
            reference_frame=frame,
        )

    def supports_pose(
        self, surface: HasSupportingSurface, pose: Pose, bounds: VolumetricBoundingBox
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
        self, surface: HasSupportingSurface, pose: Pose, bounds: VolumetricBoundingBox
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
        yaw = self._yaw_of(surface_T_object.to_np())
        turned = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        ray_height = mesh.bounds[1, 2] + bounds.height
        corners = list(
            product((bounds.min_x, bounds.max_x), (bounds.min_y, bounds.max_y))
        ) + [(float(bounds.center.x), float(bounds.center.y))]
        origins = np.array(
            [
                [
                    *(
                        np.array([float(position.x), float(position.y)])
                        + turned @ np.array([x, y])
                    ),
                    ray_height,
                ]
                for x, y in corners
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
            yaw=yaw,
            reference_frame=surface.root,
        )
        return self.world.transform(resting_pose, pose.reference_frame)
