"""
Pair-specific contact policies for grasping and supported placement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Iterator

from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import Body


# %% temporary policy lifetime
@dataclass
class TemporaryCollisionScope:
    """
    Append temporary collision rules for a bounded execution scope.
    """

    world: World
    """World whose active collision policy is temporarily extended."""
    rules: list[CollisionRule] = field(default_factory=list)
    """
    Rules applied after the caller's existing temporary rules.
    """

    @contextmanager
    def activate(self) -> Iterator[None]:
        """
        Restore the exact previous rules after completion or failure.
        """
        if not self.rules:
            yield
            return
        manager = self.world.collision_manager
        previous = list(manager.temporary_rules)
        try:
            manager.extend_temporary_rule(self.rules)
            manager.update_collision_matrix()
            yield
        finally:
            manager.clear_temporary_rules()
            manager.extend_temporary_rule(previous)
            manager.update_collision_matrix()


# %% intended manipulation contacts
@dataclass
class ManipulationContactPolicy:
    """
    Permit grasp contact and bounded contact with an object's supporting surface.
    """

    body: Body
    """
    Object intentionally contacted by the selected gripper.
    """

    gripper_bodies: list[Body]
    """
    Collision bodies of the selected end effector.
    """

    support_pose: Pose
    """
    Object pose at which a supporting surface may be contacted.
    """

    gripper_clearance: float = 0.005
    """
    Desired gripper-to-support separation in metres.
    """

    contact_tolerance: float = 0.005
    """
    Maximum support-contact penetration and geometric matching error in metres.
    """

    minimum_support_normal: float = 0.95
    """
    Minimum upward component of the supporting mesh face normal.
    """

    def supporting_bodies(self, world: World) -> list[Body]:
        """
        Find upward-facing mesh contact beneath the requested object pose.

        :param world: World providing the current external collision geometry.
        """
        mesh = self.body.combined_mesh
        if mesh is None or mesh.is_empty:
            return []
        pose = self.body._world.transform(self.support_pose, self.body._world.root)
        origin = HomogeneousTransformationMatrix(reference_frame=self.body._world.root)
        bounds = BoundingBox.from_mesh(
            mesh, pose.to_homogeneous_matrix()
        ).transform_to_origin(origin)
        root_point = np.array(
            [float(bounds.center.x), float(bounds.center.y), bounds.min_z, 1.0]
        )
        result = []
        robot_bodies = {
            part
            for robot in world.get_semantic_annotations_by_type(AbstractRobot)
            for part in robot.bodies_with_collision
        }
        for candidate in world.bodies_with_collision:
            if candidate.id == self.body.id or candidate in robot_bodies:
                continue
            if self._supports_point(candidate, root_point):
                result.append(candidate)
        return result

    def _supports_point(self, candidate: Body, root_point: np.ndarray) -> bool:
        """
        Check for upward mesh contact at one world-frame support point.

        :param candidate: Potential supporting collision body.
        :param root_point: Homogeneous point on the object's lower bounding plane.
        """
        mesh = candidate.combined_mesh
        if mesh is None or mesh.is_empty:
            return False
        root_T_candidate = candidate.global_transform.to_np()
        candidate_T_root = np.linalg.inv(root_T_candidate)
        ray_origin = root_point.copy()
        ray_origin[2] += self.contact_tolerance
        local_origin = (candidate_T_root @ ray_origin)[:3]
        local_direction = -candidate_T_root[:3, 2]
        points, _, faces = mesh.ray.intersects_location(
            [local_origin], [local_direction]
        )
        if len(points) == 0:
            return False
        heights = (root_T_candidate[:3, :3] @ points.T + root_T_candidate[:3, 3, None])[
            2
        ]
        normal_z = (root_T_candidate[:3, :3] @ mesh.face_normals[faces].T)[2]
        return bool(
            np.any(
                (np.abs(heights - root_point[2]) <= self.contact_tolerance)
                & (normal_z >= self.minimum_support_normal)
            )
        )

    def scope(self, world: World) -> TemporaryCollisionScope:
        """
        Resolve this policy against live or copied world entities.

        :param world: World in which the contact-aware motion will execute.
        """
        body = world.get_kinematic_structure_entity_by_id(self.body.id)
        gripper = [
            world.get_kinematic_structure_entity_by_id(part.id)
            for part in self.gripper_bodies
            if part.id != self.body.id
        ]
        rules: list[CollisionRule] = [
            AllowCollisionBetweenGroups(body_group_a=gripper, body_group_b=[body])
        ]
        supports = self.supporting_bodies(world)
        if supports:
            rules.extend(
                [
                    AvoidCollisionBetweenGroups(
                        body_group_a=[body],
                        body_group_b=supports,
                        buffer_zone_distance=0.0,
                        violated_distance=-self.contact_tolerance,
                    ),
                    AvoidCollisionBetweenGroups(
                        body_group_a=gripper,
                        body_group_b=supports,
                        buffer_zone_distance=self.gripper_clearance,
                        violated_distance=0.0,
                    ),
                ]
            )
        return TemporaryCollisionScope(world, rules)


@dataclass
class HasManipulationContactPolicy(ABC):
    """
    Provide the intended contacts of a manipulation action.
    """

    @property
    @abstractmethod
    def manipulation_contact_policy(self) -> ManipulationContactPolicy:
        """
        Return the selected object, end effector and support contact pose.
        """
        ...
