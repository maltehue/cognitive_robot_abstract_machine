"""
Named robot instances assembled with native semantic world specifications.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import atan2, isfinite
import re

import numpy as np

from cramera.environment_file import EnvironmentFile
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    FixedConnection,
)


# %% authoring and selection failures
@dataclass
class InvalidRobotScene(ValueError):
    """
    A scene cannot identify its robots or active selection unambiguously.
    """

    reason: str
    """Explanation of the invalid authored configuration."""

    def __str__(self) -> str:
        """
        Return the author-facing reason for rejecting the scene.
        """
        return self.reason


@dataclass
class RobotInstanceUnavailable(LookupError):
    """
    A configured instance does not resolve to exactly one robot in a world.
    """

    identifier: str
    """Instance namespace requested by the caller."""

    matches: int
    """
    Number of matching native robot annotations in the inspected world.
    """

    def __str__(self) -> str:
        """
        Explain why the instance cannot be selected safely.
        """
        return f"Robot instance {self.identifier!r} has {self.matches} matches; expected one."


@dataclass
class UnknownEnvironmentJointError(ValueError):
    """
    Raised for joint positions naming a joint the environment does not have.
    """

    names: list[str]
    """
    The joints named that the environment lacks.
    """

    def __str__(self) -> str:
        return f"the environment has no joints {self.names}"


@dataclass
class RobotPlacementNotFixedError(ValueError):
    """
    Raised for moving a robot whose localization frame is not fixed to the world, so
    that there is no placement to re-fix; a spawned robot's always is, whether or not it
    follows a real robot's odometry.
    """

    robot_name: str
    """
    The name of the robot's root body.
    """

    connection_type: type
    """
    The type of the connection attaching the robot's localization frame to the world.
    """

    def __str__(self) -> str:
        return (
            f"robot {self.robot_name} is attached to the world by a "
            f"{self.connection_type.__name__}, not a fixed placement, so it cannot be moved"
        )


# %% independently named robot instances
@dataclass(frozen=True)
class RobotInstance:
    """
    A robot model, its stable namespace, display label, and world placement.
    """

    identifier: str
    """
    Unique namespace used by the native bodies, joints and localization frame.
    """

    label: str
    """
    Visible name of the robot's semantic annotation.
    """

    robot_type: type[AbstractRobot]
    """
    Native semantic robot class responsible for loading and annotating its model.
    """

    pose: HomogeneousTransformationMatrix
    """
    World-root-relative localization pose at which the instance starts.
    """

    joint_positions: dict[str, float] = field(default_factory=dict)
    """
    Captured scalar joint positions keyed by the full namespaced connection name.
    """

    def __post_init__(self) -> None:
        """
        Reject identities that cannot serve as a single model namespace.
        """
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.identifier) is None:
            raise InvalidRobotScene(
                f"Robot identifier {self.identifier!r} must start with a letter and contain only letters, numbers or underscores."
            )
        if not self.label.strip():
            raise InvalidRobotScene("Every robot instance needs a display label.")
        for name, position in self.joint_positions.items():
            if not name.startswith(f"{self.identifier}/"):
                raise InvalidRobotScene(
                    f"Joint {name!r} does not belong to robot {self.identifier!r}."
                )
            if isinstance(position, bool) or not isfinite(position):
                raise InvalidRobotScene(f"Joint {name!r} needs a finite position.")

    def specification(self) -> RobotSpecification:
        """
        Describe this instance using native robot parsing and localization.

        :return: Specification preserving the instance namespace and start pose.
        """
        return RobotSpecification(
            semantic_annotation_type=self.robot_type,
            world_T_odom=self.pose,
            prefix=self.identifier,
        )

    def resolve(self, world: World) -> AbstractRobot:
        """
        Resolve exactly this model instance in a semantic world.

        :param world: World containing the named robot annotation.
        :return: The unique matching robot.
        :raises RobotInstanceUnavailable: If the instance is missing or duplicated.
        """
        robots = [
            robot
            for robot in world.get_semantic_annotations_by_type(self.robot_type)
            if robot.root.name.prefix == self.identifier
        ]
        if len(robots) != 1:
            raise RobotInstanceUnavailable(self.identifier, len(robots))
        return robots[0]

    def restore_joint_positions(self, world: World) -> None:
        """
        Restore captured articulation onto this instance's native joints.

        :param world: World containing the independently annotated robot.
        :raises InvalidRobotScene: If a saved name is not a scalar joint of this model.
        """
        connections = {
            str(connection.name): connection
            for connection in self.resolve(world).connections
            if isinstance(connection, ActiveConnection1DOF)
        }
        unknown_names = self.joint_positions.keys() - connections.keys()
        if unknown_names:
            raise InvalidRobotScene(
                f"Robot {self.identifier!r} has no scalar joints {sorted(unknown_names)!r}."
            )
        with world.batch_state_changes():
            for name, position in self.joint_positions.items():
                connections[name].position = position


# %% shared native worlds
@dataclass
class RobotScene:
    """
    Independent robot instances sharing one environment and an active selection.
    """

    instances: Sequence[RobotInstance]
    """
    Robot configurations whose namespaces must be unique within the world.
    """

    active_identifier: str
    """
    Namespace of the robot selected for the current plan.
    """

    environment_joint_positions: dict[str, float] = field(default_factory=dict)
    """
    Where the environment's own joints stand, by connection name - how far each door of
    a building stands open, say.
    """

    def __post_init__(self) -> None:
        """
        Validate the complete selection before any models are loaded.
        """
        self.instances = tuple(self.instances)
        if not self.instances:
            raise InvalidRobotScene("A robot scene needs at least one instance.")
        identifiers = [instance.identifier for instance in self.instances]
        if len(identifiers) != len(set(identifiers)):
            raise InvalidRobotScene("Robot instance identifiers must be unique.")
        if self.active_identifier not in identifiers:
            raise InvalidRobotScene(
                f"Selected robot {self.active_identifier!r} is absent from the scene."
            )

    def build_world(self, environment: str | EnvironmentFile | None = None) -> World:
        """
        Build an environment containing every independently annotated robot, each
        standing on the floor beneath it.

        :param environment: The environment's file or its path, or None for an empty
            environment.
        :return: Shared world with native collision geometry and robot annotations.
        """
        specifications = [instance.specification() for instance in self.instances]
        if isinstance(environment, str):
            environment = EnvironmentFile.from_path(environment)
        specification = (
            environment.specification(robots=specifications)
            if environment is not None
            else WorldSpecification(robots=specifications)
        )
        world = specification.to_domain_object()
        with world.modify_world():
            for instance in self.instances:
                instance.resolve(world).update_name(
                    PrefixedName(instance.label, prefix=instance.identifier)
                )
        for instance in self.instances:
            stand_on_the_floor(world, instance.resolve(world))
            instance.restore_joint_positions(world)
        stand_joints_at(world, self.environment_joint_positions)
        return world

    def robot(self, world: World, identifier: str) -> AbstractRobot:
        """
        Resolve an authored robot by its stable instance namespace.

        :param world: Shared world containing the configured instances.
        :param identifier: Namespace of the requested instance.
        :return: Matching robot annotation in the supplied world.
        :raises InvalidRobotScene: If the requested identifier is not configured.
        :raises RobotInstanceUnavailable: If its world annotation is not unique.
        """
        for instance in self.instances:
            if instance.identifier == identifier:
                return instance.resolve(world)
        raise InvalidRobotScene(f"Unknown robot instance {identifier!r}.")

    def selected_robot(self, world: World) -> AbstractRobot:
        """
        Resolve the robot selected for the current plan.

        :param world: Shared world containing the configured instances.
        :return: Native annotation for the selected instance.
        """
        return self.robot(world, self.active_identifier)


# %% standing on the floor
def floor_height_beneath(world: World, robot: AbstractRobot) -> float:
    """
    How high the floor is beneath a robot: the lowest top of the environment's collision
    geometry standing under the robot's root, or the ground where nothing does.

    The lowest rather than the highest top, so that a table the robot reaches over is
    not taken for the floor it stands on.

    :param world: The world the robot stands in.
    :param robot: The robot standing in it.
    :return: The floor's height in the world's root frame.
    """
    robot_bodies = {
        body
        for other in world.get_semantic_annotations_by_type(AbstractRobot)
        for body in world.get_kinematic_structure_entities_of_branch(other.root)
    }
    world_T_root = robot.root.global_pose.to_np()
    x, y = world_T_root[0, 3], world_T_root[1, 3]
    tops = [
        box.max_z
        for body in world.bodies
        if body.collision and body not in robot_bodies
        for box in body.collision.as_bounding_box_collection_in_frame(
            world.root
        ).bounding_boxes
        if box.min_x <= x <= box.max_x and box.min_y <= y <= box.max_y
    ]
    return min(tops, default=0.0)


def stand_on_the_floor(world: World, robot: AbstractRobot) -> float:
    """
    Lift a robot whose lowest point is below the floor beneath it until it stands on
    that floor: a humanoid is rooted at its pelvis rather than between its feet, and a
    scanned building's floor seldom lies at the world's origin.

    :param world: The world the robot stands in.
    :param robot: The robot to lift; its localization frame must be fixed to its parent,
        as a spawned robot's is.
    :return: How far the robot was lifted, in metres; zero for one already standing.
    """
    lift = max(
        0.0,
        floor_height_beneath(world, robot)
        - world.height_of_lowest_collision_point_of_branch(robot.root),
    )
    if not lift:
        return lift
    parent_T_odom = _placement_of(robot).origin.to_np().copy()
    parent_T_odom[2, 3] += lift
    _replace_placement(world, robot, parent_T_odom)
    return lift


def yaw_of(pose: np.ndarray) -> float:
    """
    :param pose: A homogeneous transformation matrix.
    :return: The angle it turns about the z axis, in radians.
    """
    return atan2(pose[1, 0], pose[0, 0])


def move_robot_to(
    world: World, robot: AbstractRobot, x: float, y: float, yaw: float
) -> None:
    """
    Stand a robot somewhere else on the floor, facing another way.

    It is the robot's root that is stood there. The localization frame is fixed wherever
    that takes it, since the drive between the two carries the root as far from the
    frame's origin as the odometry written into it says.

    :param world: The world the robot stands in.
    :param robot: The robot to move; its localization frame must be fixed to its parent,
        as a spawned robot's is.
    :param x: Where it stands, along the world's x axis, in metres.
    :param y: Where it stands, along the world's y axis, in metres.
    :param yaw: Which way it faces, in radians.
    """
    world_T_root = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x, y=y, yaw=yaw
    ).to_np()
    odom_T_root = robot.root.parent_connection.origin.to_np()
    world_T_odom = world_T_root @ np.linalg.inv(odom_T_root)
    _replace_placement(
        world,
        robot,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=world_T_odom[0, 3], y=world_T_odom[1, 3], yaw=yaw_of(world_T_odom)
        ).to_np(),
    )
    stand_on_the_floor(world, robot)


def _placement_of(robot: AbstractRobot) -> FixedConnection:
    """
    :param robot: A spawned robot.
    :return: The connection fixing the robot's localization frame to the world.
    :raises RobotPlacementNotFixedError: If that connection is not a fixed one.
    """
    placement = robot.root.parent_connection.parent.parent_connection
    if not isinstance(placement, FixedConnection):
        raise RobotPlacementNotFixedError(str(robot.root.name), type(placement))
    return placement


def _replace_placement(
    world: World, robot: AbstractRobot, parent_T_odom: np.ndarray
) -> None:
    """
    Fix a robot's localization frame at another pose.

    A fixed connection's pose is constant, so the new pose takes a connection of its
    own.

    :param world: The world the robot stands in.
    :param robot: The robot to place.
    :param parent_T_odom: The localization frame's new pose in its parent's frame.
    """
    placement = _placement_of(robot)
    odom = placement.child
    with world.modify_world():
        world.remove_connection(placement)
        world.add_connection(
            FixedConnection(
                parent=placement.parent,
                child=odom,
                parent_T_connection_expression=HomogeneousTransformationMatrix(
                    parent_T_odom, reference_frame=placement.parent, child_frame=odom
                ),
            )
        )


# %% the environment's own joints
def stand_joints_at(world: World, positions: dict[str, float]) -> None:
    """
    Stand joints of a world at given positions.

    :param world: The world whose joints are stood.
    :param positions: The position of each joint, by connection name.
    :raises UnknownEnvironmentJointError: If the world has no joint of a name given.
    """
    joints = {
        str(connection.name): connection
        for connection in world.connections
        if isinstance(connection, ActiveConnection1DOF)
    }
    unknown = sorted(positions.keys() - joints.keys())
    if unknown:
        raise UnknownEnvironmentJointError(unknown)
    with world.batch_state_changes():
        for name, position in positions.items():
            joints[name].position = position
