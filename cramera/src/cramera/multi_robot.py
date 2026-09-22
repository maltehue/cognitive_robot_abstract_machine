"""
Named robot instances assembled with native semantic world specifications.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
import re

from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


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

    def build_world(self, environment_path: str | None = None) -> World:
        """
        Build an environment containing every independently annotated robot.

        :param environment_path: Environment URDF, or None for an empty environment.
        :return: Shared world with native collision geometry and robot annotations.
        """
        specifications = [instance.specification() for instance in self.instances]
        specification = (
            WorldSpecification.from_urdf(environment_path, robots=specifications)
            if environment_path is not None
            else WorldSpecification(robots=specifications)
        )
        world = specification.to_domain_object()
        with world.modify_world():
            for instance in self.instances:
                instance.resolve(world).update_name(
                    PrefixedName(instance.label, prefix=instance.identifier)
                )
        for instance in self.instances:
            instance.restore_joint_positions(world)
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
