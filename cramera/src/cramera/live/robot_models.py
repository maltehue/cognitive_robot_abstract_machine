"""
Stable identities of annotated robots sharing a world.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote
from typing_extensions import Any

from cramera.body_geometry import rounded_pose
from cramera.robot_fields import RobotField
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


# %% robot publication
class UnknownRobot(ValueError):
    """
    A requested robot instance is absent from the attached world.
    """


class RobotSelectionBusy(RuntimeError):
    """
    An executing plan retains ownership of its selected robot.
    """


@dataclass(frozen=True)
class RobotModels:
    """
    Native robot annotations and their persistent scene identities.
    """

    world: World | None
    """
    Shared digital twin, when attached.
    """

    selected: AbstractRobot | None = None
    """
    Explicit robot annotation, retained for existing single-robot adapters.
    """

    def robots(self) -> list[AbstractRobot]:
        """
        Return each annotated robot, including an explicitly supplied adapter.
        """
        robots = (
            list(self.world.get_semantic_annotations_by_type(AbstractRobot))
            if self.world is not None
            else []
        )
        if self.selected is not None and self.selected not in robots:
            robots.append(self.selected)
        return robots

    @staticmethod
    def identifier(robot: AbstractRobot) -> str:
        """
        Read the robot root's namespace, falling back to its full name.

        :param robot: Annotation whose native root identifies the instance.
        :return: Stable world-instance identity.
        """
        return robot.root.name.prefix or str(robot.root.name)

    def named(self, identifier: str) -> AbstractRobot:
        """
        Resolve an instance without guessing from its robot class.

        :param identifier: Native namespace requested by the viewer.
        :return: Matching annotation.
        :raises UnknownRobot: If the attached world has no such robot.
        """
        for robot in self.robots():
            if self.identifier(robot) == identifier:
                return robot
        raise UnknownRobot(f"Unknown robot instance: {identifier}")

    def model_name(self, robot: AbstractRobot) -> str:
        """
        Choose a distinct filesystem name while preserving single-robot bundles.

        :param robot: Annotation whose model is written.
        :return: Existing class name for one robot, instance name for several robots.
        """
        if len(self.robots()) == 1:
            return type(robot).__name__.lower()
        return "robot_" + quote(self.identifier(robot), safe="")

    def root_poses(self) -> dict[str, list[float]]:
        """
        Publish all robot roots independently in world coordinates.
        """
        return {
            self.identifier(robot): rounded_pose(robot.root) for robot in self.robots()
        }

    def catalog(self) -> list[dict[str, Any]]:
        """
        Describe live instances for selection and pose capture.
        """
        return [
            {
                RobotField.IDENTIFIER: self.identifier(robot),
                RobotField.LABEL: (
                    robot.name.name
                    if isinstance(robot, AbstractRobot)
                    else type(robot).__name__
                ),
                RobotField.MODEL: type(robot).__name__,
                RobotField.POSE: rounded_pose(robot.root),
                RobotField.JOINT_POSITIONS: self.joint_positions(robot),
            }
            for robot in self.robots()
        ]

    @staticmethod
    def joint_positions(robot: AbstractRobot) -> dict[str, float]:
        """
        Capture the instance's articulated posture using native connection names.

        :param robot: Native annotation whose movable joints are read.
        :return: Current values, or an empty mapping for a legacy metadata adapter.
        """
        if not isinstance(robot, AbstractRobot):
            return {}
        return {
            str(connection.name): float(connection.position)
            for connection in robot.connections
            if isinstance(connection, ActiveConnection1DOF)
        }
