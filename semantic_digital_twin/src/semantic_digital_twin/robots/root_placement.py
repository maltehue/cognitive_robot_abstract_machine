from __future__ import annotations

from dataclasses import dataclass

from trimesh import transform_points
from typing_extensions import TYPE_CHECKING

from semantic_digital_twin.exceptions import UsageError
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot

# %% support geometry validation


@dataclass
class InvalidRobotSupportBodiesError(UsageError):
    """
    Raised when support selection lacks collision bodies belonging to the robot.
    """

    bodies: list[Body]
    """The rejected support selection."""

    def error_message(self) -> str:
        """
        Describe the invalid support selection.
        """
        return f"Support bodies must be nonempty collision bodies belonging to the robot: {self.bodies}"

    def suggest_correction(self) -> str:
        """
        Describe a valid support selection.
        """
        return "Select the robot bodies whose collision geometry contacts the support plane."


# %% initial standing placement


@dataclass
class RobotSupportPlacement:
    """
    Align explicitly selected support geometry with a horizontal world plane.

    Apply after setting the robot configuration and before motion execution.
    """

    robot: AbstractRobot
    """The robot whose root connection is translated vertically."""

    support_bodies: list[Body]
    """
    Robot bodies whose lowest collision point defines the contact height.
    """

    surface_height: float
    """Height of the horizontal support plane in the world frame."""

    def apply(self) -> None:
        """
        Set standing height without changing joint state or planar pose.
        """
        if not self.support_bodies or any(
            body not in self.robot.bodies or not body.collision.shapes
            for body in self.support_bodies
        ):
            raise InvalidRobotSupportBodiesError(self.support_bodies)
        world = self.robot._world
        lowest_point = min(
            float(
                transform_points(
                    body.collision.combined_mesh.vertices, body.global_transform.to_np()
                )[:, 2].min()
            )
            for body in self.support_bodies
        )
        displacement = Vector3(
            z=self.surface_height - lowest_point, reference_frame=world.root
        )
        connection = self.robot.root.parent_connection
        parent_displacement = world.transform(displacement, connection.parent)
        translation = HomogeneousTransformationMatrix.from_xyz_rpy(
            *parent_displacement.to_np()[:3], reference_frame=connection.parent
        )
        replacement = connection.copy_with_new_parent(
            connection.parent, translation @ connection.parent_T_connection_expression
        )
        replacement.name = connection.name
        with world.modify_world():
            world.remove_connection(connection)
            world.add_connection(replacement)
