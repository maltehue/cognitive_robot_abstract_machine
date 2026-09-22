"""
Shared selection of independently displayed world objects.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Protocol, runtime_checkable

from semantic_digital_twin.world_description.connections import Connection6DoF

from cramera.mesh_format import MeshFormat
from cramera.live.robot_models import RobotModels

if TYPE_CHECKING:
    from semantic_digital_twin.robots.robot_parts import AbstractRobot
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body


# %% loose scene objects
@runtime_checkable
class RobotBodyCollection(Protocol):
    """
    The annotated bodies a robot or recording adapter already declares.
    """

    bodies: list[Body]
    """Bodies belonging to this robot."""


@dataclass
class WorldObjects:
    """
    Objects a live query or recording can treat independently of the robot.
    """

    world: World
    """Digital twin supplying current parent connections and geometry."""
    robot: AbstractRobot | None = None
    """
    Robot whose bodies are represented by the robot model.
    """

    def free_floating(self) -> list[Body]:
        """
        Select shaped free bodies, excluding robot parts and unshaped frames.
        """
        robot_body_names = self.robot_body_names()
        return [
            body
            for body in self.world.bodies
            if isinstance(body.parent_connection, Connection6DoF)
            and str(body.name) not in robot_body_names
            and (body.visual.shapes or body.collision.shapes)
        ]

    def robot_body_names(self) -> set[str]:
        """
        Collect annotated parts of every robot in the shared world.
        """
        names = set()
        for robot in RobotModels(self.world, self.robot).robots():
            bodies = (
                robot.bodies
                if isinstance(robot, RobotBodyCollection)
                else self.world.get_kinematic_structure_entities_of_branch(robot.root)
            )
            names.update(str(body.name) for body in bodies)
        return names

    def overlay_bodies(self, previously_published: Iterable[Body] = ()) -> list[Body]:
        """
        Select live objects and retain their identity through grasp attachments.

        :param previously_published: Bodies already tracked while present in this world.
        :return: Current movable primitives, mesh-named objects and retained bodies.
        """
        tracked = {body.id for body in previously_published}
        tracked.update(body.id for body in self.free_floating())
        robot_roots = {
            robot.root for robot in RobotModels(self.world, self.robot).robots()
        }
        return [
            body
            for body in self.world.bodies
            if body not in robot_roots
            and (
                body.id in tracked
                or MeshFormat.of_path(str(body.name).split("/")[-1]) is not None
            )
        ]
