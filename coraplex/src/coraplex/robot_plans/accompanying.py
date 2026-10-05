"""
Goals that accompany an action's motions.

An action's motions end when their own goals are reached. A goal listed in an action's
:attr:`~coraplex.robot_plans.actions.base.ActionDescription.accompanied_by` runs
alongside every one of them without being part of that end: the controller works
towards it while the motion lasts, and the motion ends when the motion's own goals are
reached whether or not the accompanying one is. That is how a constraint such as "the
robot must look where it operates" is kept during a pick or a place rather than
performed before it: a camera that cannot quite reach the object - a head whose pitch
ends before the tabletop - still turns as far as it can, and the hand is not held up by
it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.tasks.pointing import Pointing
from semantic_digital_twin.robots.robot_parts import AbstractRobot, Camera
from semantic_digital_twin.spatial_types.spatial_types import Point3, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from coraplex.robot_plans.actions.base import ActionDescription


class AccompanyingGoal(ABC):
    """
    A goal an action's motions run alongside their own.
    """

    @abstractmethod
    def node(self, action: ActionDescription) -> MotionStatechartNode:
        """
        :param action: The action whose motion is about to run, grounded and attached
            to its plan, so its world, robot and parameters can be read.
        :return: The goal as a motion statechart node, built against the world as it
            is when the motion is prepared.
        """


def camera_turned_from(robot: AbstractRobot, camera: Camera, world: World) -> Body:
    """
    The body a camera is turned from to look somewhere.

    From the torso's root as a rule, which keeps the base where it is and lets a torso
    that lifts or turns help the head; from the robot's own root where there is no
    torso. A robot whose torso costs more to move than any other joint - a humanoid
    keeping its balance - does not bend to look: where it has a neck the camera hangs
    from, the camera is turned from the neck's root alone.

    :param robot: The robot looking.
    :param camera: The camera it looks with.
    :param world: The world they are in.
    :return: The body the camera is turned from.
    """
    torso = robot.get_torso_if_specified()
    neck = robot.get_neck_if_specified()
    if (
        torso is not None
        and torso.motion_cost > 1.0
        and neck is not None
        and camera.root in set(world.get_kinematic_structure_entities_of_branch(neck.root))
    ):
        return neck.root
    return torso.root if torso is not None else robot.root


@dataclass
class LookingAt(AccompanyingGoal):
    """
    The robot's camera kept on a point while the action's motions run.
    """

    target: Optional[Point3] = None
    """
    The point looked at, in any frame - a body's own, to follow the body as it moves -
    or ``None`` to look at where the action puts its object down, read off the action's
    ``target_location`` once it is grounded.
    """

    camera: Optional[Camera] = None
    """
    The camera that looks, or ``None`` for the robot's default camera.
    """

    def node(self, action: ActionDescription) -> MotionStatechartNode:
        camera = self.camera or action.robot.get_default_camera()
        target = self.target
        if target is None:
            target_location = getattr(action, "target_location", None)
            if not isinstance(target_location, Pose):
                raise ValueError(
                    f"{type(action).__name__} puts nothing down at a pose to look at"
                )
            target = target_location.to_position()
        return Pointing(
            root_link=camera_turned_from(action.robot, camera, action.world),
            tip_link=camera.root,
            goal_point=target,
            pointing_axis=camera.forward_facing_axis,
        )
