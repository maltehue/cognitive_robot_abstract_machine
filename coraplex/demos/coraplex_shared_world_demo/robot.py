"""
A giskard that holds one robot of the world the demo serves.

Run one of these per robot, next to ``demo.py``::

    python robot.py --robot Stretch

It fetches the demo's world instead of building one of its own, so that every process
holds the same bodies, connections and degrees of freedom under the same identities, and
it registers only the connections of its own robot, so the other robots stay where their
processes put them. They are part of its world all the same, and are therefore seen and
avoided rather than moved.

A robot that reports its joint states is followed rather than commanded: this process
writes what it reports into the shared world and is sent no goals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import StrEnum
from typing import List, Type

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    MirroredRobotOfManyInterface,
    OneRobotOfManyInterface,
    RobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerSettings,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.model.world_config import WorldFromFetchService
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago

# %% how the controller runs

CONTROL_FREQUENCY = 20.0
"""
Frequency in hertz the controller solves at.
"""

REAL_TIME_FACTOR = 1.0
"""
How fast the simulation runs relative to the wall clock.

Unpaced servers race each other: the robot with the fewest joints finishes its cycles
soonest and would move faster than the others.
"""

# %% the handles a robot can be dragged by


APARTMENT_ROOT_NAME = "apartment_root"
"""
Name of the body the apartment stands on, which the robots are driven relative to.
"""


@dataclass(frozen=True)
class MarkerChain:
    """
    One kinematic chain an interactive marker offers a handle for.

    Dragging the handle asks the robot's giskard to move :attr:`tip` relative to
    :attr:`root`. Both are named the way the marker node resolves them, so a link
    several robots carry has to bring its prefix.
    """

    root: str
    """
    The link the goal is expressed relative to.
    """

    tip: str
    """
    The link the handle is attached to and that the goal moves.
    """


# %% the robots of the demo


class DemoRobot(StrEnum):
    """
    The robots of the demo, each driven by a process of its own.
    """

    PR2 = "PR2"
    STRETCH = "Stretch"
    TIAGO = "Tiago"

    @property
    def annotation_type(self) -> Type[AbstractRobot]:
        """
        The annotation that marks this robot in the shared world.
        """
        match self:
            case DemoRobot.PR2:
                return PR2
            case DemoRobot.STRETCH:
                return Stretch
            case DemoRobot.TIAGO:
                return Tiago

    @property
    def joint_states_topic(self) -> str | None:
        """
        The topic this robot reports its joint positions on, or ``None`` where the
        demo's plans move it instead.
        """
        match self:
            case DemoRobot.TIAGO:
                return f"/{self.lower()}/joint_states"
            case _:
                return None

    @property
    def is_mirrored(self) -> bool:
        """
        Whether this robot is followed on its joint state topic rather than commanded.
        """
        return self.joint_states_topic is not None

    @property
    def joint_state_publisher_node_name(self) -> str:
        """
        Name of the node reporting this robot's joint positions.
        """
        return f"joint_states_of_{self.lower()}"

    @property
    def giskard_node_name(self) -> str:
        """
        Name of the giskard node that drives this robot.
        """
        return f"giskard_{self.lower()}"

    @property
    def command_action_name(self) -> str:
        """
        Name of the action this robot's giskard takes goals on.
        """
        return f"{self.giskard_node_name}/command"

    @property
    def marker_node_name(self) -> str:
        """
        Name of the node offering this robot's handles.
        """
        return f"interactive_marker_{self.lower()}"

    @property
    def marker_namespace(self) -> str:
        """
        Topic namespace this robot's handles appear under.

        Every robot keeps its own, because the markers of two robots would otherwise
        share both their topics and their marker names.
        """
        return f"{self.giskard_node_name}/{InteractiveMarkerSettings.marker_namespace}"

    def marker_chains(self, world_root_name: str) -> List[MarkerChain]:
        """
        The chains this robot offers a handle for: its base against the world it stands
        in, and every hand against its base.

        A mirrored robot offers none, because a goal sent by hand would fight whoever
        publishes its joint states.

        :param world_root_name: Name of the root of that world.
        """
        match self:
            case DemoRobot.TIAGO:
                return []
            case DemoRobot.PR2:
                return [
                    MarkerChain(root=world_root_name, tip="pr2/base_footprint"),
                    MarkerChain(root="pr2/base_footprint", tip="l_gripper_tool_frame"),
                    MarkerChain(root="pr2/base_footprint", tip="r_gripper_tool_frame"),
                ]
            case DemoRobot.STRETCH:
                return [
                    MarkerChain(
                        root=world_root_name, tip="stretch_description/base_link"
                    ),
                    MarkerChain(
                        root="stretch_description/base_link", tip="link_grasp_center"
                    ),
                ]


# %% the process


def build_robot_interface(robot: DemoRobot) -> RobotInterfaceConfig:
    """
    How this process reaches the given robot: by following what it reports, where it
    reports anything, and by moving it on a goal otherwise.

    :param robot: The robot this process holds.
    """
    if not robot.is_mirrored:
        return OneRobotOfManyInterface(robot_type=robot.annotation_type)
    return MirroredRobotOfManyInterface(
        robot_type=robot.annotation_type, joint_states_topic=robot.joint_states_topic
    )


def build_giskard(robot: DemoRobot) -> Giskard:
    """
    The giskard that holds the given robot of the fetched world.

    It neither serves that world nor draws it: a process starting late would otherwise
    fetch a copy of it rather than the original, and every copy would draw the same
    markers again.

    :param robot: The robot this process holds.
    :return: The server, not yet set up.
    """
    return Giskard(
        world_config=WorldFromFetchService(),
        robot_interface_config=build_robot_interface(robot),
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.STANDALONE,
            publishes_world=False,
            real_time_factor=REAL_TIME_FACTOR,
        ),
        qp_controller_config=QPControllerConfig(target_frequency=CONTROL_FREQUENCY),
    )


def main() -> None:
    """
    Serve goals for the robot named on the command line until ROS shuts down.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        type=DemoRobot,
        choices=list(DemoRobot),
        required=True,
        help="which robot of the demo's world this process holds",
    )
    robot = parser.parse_args().robot
    rospy.init_node(robot.giskard_node_name)
    build_giskard(robot).live()


if __name__ == "__main__":
    main()
