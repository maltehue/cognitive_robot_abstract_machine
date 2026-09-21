"""
A giskard that holds one robot of the world the demo serves.

Run one of these per robot, next to ``demo.py``::

    python robot.py --robot Stretch

It fetches the demo's world instead of building one of its own, so that every process
holds the same bodies, connections and degrees of freedom under the same identities, and
it runs the robot's own velocity interface in closed loop, in the robot's namespace. The
other robots are part of its world all the same, and are therefore seen and avoided
rather than moved.

Of everything that interface talks to, only the joint state topic is answered here: a
window publishes on it, while odometry stays silent and velocity commands go nowhere.
"""

from __future__ import annotations

import argparse
import sys
from enum import StrEnum
from typing import List, Type

import rclpy

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchTopic,
    StretchVelocityInterface,
)
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    TiagoTopic,
    TiagoVelocityInterface,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.model.world_config import WorldFromFetchService
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago

# %% how the controller runs

CONTROL_FREQUENCY = 25.0
"""
Frequency in hertz the controller solves at.
"""

PREDICTION_HORIZON = 30
"""
How many control cycles ahead the controller plans.
"""

ROS_ARGUMENTS_FLAG = "--ros-args"
"""
The flag that opens the part of a command line ROS reads itself.
"""

REMAP_FLAG = "-r"
"""
The flag that renames one node or topic of a started node.
"""

# %% the robots of the demo


class DemoRobot(StrEnum):
    """
    The robots of the demo, each held by a process of its own.
    """

    STRETCH = "Stretch"
    TIAGO = "Tiago"

    @property
    def annotation_type(self) -> Type[AbstractRobot]:
        """
        The annotation that marks this robot in the shared world.
        """
        match self:
            case DemoRobot.STRETCH:
                return Stretch
            case DemoRobot.TIAGO:
                return Tiago

    @property
    def interface(self) -> RobotInterfaceConfig:
        """
        The interface config this robot runs on its hardware.
        """
        match self:
            case DemoRobot.STRETCH:
                return StretchVelocityInterface()
            case DemoRobot.TIAGO:
                return TiagoVelocityInterface()

    @property
    def joint_states_topic_name(self) -> str:
        """
        The topic this robot's interface reads its joint positions from, relative to the
        robot's namespace.
        """
        match self:
            case DemoRobot.STRETCH:
                return StretchTopic.JOINT_STATES
            case DemoRobot.TIAGO:
                return TiagoTopic.JOINT_STATES

    @property
    def namespace(self) -> str:
        """
        The ROS namespace this robot's nodes and topics live under.
        """
        return f"/{self.lower()}"

    @property
    def joint_states_topic(self) -> str:
        """
        The topic this robot reports its joint positions on.
        """
        return f"{self.namespace}/{self.joint_states_topic_name}"

    @property
    def joint_state_publisher_node_name(self) -> str:
        """
        Name of the node reporting this robot's joint positions.
        """
        return f"joint_states_of_{self.lower()}"

    @property
    def giskard_node_name(self) -> str:
        """
        Name of the giskard node that holds this robot.
        """
        return f"giskard_{self.lower()}"

    @property
    def command_action_name(self) -> str:
        """
        Name of the action this robot's giskard takes goals on, which tells that it is
        up.
        """
        return f"{self.namespace}/{self.giskard_node_name}/command"

    def namespace_arguments(self) -> List[str]:
        """
        The ROS arguments that put a node into this robot's namespace.
        """
        return [ROS_ARGUMENTS_FLAG, REMAP_FLAG, f"__ns:={self.namespace}"]


# %% the process


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
        world_config=WorldFromFetchService(robot_type=robot.annotation_type),
        robot_interface_config=robot.interface,
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.CLOSED_LOOP, publishes_world=False
        ),
        qp_controller_config=QPControllerConfig(
            target_frequency=CONTROL_FREQUENCY, prediction_horizon=PREDICTION_HORIZON
        ),
    )


def main() -> None:
    """
    Hold the robot named on the command line, in its namespace, until ROS shuts down.
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
    rclpy.init(args=[sys.argv[0], *robot.namespace_arguments()])
    rospy.init_node(robot.giskard_node_name)
    build_giskard(robot).live()


if __name__ == "__main__":
    main()
