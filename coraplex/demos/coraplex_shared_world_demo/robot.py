"""
A giskard that holds one robot of the world the demo serves.

Run one of these per robot, next to ``demo.py``::

    python robot.py --robot Stretch

It fetches the demo's world instead of building one of its own, so that every process
holds the same bodies, connections and degrees of freedom under the same identities, and
it registers only the connections of its own robot, so the other robots stay where their
processes put them. They are part of its world all the same, and are therefore seen and
avoided rather than moved.

The robot is followed rather than commanded: whatever its joint state topic reports is
written into the shared world and announced to every other process, and no goals are
sent.
"""

from __future__ import annotations

import argparse
from enum import StrEnum
from typing import Type

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    MirroredRobotOfManyInterface,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.model.world_config import WorldFromFetchService
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago

# %% how the controller runs

CONTROL_FREQUENCY = 20.0
"""
Frequency in hertz the controller solves at.
"""

JOINT_STATES_TOPIC_NAME = "joint_states"
"""
Name of the topic a robot reports its joint positions on, within its own namespace.
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
    def namespace(self) -> str:
        """
        The ROS namespace this robot's topics live under.
        """
        return f"/{self.lower()}"

    @property
    def joint_states_topic(self) -> str:
        """
        The topic this robot reports its joint positions on.
        """
        return f"{self.namespace}/{JOINT_STATES_TOPIC_NAME}"

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
        return f"{self.giskard_node_name}/command"


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
        world_config=WorldFromFetchService(),
        robot_interface_config=MirroredRobotOfManyInterface(
            robot_type=robot.annotation_type,
            joint_states_topic=robot.joint_states_topic,
        ),
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.STANDALONE, publishes_world=False
        ),
        qp_controller_config=QPControllerConfig(target_frequency=CONTROL_FREQUENCY),
    )


def main() -> None:
    """
    Follow the robot named on the command line until ROS shuts down.
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
