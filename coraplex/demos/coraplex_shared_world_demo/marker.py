"""
The interactive marker of one robot of the shared world.

Run one of these per robot, next to ``demo.py``::

    python marker.py --robot Stretch

Dragging one of its handles in RViz and releasing it sends a Cartesian goal to that
robot's giskard, so a robot can be moved by hand while the demo's plans run. The handles
of every robot live under a namespace of their own, because two markers would otherwise
share their topics and their marker names.
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import rclpy

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerNode,
    InteractiveMarkerSettings,
    MarkerParameter,
)
from robot import APARTMENT_ROOT_NAME, DemoRobot

# %% what the node is launched with

ROS_ARGUMENTS_FLAG = "--ros-args"
"""
The flag that opens the part of a command line ROS reads itself.
"""

PARAMETER_FLAG = "-p"
"""
The flag that overrides one node parameter.
"""


def as_parameter_list(names: List[str]) -> str:
    """
    The names written as one string array override.

    :param names: The names to hand over.
    """
    return f"[{','.join(names)}]"


def marker_ros_arguments(robot: DemoRobot) -> List[str]:
    """
    The ROS arguments that hand one robot's chains, giskard and namespace to the marker
    node, which is what a launch file would otherwise do.

    :param robot: The robot whose handles this marker offers.
    """
    chains = robot.marker_chains(APARTMENT_ROOT_NAME)
    overrides = {
        MarkerParameter.ROOT_LINKS: as_parameter_list([chain.root for chain in chains]),
        MarkerParameter.TIP_LINKS: as_parameter_list([chain.tip for chain in chains]),
        MarkerParameter.GISKARD_NODE_NAME: robot.giskard_node_name,
        MarkerParameter.MARKER_NAMESPACE: robot.marker_namespace,
    }
    return [ROS_ARGUMENTS_FLAG] + [
        argument
        for parameter, value in overrides.items()
        for argument in [PARAMETER_FLAG, f"{parameter}:={value}"]
    ]


# %% the process


def main() -> None:
    """
    Offer the handles of the robot named on the command line until ROS shuts down.

    The ROS context is initialized here rather than by
    :func:`giskardpy.middleware.ros2.rospy.init_node` so that the chains can be passed
    to it; the node reads them from the context it is created in.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robot",
        type=DemoRobot,
        choices=list(DemoRobot),
        required=True,
        help="which robot of the demo's world this marker offers handles for",
    )
    robot = parser.parse_args().robot
    rclpy.init(args=[sys.argv[0], *marker_ros_arguments(robot)])
    rospy.init_node(robot.marker_node_name)
    node = InteractiveMarkerNode(InteractiveMarkerSettings.from_node(rospy.get_node()))
    node.giskard.node_handle.get_logger().info(
        f"{robot} offers {len(node.markers)} handles under {robot.marker_namespace}"
    )
    rospy.spinner_thread.join()


if __name__ == "__main__":
    main()
