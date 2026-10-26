from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
    TracyVelocityInterface,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from rclpy import Parameter
from rclpy.exceptions import ParameterUninitializedException

RECORD_CONTROL_CYCLES_PARAMETER = "record_control_cycles"
"""
ROS parameter that switches on recording every control cycle, for comparing what the
pouring model predicted with what the robot did.
"""


def main():
    rospy.init_node("giskard")

    robot_description = load_xacro(
            "package://iai_tracy_description/urdf/tracy.urdf.xacro"
        )
    records_control_cycles = (
        rospy.get_node().declare_parameter(RECORD_CONTROL_CYCLES_PARAMETER, True).value
    )
    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyVelocityInterface(),
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.CLOSED_LOOP,
            debug_mode=records_control_cycles,
            record_control_cycles=records_control_cycles,
        ),
        qp_controller_config=QPControllerConfig(
            target_frequency=80, prediction_horizon=30
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
