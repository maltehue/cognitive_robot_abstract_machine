from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    WorldWithTracyConfig,
    TracyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.utils.utils import load_xacro

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.giskard import Giskard


def main():
    rospy.init_node("giskard")
    rospy.get_node().declare_parameter("robot_description", "")
    robot_description = rospy.get_node().get_parameter("robot_description").value
    if not robot_description:
        robot_description = load_xacro(
            "package://iai_tracy_description/urdf/tracy.urdf.xacro"
        )

    giskard = Giskard(
        world_config=WorldWithTracyConfig(urdf=robot_description),
        robot_interface_config=TracyStandAloneRobotInterfaceConfig(),
        server_config=GiskardServerConfig(
            execution_mode=ExecutionMode.STANDALONE,
            debug_mode=True,
            record_control_cycles=True,
            publish_debug_expressions=False,
        ),
        qp_controller_config=QPControllerConfig(
            target_frequency=80,
            # The fill tasks predict the pour over their own window, so the horizon only
            # needs to smooth the joint motion; a short one keeps the other tasks quick.
            prediction_horizon=20,
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
