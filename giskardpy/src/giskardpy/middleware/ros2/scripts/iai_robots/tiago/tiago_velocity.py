from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    TiagoVelocityInterface,
    WorldWithTiagoConfigDiffDrive,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.tiago import Tiago


def main():
    rospy.init_node("giskard")

    robot_description = load_xacro(Tiago.get_ros_file_path())
    giskard = Giskard(
        world_config=WorldWithTiagoConfigDiffDrive(urdf=robot_description),
        robot_interface_config=TiagoVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=30
        ),
    )
    giskard.live()


if __name__ == "__main__":
    main()
