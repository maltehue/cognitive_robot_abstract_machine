from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    TiagoVelocityInterface,
    WorldWithTiagoConfigDiffDrive,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world_description.connections import DifferentialDrive

# %% velocity interface


def test_velocity_interface_sets_up_against_the_robot_description(init_rospy):
    """
    The configuration the robot actually runs comes up against the robot description:
    the hardware topics are wired and every controlled joint resolves.

    Closed-loop control needs live joint states, so this covers initialisation rather
    than motion.
    """
    giskard = Giskard(
        world_config=WorldWithTiagoConfigDiffDrive(
            urdf=load_xacro(Tiago.get_ros_file_path())
        ),
        robot_interface_config=TiagoVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=30
        ),
    )

    giskard.setup()

    world = giskard.executor.context.world
    assert world.get_connections_by_type(DifferentialDrive)
    for joint_name in TiagoVelocityInterface().velocity_controlled_joint_names():
        assert world.get_connection_by_name(joint_name) is not None
