from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.tiago.configs import (
    TiagoVelocityInterface,
    WorldWithTiagoConfigDiffDrive,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)

# %% velocity interface


def test_velocity_controlled_joints_exist_in_the_world(tiago_world_copy):
    """
    The velocity group controller addresses joints by name in a fixed order, so a joint
    renamed in the robot description would leave the controller driving nothing.
    """
    interface = TiagoVelocityInterface()

    for joint_name in interface.velocity_controlled_joint_names():
        connection = tiago_world_copy.get_connection_by_name(joint_name)
        assert isinstance(connection, ActiveConnection1DOF)


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
