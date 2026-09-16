from dataclasses import dataclass

from giskardpy.middleware.ros2.command_publishing import DriveVelocityCommandPublisher
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchStandaloneInterface,
    StretchVelocityInterface,
    WorldWithStretchConfigDiffDrive,
)
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.model.world_config import WorldConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    DifferentialDrive,
)

# %% controlled joints


def test_controlled_joints_resolve_the_drive_from_the_world(stretch_world_copy):
    """
    The base drive is looked up in the world instead of being named by a literal, so the
    interface controls the base whatever the drive connection ended up being called.
    """
    drive = stretch_world_copy.get_connections_by_type(DifferentialDrive)[0]
    interface = StretchStandaloneInterface()

    controlled_joint_names = interface.controlled_joint_names(stretch_world_copy)

    assert controlled_joint_names[0] == drive.name
    assert controlled_joint_names[1:] == interface.joint_names


def test_every_controlled_joint_exists_in_the_world(stretch_world_copy):
    """
    Registering a joint the world does not know about fails at setup time, so every
    declared name must resolve to a connection.
    """
    interface = StretchStandaloneInterface()

    for joint_name in interface.controlled_joint_names(stretch_world_copy):
        assert stretch_world_copy.get_connection_by_name(joint_name) is not None


# %% velocity interface


def test_velocity_controlled_joints_exist_in_the_world(stretch_world_copy):
    """
    The velocity group controller addresses joints by name in a fixed order, so a joint
    renamed in the robot description would leave the controller driving nothing.
    """
    interface = StretchVelocityInterface()

    for joint_name in interface.velocity_controlled_joint_names():
        connection = stretch_world_copy.get_connection_by_name(joint_name)
        assert isinstance(connection, ActiveConnection1DOF)


def test_velocity_interface_sets_up_against_the_robot_description(init_rospy):
    """
    The configuration the robot actually runs comes up against the robot description:
    the hardware topics are wired and every controlled joint resolves.

    Closed-loop control needs live joint states, so this covers initialisation rather
    than motion.
    """
    giskard = Giskard(
        world_config=WorldWithStretchConfigDiffDrive(
            urdf=load_xacro(Stretch.get_ros_file_path())
        ),
        robot_interface_config=StretchVelocityInterface(),
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=30
        ),
    )

    giskard.setup()

    world = giskard.executor.context.world
    assert world.get_connections_by_type(DifferentialDrive)
    for joint_name in StretchVelocityInterface().velocity_controlled_joint_names():
        assert world.get_connection_by_name(joint_name) is not None


# %% the velocity interface in a world shared with another robot


@dataclass
class WorldBuiltElsewhere(WorldConfig):
    """
    Holds a world that was built before the giskard, and builds nothing itself.
    """

    def setup_world(self) -> None:
        return


def test_velocity_interface_wires_its_own_robot_of_a_shared_world(
    init_rospy, world_with_two_robots: World
):
    """
    In a world holding another robot too, the drive that is commanded and the
    localization that follows tf are the Stretch's own, not the first ones of the world.
    """
    stretch = world_with_two_robots.get_semantic_annotations_by_type(Stretch)[0]
    drive = stretch.root.parent_connection
    localization = drive.parent.parent_connection
    interface = StretchVelocityInterface()
    giskard = Giskard(
        world_config=WorldBuiltElsewhere(
            world=world_with_two_robots, robot_type=Stretch
        ),
        robot_interface_config=interface,
        server_config=GiskardServerConfig(execution_mode=ExecutionMode.CLOSED_LOOP),
        qp_controller_config=QPControllerConfig(
            target_frequency=25, prediction_horizon=30
        ),
    )

    giskard.setup()
    try:
        drive_publisher = [
            publisher
            for publisher in giskard.motion_server.control_loop.command_publishers
            if isinstance(publisher, DriveVelocityCommandPublisher)
        ][0]
        assert drive_publisher.connection is drive
        assert interface.tf_frame_synchronizer.connection_to_frames == {
            localization: (str(localization.parent.name), str(localization.child.name))
        }
    finally:
        giskard.close_world_model_ros_interface()
