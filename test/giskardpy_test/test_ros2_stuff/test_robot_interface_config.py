from dataclasses import fields, is_dataclass

import pytest

from giskardpy.data_types.exceptions import RobotNotInWorldError
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    OneRobotOfManyInterface,
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.daisy.configs import (
    DaisyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    PR2VelocityMujocoInterface,
)
from giskardpy.middleware.ros2.scripts.iai_robots.stretch.configs import (
    StretchStandaloneInterface,
)
from giskardpy.middleware.ros2.scripts.iai_robots.tracy.configs import (
    TracyStandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.model.world_config import EmptyWorld
from semantic_digital_twin.robots.daisy import DAiSyJoint
from semantic_digital_twin.robots.stretch import StretchJoint
from semantic_digital_twin.robots.tracy import TracyJoint
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection

# %% the interface hierarchy is built from dataclasses


def test_the_robot_interface_base_is_a_dataclass():
    assert is_dataclass(RobotInterfaceConfig)


def test_the_interface_state_bound_after_construction_is_no_constructor_argument():
    assert [field.name for field in fields(RobotInterfaceConfig) if field.init] == []


def test_the_standalone_interface_takes_the_joint_names_positionally():
    interface = StandAloneRobotInterfaceConfig(["torso_lift_joint", "head_pan_joint"])

    assert interface.joint_names == ["torso_lift_joint", "head_pan_joint"]


def test_two_interfaces_with_the_same_joint_names_are_equal():
    assert StandAloneRobotInterfaceConfig(["head_pan_joint"]) == (
        StandAloneRobotInterfaceConfig(["head_pan_joint"])
    )


def test_the_mujoco_interface_defaults_name_every_part_of_the_drive():
    interface = PR2VelocityMujocoInterface()

    assert (
        interface.map_name,
        interface.localization_joint_name,
        interface.odom_link_name,
        interface.drive_joint_name,
    ) == ("map", "localization", "odom_combined", "brumbrum")


# %% the tf frame synchronizer is created on demand


def test_an_interface_starts_without_a_tf_frame_synchronizer():
    assert StandAloneRobotInterfaceConfig([]).tf_frame_synchronizer is None


# %% attaching binds the giskard instance the accessors read from


def test_attaching_lets_the_interface_reach_the_server_config():
    server_config = GiskardServerConfig()
    interface = StandAloneRobotInterfaceConfig([])
    giskard = Giskard(
        world_config=EmptyWorld(),
        server_config=server_config,
        robot_interface_config=interface,
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )

    interface.attach(giskard)

    assert interface.server_config is server_config


# %% robot specific interfaces declare their controlled joints


def test_the_tracy_interface_controls_both_arms():
    assert TracyStandAloneRobotInterfaceConfig().joint_names == [
        TracyJoint.LEFT_SHOULDER_PAN,
        TracyJoint.LEFT_SHOULDER_LIFT,
        TracyJoint.LEFT_ELBOW,
        TracyJoint.LEFT_WRIST_1,
        TracyJoint.LEFT_WRIST_2,
        TracyJoint.LEFT_WRIST_3,
        TracyJoint.RIGHT_SHOULDER_PAN,
        TracyJoint.RIGHT_SHOULDER_LIFT,
        TracyJoint.RIGHT_ELBOW,
        TracyJoint.RIGHT_WRIST_1,
        TracyJoint.RIGHT_WRIST_2,
        TracyJoint.RIGHT_WRIST_3,
    ]


def test_the_daisy_interface_controls_both_arms_and_both_grippers():
    assert DaisyStandAloneRobotInterfaceConfig().joint_names == [
        DAiSyJoint.LEFT_SHOULDER_PAN,
        DAiSyJoint.LEFT_SHOULDER_LIFT,
        DAiSyJoint.LEFT_ELBOW,
        DAiSyJoint.LEFT_WRIST_1,
        DAiSyJoint.LEFT_WRIST_2,
        DAiSyJoint.LEFT_WRIST_3,
        DAiSyJoint.RIGHT_SHOULDER_PAN,
        DAiSyJoint.RIGHT_SHOULDER_LIFT,
        DAiSyJoint.RIGHT_ELBOW,
        DAiSyJoint.RIGHT_WRIST_1,
        DAiSyJoint.RIGHT_WRIST_2,
        DAiSyJoint.RIGHT_WRIST_3,
        DAiSyJoint.LEFT_GRIPPER_FINGER,
        DAiSyJoint.LEFT_GRIPPER_RIGHT_FINGER,
        DAiSyJoint.RIGHT_GRIPPER_FINGER,
        DAiSyJoint.RIGHT_GRIPPER_RIGHT_FINGER,
    ]


def test_two_daisy_interfaces_do_not_share_their_joint_name_list():
    first = DaisyStandAloneRobotInterfaceConfig()
    second = DaisyStandAloneRobotInterfaceConfig()

    assert first.joint_names is not second.joint_names


def test_the_stretch_interface_controls_every_joint_except_the_drive():
    assert StretchStandaloneInterface().joint_names == [
        StretchJoint.GRIPPER_LEFT_FINGER,
        StretchJoint.GRIPPER_RIGHT_FINGER,
        StretchJoint.RIGHT_WHEEL,
        StretchJoint.LEFT_WHEEL,
        StretchJoint.LIFT,
        StretchJoint.ARM_L3,
        StretchJoint.ARM_L2,
        StretchJoint.ARM_L1,
        StretchJoint.ARM_L0,
        StretchJoint.WRIST_YAW,
        StretchJoint.HEAD_PAN,
        StretchJoint.HEAD_TILT,
    ]


# %% one robot of a world that holds several


@pytest.fixture()
def world_with_two_robots() -> World:
    """
    A world holding a PR2 and a Stretch, each on a drive of its own.
    """
    try:
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=PR2,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=1.0),
                ),
                RobotSpecification(
                    semantic_annotation_type=Stretch,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=-1.0),
                ),
            ],
        ).to_domain_object()
    except ParsingError as error:
        pytest.skip(f"Robot URDF not available: {error}")


def test_the_interface_controls_its_robots_connections_and_its_drive(
    world_with_two_robots: World,
):
    robot = world_with_two_robots.get_semantic_annotations_by_type(PR2)[0]

    connections = OneRobotOfManyInterface(robot_type=PR2).connections_to_control(
        world_with_two_robots
    )

    assert set(connections) == {
        connection
        for connection in robot.connections
        if isinstance(connection, ActiveConnection)
    } | {robot.root.parent_connection}


def test_two_interfaces_of_one_world_control_different_connections(
    world_with_two_robots: World,
):
    pr2_connections = OneRobotOfManyInterface(robot_type=PR2).connections_to_control(
        world_with_two_robots
    )
    stretch_connections = OneRobotOfManyInterface(
        robot_type=Stretch
    ).connections_to_control(world_with_two_robots)

    assert set(pr2_connections).isdisjoint(set(stretch_connections))
    assert set(pr2_connections) | set(stretch_connections) == set(
        world_with_two_robots.get_connections_by_type(ActiveConnection)
    )


def test_an_interface_for_a_robot_that_is_not_there_says_so(
    world_with_two_robots: World,
):
    with pytest.raises(RobotNotInWorldError):
        OneRobotOfManyInterface(robot_type=Tiago).connections_to_control(
            world_with_two_robots
        )
