from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import List, Union

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
    RobotInterfaceConfig,
)
from giskardpy.model.world_config import (
    WorldWithOmniDriveRobot,
    WorldWithDiffDriveRobot,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.stretch import Stretch, StretchJoint
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
)


@dataclass
class StretchStandaloneInterface(StandAloneRobotInterfaceConfig):
    """
    Simulates the arm, gripper, head and drive of Stretch without talking to hardware.
    """

    joint_names: List[StretchJoint] = field(init=False, default_factory=list)
    """
    The arm, gripper, wheel and head joints of Stretch, without the base drive.
    """

    def __post_init__(self) -> None:
        self.joint_names = [
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

    def controlled_joint_names(self, world: World) -> List[Union[str, PrefixedName]]:
        """
        The joints to control, with the base drive resolved from ``world``.

        :param world: The world holding the robot and its base drive connection.
        """
        drive = world.get_connections_by_type(DifferentialDrive)[0]
        return [drive.name, *self.joint_names]

    def setup(self):
        self.register_controlled_joints(self.controlled_joint_names(self.world))


class StretchTopic(StrEnum):
    """
    The topics the Stretch's controllers talk on, relative to the robot's namespace.
    """

    JOINT_STATES = "joint_states"
    """
    Where the robot reports its joint positions.
    """

    ODOMETRY = "odom"
    """
    Where the robot reports the pose of its base.
    """

    BASE_VELOCITY_COMMAND = "cmd_vel"
    """
    Where the base takes its velocity.
    """

    JOINT_VELOCITY_COMMAND = "joint_velocity_cmd"
    """
    Where the velocity group controller takes the velocities of its joints.
    """


@dataclass
class StretchVelocityInterface(RobotInterfaceConfig):
    """
    Commands the arm, head and drive of Stretch through their velocity controllers.

    The drive and the localization are those of the robot this interface belongs to,
    so the interface works in a world holding other robots as well; the tf frames of
    the localization are the names its bodies carry in the world.
    """

    @staticmethod
    def velocity_controlled_joint_names() -> List[StretchJoint]:
        """
        The joints driven by the velocity group controller.

        Their order matches the controller's command layout, so it is significant.
        """
        return [
            StretchJoint.ARM_L0,  # 0
            StretchJoint.LIFT,  # 1
            StretchJoint.WRIST_YAW,  # 2
            StretchJoint.WRIST_PITCH,  # 3
            StretchJoint.WRIST_ROLL,  # 4
            StretchJoint.HEAD_PAN,  # 5
            StretchJoint.HEAD_TILT,  # 6
            StretchJoint.GRIPPER_LEFT_FINGER,  # 7
            StretchJoint.RIGHT_WHEEL,  # 8
            StretchJoint.LEFT_WHEEL,  # 9
        ]

    def setup(self):
        diff_drive = self.robot.root.parent_connection
        localization = diff_drive.parent.parent_connection
        self.sync_6dof_joint_with_tf_frame(
            joint=localization,
            tf_parent_frame=str(localization.parent.name),
            tf_child_frame=str(localization.child.name),
        )
        self.sync_odometry_topic(StretchTopic.ODOMETRY, diff_drive)
        self.add_base_cmd_velocity(
            cmd_vel_topic=StretchTopic.BASE_VELOCITY_COMMAND, joint=diff_drive
        )

        self.sync_joint_state_topic(StretchTopic.JOINT_STATES)
        self.add_joint_velocity_group_controller(
            cmd_topic=StretchTopic.JOINT_VELOCITY_COMMAND,
            connections=self.velocity_controlled_joint_names(),
            minimum_valid_velocity=0.03,
            minimum_velocity_overrides={
                StretchJoint.LIFT: 0.0,
                StretchJoint.ARM_L0: 0.0,
                StretchJoint.GRIPPER_LEFT_FINGER: 0.0,
            },
        )


@dataclass
class WorldWithStretchConfig(WorldWithOmniDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Stretch, init=False)

    def setup_collision_config(self):
        pass


@dataclass
class WorldWithStretchConfigDiffDrive(WorldWithDiffDriveRobot):
    urdf_view: AbstractRobot = field(kw_only=True, default=Stretch, init=False)

    def setup_collision_config(self):
        pass
