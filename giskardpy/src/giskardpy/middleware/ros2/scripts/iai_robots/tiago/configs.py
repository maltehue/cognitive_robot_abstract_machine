from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List

from giskardpy.middleware.ros2.robot_interface_config import RobotInterfaceConfig
from giskardpy.model.world_config import WorldWithDiffDriveRobot
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago, TiagoJoint

# %% the topics of the robot


class TiagoTopic(StrEnum):
    """
    The topics the Tiago's controllers talk on, relative to the robot's namespace.
    """

    JOINT_STATES = "joint_states"
    """
    Where the robot reports its joint positions.
    """

    ODOMETRY = "mobile_base_controller/odom"
    """
    Where the base controller reports the pose of the base.
    """

    BASE_VELOCITY_COMMAND = "mobile_base_controller/cmd_vel"
    """
    Where the base controller takes its velocity.
    """

    JOINT_VELOCITY_COMMAND = "joint_velocity_controller/commands"
    """
    Where the velocity group controller takes the velocities of its joints.
    """


# %% commanding the robot


@dataclass
class TiagoVelocityInterface(RobotInterfaceConfig):
    """
    Commands the torso, head, both arms, both grippers and the drive of Tiago through
    their velocity controllers.

    The drive and the localization are those of the robot this interface belongs to,
    so the interface works in a world holding other robots as well; the tf frames of
    the localization are the names its bodies carry in the world.
    """

    @staticmethod
    def velocity_controlled_joint_names() -> List[TiagoJoint]:
        """
        The joints driven by the velocity group controller.

        Their order matches the controller's command layout, so it is significant.
        """
        return [
            TiagoJoint.TORSO_LIFT,
            TiagoJoint.HEAD_1,
            TiagoJoint.HEAD_2,
            TiagoJoint.LEFT_ARM_1,
            TiagoJoint.LEFT_ARM_2,
            TiagoJoint.LEFT_ARM_3,
            TiagoJoint.LEFT_ARM_4,
            TiagoJoint.LEFT_ARM_5,
            TiagoJoint.LEFT_ARM_6,
            TiagoJoint.LEFT_ARM_7,
            TiagoJoint.LEFT_GRIPPER_FINGER,
            TiagoJoint.RIGHT_ARM_1,
            TiagoJoint.RIGHT_ARM_2,
            TiagoJoint.RIGHT_ARM_3,
            TiagoJoint.RIGHT_ARM_4,
            TiagoJoint.RIGHT_ARM_5,
            TiagoJoint.RIGHT_ARM_6,
            TiagoJoint.RIGHT_ARM_7,
            TiagoJoint.RIGHT_GRIPPER_FINGER,
        ]

    def setup(self):
        diff_drive = self.robot.root.parent_connection
        localization = diff_drive.parent.parent_connection
        self.sync_6dof_joint_with_tf_frame(
            joint=localization,
            tf_parent_frame=str(localization.parent.name),
            tf_child_frame=str(localization.child.name),
        )
        self.sync_odometry_topic(TiagoTopic.ODOMETRY, diff_drive)
        self.add_base_cmd_velocity(
            cmd_vel_topic=TiagoTopic.BASE_VELOCITY_COMMAND, joint=diff_drive
        )

        self.sync_joint_state_topic(TiagoTopic.JOINT_STATES)
        self.add_joint_velocity_group_controller(
            cmd_topic=TiagoTopic.JOINT_VELOCITY_COMMAND,
            connections=self.velocity_controlled_joint_names(),
        )


# %% the world the robot stands in


@dataclass
class WorldWithTiagoConfigDiffDrive(WorldWithDiffDriveRobot):
    """
    A Tiago alone on a differential drive.
    """

    urdf_view: AbstractRobot = field(kw_only=True, default=Tiago, init=False)

    def setup_collision_config(self):
        pass
