"""
The arm motions the Tracy pouring demo and the calibration ramp share: parking both arms
and holding the left gripper's cup upright above the table.
"""

from __future__ import annotations

import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose

# %% robot and table

TABLE_SURFACE_Z = 0.875 + 0.1
"""
Height of the table top in the world root, in metres.
"""

LEFT_TOOL_FRAME = "l_gripper_tool_frame"
"""
The frame the source cup is held in; its z axis is the axis the cup tilts about.
"""

LEFT_TILT_JOINT = "left_wrist_3_joint"
"""
The joint that turns the left tool frame about its own z axis without moving it, so
turning it alone tilts the held cup.
"""

_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    LEFT_TILT_JOINT,
]
_LEFT_ARM_PARK_POSITIONS = [2.62, -1.035, 1.13, -0.966, -0.88, 2.07]

_RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pan_joint",
    "right_shoulder_lift_joint",
    "right_elbow_joint",
    "right_wrist_1_joint",
    "right_wrist_2_joint",
    "right_wrist_3_joint",
]
_RIGHT_ARM_PARK_POSITIONS = [3.72, -2.07, -1.17, 4.0, 0.82, 0.75]

# %% connection


def connect_to_giskard(node_name: str) -> GiskardWrapper:
    """
    Start ROS, spin a node of this name in a background thread and connect it to the
    running Giskard server.
    """
    rclpy.init()
    node = rclpy.create_node(node_name)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()
    return GiskardWrapper(node)


# %% motions


def park_arms(giskard: GiskardWrapper) -> None:
    """
    Move both arms to their park positions.
    """
    world = giskard.world
    park_state = JointState.from_mapping(
        mapping={
            world.get_connection_by_name(name): value
            for name, value in zip(
                _LEFT_ARM_JOINT_NAMES + _RIGHT_ARM_JOINT_NAMES,
                _LEFT_ARM_PARK_POSITIONS + _RIGHT_ARM_PARK_POSITIONS,
            )
        }
    )
    motion = MotionStatechart()
    park_task = JointPositionList(goal_state=park_state)
    motion.add_node(park_task)
    motion.add_node(EndMotion.when_true(park_task))
    giskard.execute(motion)


def upright_carry_pose(giskard: GiskardWrapper, tilt: float = 0.0) -> Pose:
    """
    :param tilt: Rotation about the tool's z axis, towards the receiver, in radians.
    :return: The left tool pose that holds the cup above the table, upright when
        ``tilt`` is zero.
    """
    world = giskard.world
    root_T_upright = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=1,
        pos_y=0.3,
        pos_z=TABLE_SURFACE_Z + 0.15,
        quat_z=0.5,
        quat_x=0.5,
        quat_y=0.5,
        quat_w=0.5,
        reference_frame=world.root,
    )
    upright_T_tilted = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=tilt)
    return (root_T_upright @ upright_T_tilted).to_pose()


def move_left_tool_to(giskard: GiskardWrapper, pose: Pose) -> None:
    """
    Move the left tool frame to ``pose`` and stop once it can get no closer.
    """
    world = giskard.world
    motion = MotionStatechart()
    motion.add_node(
        CartesianPose(
            root_link=world.root,
            tip_link=world.get_body_by_name(LEFT_TOOL_FRAME),
            goal_pose=pose,
        )
    )
    motion.add_node(minimum_reached := LocalMinimumReached())
    motion.add_node(EndMotion.when_true(minimum_reached))
    giskard.execute(motion)
