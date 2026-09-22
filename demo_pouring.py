import threading
import time
from dataclasses import dataclass
from pathlib import Path

import math
import rclpy
from rclpy.executors import SingleThreadedExecutor

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.joint_state import JointState
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPosition,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Connection
from importlib.resources import files

# ------ Constants ----
_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
_JEROEN_CUP_SCALE = Scale(1, 1, 1)
_TABLE_SURFACE_Z = 0.9
IS_SIM = False


def _spawn_jeroen_cup_body(name: str) -> Body:
    """
    Create a Body with the Jeroen cup mesh geometry.
    """
    mesh = Mesh(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
        filename=_JEROEN_CUP_STL,
        scale=_JEROEN_CUP_SCALE,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
    )


# --------------------------
rclpy.init()

# World with real robot:
rclpy_node = rclpy.create_node("tracy_demo")

executor = SingleThreadedExecutor()
executor.add_node(rclpy_node)
print(f"Executor started")
thread = threading.Thread(
    target=executor.spin,
    daemon=True,
    name="rclpy-executor",
)
thread.start()


giskard = GiskardWrapper(rclpy_node)

world = giskard.world
print(f"world root: {world.root.name.name}")

# ----- Park arms -----
_LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pan_joint",
    "left_shoulder_lift_joint",
    "left_elbow_joint",
    "left_wrist_1_joint",
    "left_wrist_2_joint",
    "left_wrist_3_joint",
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

park_state = JointState.from_mapping(
    mapping={
        world.get_connection_by_name(name): value
        for name, value in zip(
            _LEFT_ARM_JOINT_NAMES + _RIGHT_ARM_JOINT_NAMES,
            _LEFT_ARM_PARK_POSITIONS + _RIGHT_ARM_PARK_POSITIONS,
        )
    }
)
msc_park = MotionStatechart()
park_task = JointPositionList(goal_state=park_state)
msc_park.add_node(park_task)
msc_park.add_node(EndMotion.when_true(park_task))
giskard.execute(msc_park)

# ------ Create cartesian init pose ----
left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

root_T_upright_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
    pos_x=1,
    pos_y=0.2,
    pos_z=_TABLE_SURFACE_Z + 0.3,
    quat_z=0.5,
    quat_x=0.5,
    quat_y=0.5,
    quat_w=0.5,
    reference_frame=world.root,
)
upright_pose_T_rotated = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=0.1)
upright_pose = (root_T_upright_pose @ upright_pose_T_rotated).to_pose()

msc_cartesian = MotionStatechart()
cartesian_task = CartesianPose(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_pose=upright_pose,
    # reference_linear_velocity=0.05,
    # reference_angular_velocity=0.05
)
msc_cartesian.add_node(cartesian_task)
msc_cartesian.add_node(min_reached := LocalMinimumReached())
msc_cartesian.add_node(EndMotion.when_true(min_reached))

giskard.execute(msc_cartesian)

# ----- Add Cup to the gripper -----

old_number_bodies = len(world.bodies)

grasped_cup_body = _spawn_jeroen_cup_body("grasped_cup")
with world.modify_world():
    connection = FixedConnection.create_with_dofs(
        world=world,
        parent=left_tool_frame,
        child=grasped_cup_body,
        name=PrefixedName("l_gripper_T_grasped_cup"),
        parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
            roll=-math.pi / 2.0, y=-0.0
        ),
    )
    world.add_connection(connection)

assert len(world.bodies) > old_number_bodies

grasped_cup = HasFillLevel(name=PrefixedName("grasped_cup"), root=grasped_cup_body)

with world.modify_world():
    world.add_semantic_annotation(grasped_cup)

grasped_cup.initialize_fill_level(
    world=world, initial_fill=1.0, outflow_rate_constant=1.0
)

# ----- World setup Done -----
goal_fill = 0.5
tolerance = 0.05

msc_pouring = MotionStatechart()
pouring_task = PouringTask(
    fill_equation=grasped_cup.fill_equation,
    fill_connection=grasped_cup.fill_connection,
    root_link=world.root,
    tip_link=grasped_cup_body,
    goal_value=goal_fill,
    fill_level_tolerance=tolerance,
    reference_velocity=0.03,
)
keep_position = CartesianPosition(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_point=Point3(reference_frame=left_tool_frame),
    weight=DefaultWeights.WEIGHT_ABOVE_CA,
)
motion = Parallel([pouring_task, keep_position])
msc_pouring.add_node(motion)
msc_pouring.add_node(EndMotion.when_true(motion))

print("Start pouring.")

giskard.execute(msc_pouring)
