"""
Cup-to-cup transfer demo with a watering can as the source.

Same motion as :mod:`demo_pouring_transfer`, but the held container is a watering can
built from primitives: a cylindrical body, a spout attached to its side and a handle
over the top. The liquid leaves through the spout outlet, so the aim and the rim
clearance work from the spout rather than from the body's rim.
"""

from __future__ import annotations

import math
import threading
import time
from importlib.resources import files
from pathlib import Path

import rclpy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.templates import Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from rclpy.executors import SingleThreadedExecutor
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, HasSpout
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% constants shared with the cup demo

_JEROEN_CUP_STL = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
_JEROEN_CUP_SCALE = Scale(1, 1, 1)
_TABLE_SURFACE_Z = 0.875 + 0.1

START_FILL = 1.0
GOAL_FILL_CONST = 0.7
START_YAW = 0.1

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

# %% watering can geometry

CAN_BODY_WIDTH = 0.12
"""
Diameter of the can's cylindrical body, in metres.
"""

CAN_BODY_HEIGHT = 0.16
"""
Height of the can's body, in metres; the body frame sits at its base like the cup's.
"""

SPOUT_WIDTH = 0.02
"""
Diameter of the spout tube, in metres.
"""

SPOUT_LENGTH = 0.13
"""
Length of the spout tube from the body wall to the outlet, in metres.
"""

SPOUT_PITCH = -math.pi / 4
"""
Pitch of the spout frame in the body frame: its ``z`` axis, the outflow direction,
points up and towards negative ``x``, the side the can tilts to.
"""

SPOUT_OUTLET_X = -0.15
"""
Horizontal offset of the spout outlet from the body axis, in metres.
"""

SPOUT_OUTLET_HEIGHT = 0.15
"""
Height of the spout outlet above the body's base, in metres, just below the rim.
"""

HANDLE_THICKNESS = 0.02
"""
Edge length of the square handle bars, in metres.
"""

HANDLE_HEIGHT = 0.05
"""
How far the handle arch rises above the body's rim, in metres.
"""

CAN_COLOR = Color(R=0.2, G=0.5, B=0.2, A=1.0)
"""
Colour of the can's body, spout and handle.
"""

CAN_OUTFLOW_RATE_CONSTANT = 0.8
"""
Outflow rate constant of the can's drain, the cup demo's value.
"""

SPOUT_DISCHARGE_COEFFICIENT = 0.7
"""
Discharge coefficient of the spout outlet.

An orifice at the end of a tube discharges at about 0.6 to 0.8 of Torricelli speed,
unlike liquid sliding over a rim, which the cup demo's far smaller value describes.
"""


def _watering_can_body() -> Body:
    """
    The can's cylindrical body, standing on its frame origin.
    """
    body_cylinder = Cylinder(
        width=CAN_BODY_WIDTH,
        height=CAN_BODY_HEIGHT,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=CAN_BODY_HEIGHT / 2),
        color=CAN_COLOR,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([body_cylinder]),
        name=PrefixedName("watering_can"),
    )


def _watering_can_spout() -> Body:
    """
    The spout tube; the body frame is the outlet with ``z`` along the outflow.

    The tube spans from the outlet back to the body wall along negative ``z``.
    """
    spout_tube = Cylinder(
        width=SPOUT_WIDTH,
        height=SPOUT_LENGTH,
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-SPOUT_LENGTH / 2),
        color=CAN_COLOR,
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([spout_tube]),
        name=PrefixedName("watering_can_spout"),
    )


def _watering_can_handle() -> Body:
    """
    An arch over the body's rim, built from three bars.
    """
    arch_top = CAN_BODY_HEIGHT + HANDLE_HEIGHT
    post_height = HANDLE_HEIGHT + HANDLE_THICKNESS
    half_width = CAN_BODY_WIDTH / 2 - HANDLE_THICKNESS / 2
    bars = [
        Box(
            scale=Scale(CAN_BODY_WIDTH, HANDLE_THICKNESS, HANDLE_THICKNESS),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=arch_top),
            color=CAN_COLOR,
        ),
        Box(
            scale=Scale(HANDLE_THICKNESS, HANDLE_THICKNESS, post_height),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=half_width, z=CAN_BODY_HEIGHT + post_height / 2
            ),
            color=CAN_COLOR,
        ),
        Box(
            scale=Scale(HANDLE_THICKNESS, HANDLE_THICKNESS, post_height),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-half_width, z=CAN_BODY_HEIGHT + post_height / 2
            ),
            color=CAN_COLOR,
        ),
    ]
    return Body.from_shape_collection(
        shape_collection=ShapeCollection(bars),
        name=PrefixedName("watering_can_handle"),
    )


def attach_watering_can(world: World, left_tool_frame: Body) -> HasSpout:
    """
    Build the watering can in the left gripper and annotate it as a spouted container.

    The body hangs from the gripper like the cup does, the spout leaves the body on the
    side the can tilts to, and the handle arches over the rim.
    """
    body = _watering_can_body()
    spout = _watering_can_spout()
    handle = _watering_can_handle()
    with world.modify_world():
        world.add_body(body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=left_tool_frame,
                child=body,
                name=PrefixedName("l_gripper_T_watering_can"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0, y=-0.05
                ),
            )
        )
        world.add_body(spout)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=body,
                child=spout,
                name=PrefixedName("watering_can_T_spout"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=SPOUT_OUTLET_X, z=SPOUT_OUTLET_HEIGHT, pitch=SPOUT_PITCH
                ),
            )
        )
        world.add_body(handle)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=body,
                child=handle,
                name=PrefixedName("watering_can_T_handle"),
            )
        )
    can = HasSpout(name=PrefixedName("watering_can"), root=body, spout=spout)
    with world.modify_world():
        world.add_semantic_annotation(can)
    return can


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


# %% giskard connection

rclpy.init()
rclpy_node = rclpy.create_node("tracy_watering_can_demo")
executor = SingleThreadedExecutor()
executor.add_node(rclpy_node)
threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

giskard = GiskardWrapper(rclpy_node)
world = giskard.world

# %% park arms

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

# %% move the left gripper to the upright carry pose

left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

root_T_upright_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
    pos_x=1,
    pos_y=0.3,
    pos_z=_TABLE_SURFACE_Z + 0.15,
    quat_z=0.5,
    quat_x=0.5,
    quat_y=0.5,
    quat_w=0.5,
    reference_frame=world.root,
)
upright_pose_T_rotated = HomogeneousTransformationMatrix.from_xyz_rpy(yaw=START_YAW)
upright_pose = (root_T_upright_pose @ upright_pose_T_rotated).to_pose()

msc_cartesian = MotionStatechart()
cartesian_task = CartesianPose(
    root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
)
msc_cartesian.add_node(cartesian_task)
msc_cartesian.add_node(min_reached := LocalMinimumReached())
msc_cartesian.add_node(EndMotion.when_true(min_reached))
giskard.execute(msc_cartesian)

# %% set up the can and the cup on the first run, reuse them afterwards

# Re-running against an already-running Giskard fetches a world that still contains the containers
# of a previous run; re-adding them would fail on duplicate bodies, so on reuse only the fill levels
# are reset.
containers_already_present = bool(
    world.get_semantic_annotations_by_name("watering_can")
)

if containers_already_present:
    print("Containers already present; reusing them and resetting fill levels.")
    watering_can = world.get_semantic_annotation_by_name("watering_can")
    receiving_cup = world.get_semantic_annotation_by_name("receiving_cup")
    # The fetched annotations reference detached copies of their fill connections; re-resolve the
    # world-resident ones so their fill state can be read and written.
    for container in (watering_can, receiving_cup):
        container.fill_connection = world.get_connection(
            container.fill_connection.parent, container.fill_connection.child
        )
    with world.modify_world():
        JointState.from_mapping(
            {
                watering_can.fill_connection: START_FILL,
                receiving_cup.fill_connection: 0.0,
            }
        ).apply_to(world)
    time.sleep(0.5)
    # Rebuild the coupling from the current constants, so a changed drain parameter takes effect
    # without restarting Giskard; the published equation swap makes Giskard rebuild its side too.
    receiving_cup.recouple_outflow_from(
        source=watering_can,
        world=world,
        fill_equation=watering_can.create_pouring_equation(
            world, CAN_OUTFLOW_RATE_CONSTANT, SPOUT_DISCHARGE_COEFFICIENT
        ),
    )
else:
    watering_can = attach_watering_can(world, left_tool_frame)
    watering_can.initialize_fill_level(
        world=world,
        initial_fill=START_FILL,
        outflow_rate_constant=CAN_OUTFLOW_RATE_CONSTANT,
        discharge_coefficient=SPOUT_DISCHARGE_COEFFICIENT,
    )

    receiving_cup_body = _spawn_jeroen_cup_body("receiving_cup")
    with world.modify_world():
        world.add_body(receiving_cup_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                receiving_cup_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.099, _TABLE_SURFACE_Z
                ),
            )
        )
    receiving_cup = HasFillLevel(
        name=PrefixedName("receiving_cup"), root=receiving_cup_body
    )
    with world.modify_world():
        world.add_semantic_annotation(receiving_cup)
    time.sleep(0.2)
    receiving_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )
    receiving_cup.receive_outflow_from(source=watering_can, world=world)

time.sleep(0.2)

assert watering_can.fill_level == START_FILL
assert receiving_cup.fill_level == 0.0

# %% transfer

goal_fill = GOAL_FILL_CONST
tolerance = 0.05
transfer_task = FillByTransferTask(
    receiver=receiving_cup,
    goal_value=goal_fill,
    fill_level_tolerance=tolerance,
    reference_velocity=0.03,
)
# Keep the stream's landing point in the receiver; with a spout the stream leaves at the outlet, so
# the optimizer positions the outlet rather than the body's rim.
no_spill = KeepProjectileInReceiver(
    receiver=receiving_cup,
    source=watering_can,
    weight=DefaultWeights.WEIGHT_MAXIMUM,
    reference_velocity=0.1,
)
# The spout outlet is the can's lowest lip while pouring, so this keeps the outlet above the cup's
# rim however far the can tilts.
keep_above = KeepSourceRimAboveReceiverRim(
    receiver=receiving_cup,
    source=watering_can,
    minimum_clearance=0.07,
    clearance_band=0.05,
    weight=DefaultWeights.WEIGHT_MAXIMUM,
)
keep_plane = AlignPlanes(
    root_link=world.root,
    tip_link=left_tool_frame,
    goal_normal=Vector3.X(reference_frame=world.root),
    tip_normal=Vector3.Z(reference_frame=left_tool_frame),
)
motion = Parallel([no_spill, keep_above, keep_plane])
msc_transfer = MotionStatechart()
msc_transfer.add_node(motion)
msc_transfer.add_node(EndMotion.when_true(motion))
giskard.execute(msc_transfer)

extended_motion = Parallel([transfer_task, no_spill, keep_above, keep_plane])
msc = MotionStatechart()
msc.add_node(extended_motion)
msc.add_node(EndMotion.when_true(extended_motion))
print("Start transfer.")
giskard.execute(msc)

# ``giskard.execute`` returns on the action result, but the final fill levels are synchronized
# back from the Giskard process asynchronously. Wait until the receiver's fill reflects the
# completed transfer and settles.
settle_deadline = time.time() + 10.0
previous_fill = None
while time.time() < settle_deadline:
    current_fill = receiving_cup.fill_level
    transfer_reflected = current_fill > tolerance
    settled = previous_fill is not None and abs(current_fill - previous_fill) < 1e-4
    if transfer_reflected and settled:
        break
    previous_fill = current_fill
    time.sleep(0.1)

print(f"receiving cup fill level: {receiving_cup.fill_level}")
print(f"watering can fill level: {watering_can.fill_level}")

# %% return to the carry pose

msc_cartesian = MotionStatechart()
cartesian_task = CartesianPose(
    root_link=world.root, tip_link=left_tool_frame, goal_pose=upright_pose
)
msc_cartesian.add_node(cartesian_task)
msc_cartesian.add_node(min_reached := LocalMinimumReached())
msc_cartesian.add_node(EndMotion.when_true(min_reached))
giskard.execute(msc_cartesian)
