"""
Two-handed transfer demo: a watering can in the left hand pours into a cup in the right.

The can is the one of :mod:`demo_pouring_transfer_watering_can`; the cup hangs from the
right gripper instead of standing on the table. The aim, the rim clearance and the fill
task may move either hand, and the right hand is kept upright throughout.

Two cases run one after the other. In the first the right hand holds the cup at its carry
position and helps close the aiming gap with :class:`ShareAimWithReceiver`, since the
aiming task only ever differentiates the source's side of that gap and so never gives the
cup a reason to move on its own. In the second the right hand carries the cup around a
circle for as long as the pour lasts, so the can has to follow it to keep the stream
inside.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
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
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
    CartesianPositionTrajectory,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
    ShareAimWithReceiver,
)
from rclpy.executors import SingleThreadedExecutor
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, HasSpout
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
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
FILL_TOLERANCE = 0.05
"""
How far short of the goal fill level the transfer counts as done.
"""

CARRY_ORIENTATION = HomogeneousTransformationMatrix.from_xyz_quaternion(
    quat_x=0.5, quat_y=0.5, quat_z=0.5, quat_w=0.5
)
"""
Orientation of a gripper tool frame while carrying: its ``y`` axis points up and its
``z`` axis forward, so a container hanging from it stands upright.
"""

RIGHT_GRASP_ROTATION = math.pi / 2
"""
Rotation about the right tool frame's own ``y`` axis (the vertical carry axis), in
radians, so the right gripper approaches the cup from its right side instead of head-on
like the left gripper approaches the can.

Rotating about the tool's own ``y`` axis leaves that axis pointing where it already
does, so the cup's grasp transform and the upright task are unaffected by this angle.
"""

RIGHT_CARRY_ORIENTATION = (
    CARRY_ORIENTATION
    @ HomogeneousTransformationMatrix.from_xyz_rpy(pitch=RIGHT_GRASP_ROTATION)
)
"""
Orientation of the right tool frame while carrying: the shared carry orientation turned
about its own ``y`` axis by :data:`RIGHT_GRASP_ROTATION`.
"""

LEFT_CARRY_POSITION = (1.0, 0.3, _TABLE_SURFACE_Z + 0.3)
"""
Where the left tool frame carries the can, in the world root frame, in metres.
"""

RIGHT_CARRY_POSITION = (1.0, -0.2, _TABLE_SURFACE_Z + 0.15)
"""
Where the right tool frame carries the cup, in the world root frame, in metres; below
the can and on the side its spout points to.
"""

POUR_START_TILT_MARGIN = 0.05
"""
How far past the tilt at which the liquid reaches the spout the can is tilted before the
pour, in radians; below that tilt the drain has neither flow nor gradient.
"""

CUP_CIRCLE_RADIUS = 0.08
"""
Radius of the circle the right hand carries the cup around while pouring, in metres.
"""

CUP_CIRCLE_TURNS = 3
"""
How many times the cup is carried around that circle; the pour reaches its fill goal
while the cup is still moving.
"""

CUP_CIRCLE_POINT_SPACING = 0.005
"""
Distance between neighbouring points of the circle, in metres; the trajectory task
assumes a dense path.
"""

CUP_CIRCLE_REFERENCE_VELOCITY = 0.05
"""
Reference velocity of the task carrying the cup around the circle, in m/s.
"""

CUP_CIRCLE_LOOK_AHEAD = 0.02
"""
How far ahead of the cup its target on the circle is kept, in metres.
"""

CUP_CIRCLE_SKIP_AHEAD_POINTS = 10
"""
How many points of the circle the cup may advance by in one control cycle; a closed path
needs this cap, since the point nearest to the cup can also be one lap further along.
"""

CUP_CIRCLE_WEIGHT = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
"""
Weight of the task carrying the cup around: above the tasks that only keep the scene
tidy, below the aim's :attr:`DefaultWeights.WEIGHT_MAXIMUM`, so the aim wins wherever the
can cannot follow.
"""

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


# %% carry poses, motions and the held cup


def carry_pose(
    world: World,
    position: tuple[float, float, float],
    orientation: HomogeneousTransformationMatrix = CARRY_ORIENTATION,
    tilt: float = 0.0,
) -> Pose:
    """
    The carry pose of a gripper tool frame at the given position.

    :param orientation: Orientation of the tool frame while carrying.
    :param tilt: Rotation about the tool's ``z`` axis, the pouring axis, in radians; a
        positive value lowers the can's spout.
    """
    x, y, z = position
    return (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x, y=y, z=z, reference_frame=world.root
        )
        @ orientation
        @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=tilt)
    ).to_pose()


def pour_start_tilt(can: HasSpout) -> float:
    """
    The tilt at which the can's liquid reaches the spout at its current fill, plus a
    margin, so the drain has flow and gradient from the start.
    """
    equation = can.fill_equation.ungated()
    dry_height = equation.container_height * (1.0 - can.fill_level)
    return math.atan2(dry_height, equation.lip_offset) + POUR_START_TILT_MARGIN


def reset_fill_levels(
    world: World, source: HasFillLevel, receiver: HasFillLevel
) -> None:
    """
    Fill the source and empty the receiver, so every case pours from the same state.
    """
    with world.modify_world():
        JointState.from_mapping(
            {source.fill_connection: START_FILL, receiver.fill_connection: 0.0}
        ).apply_to(world)
    time.sleep(0.5)


def execute_pose(
    giskard: GiskardWrapper, world: World, tip_link: Body, goal_pose: Pose
) -> None:
    """
    Move ``tip_link`` to ``goal_pose`` and stop at the local minimum.
    """
    statechart = MotionStatechart()
    cartesian_task = CartesianPose(
        root_link=world.root, tip_link=tip_link, goal_pose=goal_pose
    )
    statechart.add_node(cartesian_task)
    statechart.add_node(minimum_reached := LocalMinimumReached())
    statechart.add_node(EndMotion.when_true(minimum_reached))
    giskard.execute(statechart)


def attach_cup(world: World, right_tool_frame: Body) -> HasFillLevel:
    """
    Hang the receiving cup from the right gripper, held like the can is held.
    """
    cup_body = _spawn_jeroen_cup_body("held_cup")
    with world.modify_world():
        world.add_body(cup_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=right_tool_frame,
                child=cup_body,
                name=PrefixedName("r_gripper_T_held_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0, y=-0.05
                ),
            )
        )
    cup = HasFillLevel(name=PrefixedName("held_cup"), root=cup_body)
    with world.modify_world():
        world.add_semantic_annotation(cup)
    return cup


# %% the two cases and the transfer that runs one


def circling_path(
    start: tuple[float, float, float], radius: float, turns: float, point_spacing: float
) -> list[tuple[float, float, float]]:
    """
    A dense horizontal circle that begins at ``start`` and curves towards the robot, so
    ``start`` stays the point of the circle the arm is stretched furthest to reach.

    :param radius: Radius of the circle, in metres.
    :param turns: How many times the circle is walked.
    :param point_spacing: Distance between neighbouring points, in metres.
    """
    x, y, z = start
    center_x = x - radius
    total_angle = turns * 2.0 * math.pi
    steps = max(1, round(total_angle * radius / point_spacing))
    angles = [total_angle * step / steps for step in range(steps + 1)]
    return [
        (center_x + radius * math.cos(angle), y + radius * math.sin(angle), z)
        for angle in angles
    ]


@dataclass
class TransferCase:
    """
    One of the transfers the demo runs, told apart by what the right hand does with the
    cup while the can pours into it.
    """

    name: str
    """
    What this case is called in the demo's output.
    """

    cup_path: list[tuple[float, float, float]] = field(default_factory=list)
    """
    Positions the right hand carries the cup through while pouring, in the world root
    frame and in metres; empty leaves the cup at :data:`RIGHT_CARRY_POSITION`.
    """

    receiver_shares_aim: bool = True
    """
    Whether the cup moves towards the pour to help close the aiming gap; a cup carried
    along :attr:`cup_path` does not, because the path decides where it goes.
    """


TRANSFER_CASES = [
    TransferCase(name="the cup is held still and helps the aim"),
    TransferCase(
        name="the cup is carried around and the can follows",
        cup_path=circling_path(
            start=RIGHT_CARRY_POSITION,
            radius=CUP_CIRCLE_RADIUS,
            turns=CUP_CIRCLE_TURNS,
            point_spacing=CUP_CIRCLE_POINT_SPACING,
        ),
        receiver_shares_aim=False,
    ),
]


def run_transfer(
    giskard: GiskardWrapper,
    world: World,
    case: TransferCase,
    source: HasSpout,
    receiver: HasFillLevel,
    source_tool_frame: Body,
    receiver_tool_frame: Body,
) -> None:
    """
    Pour from ``source`` into ``receiver`` until it holds :data:`GOAL_FILL_CONST`, with
    the right hand doing what ``case`` asks of it.
    """
    transfer_task = FillByTransferTask(
        receiver=receiver,
        goal_value=GOAL_FILL_CONST,
        fill_level_tolerance=FILL_TOLERANCE,
        reference_velocity=0.03,
    )
    # The landing point depends on both hands, so the optimizer may move the can or the cup to
    # keep the stream in the cup.
    no_spill = KeepProjectileInReceiver(
        receiver=receiver,
        source=source,
        weight=DefaultWeights.WEIGHT_MAXIMUM,
        reference_velocity=0.1,
    )
    keep_above = KeepSourceRimAboveReceiverRim(
        receiver=receiver,
        source=source,
        minimum_clearance=0.07,
        clearance_band=0.05,
        weight=DefaultWeights.WEIGHT_MAXIMUM,
    )
    keep_plane = AlignPlanes(
        name="keep pouring plane",
        root_link=world.root,
        tip_link=source_tool_frame,
        goal_normal=Vector3.X(reference_frame=world.root),
        tip_normal=Vector3.Z(reference_frame=source_tool_frame),
    )
    # The cup hangs from the right tool with its up axis along the tool's y axis; keep that
    # vertical so the cup does not tip while the arm follows the pour or carries it around.
    keep_cup_upright = AlignPlanes(
        name="keep cup upright",
        root_link=world.root,
        tip_link=receiver_tool_frame,
        goal_normal=Vector3.Z(reference_frame=world.root),
        tip_normal=Vector3.Y(reference_frame=receiver_tool_frame),
    )

    pouring = Parallel([transfer_task, no_spill, keep_above, keep_plane])
    statechart = MotionStatechart()
    statechart.add_node(pouring)
    statechart.add_node(keep_cup_upright)
    if case.receiver_shares_aim:
        # Drives the cup itself towards the landing point, since KeepProjectileInReceiver only
        # ever differentiates the source's side of that same gap and so never gives the cup a
        # reason to move.
        statechart.add_node(ShareAimWithReceiver(receiver=receiver, source=source))
    if case.cup_path:
        statechart.add_node(
            CartesianPositionTrajectory(
                name="carry the cup around",
                root_link=world.root,
                tip_link=receiver_tool_frame,
                goal_points=[
                    Point3(x, y, z, reference_frame=world.root)
                    for x, y, z in case.cup_path
                ],
                maximum_skip_ahead=CUP_CIRCLE_SKIP_AHEAD_POINTS,
                look_ahead_distance=CUP_CIRCLE_LOOK_AHEAD,
                reference_velocity=CUP_CIRCLE_REFERENCE_VELOCITY,
                weight=CUP_CIRCLE_WEIGHT,
            )
        )
    statechart.add_node(EndMotion.when_true(pouring))
    print(f"Start transfer: {case.name}.")
    giskard.execute(statechart)

    # ``giskard.execute`` returns on the action result, but the final fill levels are
    # synchronized back from the Giskard process asynchronously. Wait until the receiver's fill
    # reflects the completed transfer and settles.
    settle_deadline = time.time() + 10.0
    previous_fill = None
    while time.time() < settle_deadline:
        current_fill = receiver.fill_level
        transfer_reflected = current_fill > FILL_TOLERANCE
        settled = previous_fill is not None and abs(current_fill - previous_fill) < 1e-4
        if transfer_reflected and settled:
            break
        previous_fill = current_fill
        time.sleep(0.1)

    print(f"receiver fill level: {receiver.fill_level}")
    print(f"source fill level: {source.fill_level}")


# %% giskard connection

rclpy.init()
rclpy_node = rclpy.create_node("tracy_two_hands_demo")
executor = SingleThreadedExecutor()
executor.add_node(rclpy_node)
threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

giskard = GiskardWrapper(rclpy_node)
world = giskard.world
left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")
right_tool_frame = world.get_body_by_name("r_gripper_tool_frame")

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

# %% carry both containers

execute_pose(giskard, world, left_tool_frame, carry_pose(world, LEFT_CARRY_POSITION))
execute_pose(
    giskard,
    world,
    right_tool_frame,
    carry_pose(world, RIGHT_CARRY_POSITION, RIGHT_CARRY_ORIENTATION),
)

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
    held_cup = world.get_semantic_annotation_by_name("held_cup")
    for container in (watering_can, held_cup):
        container.fill_connection = world.get_connection(
            container.fill_connection.parent, container.fill_connection.child
        )
    reset_fill_levels(world, watering_can, held_cup)
    held_cup.recouple_outflow_from(
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
    held_cup = attach_cup(world, right_tool_frame)
    time.sleep(0.2)
    held_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )
    held_cup.receive_outflow_from(source=watering_can, world=world)

time.sleep(0.2)

# %% run both cases

for case in TRANSFER_CASES:
    print(f"Case: {case.name}.")
    # Both hands start each case from their carry poses, which also brings them back from
    # the previous one.
    execute_pose(
        giskard, world, left_tool_frame, carry_pose(world, LEFT_CARRY_POSITION)
    )
    execute_pose(
        giskard,
        world,
        right_tool_frame,
        carry_pose(world, RIGHT_CARRY_POSITION, RIGHT_CARRY_ORIENTATION),
    )
    reset_fill_levels(world, watering_can, held_cup)
    assert watering_can.fill_level == START_FILL
    assert held_cup.fill_level == 0.0
    # Tilt the can to the brink of pouring, where its drain has flow and gradient.
    execute_pose(
        giskard,
        world,
        left_tool_frame,
        carry_pose(world, LEFT_CARRY_POSITION, tilt=pour_start_tilt(watering_can)),
    )
    run_transfer(
        giskard=giskard,
        world=world,
        case=case,
        source=watering_can,
        receiver=held_cup,
        source_tool_frame=left_tool_frame,
        receiver_tool_frame=right_tool_frame,
    )

# %% return both hands to their carry poses

execute_pose(giskard, world, left_tool_frame, carry_pose(world, LEFT_CARRY_POSITION))
execute_pose(
    giskard,
    world,
    right_tool_frame,
    carry_pose(world, RIGHT_CARRY_POSITION, RIGHT_CARRY_ORIENTATION),
)
