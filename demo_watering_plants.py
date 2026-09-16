"""
Watering demo: fill a watering can at a faucet, then water a flower pot.

Tracy carries a watering can with a side handle, stands it under a faucet, grasps the
faucet's valve lever and lets the fill task turn the valve until the can holds enough
water. It shuts the valve, picks the can up again by its handle, carries it to a flower
pot and pours with the cup-to-cup transfer tasks.
"""

from __future__ import annotations

import math
import threading
import time

import rclpy
from giskardpy.middleware.ros2.python_interface import GiskardWrapper
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.open_close import Close
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
from semantic_digital_twin.semantic_annotations.semantic_annotations import Faucet
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Box, Color, Cylinder, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% constants shared with the cup demo

_TABLE_SURFACE_Z = 0.875 + 0.1

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

WORLD_SYNC_PAUSE = 0.5
"""
Seconds to give a world change time to reach the Giskard process.
"""

CARRY_ORIENTATION = HomogeneousTransformationMatrix.from_xyz_quaternion(
    quat_x=0.5, quat_y=0.5, quat_z=0.5, quat_w=0.5
)
"""
Orientation of the gripper tool frame while carrying: its ``y`` axis points up and its
``z`` axis forward, so a container hanging from it stands upright.
"""

# %% watering can geometry

CAN_BODY_WIDTH = 0.12
"""
Diameter of the can's cylindrical body, in metres.
"""

CAN_BODY_HEIGHT = 0.16
"""
Height of the can's body, in metres; the body frame sits at its base.
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

HANDLE_OFFSET = 0.09
"""
Distance of the side handle's grip bar from the body axis, in metres, on the side
opposite the spout.
"""

HANDLE_BOTTOM = 0.04
"""
Height of the handle's lower connector above the body's base, in metres.
"""

HANDLE_TOP = 0.14
"""
Height of the handle's upper connector above the body's base, in metres.
"""

CAN_COLOR = Color(R=0.2, G=0.5, B=0.2, A=1.0)
"""
Colour of the can's body, spout and handle.
"""

CAN_OUTFLOW_RATE_CONSTANT = 0.8
"""
Outflow rate constant of the can's drain.
"""

SPOUT_DISCHARGE_COEFFICIENT = 0.7
"""
Discharge coefficient of the spout outlet, an orifice value.
"""

CAN_T_TOOL = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=HANDLE_OFFSET, z=(HANDLE_BOTTOM + HANDLE_TOP) / 2
) @ HomogeneousTransformationMatrix.from_xyz_rpy(roll=math.pi / 2)
"""
Pose of the gripper tool frame in the can frame while the can is held by its side
handle: the tool sits on the grip bar with its ``y`` axis along the can's up axis.
"""

# %% faucet geometry

FAUCET_X = 1.05
"""
Forward position of the faucet post on the table, in metres.
"""

FAUCET_Y = 0.45
"""
Sideways position of the faucet post on the table, in metres.
"""

FAUCET_POST_WIDTH = 0.05
"""
Edge length of the square faucet post, in metres.
"""

FAUCET_POST_HEIGHT = 0.55
"""
Height of the faucet post above the table, in metres.
"""

FAUCET_ARM_LENGTH = 0.2
"""
Length of the outlet arm from the post's axis to the outlet, in metres, towards the
robot.
"""

FAUCET_ARM_HEIGHT = 0.48
"""
Height of the outlet arm above the table, in metres.
"""

FAUCET_ARM_WIDTH = 0.04
"""
Edge length of the square outlet arm, in metres.
"""

VALVE_HEIGHT = 0.3
"""
Height of the valve joint above the table, in metres.
"""

VALVE_LEVER_LENGTH = 0.12
"""
Length of the valve lever from its joint to the grip at its tip, in metres.
"""

VALVE_OPEN_POSITION = math.pi / 2
"""
Valve position at which the faucet is fully open, in radians; it is shut at zero.
"""

FAUCET_VOLUME_RATE = 0.002
"""
Volume rate of the fully open faucet, in cubic metres per second.
"""

FAUCET_COLOR = Color(R=0.6, G=0.6, B=0.65, A=1.0)
"""
Colour of the faucet's post, arm and lever.
"""

# %% flower pot geometry

POT_X = 1.0
"""
Forward position of the flower pot on the table, in metres.
"""

POT_Y = 0.1
"""
Sideways position of the flower pot on the table, in metres.
"""

POT_WIDTH = 0.2
"""
Diameter of the flower pot, in metres.
"""

POT_HEIGHT = 0.15
"""
Height of the flower pot, in metres; the pot frame sits at its base.
"""

PLANT_STEM_WIDTH = 0.02
"""
Diameter of the plant's stem, in metres.
"""

PLANT_STEM_HEIGHT = 0.2
"""
Height of the plant's stem above the pot's rim, in metres.
"""

PLANT_CROWN_WIDTH = 0.16
"""
Diameter of the plant's crown, in metres.
"""

PLANT_CROWN_HEIGHT = 0.06
"""
Height of the plant's crown, in metres.
"""

POT_COLOR = Color(R=0.6, G=0.35, B=0.2, A=1.0)
"""
Colour of the flower pot.
"""

PLANT_COLOR = Color(R=0.1, G=0.6, B=0.1, A=1.0)
"""
Colour of the plant.
"""

# %% fill goals

CAN_FILL_GOAL = 0.9
"""
Fill level the can is filled to at the faucet.
"""

POT_START_FILL = 0.2
"""
Fill level of the flower pot before watering; the soil's moisture.
"""

POT_FILL_GOAL = 0.6
"""
Fill level the flower pot is watered to.
"""

FILL_TOLERANCE = 0.05
"""
Tolerance around both fill goals.
"""

POUR_START_TILT_MARGIN = 0.05
"""
How far past the tilt at which the liquid reaches the spout the can is carried to the
pot, in radians.

Below that tilt the drain model has no flow and no gradient, so the fill task could not
discover that tilting further starts the pour.
"""

# %% watering can


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
    A side handle opposite the spout: a vertical grip bar joined to the body by two
    connectors.
    """
    wall = CAN_BODY_WIDTH / 2
    connector_length = HANDLE_OFFSET - wall + HANDLE_THICKNESS / 2
    connector_x = wall + connector_length / 2 - HANDLE_THICKNESS / 2
    bars = [
        Box(
            scale=Scale(HANDLE_THICKNESS, HANDLE_THICKNESS, HANDLE_TOP - HANDLE_BOTTOM),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=HANDLE_OFFSET, z=(HANDLE_BOTTOM + HANDLE_TOP) / 2
            ),
            color=CAN_COLOR,
        ),
        Box(
            scale=Scale(connector_length, HANDLE_THICKNESS, HANDLE_THICKNESS),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=connector_x, z=HANDLE_BOTTOM
            ),
            color=CAN_COLOR,
        ),
        Box(
            scale=Scale(connector_length, HANDLE_THICKNESS, HANDLE_THICKNESS),
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=connector_x, z=HANDLE_TOP
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
    Build the watering can hanging from the gripper by its side handle and annotate it
    as a spouted container.
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
                parent_T_connection_expression=CAN_T_TOOL.inverse(),
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


# %% faucet


def build_faucet(world: World) -> Faucet:
    """
    Stand a faucet on the table: a post with an outlet arm reaching towards the robot
    and a lever valve on the post's near side, shut with the lever pointing at the
    robot.
    """
    post = Body.from_shape_collection(
        shape_collection=ShapeCollection(
            [
                Box(
                    scale=Scale(
                        FAUCET_POST_WIDTH, FAUCET_POST_WIDTH, FAUCET_POST_HEIGHT
                    ),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=FAUCET_POST_HEIGHT / 2
                    ),
                    color=FAUCET_COLOR,
                ),
                Box(
                    scale=Scale(FAUCET_ARM_LENGTH, FAUCET_ARM_WIDTH, FAUCET_ARM_WIDTH),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-FAUCET_ARM_LENGTH / 2, z=FAUCET_ARM_HEIGHT
                    ),
                    color=FAUCET_COLOR,
                ),
            ]
        ),
        name=PrefixedName("faucet"),
    )
    outlet = Body(name=PrefixedName("faucet_outlet"))
    lever = Body.from_shape_collection(
        shape_collection=ShapeCollection(
            [
                Box(
                    scale=Scale(VALVE_LEVER_LENGTH, HANDLE_THICKNESS, HANDLE_THICKNESS),
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        x=-VALVE_LEVER_LENGTH / 2
                    ),
                    color=FAUCET_COLOR,
                )
            ]
        ),
        name=PrefixedName("faucet_valve_lever"),
    )
    lever_grip = Body(name=PrefixedName("faucet_valve_grip"))
    with world.modify_world():
        world.add_body(post)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=post,
                name=PrefixedName("table_T_faucet"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=FAUCET_X, y=FAUCET_Y, z=_TABLE_SURFACE_Z
                ),
            )
        )
        world.add_body(outlet)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=post,
                child=outlet,
                name=PrefixedName("faucet_T_outlet"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-FAUCET_ARM_LENGTH, z=FAUCET_ARM_HEIGHT - FAUCET_ARM_WIDTH / 2
                ),
            )
        )
        world.add_body(lever)
        world.add_connection(
            RevoluteConnection.create_with_dofs(
                world=world,
                parent=post,
                child=lever,
                name=PrefixedName("faucet_T_valve"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-FAUCET_POST_WIDTH / 2, z=VALVE_HEIGHT
                ),
                axis=Vector3.Z(),
                dof_limits=DegreeOfFreedomLimits(
                    lower=DerivativeMap(position=0.0, velocity=-1.0),
                    upper=DerivativeMap(position=VALVE_OPEN_POSITION, velocity=1.0),
                ),
            )
        )
        world.add_body(lever_grip)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=lever,
                child=lever_grip,
                name=PrefixedName("valve_T_grip"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-VALVE_LEVER_LENGTH
                ),
            )
        )
    faucet = Faucet(
        name=PrefixedName("faucet"),
        root=post,
        outlet=outlet,
        valve=lever,
        maximum_volume_rate=FAUCET_VOLUME_RATE,
    )
    with world.modify_world():
        world.add_semantic_annotation(faucet)
    return faucet


# %% flower pot


def build_flower_pot(world: World) -> HasFillLevel:
    """
    Stand a flower pot with a plant on the table and annotate the pot as a container.
    """
    pot_body = Body.from_shape_collection(
        shape_collection=ShapeCollection(
            [
                Cylinder(
                    width=POT_WIDTH,
                    height=POT_HEIGHT,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=POT_HEIGHT / 2
                    ),
                    color=POT_COLOR,
                )
            ]
        ),
        name=PrefixedName("flower_pot"),
    )
    plant = Body.from_shape_collection(
        shape_collection=ShapeCollection(
            [
                Cylinder(
                    width=PLANT_STEM_WIDTH,
                    height=PLANT_STEM_HEIGHT,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=PLANT_STEM_HEIGHT / 2
                    ),
                    color=PLANT_COLOR,
                ),
                Cylinder(
                    width=PLANT_CROWN_WIDTH,
                    height=PLANT_CROWN_HEIGHT,
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                        z=PLANT_STEM_HEIGHT + PLANT_CROWN_HEIGHT / 2
                    ),
                    color=PLANT_COLOR,
                ),
            ]
        ),
        name=PrefixedName("plant"),
    )
    with world.modify_world():
        world.add_body(pot_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=pot_body,
                name=PrefixedName("table_T_flower_pot"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=POT_X, y=POT_Y, z=_TABLE_SURFACE_Z
                ),
            )
        )
        world.add_body(plant)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=pot_body,
                child=plant,
                name=PrefixedName("flower_pot_T_plant"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=POT_HEIGHT
                ),
            )
        )
    pot = HasFillLevel(name=PrefixedName("flower_pot"), root=pot_body)
    with world.modify_world():
        world.add_semantic_annotation(pot)
    return pot


# %% motions


def execute_pose(
    giskard: GiskardWrapper,
    world: World,
    tip_link: Body,
    goal_pose: Pose,
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


def carry_pose(world: World, x: float, y: float, z: float, tilt: float = 0.0) -> Pose:
    """
    The gripper tool frame's carry pose at the given position.

    :param tilt: Rotation about the tool's ``z`` axis, the pouring axis, in radians; a
        positive value lowers the spout.
    """
    return (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x, y=y, z=z, reference_frame=world.root
        )
        @ CARRY_ORIENTATION
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


def standing_pose(world: World, x: float, y: float) -> Pose:
    """
    The pose of a container standing upright on the table at the given position, with
    its ``x`` axis along the world's ``y``, as it hangs from the carry pose.
    """
    return HomogeneousTransformationMatrix.from_xyz_rpy(
        x=x, y=y, z=_TABLE_SURFACE_Z, yaw=math.pi / 2, reference_frame=world.root
    ).to_pose()


def grasp_pose(
    world: World, grip: Body, grip_T_tool: HomogeneousTransformationMatrix
) -> Pose:
    """
    The gripper tool frame's pose that holds ``grip`` with the given relative pose.
    """
    world_T_grip = HomogeneousTransformationMatrix(
        world.compute_forward_kinematics_np(world.root, grip),
        reference_frame=world.root,
    )
    return (world_T_grip @ grip_T_tool).to_pose()


def hand_over(world: World, branch_root: Body, new_parent: Body) -> None:
    """
    Re-parent ``branch_root`` under ``new_parent`` at its current pose and give the
    change time to reach Giskard.
    """
    world.move_branch(branch_root, new_parent)
    time.sleep(WORLD_SYNC_PAUSE)


def wait_for_synchronized_fill(container: HasFillLevel, above: float) -> None:
    """
    Wait until the container's fill reflects the finished transfer and stops changing.
    """
    deadline = time.time() + 10.0
    previous_fill = None
    while time.time() < deadline:
        current_fill = container.fill_level
        settled = previous_fill is not None and abs(current_fill - previous_fill) < 1e-4
        if current_fill > above and settled:
            return
        previous_fill = current_fill
        time.sleep(0.1)


# %% giskard connection

rclpy.init()
rclpy_node = rclpy.create_node("tracy_watering_demo")
executor = SingleThreadedExecutor()
executor.add_node(rclpy_node)
threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor").start()

giskard = GiskardWrapper(rclpy_node)
world = giskard.world
left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")

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

# %% set up the scene on the first run, reuse it afterwards

scene_already_present = bool(world.get_semantic_annotations_by_name("watering_can"))

if scene_already_present:
    print("Scene already present; reusing it and resetting fill levels.")
    watering_can = world.get_semantic_annotation_by_name("watering_can")
    faucet = world.get_semantic_annotation_by_name("faucet")
    flower_pot = world.get_semantic_annotation_by_name("flower_pot")
    for container in (watering_can, flower_pot):
        container.fill_connection = world.get_connection(
            container.fill_connection.parent, container.fill_connection.child
        )
    with world.modify_world():
        JointState.from_mapping(
            {
                watering_can.fill_connection: 0.0,
                flower_pot.fill_connection: POT_START_FILL,
                faucet.valve_connection: 0.0,
            }
        ).apply_to(world)
    if watering_can.root.parent_kinematic_structure_entity is not left_tool_frame:
        execute_pose(
            giskard,
            world,
            left_tool_frame,
            grasp_pose(world, watering_can.root, CAN_T_TOOL),
        )
        hand_over(world, watering_can.root, left_tool_frame)
    time.sleep(WORLD_SYNC_PAUSE)
else:
    watering_can = attach_watering_can(world, left_tool_frame)
    watering_can.initialize_fill_level(
        world=world,
        initial_fill=0.0,
        outflow_rate_constant=CAN_OUTFLOW_RATE_CONSTANT,
        discharge_coefficient=SPOUT_DISCHARGE_COEFFICIENT,
    )
    faucet = build_faucet(world)
    flower_pot = build_flower_pot(world)
    time.sleep(WORLD_SYNC_PAUSE)
    flower_pot.initialize_fill_level(
        world=world, initial_fill=POT_START_FILL, outflow_rate_constant=1.0
    )
    watering_can.receive_outflow_from(source=faucet, world=world)
    flower_pot.receive_outflow_from(source=watering_can, world=world)
    time.sleep(WORLD_SYNC_PAUSE)

# %% stand the can under the faucet and let go of it

execute_pose(
    giskard, world, left_tool_frame, carry_pose(world, 0.9, 0.3, _TABLE_SURFACE_Z + 0.2)
)
outlet_x, outlet_y = FAUCET_X - FAUCET_ARM_LENGTH, FAUCET_Y
execute_pose(
    giskard, world, watering_can.root, standing_pose(world, outlet_x, outlet_y)
)
hand_over(world, watering_can.root, world.root)

# %% grasp the valve lever and let the fill task turn it

lever_grip = world.get_body_by_name("faucet_valve_grip")
execute_pose(
    giskard, world, left_tool_frame, grasp_pose(world, lever_grip, CARRY_ORIENTATION)
)

fill_can = FillByTransferTask(
    receiver=watering_can,
    goal_value=CAN_FILL_GOAL,
    fill_level_tolerance=FILL_TOLERANCE,
    reference_velocity=0.05,
)
hold_lever = CartesianPose(
    name="hold lever",
    root_link=lever_grip,
    tip_link=left_tool_frame,
    goal_pose=Pose(reference_frame=left_tool_frame),
    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
)
msc_fill = MotionStatechart()
msc_fill.add_node(fill_can)
msc_fill.add_node(hold_lever)
msc_fill.add_node(EndMotion.when_true(fill_can))
print("Filling the can at the faucet.")
giskard.execute(msc_fill)
wait_for_synchronized_fill(watering_can, above=FILL_TOLERANCE)
print(f"watering can fill level: {watering_can.fill_level}")

msc_shut = MotionStatechart()
shut_valve = Close(tip_link=left_tool_frame, environment_link=lever_grip)
msc_shut.add_node(shut_valve)
msc_shut.add_node(EndMotion.when_true(shut_valve))
giskard.execute(msc_shut)

# %% pick the can up again and carry it to the flower pot

execute_pose(
    giskard, world, left_tool_frame, grasp_pose(world, watering_can.root, CAN_T_TOOL)
)
hand_over(world, watering_can.root, left_tool_frame)
execute_pose(
    giskard,
    world,
    left_tool_frame,
    carry_pose(
        world,
        POT_X,
        POT_Y + 0.3,
        _TABLE_SURFACE_Z + 0.25,
        tilt=pour_start_tilt(watering_can),
    ),
)

# %% water the plant

water_plant = FillByTransferTask(
    receiver=flower_pot,
    goal_value=POT_FILL_GOAL,
    fill_level_tolerance=FILL_TOLERANCE,
    reference_velocity=0.03,
)
no_spill = KeepProjectileInReceiver(
    receiver=flower_pot,
    source=watering_can,
    weight=DefaultWeights.WEIGHT_MAXIMUM,
    reference_velocity=0.1,
)
keep_above = KeepSourceRimAboveReceiverRim(
    receiver=flower_pot,
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
aiming = Parallel([no_spill, keep_above, keep_plane])
msc_aim = MotionStatechart()
msc_aim.add_node(aiming)
msc_aim.add_node(EndMotion.when_true(aiming))
giskard.execute(msc_aim)

watering = Parallel([water_plant, no_spill, keep_above, keep_plane])
msc_water = MotionStatechart()
msc_water.add_node(watering)
msc_water.add_node(EndMotion.when_true(watering))
print("Watering the plant.")
giskard.execute(msc_water)
wait_for_synchronized_fill(flower_pot, above=POT_START_FILL + FILL_TOLERANCE)

print(f"flower pot fill level: {flower_pot.fill_level}")
print(f"watering can fill level: {watering_can.fill_level}")

# %% return to the carry pose

execute_pose(
    giskard,
    world,
    left_tool_frame,
    carry_pose(world, POT_X, POT_Y + 0.3, _TABLE_SURFACE_Z + 0.25),
)
