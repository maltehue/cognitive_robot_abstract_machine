"""The liquid-transfer scenario the knowledge-servoing demonstration runs on.

Building the world here rather than in a test fixture lets the same scenario back both the
demonstration script that produces the thesis figures and the tests that assert its behaviour, so
the thing shown and the thing verified cannot drift apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Mesh, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

TABLE_SURFACE_HEIGHT = 0.975
"""Height of the table surface the receiving cup stands on, in metres."""

_JEROEN_CUP_MESH_PATH = str(
    Path(files("semantic_digital_twin")).parent.parent
    / "resources"
    / "stl"
    / "jeroen_cup.stl"
)
"""The cup mesh both containers use, as in the Tracy liquid-transfer demo."""

CARRY_POSE_YAW = 0.1
"""Yaw of the carry pose, giving the pour a defined tilt direction from the start."""

POURING_OUTFLOW_RATE_CONSTANT = 0.08
"""Outflow rate the transfer is coupled at.

Slow enough that the arm can tilt the source back before the receiver overfills; a faster coupling
empties the source in under two seconds, which no controller could track.
"""


@dataclass
class TransferScenario:
    """A world with a robot holding a source cup above a receiving cup, ready to pour."""

    world: World
    """The world everything lives in."""

    source_cup: HasFillLevel
    """The filled container attached to the robot's gripper."""

    receiving_cup: HasFillLevel
    """The empty container on the table, coupled to the source."""

    sensitive_body: Body
    """A body beside the receiving cup that must not be spilled on."""

    balance_body: Body
    """A laboratory balance standing apart from the cups, to be kept clear of."""

    tool_frame: Body
    """The gripper frame the source cup is attached to."""


def _move_gripper_to_carry_pose(world: World, tool_frame: Body) -> None:
    """Drives the gripper to an upright carry pose above the table before the cup is attached.

    The demonstration is about the pour, not about reaching, so the reaching motion is solved once
    here and the reasoner-driven statechart starts from a pose where a transfer is plausible.

    :param world: The world to move the robot in.
    :param tool_frame: The gripper frame to position.
    """
    upright_orientation = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=1.0,
        pos_y=0.3,
        pos_z=TABLE_SURFACE_HEIGHT + 0.15,
        quat_x=0.5,
        quat_y=0.5,
        quat_z=0.5,
        quat_w=0.5,
        reference_frame=world.root,
    )
    carry_pose = (
        upright_orientation
        @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=CARRY_POSE_YAW)
    ).to_pose()
    statechart = MotionStatechart()
    reach = CartesianPose(
        root_link=world.root, tip_link=tool_frame, goal_pose=carry_pose
    )
    statechart.add_node(reach)
    statechart.add_node(EndMotion.when_true(reach))
    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    executor.tick_until_end(timeout=1000)


def _cup_body(name: str) -> Body:
    """Builds a cup body with the Jeroen cup mesh, as in the Tracy liquid-transfer demo.

    :param name: Name of the body.
    :return: The body.
    """
    mesh = Mesh(
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(),
        filename=_JEROEN_CUP_MESH_PATH,
        scale=Scale(1, 1, 1),
    )
    return Body.from_shape_collection(
        shape_collection=ShapeCollection([mesh]), name=PrefixedName(name)
    )


def build_transfer_scenario(
    source_fill_level: float = 1.0,
) -> TransferScenario:
    """Builds the demonstration world with both cups coupled for transfer.

    :param source_fill_level: Initial normalized fill level of the source cup.
    :return: The assembled scenario.
    """
    world = URDFParser.from_file(file_path=Tracy.get_ros_file_path()).parse()
    tracy = Tracy.from_world(world)

    JointState.from_mapping(
        dict(tracy.left_arm.get_joint_state_by_type(StaticJointState.PARK).items())
    ).apply_to(world)
    JointState.from_mapping(
        dict(tracy.right_arm.get_joint_state_by_type(StaticJointState.PARK).items())
    ).apply_to(world)

    tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    _move_gripper_to_carry_pose(world, tool_frame)
    source_cup_body = _cup_body("source_cup")
    with world.modify_world():
        world.add_body(source_cup_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=tool_frame,
                child=source_cup_body,
                name=PrefixedName("l_gripper_T_source_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    roll=-math.pi / 2.0
                ),
            )
        )
    source_cup = HasFillLevel(name=PrefixedName("source_cup"), root=source_cup_body)
    with world.modify_world():
        world.add_semantic_annotation(source_cup)
    source_cup.initialize_fill_level(
        world=world,
        initial_fill=source_fill_level,
        outflow_rate_constant=0.8,
        discharge_coefficient=0.2,
    )

    receiving_cup_body = _cup_body("receiving_cup")
    with world.modify_world():
        world.add_body(receiving_cup_body)
        world.add_connection(
            Connection6DoF.create_with_dofs(
                world,
                world.root,
                receiving_cup_body,
                name=PrefixedName("table_T_receiving_cup"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.099, TABLE_SURFACE_HEIGHT
                ),
            )
        )
    receiving_cup = HasFillLevel(
        name=PrefixedName("receiving_cup"), root=receiving_cup_body
    )
    with world.modify_world():
        world.add_semantic_annotation(receiving_cup)
    receiving_cup.initialize_fill_level(
        world=world, initial_fill=0.0, outflow_rate_constant=1.0
    )
    receiving_cup.receive_outflow_from(source=source_cup, world=world)

    source_equation = source_cup.fill_equation
    receiving_cup.recouple_outflow_from(
        source=source_cup,
        world=world,
        fill_equation=ArticulatedPouringEquation(
            container_height=source_equation.container_height,
            container_width=source_equation.container_width,
            outflow_rate_constant=POURING_OUTFLOW_RATE_CONSTANT,
            discharge_coefficient=source_equation.discharge_coefficient,
        ),
    )

    sensitive_body = Body.from_shape_collection(
        shape_collection=ShapeCollection([Box(scale=Scale(0.25, 0.2, 0.02))]),
        name=PrefixedName("laptop"),
    )
    with world.modify_world():
        world.add_body(sensitive_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=receiving_cup_body,
                child=sensitive_body,
                name=PrefixedName("receiving_cup_T_laptop"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=-0.25
                ),
            )
        )

    balance_body = Body.from_shape_collection(
        shape_collection=ShapeCollection([Box(scale=Scale(0.2, 0.2, 0.05))]),
        name=PrefixedName("balance"),
    )
    with world.modify_world():
        world.add_body(balance_body)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=balance_body,
                name=PrefixedName("table_T_balance"),
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    1.0, 0.5, TABLE_SURFACE_HEIGHT
                ),
            )
        )

    return TransferScenario(
        world=world,
        source_cup=source_cup,
        receiving_cup=receiving_cup,
        sensitive_body=sensitive_body,
        balance_body=balance_body,
        tool_frame=tool_frame,
    )


def pouring_plane_stabilization(scenario: TransferScenario) -> AlignPlanes:
    """The wrist stabilization the Tracy liquid-transfer demo runs throughout the motion.

    Keeps the tool frame's z-axis along the world's x-axis, so the only rotational freedom left is
    the tilt the pour needs. Without it the wrist is unconstrained about the pour axis and the
    optimizer wanders it, which shows up as jitter.

    This is a statement about the embodiment and the scene, not about any theory, so the caller
    adds it to the chart the way it adds termination.

    :param scenario: The scenario whose tool frame is stabilized.
    :return: The always-active alignment task.
    """
    return AlignPlanes(
        name="pouring_plane_stabilization",
        root_link=scenario.world.root,
        tip_link=scenario.tool_frame,
        goal_normal=Vector3.X(reference_frame=scenario.world.root),
        tip_normal=Vector3.Z(reference_frame=scenario.tool_frame),
    )
