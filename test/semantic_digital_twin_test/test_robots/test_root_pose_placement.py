from enum import StrEnum

import numpy as np
import pytest

from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body


def _pr2_behind_an_odom(
    world_T_odom: HomogeneousTransformationMatrix,
) -> tuple[World, AbstractRobot]:
    """
    A world holding nothing but a PR2, reached through an odom frame at
    ``world_T_odom``.

    The specification path is what puts the odom between the world root and the robot,
    so the placement is exercised against the same chain a specification builds.
    """
    world = World.create_with_root_body("root")
    try:
        robot = RobotSpecification(
            semantic_annotation_type=PR2, world_T_odom=world_T_odom
        ).spawn(world)
    except ParsingError as error:
        pytest.skip(f"PR2 URDF not available: {error}")
    return world, robot


# %% placing the root through a displaced odom

# The drive is an OmniDrive, which represents x, y and yaw only, so the odom is
# displaced within that plane. A z or roll offset would make these tests assert the
# drive's limits instead of the frame conversion.
_DISPLACED_ODOM = HomogeneousTransformationMatrix.from_xyz_rpy(
    0.5, 0.5, 0, yaw=np.pi / 2
)


def test_root_reaches_a_world_pose_through_a_displaced_odom():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


def test_root_reaches_a_world_pose_through_an_undisplaced_odom():
    world, robot = _pr2_behind_an_odom(HomogeneousTransformationMatrix())
    target = Pose.from_xyz_rpy(1.3, 2.0, 0.0, yaw=0.25, reference_frame=world.root)

    robot.set_root_pose(target)

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=1e-9
    )


# %% pose already expressed in the root connection's parent frame


def test_pose_in_the_root_connection_parent_frame_is_applied_unchanged():
    world, robot = _pr2_behind_an_odom(_DISPLACED_ODOM)
    connection = robot.root.parent_connection
    target = Pose.from_xyz_rpy(
        1.3, 2.0, 0.0, yaw=0.25, reference_frame=connection.parent
    )

    robot.set_root_pose(target)

    np.testing.assert_allclose(connection.origin.to_np(), target.to_np(), atol=1e-9)


# %% standing-height fixture


class FrameName(StrEnum):
    """
    Frames separating the world, drive plane and requested pose.
    """

    WORLD = "world"
    TARGET = "target"


@pytest.fixture
def standing_robot(world_with_two_bodies: tuple[World, Body, Body]) -> MinimalRobot:
    """
    A synthetic planar robot whose root stands above a displaced drive frame.
    """
    world, parent, child = world_with_two_bodies
    world_frame = Body(name=PrefixedName(FrameName.WORLD))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world_frame,
                child=parent,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, z=0.4, yaw=0.3
                ),
            )
        )
        world.add_connection(
            OmniDrive.create_with_dofs(
                world=world,
                parent=parent,
                child=child,
                connection_T_child_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=0.9
                ),
            )
        )
    return MinimalRobot.from_branch_in_world(child)


# %% projecting navigation targets into the drive plane


@pytest.mark.parametrize("pitch", [0.0, 0.4])
def test_standing_pose_keeps_drive_height_in_other_frames(
    standing_robot: MinimalRobot, pitch: float
) -> None:
    """
    A target keeps its planar position and orientation at the root's drive height.
    """
    world = standing_robot._world
    drive_frame = standing_robot.root.parent_kinematic_structure_entity
    target_frame = Body(name=PrefixedName(FrameName.TARGET))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=target_frame,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.2, z=0.7, pitch=pitch
                ),
            )
        )
    target = Pose.from_xyz_rpy(1.3, 2.0, yaw=0.2, reference_frame=target_frame)
    drive_target = world.transform(target, drive_frame)
    drive_root = world.compute_forward_kinematics(drive_frame, standing_robot.root)
    expected = HomogeneousTransformationMatrix.from_point_rotation_matrix(
        Pose.from_xyz_rpy(
            float(drive_target.x),
            float(drive_target.y),
            float(drive_root.z),
            reference_frame=drive_frame,
        ).position,
        drive_target.rotation_matrix,
        reference_frame=drive_frame,
    ).pose

    result = standing_robot.resolve_standing_pose(target)

    np.testing.assert_allclose(
        world.transform(result, drive_frame).to_np(), expected.to_np(), atol=1e-9
    )
    assert result.reference_frame is target.reference_frame


def test_standing_pose_without_parent_preserves_target(mini_world: World) -> None:
    """
    A root without a parent frame imposes no standing height.
    """
    robot = MinimalRobot.from_branch_in_world(mini_world.root)
    target = Pose.from_xyz_rpy(z=0.7, reference_frame=mini_world.root)

    assert robot.resolve_standing_pose(target) is target
