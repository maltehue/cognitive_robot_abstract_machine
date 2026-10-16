from enum import StrEnum

import numpy as np
import pytest

from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.robots import root_placement
from semantic_digital_twin.world_description.geometry import Scale
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
    SUPPORT = "support"
    LOWER_LINK = "lower_link"


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


# %% support-body placement


@pytest.fixture
def support_body(standing_robot: MinimalRobot) -> Body:
    """
    A support body below the standing robot's elevated root.
    """
    return BodySpecification.box(
        FrameName.SUPPORT,
        Scale(0.25, 0.25, 0.125),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.75),
    ).spawn(standing_robot._world, parent=standing_robot.root)


def test_support_placement_aligns_selected_geometry_and_preserves_planar_pose(
    standing_robot: MinimalRobot, support_body: Body
) -> None:
    """
    Placement changes height while retaining horizontal position and orientation.
    """
    world = standing_robot._world
    before = standing_robot.root.global_transform.to_np()
    surface_height = 0.25
    root_placement.RobotSupportPlacement(
        standing_robot, [support_body], surface_height
    ).apply()
    after = standing_robot.root.global_transform.to_np()

    bounds = support_body.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    assert float(bounds.min_z) == pytest.approx(surface_height)
    np.testing.assert_allclose(after[:2, 3], before[:2, 3])
    np.testing.assert_allclose(after[:3, :3], before[:3, :3])


def test_support_placement_ignores_lower_links_outside_the_support_selection(
    standing_robot: MinimalRobot, support_body: Body
) -> None:
    """
    An unrelated low link cannot determine the feet's contact height.
    """
    world = standing_robot._world
    lower = BodySpecification.box(
        FrameName.LOWER_LINK,
        Scale(0.1, 0.1, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=-2),
    ).spawn(world, parent=standing_robot.root)
    root_placement.RobotSupportPlacement(standing_robot, [support_body], 0.0).apply()

    bounds = support_body.collision.as_bounding_box_collection_in_frame(
        world.root
    ).bounding_box()
    assert float(bounds.min_z) == pytest.approx(0.0)
    assert float(
        lower.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
    ) < float(bounds.min_z)


def test_support_placement_is_retained_during_planar_navigation(
    standing_robot: MinimalRobot, support_body: Body
) -> None:
    """
    A navigation pose retains the standing height established from support geometry.
    """
    world = standing_robot._world
    root_placement.RobotSupportPlacement(standing_robot, [support_body], 0.0).apply()
    before = standing_robot.root.global_pose.position.z
    standing_robot.set_root_pose(
        standing_robot.resolve_standing_pose(
            Pose.from_xyz_rpy(x=2, y=1, yaw=0.8, reference_frame=world.root)
        )
    )

    assert float(standing_robot.root.global_pose.position.z) == pytest.approx(
        float(before)
    )
    assert float(
        support_body.collision.as_bounding_box_collection_in_frame(world.root)
        .bounding_box()
        .min_z
    ) == pytest.approx(0.0)


@pytest.mark.parametrize("case", [FrameName.WORLD, FrameName.TARGET, FrameName.SUPPORT])
def test_support_placement_rejects_missing_or_invalid_geometry(
    standing_robot: MinimalRobot, support_body: Body, case: FrameName
) -> None:
    """
    A support selection must contain collision bodies belonging to the robot.
    """
    bodies = (
        []
        if case is FrameName.WORLD
        else (
            [standing_robot._world.root]
            if case is FrameName.TARGET
            else [standing_robot.root]
        )
    )
    before = standing_robot.root.global_transform.to_np()

    with pytest.raises(root_placement.InvalidRobotSupportBodiesError):
        root_placement.RobotSupportPlacement(standing_robot, bodies, 0.0).apply()

    np.testing.assert_allclose(standing_robot.root.global_transform.to_np(), before)
