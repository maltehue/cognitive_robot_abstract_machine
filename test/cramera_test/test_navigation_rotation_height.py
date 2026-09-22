"""
Rotational clearance follows the height of each robot part and attachment.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.locations.navigation import (
    NavigationFailureReason,
    NavigationPath,
    NavigationPathUnavailable,
)
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Scale

# %% elevated robot geometry


@pytest.fixture()
def elevated_robot_world(cylinder_bot_world: World) -> World:
    """
    Attach a wide upper body above the narrow mobile base.

    :param cylinder_bot_world: Existing annotated mobile robot and obstacles.
    :return: World containing the robot with its elevated collision geometry.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    obstacle = world.get_body_by_name("environment2")
    with world.modify_world():
        world.remove_connection(obstacle.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=obstacle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=5
                ),
            )
        )
    upper_body = BodySpecification.box(
        "upper_body",
        Scale(0.2, 1.2, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1),
    ).spawn(world)
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            branch_root=upper_body, new_parent=robot.root
        )
    return world


@pytest.mark.parametrize("obstacle_height", [0.0, 1.0])
def test_heading_change_respects_each_parts_height(
    elevated_robot_world: World, obstacle_height: float
) -> None:
    """
    A wide upper body can rotate above low barriers but not through high ones.

    :param elevated_robot_world: Mobile robot with narrow lower and wide upper parts.
    :param obstacle_height: Height of the corridor barriers relative to the base.
    """
    world = elevated_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    for name, position in (("upper_barrier", 2.3), ("lower_barrier", -2.3)):
        BodySpecification.box(
            name,
            Scale(0.2, 4, 0.2),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-1, y=position, z=obstacle_height
            ),
        ).spawn(world)
    target = Pose.from_xyz_rpy(-2, yaw=0.4, reference_frame=world.root)
    path = NavigationPath(world, robot, target, keep_joint_states=True)
    if obstacle_height:
        with pytest.raises(NavigationPathUnavailable) as failure:
            path.plan()
        assert failure.value.reason is NavigationFailureReason.DISCONNECTED
        return
    poses = path.plan()
    assert len(poses) == 1
    np.testing.assert_allclose(poses[0].to_np(), target.to_np())


def test_lower_attachment_closes_rotational_clearance(
    elevated_robot_world: World,
) -> None:
    """
    A carried object contributes its own height-specific turning radius.

    :param elevated_robot_world: Robot whose upper body clears the low barriers.
    """
    world = elevated_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    for name, position in (("upper_barrier", 2.3), ("lower_barrier", -2.3)):
        BodySpecification.box(
            name,
            Scale(0.2, 4, 0.2),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-1, y=position
            ),
        ).spawn(world)
    target = Pose.from_xyz_rpy(-2, yaw=0.4, reference_frame=world.root)
    assert len(NavigationPath(world, robot, target).plan()) == 1
    payload = BodySpecification.box("carried_payload", Scale(0.1, 0.5, 0.1)).spawn(
        world
    )
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            branch_root=payload, new_parent=robot.root
        )
    assert payload in robot.bodies_with_collision
    with pytest.raises(NavigationPathUnavailable) as failure:
        NavigationPath(world, robot, target).plan()
    assert failure.value.reason is NavigationFailureReason.DISCONNECTED
