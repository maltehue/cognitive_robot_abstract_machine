"""
Whole-robot navigation through bounded, height-aware free space.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.locations.navigation import (
    NavigationPath,
    NavigationPathUnavailable,
    NavigationFailureReason,
)
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.semantic_annotations.semantic_annotations import Agent

# %% route geometry


def test_route_preserves_base_height_and_goal_orientation(
    cylinder_bot_world: World,
) -> None:
    """
    Projected planning points become planar base poses with the requested yaw.

    :param cylinder_bot_world: Annotated mobile robot and existing obstacles.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(-2, 0, 0, yaw=0.4, reference_frame=world.root)
    BodySpecification.box(
        "barrier",
        Scale(0.4, 0.8, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-1),
    ).spawn(world)
    path = NavigationPath(world=world, robot=robot, target=target)
    poses = path.plan()
    assert len(poses) > 1
    assert all(pose.z == robot.root.global_pose.z for pose in poses)
    np.testing.assert_allclose(poses[-1].to_np(), target.to_np())


@pytest.mark.parametrize("height", [-2.0, 2.0])
def test_obstacles_outside_robot_height_allow_a_direct_route(
    cylinder_bot_world: World, height: float
) -> None:
    """
    Geometry below or above the robot does not block base navigation.

    :param cylinder_bot_world: Mobile robot whose collision height is finite.
    :param height: Vertical center of a broad obstacle.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    BodySpecification.box(
        "outside_height",
        Scale(10, 10, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=height),
    ).spawn(world)
    target = Pose.from_xyz_rpy(-1, reference_frame=world.root)
    poses = NavigationPath(world, robot, target).plan()
    assert len(poses) == 1
    np.testing.assert_allclose(poses[0].to_np(), target.to_np())


def test_blocked_route_fails_without_moving_the_robot(
    cylinder_bot_world: World,
) -> None:
    """
    A wall spanning the bounded map produces a plan failure.

    :param cylinder_bot_world: Annotated robot used to test a disconnected route.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    start = robot.root.global_pose.to_np().copy()
    BodySpecification.box(
        "wall",
        Scale(0.4, 10, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-1),
    ).spawn(world)
    path = NavigationPath(
        world, robot, Pose.from_xyz_rpy(-2, reference_frame=world.root)
    )
    with pytest.raises(NavigationPathUnavailable):
        path.plan()
    np.testing.assert_array_equal(robot.root.global_pose.to_np(), start)


def test_relative_goal_is_resolved_in_the_world(cylinder_bot_world: World) -> None:
    """
    A goal expressed at the robot base retains its world-frame meaning.

    :param cylinder_bot_world: Robot used as the requested goal's reference frame.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(-1, reference_frame=robot.root)
    poses = NavigationPath(world, robot, target).plan()
    assert poses[-1].reference_frame is world.root
    np.testing.assert_allclose(
        poses[-1].to_np(), world.transform(target, world.root).to_np()
    )


@pytest.mark.parametrize("protrusion", [0.0, 0.005])
@pytest.mark.parametrize("keep_joint_states", [False, True])
def test_support_contact_does_not_hide_low_obstacles(
    cylinder_bot_world: World, protrusion: float, keep_joint_states: bool
) -> None:
    """
    A touching floor is free, but geometry entering the robot's height is occupied.

    :param cylinder_bot_world: Cylinder with a known lower collision boundary.
    :param protrusion: Obstacle height above the robot's lowest point.
    :param keep_joint_states: Whether fixed-orientation footprint planning is enabled.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    lower = min(
        box.min_z for box in robot.as_bounding_box_collection_in_frame(world.root)
    )
    BodySpecification.box(
        "support",
        Scale(10, 10, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            z=lower - 0.05 + protrusion
        ),
    ).spawn(world)
    path = NavigationPath(
        world,
        robot,
        Pose.from_xyz_rpy(-1, reference_frame=world.root),
        keep_joint_states=keep_joint_states,
    )
    if protrusion:
        with pytest.raises(NavigationPathUnavailable):
            path.plan()
    else:
        assert len(path.plan()) == 1


def test_nonplanar_orientation_is_rejected(cylinder_bot_world: World) -> None:
    """
    A mobile base cannot satisfy a roll goal by driving on a plane.

    :param cylinder_bot_world: Existing mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    path = NavigationPath(
        world, robot, Pose.from_xyz_rpy(-1, roll=0.2, reference_frame=world.root)
    )
    with pytest.raises(NavigationPathUnavailable):
        path.plan()


def test_other_agents_remain_obstacles(cylinder_bot_world: World) -> None:
    """
    Only the controlled robot is excluded from the obstacle map.

    :param cylinder_bot_world: Robot used to navigate past another annotated agent.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    obstacle = BodySpecification.box(
        "other_agent",
        Scale(0.4, 0.8, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-1),
    ).spawn(world)
    with world.modify_world():
        world.add_semantic_annotation(Agent(root=obstacle))
    path = NavigationPath(
        world, robot, Pose.from_xyz_rpy(-2, reference_frame=world.root)
    )
    assert len(path.plan()) > 1


@pytest.mark.parametrize("keep_joint_states", [False, True])
def test_attached_payload_closes_a_previously_free_corridor(
    cylinder_bot_world: World,
    keep_joint_states: bool,
) -> None:
    """
    Include carried geometry when determining route clearance.

    :param cylinder_bot_world: Existing mobile robot with collision geometry.
    :param keep_joint_states: Whether fixed-orientation footprint planning is enabled.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    for name, y in (("upper_wall", 2.21), ("lower_wall", -2.21)):
        BodySpecification.box(
            name,
            Scale(0.2, 4.0, 1.0),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=-1.0, y=y),
        ).spawn(world)
    target = Pose.from_xyz_rpy(-2.0, reference_frame=world.root)
    assert (
        len(
            NavigationPath(
                world, robot, target, keep_joint_states=keep_joint_states
            ).plan()
        )
        == 1
    )
    payload = BodySpecification.box(
        "carried_payload",
        Scale(0.12, 0.22, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.35),
    ).spawn(world)
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            branch_root=payload, new_parent=robot.root
        )
    assert payload in robot.bodies_with_collision
    with pytest.raises(NavigationPathUnavailable) as failure:
        NavigationPath(world, robot, target, keep_joint_states=keep_joint_states).plan()
    assert failure.value.reason is NavigationFailureReason.DISCONNECTED


def test_translation_route_keeps_nonzero_heading(cylinder_bot_world: World) -> None:
    """
    A translated, rotated base retains its orientation around an obstacle.

    :param cylinder_bot_world: Omnidirectional robot with mutable initial pose.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    robot.drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1, 2, yaw=0.7)
    target = Pose.from_xyz_rpy(-1, 2, yaw=0.7, reference_frame=world.root)
    BodySpecification.box(
        "rotated_heading_barrier",
        Scale(0.4, 0.8, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(0, 2),
    ).spawn(world)
    poses = NavigationPath(world, robot, target, keep_joint_states=True).plan()
    assert len(poses) > 1
    for pose in poses:
        np.testing.assert_allclose(pose.to_np()[:3, :3], target.to_np()[:3, :3])
    np.testing.assert_allclose(poses[-1].to_np(), target.to_np())
