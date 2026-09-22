"""
Fixed-heading departures and approaches join routes with room to turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.locations.navigation import NavigationPath, NavigationPathUnavailable
from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions.navigation import MoveMotion
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from .test_navigation_rotation_height import elevated_robot_world
from .test_navigation_heading import TravelHeadingTrajectory

# %% narrow endpoint geometry


@pytest.fixture()
def close_wall_robot_world(elevated_robot_world: World) -> World:
    """
    Place a long robot beside a wall where it can translate but cannot turn.

    :param elevated_robot_world: Existing mobile robot with distant fixture obstacles.
    :return: World with a narrow fixed-heading departure and approach.
    """
    world = elevated_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    body = BodySpecification.box("long_lower_body", Scale(0.2, 1.2, 0.1)).spawn(world)
    with world.modify_world():
        world.move_branch_with_fixed_connection(branch_root=body, new_parent=robot.root)
    BodySpecification.box(
        "approach_wall",
        Scale(0.2, 4, 0.5),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.4),
    ).spawn(world)
    return world


@pytest.mark.parametrize("departure", [True, False])
def test_close_endpoint_connects_without_rotating_beside_wall(
    close_wall_robot_world: World, departure: bool
) -> None:
    """
    Heading changes occur after departure or before the final narrow approach.

    :param close_wall_robot_world: Long mobile robot parked beside a wall.
    :param departure: Whether the narrow endpoint is the start or the destination.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(-2, yaw=np.pi / 2, reference_frame=world.root)
    if not departure:
        robot.drive.origin = target.to_homogeneous_matrix()
        target = Pose.from_xyz_rpy(reference_frame=world.root)
    start = robot.root.global_pose.to_np().copy()
    path = NavigationPath(
        world, robot, target, keep_joint_states=True, face_travel_direction=True
    )
    poses = path.plan()
    assert len(poses) >= 3
    endpoint_pose = poses[0] if departure else poses[-2]
    np.testing.assert_allclose(
        endpoint_pose.to_np()[:3, :3],
        start[:3, :3] if departure else target.to_np()[:3, :3],
    )
    np.testing.assert_allclose(poses[-1].to_np(), target.to_np())
    np.testing.assert_array_equal(robot.root.global_pose.to_np(), start)


def test_endpoint_connector_rejects_an_actually_occupied_pose(
    close_wall_robot_world: World,
) -> None:
    """
    Fixed-heading approaches still reject a destination intersecting the wall.

    :param close_wall_robot_world: Long mobile robot near a solid wall.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    robot.drive.origin = HomogeneousTransformationMatrix.from_xyz_rpy(-2, yaw=np.pi / 2)
    target = Pose.from_xyz_rpy(0.35, reference_frame=world.root)
    with pytest.raises(NavigationPathUnavailable):
        NavigationPath(world, robot, target, keep_joint_states=True).plan()


def test_tight_pure_turn_departs_and_returns_through_the_same_portal(
    close_wall_robot_world: World,
) -> None:
    """
    Equal rotation-graph endpoints still include a complete heading-change stage.

    :param close_wall_robot_world: Long robot that cannot turn beside its nearby wall.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(yaw=np.pi, reference_frame=world.root)
    route = NavigationPath(
        world, robot, target, keep_joint_states=True, face_travel_direction=True
    ).plan_route()
    assert min(float(pose.x) for pose in route.poses) < -0.4
    assert len(route.stages) >= 3
    np.testing.assert_allclose(route.poses[-1].to_np(), target.to_np())
    assert route.stages[-2].complete_orientation
    np.testing.assert_allclose(route.poses[-2].to_np()[:3, :3], target.to_np()[:3, :3])


# %% controller orientation boundaries


@pytest.mark.parametrize("departure", [True, False])
def test_controller_holds_heading_near_the_wall(
    close_wall_robot_world: World, departure: bool
) -> None:
    """
    Native execution waits for the safe heading before entering a narrow approach.

    :param close_wall_robot_world: Long mobile robot beside the solid wall.
    :param departure: Whether the narrow endpoint starts or ends the trajectory.
    """
    world = close_wall_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(-2, yaw=np.pi / 2, reference_frame=world.root)
    if not departure:
        robot.drive.origin = target.to_homogeneous_matrix()
        target = Pose.from_xyz_rpy(reference_frame=world.root)
    trajectory = TravelHeadingTrajectory(robot=robot)
    plan = execute_single(
        MoveMotion(target, keep_joint_states=True),
        context=Context(world=world, robot=robot, _debug=False),
    ).plan
    plan.node_callbacks.append(trajectory)
    with simulated_robot:
        plan.perform()
    positions = np.asarray(trajectory.positions)
    headings = np.asarray(trajectory.headings)
    near_wall = positions[:, 0] > -0.15
    assert np.any(near_wall)
    np.testing.assert_allclose(headings[near_wall], 0, atol=0.01)
    assert trajectory.avoidance_counts == {1}
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )
