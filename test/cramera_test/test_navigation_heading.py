"""
Travel-facing orientation during continuous base navigation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.locations.navigation import NavigationPath, NavigationPathUnavailable
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.motions.navigation import MoveMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World

from .test_navigation_motion import NavigationTrajectory

# %% observed heading


@dataclass
class TravelHeadingTrajectory(NavigationTrajectory):
    """
    Observe base yaw together with the continuously driven positions.
    """

    headings: list[float] = field(default_factory=list)
    """
    World-frame yaw at each controller tick, in radians.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """
        Record position, avoidance and heading after a controller tick.

        :param statechart: Controller that advanced the robot.
        """
        super().on_motion_tick(statechart)
        rotation = self.robot.root.global_pose.to_np()
        self.headings.append(float(np.arctan2(rotation[1, 0], rotation[0, 0])))


def test_default_navigation_turns_while_driving(cylinder_bot_world: World) -> None:
    """
    Normal navigation faces lateral travel and restores the requested final yaw.

    :param cylinder_bot_world: Existing omnidirectional mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(0, -2, reference_frame=world.root)
    trajectory = TravelHeadingTrajectory(robot=robot)
    plan = execute_single(
        NavigateAction(target), context=Context(world=world, robot=robot, _debug=False)
    ).plan
    plan.node_callbacks.append(trajectory)
    with simulated_robot:
        plan.perform()
    positions = np.asarray(trajectory.positions)
    headings = np.unwrap(trajectory.headings)
    moving = (positions[:, 1] < -0.2) & (positions[:, 1] > -1.8)
    assert np.any(headings[moving] < -np.pi / 4)
    assert np.max(np.abs(np.diff(headings))) < 0.05
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    assert trajectory.avoidance_counts == {1}
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )


def test_travel_facing_can_be_disabled(cylinder_bot_world: World) -> None:
    """
    A caller can explicitly retain sideways omnidirectional driving.

    :param cylinder_bot_world: Existing omnidirectional mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    trajectory = TravelHeadingTrajectory(robot=robot)
    target = Pose.from_xyz_rpy(0, -0.5, reference_frame=world.root)
    plan = execute_single(
        NavigateAction(target, face_travel_direction=False),
        context=Context(world=world, robot=robot, _debug=False),
    ).plan
    plan.node_callbacks.append(trajectory)
    with simulated_robot:
        plan.perform()
    np.testing.assert_allclose(trajectory.headings, 0, atol=0.005)
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )


@pytest.mark.parametrize("yaw", [0.0, 0.4])
def test_travel_facing_without_translation(
    cylinder_bot_world: World, yaw: float
) -> None:
    """
    No-op and pure-turn requests preserve their requested final pose.

    :param cylinder_bot_world: Existing omnidirectional mobile robot.
    :param yaw: Requested base rotation without translation.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(yaw=yaw, reference_frame=world.root)
    plan = execute_single(
        MoveMotion(target), context=Context(world=world, robot=robot, _debug=False)
    ).plan
    with simulated_robot:
        plan.perform()
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )


# %% turning clearance


def test_travel_facing_retains_short_route_segments(cylinder_bot_world: World) -> None:
    """
    Short obstacle-corner segments remain part of the verified route.

    :param cylinder_bot_world: Existing omnidirectional mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    corner = Pose.from_xyz_rpy(0.004, reference_frame=world.root)
    target = Pose.from_xyz_rpy(0.004, 1, reference_frame=world.root)
    path = NavigationPath(world, robot, target, face_travel_direction=True)
    poses = path.face_waypoints(robot.root.global_pose, [corner, target])
    np.testing.assert_allclose(
        poses[0].to_position().to_np(), corner.to_position().to_np()
    )


def test_travel_facing_preserves_a_tilted_base_plane(cylinder_bot_world: World) -> None:
    """
    Travel heading does not introduce roll or pitch changes to a tilted base.

    :param cylinder_bot_world: Existing omnidirectional mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    start = Pose.from_xyz_rpy(roll=0.1, reference_frame=world.root)
    target = Pose.from_xyz_rpy(0, -1, roll=0.1, reference_frame=world.root)
    path = NavigationPath(world, robot, target, face_travel_direction=True)
    poses = path.face_waypoints(start, [target])
    for pose in poses:
        np.testing.assert_allclose(pose.to_np()[:3, 2], start.to_np()[:3, 2])


def test_route_faces_travel_before_final_orientation(cylinder_bot_world: World) -> None:
    """
    Arrival heading follows the route before a separate final orientation goal.

    :param cylinder_bot_world: Existing omnidirectional mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(0, -2, reference_frame=world.root)
    poses = NavigationPath(world, robot, target, face_travel_direction=True).plan()
    expected_arrival = Pose.from_xyz_rpy(
        0, -2, yaw=-np.pi / 2, reference_frame=world.root
    )
    assert len(poses) == 2
    np.testing.assert_allclose(poses[0].to_np(), expected_arrival.to_np(), atol=1e-10)
    np.testing.assert_allclose(poses[-1].to_np(), target.to_np(), atol=1e-10)


@pytest.mark.parametrize(
    "keep_joint_states, yaw", [(True, 0.0), (False, 0.0), (True, 0.4)]
)
def test_narrow_route_only_falls_back_with_a_fixed_footprint(
    pr2_apartment_state_reset: World, keep_joint_states: bool, yaw: float
) -> None:
    """
    Tight-space sideways driving requires held joints and unchanged base yaw.

    :param pr2_apartment_state_reset: Annotated apartment with a narrow start area.
    :param keep_joint_states: Whether the arm posture is held throughout navigation.
    :param yaw: Requested final base orientation.
    """
    world = pr2_apartment_state_reset
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    target = Pose.from_xyz_rpy(1.1, 2, yaw=yaw, reference_frame=world.root)
    path = NavigationPath(
        world,
        robot,
        target,
        clearance=0.05,
        keep_joint_states=keep_joint_states,
        face_travel_direction=True,
    )
    if not keep_joint_states or yaw:
        with pytest.raises(NavigationPathUnavailable):
            path.plan()
        return
    poses = path.plan()
    for pose in poses:
        np.testing.assert_allclose(
            pose.to_np()[:3, :3], robot.root.global_pose.to_np()[:3, :3]
        )
    np.testing.assert_allclose(poses[-1].to_np(), target.to_np())
