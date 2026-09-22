"""
Fixed-heading transit can join separate regions with room to turn.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.locations.navigation import NavigationPath, NavigationPathUnavailable
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from .test_navigation_heading import TravelHeadingTrajectory
from .test_navigation_rotation_height import elevated_robot_world


# %% separated turning regions
@pytest.fixture()
def corridor_robot_world(elevated_robot_world: World) -> World:
    """
    Add a passage which admits the long robot only after a quarter turn.

    :param elevated_robot_world: Existing mobile robot with a long upper body.
    :return: Two rooms joined by a 0.8 m wide and 2 m long passage.
    """
    world = elevated_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    with world.modify_world():
        for name in ("environment", "environment2"):
            world.remove_branch_from_world(world.get_body_by_name(name))
    for name, y in (("north_wall", 2.4), ("south_wall", -2.4)):
        BodySpecification.box(
            name,
            Scale(2, 4, 3),
            parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(2, y, 1.2),
        ).spawn(world)
    payload = BodySpecification.box(
        "carried_payload",
        Scale(0.2, 1.4, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1.2),
    ).spawn(world)
    with world.modify_world():
        world.move_branch_with_fixed_connection(payload, robot.root)
    return world


def test_transit_turns_in_free_rooms_before_entering_the_passage(
    corridor_robot_world: World,
) -> None:
    """
    Hold the feasible corridor heading and restore the exact requested yaw.

    :param corridor_robot_world: Long robot and payload between separated turning
        regions.
    """
    world = corridor_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    start = robot.root.global_pose.to_np().copy()
    target = Pose.from_xyz_rpy(4, reference_frame=world.root)
    route = NavigationPath(
        world,
        robot,
        target,
        keep_joint_states=True,
        face_travel_direction=True,
    ).plan_route()
    transit = [
        stage for stage in route.stages if abs(float(stage.target.to_np()[0, 0])) < 0.01
    ]
    assert len(transit) >= 2
    assert float(transit[0].target.x) < 0.25
    assert float(transit[-1].target.x) > 3.75
    assert all(stage.complete_orientation for stage in route.stages)
    np.testing.assert_array_equal(robot.root.global_pose.to_np(), start)
    np.testing.assert_allclose(route.poses[-1].to_np(), target.to_np())


def test_transit_rejects_an_attachment_too_wide_at_every_heading(
    corridor_robot_world: World,
) -> None:
    """
    A payload which cannot fit must not disappear from the footprint.

    :param corridor_robot_world: Existing orientation-constrained corridor.
    """
    world = corridor_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    payload = BodySpecification.box(
        "oversized_payload",
        Scale(1, 1, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1.4),
    ).spawn(world)
    with world.modify_world():
        world.move_branch_with_fixed_connection(payload, robot.root)
    with pytest.raises(NavigationPathUnavailable):
        NavigationPath(
            world,
            robot,
            Pose.from_xyz_rpy(4, reference_frame=world.root),
            keep_joint_states=True,
            face_travel_direction=True,
        ).plan()


def test_native_transit_holds_the_corridor_heading_with_avoidance(
    corridor_robot_world: World,
) -> None:
    """
    Controller ticks preserve the transit heading until the robot exits.

    :param corridor_robot_world: Mobile robot carrying an elongated payload.
    """
    world = corridor_robot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(4, reference_frame=world.root)
    trajectory = TravelHeadingTrajectory(robot)
    plan = execute_single(
        NavigateAction(target, keep_joint_states=True),
        context=Context(world=world, robot=robot, _debug=False),
    ).plan
    plan.node_callbacks.append(trajectory)
    with simulated_robot_advanced:
        plan.perform()
    positions = np.asarray(trajectory.positions)
    headings = np.asarray(trajectory.headings)
    inside = (positions[:, 0] > 0.8) & (positions[:, 0] < 3.2)
    assert np.any(inside)
    np.testing.assert_allclose(np.abs(headings[inside]), np.pi / 2, atol=0.005)
    assert trajectory.avoidance_counts == {1}
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )
