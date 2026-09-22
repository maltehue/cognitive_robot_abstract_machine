"""
Execute one robot's native motions while another remains in the shared world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.locations.pose_validator import AreReachableBy
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.robot_plans.motions.navigation import MoveMotion
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World


# %% shared world
@pytest.fixture
def two_robot_world(cylinder_bot_world: World, pr2_world_copy: World) -> World:
    """
    Keep both fixture robots independently controlled in one scene.

    :param cylinder_bot_world: Existing small mobile robot and obstacle fixture.
    :param pr2_world_copy: Independent native PR2 model with its drive and annotation.
    :return: Shared world with separated robot instances.
    """
    cylinder_bot_world.merge_world_at_pose(
        pr2_world_copy, HomogeneousTransformationMatrix.from_xyz_rpy(3, 0, 0)
    )
    return cylinder_bot_world


@dataclass
class RobotIsolationTrajectory(PlanCallback):
    """
    Observe both robot motion and the collision goal's ownership on every tick.
    """

    active_robot: AbstractRobot
    """Robot assigned the executed plan."""

    idle_robot: AbstractRobot
    """
    Robot whose joints and root must remain unchanged.
    """

    active_positions: list[np.ndarray] = field(default_factory=list)
    """
    Intermediate base positions proving continuous motion.
    """

    idle_transforms: list[np.ndarray] = field(default_factory=list)
    """
    All idle body transforms at each controller tick.
    """

    avoidance_robots: set[AbstractRobot] = field(default_factory=set)
    """
    Robots explicitly assigned to the native collision goals.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """
        Capture authoritative motion and ownership after a controller tick.

        :param statechart: Chart currently controlling the shared world.
        """
        self.active_positions.append(self.active_robot.root.global_pose.to_np()[:3, 3])
        self.idle_transforms.append(
            np.asarray(
                [
                    self.idle_robot._world.compute_forward_kinematics_np(
                        self.idle_robot._world.root, body
                    )
                    for body in self.idle_robot.bodies
                ]
            )
        )
        self.avoidance_robots.update(
            goal.robot
            for goal in statechart.get_nodes_by_type(ExternalCollisionAvoidance)
        )


# %% active robot execution
@pytest.mark.parametrize("select_pr2", [False, True])
def test_navigation_controls_only_the_selected_robot(
    two_robot_world: World, select_pr2: bool
) -> None:
    """
    Selection controls either robot while all bodies of the other stay fixed.

    :param two_robot_world: Shared annotated native world.
    :param select_pr2: Select the second robot rather than the first annotation.
    """
    robots = two_robot_world.get_semantic_annotations_by_type(AbstractRobot)
    robot = next(item for item in robots if isinstance(item, PR2) is select_pr2)
    idle = next(item for item in robots if item is not robot)
    initial_idle = np.asarray([body.global_pose.to_np() for body in idle.bodies])
    target = Pose.from_xyz_rpy(
        float(robot.root.global_pose.x) + (0.3 if select_pr2 else -0.3),
        float(robot.root.global_pose.y),
        reference_frame=two_robot_world.root,
    )
    plan = execute_single(
        MoveMotion(target),
        context=Context(world=two_robot_world, robot=robot, _debug=False),
    ).plan
    trajectory = RobotIsolationTrajectory(robot, idle)
    plan.node_callbacks.append(trajectory)
    with simulated_robot_advanced:
        plan.perform()
    positions = np.asarray(trajectory.active_positions)
    assert len(positions) > 2
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )
    for transforms in trajectory.idle_transforms:
        np.testing.assert_allclose(transforms, initial_idle, atol=1e-12)
    assert trajectory.avoidance_robots == {robot}


def test_reachability_assigns_collision_avoidance_to_its_robot(
    two_robot_world: World,
) -> None:
    """
    Candidate validation retains the selected robot in a multi-robot chart.

    :param two_robot_world: Shared annotated native world.
    """
    [robot] = two_robot_world.get_semantic_annotations_by_type(PR2)
    tip = robot.left_arm.end_effector.tool_frame
    validator = AreReachableBy(
        [tip.global_pose], tip, context=Context(world=two_robot_world, robot=robot)
    )
    with simulated_robot_advanced:
        chart = validator.create_msc()
        [avoidance] = chart.get_nodes_by_type(ExternalCollisionAvoidance)
        assert avoidance.robot is robot
        assert validator()


def test_navigation_routes_around_the_idle_robot(
    cylinder_bot_world: World, pr2_world_copy: World
) -> None:
    """
    An unselected robot remains stationary and obstructs the driven route.

    :param cylinder_bot_world: Existing small mobile robot and obstacle fixture.
    :param pr2_world_copy: Native second robot placed across the direct route.
    """
    cylinder_bot_world.merge_world_at_pose(
        pr2_world_copy, HomogeneousTransformationMatrix.from_xyz_rpy(-1.5, 0, 0)
    )
    robots = cylinder_bot_world.get_semantic_annotations_by_type(AbstractRobot)
    robot = next(item for item in robots if not isinstance(item, PR2))
    idle = next(item for item in robots if isinstance(item, PR2))
    initial_idle = np.asarray([body.global_pose.to_np() for body in idle.bodies])
    target = Pose.from_xyz_rpy(-3, reference_frame=cylinder_bot_world.root)
    plan = execute_single(
        MoveMotion(target),
        context=Context(world=cylinder_bot_world, robot=robot, _debug=False),
    ).plan
    trajectory = RobotIsolationTrajectory(robot, idle)
    plan.node_callbacks.append(trajectory)
    with simulated_robot_advanced:
        plan.perform()
    positions = np.asarray(trajectory.active_positions)
    assert np.max(np.abs(positions[:, 1])) > 0.4
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )
    for transforms in trajectory.idle_transforms:
        np.testing.assert_allclose(transforms, initial_idle, atol=1e-12)
    assert trajectory.avoidance_robots == {robot}
