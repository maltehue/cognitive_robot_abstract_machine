"""
Continuous navigation through CRAM's existing motion controller.
"""

from __future__ import annotations

import pytest
import numpy as np
from dataclasses import dataclass, field
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan import Plan
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.robots.robot_parts import AbstractRobot

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot, real_robot
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.datastructures.enums import ExecutionType
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.motions.navigation import MoveMotion
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from semantic_digital_twin.datastructures.definitions import StaticJointState
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from giskardpy.motion_statechart.goals.templates import Sequence

# %% controller selection


@pytest.mark.parametrize("execution_environment", [simulated_robot, real_robot])
def test_navigation_uses_a_continuous_controller(
    pr2_world_copy: World, execution_environment: ExecutionEnvironment
) -> None:
    """
    Both execution modes drive the base through the controller's motion ticks.

    :param pr2_world_copy: Independent annotated PR2 world.
    :param execution_environment: Execution mode whose motion chart is inspected.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    target = Pose.from_xyz_rpy(0.5, 0.2, 0.0, reference_frame=world.root)
    motion = MoveMotion(target=target)
    execute_single(motion, context=Context(world=world, robot=robot))

    with execution_environment:
        chart = motion.motion_chart

    if execution_environment is simulated_robot:
        assert isinstance(chart, Sequence)
    else:
        assert isinstance(chart, CartesianPose)
        assert chart.goal_pose is target
        assert chart.root_link is world.root
        assert chart.tip_link is robot.root


# %% observed execution


@dataclass
class NavigationTrajectory(PlanCallback):
    """
    Record authoritative base positions emitted by a simulated plan.
    """

    robot: AbstractRobot
    """
    The robot whose base position is recorded.
    """

    positions: list[np.ndarray] = field(default_factory=list)
    """
    World-frame positions observed at each controller tick.
    """

    avoidance_counts: set[int] = field(default_factory=set)
    """
    Number of active avoidance goals observed in each motion chart.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """
        Record the robot position after the control tick.

        :param statechart: The statechart that advanced the robot.
        """
        self.positions.append(
            self.robot.root.global_pose.to_position().to_np()[:3].copy()
        )
        self.avoidance_counts.add(
            len(statechart.get_nodes_by_type(ExternalCollisionAvoidance))
        )


def test_simulated_navigation_publishes_intermediate_positions(
    cylinder_bot_world: World,
) -> None:
    """
    The plan's ticks describe a driven trajectory that reaches the goal.

    :param cylinder_bot_world: Existing minimal robot with an active planar drive.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    target = Pose.from_xyz_rpy(-0.3, 0, 0, reference_frame=world.root)
    motion = MoveMotion(target=target)
    plan = execute_single(
        motion, context=Context(world=world, robot=robot, _debug=False)
    ).plan
    trajectory = NavigationTrajectory(robot=robot)
    plan.node_callbacks.append(trajectory)

    with simulated_robot:
        plan.perform()

    positions = np.asarray(trajectory.positions)
    assert np.any((positions[:, 0] < -0.05) & (positions[:, 0] > -0.25))
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    np.testing.assert_allclose(
        positions[-1], target.to_position().to_np()[:3], atol=0.01
    )


@pytest.mark.parametrize(
    "world_fixture", ["cylinder_bot_world", "cylinder_bot_diff_world"]
)
def test_navigation_drives_around_an_obstacle(
    request: pytest.FixtureRequest, world_fixture: str
) -> None:
    """
    Drive a continuous detour with clearance from the blocking box.

    :param request: Access to the existing mobile-robot fixtures.
    :param world_fixture: Robot with either an omnidirectional or differential drive.
    """
    world = request.getfixturevalue(world_fixture)
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    BodySpecification.box(
        "navigation_barrier",
        Scale(0.4, 0.8, 1.0),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-1, 0, 0),
    ).spawn(world)
    target = Pose.from_xyz_rpy(-2, 0, 0, yaw=0.4, reference_frame=world.root)
    trajectory = NavigationTrajectory(robot=robot)
    plan = execute_single(
        MoveMotion(target), context=Context(world=world, robot=robot, _debug=False)
    ).plan
    plan.node_callbacks.append(trajectory)
    with simulated_robot:
        plan.perform()
    positions = np.asarray(trajectory.positions)
    assert np.max(np.abs(positions[:, 1])) > 0.5
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    # Distance from the robot center to the box footprint must exceed its radius.
    distances = np.linalg.norm(
        np.maximum(np.abs(positions[:, :2] - [-1, 0]) - [0.2, 0.4], 0), axis=1
    )
    assert np.min(distances) > 0.05
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )


@pytest.mark.parametrize("yaw", [0.0, 0.4])
def test_differential_navigation_can_turn_without_translation(
    cylinder_bot_diff_world: World, yaw: float
) -> None:
    """
    No-op and pure rotation goals complete without normalizing a zero vector.

    :param cylinder_bot_diff_world: Existing robot with a differential drive.
    :param yaw: Requested change of heading at the current base position.
    """
    world = cylinder_bot_diff_world
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


def test_sequential_navigation_plans_from_the_actual_previous_pose(
    cylinder_bot_world: World,
) -> None:
    """
    Each relative navigation starts after its predecessor has moved the base.

    :param cylinder_bot_world: Existing annotated mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    plan = sequential(
        [
            MoveMotion(Pose.from_xyz_rpy(-0.3, reference_frame=robot.root)),
            MoveMotion(Pose.from_xyz_rpy(-0.3, reference_frame=robot.root)),
        ],
        context=Context(world=world, robot=robot, _debug=False),
    ).plan
    with simulated_robot:
        plan.perform()
    assert float(robot.root.global_pose.x) == pytest.approx(-0.6, abs=0.02)


@dataclass
class NavigationInterrupt(PlanCallback):
    """
    Interrupt a sequence while its first navigation is actively driving.
    """

    plan: Plan
    """
    Plan whose trailing navigation must be skipped.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """
        Request interruption after the current navigation has started.

        :param statechart: Active controller triggering the callback.
        """
        self.plan.root.interrupt()


def test_interrupt_finishes_navigation_and_skips_the_next_route(
    cylinder_bot_world: World,
) -> None:
    """
    A separately planned route must not restart an interrupted sequence.

    :param cylinder_bot_world: Existing annotated mobile robot.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    plan = sequential(
        [
            MoveMotion(Pose.from_xyz_rpy(-0.3, reference_frame=world.root)),
            MoveMotion(Pose.from_xyz_rpy(-0.6, reference_frame=world.root)),
        ],
        context=Context(world=world, robot=robot, _debug=False),
    ).plan
    plan.node_callbacks.append(NavigationInterrupt(plan))
    with simulated_robot:
        plan.perform()
    np.testing.assert_allclose(
        robot.root.global_pose.to_np()[:3, 3], [-0.3, 0, 0], atol=0.01
    )


@pytest.mark.parametrize("global_avoidance", [False, True])
def test_navigation_always_has_one_collision_avoidance_goal(
    cylinder_bot_world: World, global_avoidance: bool
) -> None:
    """
    Navigation supplies avoidance without duplicating an outer collision goal.

    :param cylinder_bot_world: Existing mobile robot and environment geometry.
    :param global_avoidance: Whether the execution environment already adds avoidance.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    plan = execute_single(
        MoveMotion(Pose.from_xyz_rpy(-0.3, reference_frame=world.root)),
        context=Context(world=world, robot=robot, _debug=False),
    ).plan
    trajectory = NavigationTrajectory(robot)
    plan.node_callbacks.append(trajectory)
    with ExecutionEnvironment(
        ExecutionType.SIMULATED, collision_avoidance=global_avoidance
    ):
        plan.perform()
    assert trajectory.avoidance_counts == {1}


def test_navigation_keeps_requested_non_drive_joint_states(
    pr2_world_copy: World,
) -> None:
    """
    Posture preservation stays active while the base follows its route.

    :param pr2_world_copy: Mobile manipulator with controlled arm and torso joints.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    expected = {
        connection: connection.position
        for connection in robot.controlled_connections
        if isinstance(connection, ActiveConnection1DOF)
    }
    motion = MoveMotion(
        Pose.from_xyz_rpy(-0.3, reference_frame=world.root), keep_joint_states=True
    )
    execute_single(motion, context=Context(world=world, robot=robot))
    chart = MotionStatechart()
    with simulated_robot:
        chart.add_node(motion.motion_chart)
    executor = Executor(context=MotionStatechartContext(world=world))
    executor.compile(chart)
    goals = chart.get_nodes_by_type(JointPositionList)
    assert len(goals) == 1
    assert dict(goals[0].goal_state.items()) == expected
    chart.cleanup_nodes(executor.context)
    executor.context.cleanup()


def test_long_detour_gets_time_for_each_driving_stage(
    cylinder_bot_diff_world: World,
) -> None:
    """
    A valid multi-stage route can take longer than one short motion's budget.

    :param cylinder_bot_diff_world: Existing differential-drive robot.
    """
    world = cylinder_bot_diff_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    BodySpecification.box(
        "long_route_barrier",
        Scale(0.4, 0.8, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-4),
    ).spawn(world)
    target = Pose.from_xyz_rpy(-8, reference_frame=world.root)
    plan = execute_single(
        MoveMotion(target), context=Context(world=world, robot=robot, _debug=False)
    ).plan
    with simulated_robot:
        plan.perform()
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )


def test_navigation_drives_on_a_touching_support(cylinder_bot_world: World) -> None:
    """
    Controller execution can retain supporting contact during planar driving.

    :param cylinder_bot_world: Mobile robot with a known lower collision boundary.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    lower = min(
        box.min_z for box in robot.as_bounding_box_collection_in_frame(world.root)
    )
    BodySpecification.box(
        "supporting_floor",
        Scale(10, 10, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=lower - 0.05),
    ).spawn(world)
    target = Pose.from_xyz_rpy(-0.3, reference_frame=world.root)
    plan = execute_single(
        MoveMotion(target), context=Context(world=world, robot=robot, _debug=False)
    ).plan
    with simulated_robot:
        plan.perform()
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )


def test_differential_waypoints_keep_the_arrival_heading(
    cylinder_bot_diff_world: World,
) -> None:
    """
    Intermediate waypoints do not request redundant turns toward a fixed yaw.

    :param cylinder_bot_diff_world: Robot whose obstacle route has intermediate points.
    """
    world = cylinder_bot_diff_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    BodySpecification.box(
        "heading_barrier",
        Scale(0.4, 0.8, 1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(-1),
    ).spawn(world)
    motion = MoveMotion(Pose.from_xyz_rpy(-2, reference_frame=world.root))
    execute_single(motion, context=Context(world=world, robot=robot))
    chart = MotionStatechart()
    with simulated_robot:
        chart.add_node(motion.motion_chart)
    executor = Executor(context=MotionStatechartContext(world=world))
    executor.compile(chart)
    first = chart.get_nodes_by_type(DifferentialDriveBaseGoal)[0]
    position = first.goal_pose.to_np()[:3, 3]
    expected = Pose.from_xyz_rpy(
        yaw=np.arctan2(position[1], position[0]), reference_frame=world.root
    )
    np.testing.assert_allclose(
        first.goal_pose.to_np()[:3, :3], expected.to_np()[:3, :3], atol=1e-10
    )
    chart.cleanup_nodes(executor.context)
    executor.context.cleanup()


def test_navigation_in_the_apartment_with_parked_arms(
    pr2_apartment_state_reset: World,
) -> None:
    """
    The normal Navigate action drives a folded PR2 inside the existing apartment.

    :param pr2_apartment_state_reset: Independent copy of the annotated apartment world.
    """
    world = pr2_apartment_state_reset
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    target = Pose.from_xyz_rpy(1.1, 2, reference_frame=world.root)
    plan = execute_single(
        NavigateAction(target), context=Context(world=world, robot=robot, _debug=False)
    ).plan
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        plan.perform()
    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )
