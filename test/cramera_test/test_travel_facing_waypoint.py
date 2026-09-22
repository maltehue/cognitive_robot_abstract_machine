"""
Position-driven transitions between travel-facing navigation waypoints.
"""

from __future__ import annotations

from collections.abc import Iterator
from math import pi

import pytest

from coraplex.robot_plans.motions.navigation import TravelFacingWaypoint
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.nodes_for_testing.nodes_for_testing import (
    ConstFalseNode,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianOrientation,
    CartesianPosition,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World

# %% native execution


@pytest.fixture
def waypoint_executor(cylinder_bot_world: World) -> Iterator[Executor]:
    """
    Release controller resources after observing a waypoint transition.

    :param cylinder_bot_world: Existing world with an omnidirectional mobile robot.
    :return: Native executor bound to the fixture world.
    """
    executor = Executor(context=MotionStatechartContext(world=cylinder_bot_world))
    yield executor
    if executor.motion_statechart is not None:
        executor.motion_statechart.cleanup_nodes(executor.context)
    executor.context.cleanup()


def execute_waypoint(executor: Executor, target: Pose) -> Sequence:
    """
    Advance a waypoint and its successor through the initial lifecycle updates.

    :param executor: Executor whose world supplies the mobile robot.
    :param target: Waypoint pose whose position controls the transition.
    :return: Compiled sequence after its observations and transitions settle.
    """
    world = executor.context.world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    waypoint = TravelFacingWaypoint(
        root_link=world.root, tip_link=robot.root, goal_pose=target
    )
    sequence = Sequence(nodes=[waypoint, ConstFalseNode()])
    chart = MotionStatechart()
    chart.add_node(sequence)
    executor.compile(chart)
    observation_settling_cycles = 5
    for _ in range(observation_settling_cycles):
        executor.tick()
    return sequence


# %% waypoint completion


def test_reached_heading_keeps_unreached_position_active(
    waypoint_executor: Executor,
) -> None:
    """
    Reaching yaw alone keeps both waypoint tasks active and the successor waiting.

    :param waypoint_executor: Native executor for the mobile robot fixture.
    """
    world = waypoint_executor.context.world
    target = Pose.from_xyz_rpy(x=-1, reference_frame=world.root)
    sequence = execute_waypoint(waypoint_executor, target)
    waypoint = sequence.nodes[0]
    position = sequence.motion_statechart.get_nodes_by_type(CartesianPosition)[0]
    orientation = sequence.motion_statechart.get_nodes_by_type(CartesianOrientation)[0]

    assert orientation.observation_state == ObservationStateValues.TRUE
    assert position.observation_state == ObservationStateValues.FALSE
    assert waypoint.observation_state == ObservationStateValues.FALSE
    assert waypoint.life_cycle_state == LifeCycleValues.RUNNING
    assert position.life_cycle_state == LifeCycleValues.RUNNING
    assert orientation.life_cycle_state == LifeCycleValues.RUNNING
    assert sequence.nodes[-1].life_cycle_state == LifeCycleValues.NOT_STARTED


def test_reached_position_advances_with_heading_unfinished(
    waypoint_executor: Executor,
) -> None:
    """
    Position arrival ends both waypoint tasks and starts the following stage.

    :param waypoint_executor: Native executor for the mobile robot fixture.
    """
    world = waypoint_executor.context.world
    target = Pose.from_xyz_rpy(yaw=pi / 2, reference_frame=world.root)
    sequence = execute_waypoint(waypoint_executor, target)
    waypoint = sequence.nodes[0]
    position = sequence.motion_statechart.get_nodes_by_type(CartesianPosition)[0]
    orientation = sequence.motion_statechart.get_nodes_by_type(CartesianOrientation)[0]

    assert position.observation_state == ObservationStateValues.TRUE
    assert orientation.observation_state == ObservationStateValues.FALSE
    assert waypoint.observation_state == ObservationStateValues.TRUE
    assert waypoint.life_cycle_state == LifeCycleValues.DONE
    assert position.life_cycle_state == LifeCycleValues.DONE
    assert orientation.life_cycle_state == LifeCycleValues.DONE
    assert sequence.nodes[-1].life_cycle_state == LifeCycleValues.RUNNING
