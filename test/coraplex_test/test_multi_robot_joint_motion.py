"""
Joint motions resolve names within the robot selected by the plan context.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, TaskStatus
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from giskardpy.motion_statechart.goals.templates import Parallel
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import WorldEntityNotFoundError
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

from .test_multi_robot_motion import RobotIsolationTrajectory


# %% same-model native world
@pytest.fixture
def duplicate_robot_context(pr2_world_copy: World) -> Context:
    """
    Select the second of two distant native PR2 instances.

    :param pr2_world_copy: Existing availability guard for the installed PR2 model.
    :return: Plan context selecting the second native annotation.
    """
    world = WorldSpecification(
        robots=[
            RobotSpecification(
                PR2,
                prefix=f"robot_{index}",
                world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=index * 4),
            )
            for index in range(2)
        ]
    ).to_domain_object()
    robot = world.get_semantic_annotations_by_type(PR2)[1]
    return Context(world=world, robot=robot, _debug=False)


# %% scoped native connection lookup
@pytest.mark.parametrize("name_format", ["plain", "prefixed", "qualified"])
def test_joint_motion_resolves_only_the_selected_instance(
    duplicate_robot_context: Context, name_format: str
) -> None:
    """
    Local and explicitly qualified names target the selected robot's exact joint.

    :param duplicate_robot_context: Context selecting the second model instance.
    :param name_format: Supported representation of the native connection name.
    """
    robot = duplicate_robot_context.robot
    joint_name = PrefixedName(PR2Joint.TORSO_LIFT, prefix=robot.root.name.prefix)
    name = {
        "plain": joint_name.name,
        "prefixed": joint_name,
        "qualified": str(joint_name),
    }[name_format]
    motion = MoveJointsMotion([name], [0.2], max_joint_velocity=0.3)
    execute_single(motion, context=duplicate_robot_context)
    chart = motion.motion_chart
    assert isinstance(chart, Parallel)
    expected = duplicate_robot_context.world.get_connection_by_name(joint_name)
    assert chart.nodes[0].goal_state.connections == [expected]
    assert chart.nodes[1].connections == [expected]


@pytest.mark.parametrize("qualified", [False, True])
def test_explicit_foreign_joint_cannot_escape_the_selected_robot(
    duplicate_robot_context: Context, qualified: bool
) -> None:
    """
    Naming a different instance must fail before a motion chart is produced.

    :param duplicate_robot_context: Context selecting only the second robot.
    :param qualified: Whether the foreign name is a string or structured name.
    """
    idle = next(
        robot
        for robot in duplicate_robot_context.world.get_semantic_annotations_by_type(PR2)
        if robot is not duplicate_robot_context.robot
    )
    foreign = PrefixedName(PR2Joint.TORSO_LIFT, prefix=idle.root.name.prefix)
    motion = MoveJointsMotion([str(foreign) if qualified else foreign], [0.2])
    execute_single(motion, context=duplicate_robot_context)
    with pytest.raises(WorldEntityNotFoundError):
        motion.motion_chart


# %% native action execution
def test_parking_and_torso_keep_the_other_instance_stationary(
    duplicate_robot_context: Context,
) -> None:
    """
    Native Park and torso actions finish while every idle body stays fixed.

    :param duplicate_robot_context: Shared world and second robot selected for
        execution.
    """
    robot = duplicate_robot_context.robot
    idle = next(
        candidate
        for candidate in duplicate_robot_context.world.get_semantic_annotations_by_type(
            PR2
        )
        if candidate is not robot
    )
    initial_idle = np.asarray([body.global_pose.to_np() for body in idle.bodies])
    plan = sequential(
        [ParkArmsAction(Arms.BOTH), MoveTorsoAction(TorsoState.HIGH)],
        context=duplicate_robot_context,
    ).plan
    trajectory = RobotIsolationTrajectory(robot, idle)
    plan.node_callbacks.append(trajectory)
    with simulated_robot_advanced:
        plan.perform()
    for action_type in (ParkArmsAction, MoveTorsoAction):
        [action] = plan.get_nodes_by_designator_type(action_type)
        assert action.status is TaskStatus.SUCCEEDED
    assert trajectory.idle_transforms
    for transforms in trajectory.idle_transforms:
        np.testing.assert_allclose(transforms, initial_idle, atol=1e-12)
    assert trajectory.avoidance_robots == {robot}
