from enum import StrEnum

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.motions.navigation import MoveMotion
from krrood.entity_query_language.factories import evaluate_condition
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% navigation targets for roots above the drive plane


def test_navigation_motion_preserves_root_height(pr2_world_copy: World) -> None:
    """
    Navigation sends an elevated root to its standing height instead of the floor.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    with world.modify_world():
        robot.root.parent_connection.connection_T_child_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=0.9)
        )
    target = Pose.from_xyz_rpy(1.3, 2.0, reference_frame=world.root)
    action = NavigateAction(target)
    execute_single(action, context=Context(world=world, robot=robot))

    motion = action._action_plan.plan.get_nodes_by_designator_type(MoveMotion)[
        0
    ].designator

    assert float(motion.target.z) == pytest.approx(float(robot.root.global_pose.z))


class NavigationParameter(StrEnum):
    """
    Parameters consumed by navigation conditions.
    """

    TARGET_LOCATION = "target_location"


def test_navigation_completion_accepts_standing_root(pr2_world_copy: World) -> None:
    """
    A root above the floor satisfies completion when its planar target is reached.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    with world.modify_world():
        robot.root.parent_connection.connection_T_child_expression = (
            HomogeneousTransformationMatrix.from_xyz_rpy(z=0.9)
        )
    target = Pose.from_xyz_rpy(reference_frame=world.root)

    condition = NavigateAction.post_condition(
        {},
        Context(world=world, robot=robot),
        {NavigationParameter.TARGET_LOCATION: target},
    )

    assert evaluate_condition(condition) is True
