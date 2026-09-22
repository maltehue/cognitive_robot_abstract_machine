"""
Navigation permits standing clearance while checking physical floor contact.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot
from coraplex.locations.navigation import RobotNavigationPath
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions.navigation import MoveMotion
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
)
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Floor
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


# %% supporting floor
@pytest.fixture()
def robot_with_floor(cylinder_bot_world: World) -> World:
    """
    Add an annotated floor a centimetre below the robot's collision geometry.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    bottom = robot.as_bounding_box_collection_in_frame(world.root).bounding_box().min_z
    floor = BodySpecification.box(
        "supporting_floor",
        Scale(10, 10, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=bottom - 0.06),
    ).spawn(world)
    with world.modify_world():
        world.add_semantic_annotation(Floor(root=floor))
    return world


def test_support_policy_retains_collision_checks(robot_with_floor: World) -> None:
    """
    Only the floor pair receives the physical-contact threshold.
    """
    world = robot_with_floor
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    path = RobotNavigationPath(world, robot, robot.root.global_pose)
    rules = path.support_contact_rules()

    assert len(rules) == 1
    rule = rules[0]
    assert isinstance(rule, AvoidCollisionBetweenGroups)
    assert set(rule.body_group_a) == set(robot.bodies_with_collision)
    assert rule.body_group_b == [world.get_body_by_name("supporting_floor")]
    assert rule.buffer_zone_distance == 0.0
    assert rule.violated_distance == -path.geometry_tolerance


def test_floor_contact_scope_is_restored_after_driving(robot_with_floor: World) -> None:
    """
    Driving beside the supporting floor does not leak its contact policy.
    """
    world = robot_with_floor
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    previous = list(world.collision_manager.temporary_rules)
    target = Pose.from_xyz_rpy(-0.3, reference_frame=world.root)
    node = execute_single(MoveMotion(target), context=Context.from_world(world))

    with simulated_robot:
        node.perform()

    np.testing.assert_allclose(
        robot.root.global_pose.to_np(), target.to_np(), atol=0.01
    )
    assert world.collision_manager.temporary_rules == previous
