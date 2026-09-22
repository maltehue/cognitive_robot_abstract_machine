"""
Captured robot articulation survives rebuilding a shared scene.
"""

from __future__ import annotations

from dataclasses import replace
from math import inf, nan

import pytest

from cramera.multi_robot import InvalidRobotScene, RobotInstance, RobotScene
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2, PR2Joint
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


# %% captured articulation
@pytest.fixture
def captured_instance() -> RobotInstance:
    """
    Configure a lifted torso in the first named PR2 namespace.
    """
    return RobotInstance(
        identifier="robot_1",
        label="First PR2",
        robot_type=PR2,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=3),
        joint_positions={str(PrefixedName(PR2Joint.TORSO_LIFT, prefix="robot_1")): 0.2},
    )


def test_rebuilding_preserves_each_instances_joint_positions(
    captured_instance: RobotInstance, pr2_world_copy: World
) -> None:
    """
    Duplicate robot models restore their own independent captured torso state.

    :param captured_instance: First named robot with an authored torso position.
    :param pr2_world_copy: Existing guard for installed robot model availability.
    """
    second = replace(
        captured_instance,
        identifier="robot_2",
        label="Second PR2",
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=-3),
        joint_positions={str(PrefixedName(PR2Joint.TORSO_LIFT, prefix="robot_2")): 0.3},
    )
    scene = RobotScene([captured_instance, second], active_identifier=second.identifier)
    world = scene.build_world()
    for instance in scene.instances:
        connection = world.get_connection_by_name(
            PrefixedName(PR2Joint.TORSO_LIFT, prefix=instance.identifier)
        )
        assert connection.position == instance.joint_positions[str(connection.name)]


@pytest.mark.parametrize("position", [inf, -inf, nan, True])
def test_captured_joint_rejects_invalid_position(
    captured_instance: RobotInstance, position: float
) -> None:
    """
    Invalid numeric state is rejected before native world construction.

    :param captured_instance: Robot carrying a valid captured joint entry.
    :param position: Non-finite or boolean value that cannot represent a joint pose.
    """
    with pytest.raises(InvalidRobotScene):
        replace(
            captured_instance,
            joint_positions={next(iter(captured_instance.joint_positions)): position},
        )


def test_captured_joint_rejects_another_instance_namespace(
    captured_instance: RobotInstance,
) -> None:
    """
    One robot's snapshot cannot command a joint belonging to another instance.

    :param captured_instance: Robot whose captured names must use its namespace.
    """
    with pytest.raises(InvalidRobotScene):
        replace(
            captured_instance,
            joint_positions={
                str(PrefixedName(PR2Joint.TORSO_LIFT, prefix="robot_2")): 0.2
            },
        )


@pytest.mark.parametrize("connection", ["missing", "base_link_joint"])
def test_captured_joint_rejects_unknown_or_fixed_connection(
    captured_instance: RobotInstance, pr2_world_copy: World, connection: str
) -> None:
    """
    A snapshot must resolve to an actuated scalar joint of its exact model.

    :param captured_instance: Robot whose saved connection is replaced.
    :param pr2_world_copy: Existing guard for installed robot model availability.
    :param connection: Unknown or fixed model connection.
    """
    instance = replace(
        captured_instance,
        joint_positions={
            str(PrefixedName(connection, prefix=captured_instance.identifier)): 0.2
        },
    )
    with pytest.raises(InvalidRobotScene):
        RobotScene([instance], active_identifier=instance.identifier).build_world()
