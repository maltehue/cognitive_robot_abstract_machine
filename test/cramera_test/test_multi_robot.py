"""
Named robot instances reuse native worlds and remain independently selectable.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cramera.multi_robot import (
    InvalidRobotScene,
    RobotInstance,
    RobotInstanceUnavailable,
    RobotScene,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


# %% reusable authoring fixtures
@pytest.fixture
def robot_instance() -> RobotInstance:
    """
    Describe a named PR2 with an independently authored localization pose.
    """
    return RobotInstance(
        identifier="robot_1",
        label="First PR2",
        robot_type=PR2,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=3, y=1, yaw=0.5),
    )


@pytest.fixture
def robot_scene(robot_instance: RobotInstance) -> RobotScene:
    """
    Select the second of two same-model robots in the common world.

    :param robot_instance: First robot and shared model description.
    """
    second = replace(
        robot_instance,
        identifier="robot_2",
        label="Second PR2",
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(x=-3, y=-1, yaw=-0.5),
    )
    return RobotScene(
        instances=[robot_instance, second], active_identifier=second.identifier
    )


# %% invalid authoring is rejected before loading models
@pytest.mark.parametrize("identifier", ["", "robot/name", "robot name", "2robot"])
def test_invalid_instance_identifier(
    robot_instance: RobotInstance, identifier: str
) -> None:
    """
    An identifier must remain a single usable model namespace.

    :param robot_instance: Valid instance whose namespace is replaced.
    :param identifier: Invalid namespace submitted by an author.
    """
    with pytest.raises(InvalidRobotScene):
        replace(robot_instance, identifier=identifier)


def test_empty_instance_label(robot_instance: RobotInstance) -> None:
    """
    Every instance has a meaningful visible selection label.

    :param robot_instance: Valid instance whose label is replaced.
    """
    with pytest.raises(InvalidRobotScene):
        replace(robot_instance, label="  ")


def test_empty_scene() -> None:
    """
    A selected robot cannot exist in an empty robot scene.
    """
    with pytest.raises(InvalidRobotScene):
        RobotScene(instances=[], active_identifier="robot_1")


def test_duplicate_instance_identifier(robot_instance: RobotInstance) -> None:
    """
    Repeated model types are allowed, repeated instance namespaces are not.

    :param robot_instance: Instance submitted twice under the same identity.
    """
    with pytest.raises(InvalidRobotScene):
        RobotScene(
            instances=[robot_instance, robot_instance],
            active_identifier=robot_instance.identifier,
        )


def test_unknown_selected_instance(robot_instance: RobotInstance) -> None:
    """
    Selection must resolve to an explicitly authored instance.

    :param robot_instance: Only available instance.
    """
    with pytest.raises(InvalidRobotScene):
        RobotScene(instances=[robot_instance], active_identifier="missing")


# %% native world materialization and selection
@pytest.mark.parametrize("environment", [False, True])
def test_scene_resolves_selected_duplicate_and_preserves_poses(
    robot_scene: RobotScene, pr2_world_copy: World, environment: bool
) -> None:
    """
    Both world constructors retain identity, pose, annotation, and active selection.

    :param robot_scene: Two differently named copies of the same model.
    :param pr2_world_copy: Existing fixture checking installed robot availability.
    :param environment: Whether to merge robots into the existing table environment.
    """
    environment_path = (
        str(
            Path(__file__).resolve().parents[2]
            / "semantic_digital_twin"
            / "resources"
            / "urdf"
            / "table.urdf"
        )
        if environment
        else None
    )
    world = robot_scene.build_world(environment_path)
    assert len(world.get_semantic_annotations_by_type(PR2)) == len(
        robot_scene.instances
    )
    for instance in robot_scene.instances:
        robot = robot_scene.robot(world, instance.identifier)
        assert robot.name.name == instance.label
        assert robot.name.prefix == instance.identifier
        assert robot.root.name.prefix == instance.identifier
        assert (
            robot.root.parent_connection.parent.parent_connection.parent is world.root
        )
        np.testing.assert_allclose(
            robot.root.global_transform.to_np(), instance.pose.to_np()
        )
    assert robot_scene.selected_robot(world) is robot_scene.robot(
        world, robot_scene.active_identifier
    )


def test_scene_selection_rejects_a_missing_world_instance(
    robot_scene: RobotScene,
) -> None:
    """
    A world without the configured robot cannot silently select another one.

    :param robot_scene: Expected identities absent from the empty world.
    """
    with pytest.raises(RobotInstanceUnavailable) as raised:
        robot_scene.selected_robot(World.create_with_root_body())
    assert raised.value.identifier == robot_scene.active_identifier
    assert raised.value.matches == 0


def test_scene_selection_rejects_an_unknown_requested_identifier(
    robot_scene: RobotScene,
) -> None:
    """
    Callers cannot select an instance absent from the authored configuration.

    :param robot_scene: Configured instances and active selection.
    """
    with pytest.raises(InvalidRobotScene):
        robot_scene.robot(World.create_with_root_body(), "missing")
