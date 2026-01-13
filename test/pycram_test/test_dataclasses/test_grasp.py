import os
from copy import deepcopy

import pytest
import rclpy

from pycram.datastructures.enums import ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import NewGraspDescription
from pycram.datastructures.pose import PoseStamped
from semantic_digital_twin.adapters.mesh import STLParser

from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection


@pytest.fixture(scope="session")
def tracy_milk_world(tracy_world):
    tracy_copy = deepcopy(tracy_world)
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "pycram",
            "resources",
            "objects",
            "milk.stl",
        )
    ).parse()
    connection = FixedConnection(
        tracy_copy.get_body_by_name("l_gripper_tool_frame"),
        milk_world.root,
    )
    tracy_copy.merge_world(milk_world, connection)
    return tracy_copy, Tracy.from_world(tracy_copy)


@pytest.fixture(scope="session")
def immutable_simple_pr2_holding_world(simple_pr2_world_setup):
    world, robot_view, context = simple_pr2_world_setup

    milk = world.get_body_by_name("milk.stl")
    tcp = world.get_body_by_name("l_gripper_tool_frame")
    with world.modify_world():
        world.move_branch(milk, tcp)
    return world, robot_view, context


def test_grasp_pose_front(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == [0, 0, 0, 1]
    assert grasp_pose.position.to_list() == [0, 0, 0]

    offset_pose = grasp_desc.grasp_pose_new(
        world.get_body_by_name("milk.stl"), grasp_edge=True
    )

    assert grasp_pose.orientation.to_list() == [0, 0, 0, 1]
    assert offset_pose.position.to_list() == pytest.approx([-0.03, 0, 0], abs=0.01)


def test_grasp_pose_right(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, 0.707, 0.707], abs=0.001
    )

    offset_pose = grasp_desc.grasp_pose_new(
        world.get_body_by_name("milk.stl"), grasp_edge=True
    )

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, 0.707, 0.707], abs=0.001
    )
    assert offset_pose.position.to_list() == pytest.approx([-0.03, 0, 0], abs=0.01)


def test_grasp_pose_left(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )

    offset_pose = grasp_desc.grasp_pose_new(
        world.get_body_by_name("milk.stl"), grasp_edge=True
    )

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0, -0.707, 0.707], abs=0.001
    )
    assert offset_pose.position.to_list() == pytest.approx([-0.03, 0, 0], abs=0.01)


def test_grasp_pose_top(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0, 0.707, 0, 0.707], abs=0.001
    )


def test_grasp_front_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )


def test_grasp_back_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.BACK,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [-0.5, 0.5, 0.5, -0.5], abs=0.001
    )


def test_grasp_top_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.TOP,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0.707, 0.707, 0.0, 0.0], abs=0.001
    )


def test_grasp_left(tracy_milk_world):
    world, robot_view = tracy_milk_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.LEFT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_pose = grasp_desc.grasp_pose_new(world.get_body_by_name("milk.stl"))

    assert grasp_pose.orientation.to_list() == pytest.approx(
        [0.707, 0.0, 0.0, 0.707], abs=0.001
    )


def test_grasp_sequence_front(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world

    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    grasp_sequence = grasp_desc.grasp_pose_sequence(world.get_body_by_name("milk.stl"))

    assert grasp_sequence[0].orientation.to_list() == [0, 0, 0, 1]
    assert grasp_sequence[1].orientation.to_list() == [0, 0, 0, 1]
    assert grasp_sequence[2].orientation.to_list() == [0, 0, 0, 1]

    assert grasp_sequence[0].position.to_list() == pytest.approx(
        [-0.082, 0, 0], abs=0.01
    )
    assert grasp_sequence[1].position.to_list() == pytest.approx([0, 0, 0], abs=0.01)
    assert grasp_sequence[2].position.to_list() == pytest.approx([0, 0, 0.05], abs=0.01)


def test_man_axis(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.manipulation_axis() == [1, 0, 0]


def test_lift_axis(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.lift_axis() == [0, 0, 1]


def test_lift_axis_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.lift_axis() == [0, 1, 0]


def test_man_axis_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.manipulation_axis() == [0, 0, 1]


def test_man_axis_tracy_right(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    assert grasp_desc.manipulation_axis() == [0, 0, 1]


def test_grasp_sequence(immutable_simple_pr2_world):
    world, robot_view, context = immutable_simple_pr2_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(
        world.get_body_by_name("milk.stl"),
    )

    assert sequence[0].orientation.to_list() == pytest.approx([0, 0, 0, 1], abs=0.001)
    assert sequence[1].orientation.to_list() == pytest.approx([0, 0, 0, 1], abs=0.001)
    assert sequence[2].orientation.to_list() == pytest.approx([0, 0, 0, 1], abs=0.001)

    assert sequence[0].position.to_list() == pytest.approx([-0.082, 0, 0], abs=0.01)
    assert sequence[1].position.to_list() == pytest.approx([0, 0, 0.0], abs=0.01)
    assert sequence[2].position.to_list() == pytest.approx([0, 0.0, 0.05], abs=0.01)


def test_grasp_sequence_reverse(immutable_simple_pr2_holding_world):
    world, robot_view, context = immutable_simple_pr2_holding_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.place_pose_sequence(
        PoseStamped.from_list(frame=world.get_body_by_name("milk.stl"))
    )

    assert sequence[2].orientation.to_list() == pytest.approx([0, 0, 0, 1], abs=0.001)
    assert sequence[1].orientation.to_list() == pytest.approx([0, 0, 0, 1], abs=0.001)
    assert sequence[0].orientation.to_list() == pytest.approx([0, 0, 0, 1], abs=0.001)

    assert sequence[2].position.to_list() == pytest.approx([-0.082, 0, 0], abs=0.01)
    assert sequence[1].position.to_list() == pytest.approx([0, 0, 0.0], abs=0.01)
    assert sequence[0].position.to_list() == pytest.approx([0, 0.0, 0.05], abs=0.01)


def test_grasp_sequence_front_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(
        world.get_body_by_name("milk.stl"),
    )

    assert sequence[0].orientation.to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )
    assert sequence[1].orientation.to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )
    assert sequence[2].orientation.to_list() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5], abs=0.001
    )

    assert sequence[0].position.to_list() == pytest.approx([-0.082, 0, 0], abs=0.01)
    assert sequence[1].position.to_list() == pytest.approx([0, 0, 0.0], abs=0.01)
    assert sequence[2].position.to_list() == pytest.approx([0, 0.0, 0.05], abs=0.01)


def test_grasp_sequence_right_tracy(tracy_milk_world):
    world, robot_view = tracy_milk_world
    man = robot_view.left_arm.manipulator

    grasp_desc = NewGraspDescription(
        ApproachDirection.RIGHT,
        VerticalAlignment.NoAlignment,
        man,
    )

    sequence = grasp_desc.grasp_pose_sequence(world.get_body_by_name("milk.stl"))

    assert sequence[0].orientation.to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )
    assert sequence[1].orientation.to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )
    assert sequence[2].orientation.to_list() == pytest.approx(
        [0.0, 0.707, 0.707, 0.0], abs=0.001
    )

    assert sequence[0].position.to_list() == pytest.approx([0, -0.082, 0], abs=0.01)
    assert sequence[1].position.to_list() == pytest.approx([0, 0, 0.0], abs=0.01)
    assert sequence[2].position.to_list() == pytest.approx([0, 0.0, 0.05], abs=0.01)
