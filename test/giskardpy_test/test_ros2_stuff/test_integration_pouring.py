"""
Integration tests of pouring on a Giskard behind its action server, with the containers
built and coupled in the client's world.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from cramph.composites import Parallel
from cramph.statechart import Statechart
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.giskard_tester import GiskardTester
from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.motion_statechart.graph_node import EndMotion, MotionStatechartNode
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.pouring import PouringTask
from giskardpy.motion_statechart.weights import DefaultWeights
from krrood_spatial_math.spatial_types import Point3
from semantic_digital_twin.datastructures.definitions import StaticJointState
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.tracy import Tracy
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

from ..test_motion_statechart.test_pouring import (
    PouringSetup,
    TransferMotion,
)

# %% a Giskard simulating Tracy


@dataclass(kw_only=True)
class TracyTester(GiskardTester):
    """
    A standalone Giskard simulating Tracy at the control rate the pouring tests use.
    """

    def setup_giskard(self) -> Giskard:
        return Giskard(
            world_source=WorldSpecification(
                robots=[RobotSpecification.specify_real_robot(Tracy)]
            ),
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(),
            qp_controller_config=PouringSetup().controller_config(),
        )


@pytest.fixture()
def robot(init_background_node) -> Iterator[TracyTester]:
    """
    :return: A Giskard simulating Tracy and a client talking to it.
    """
    tester = TracyTester()
    try:
        yield tester
    finally:
        tester.close()


def _execute_on_the_server(*, robot: TracyTester, node: MotionStatechartNode) -> None:
    """
    Send a statechart running `node` to Giskard and wait until it ended there and its
    state reached the client.
    """
    statechart = Statechart(context=robot.api.context)
    statechart.add_node(node=node)
    statechart.add_node(node=EndMotion.when_true(node=node))
    robot.api.compile(statechart=statechart)
    robot.api.execute()
    robot.wait_for_cycles()


@pytest.fixture()
def tracy_in_pouring_pose(robot: TracyTester, tracy_world_copy: World) -> TracyTester:
    """
    :return: The tester, with Tracy in the pose the local pouring tests start from: both
        arms parked, the left tool upright above the table and its wrist pre-tilted.
    """
    pouring_setup = PouringSetup()
    local_world = tracy_world_copy
    [tracy] = local_world.get_semantic_annotations_by_type(Tracy)
    for arm in (tracy.left_arm, tracy.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(local_world)
    pouring_setup.move_left_tool_upright(
        world=local_world,
        left_tool_frame=local_world.get_body_by_name("l_gripper_tool_frame"),
    )
    pouring_setup.pre_tilt_left_wrist(world=local_world)
    start_configuration = {
        robot.api.world.get_connection_by_name(connection.name): connection.position
        for connection in local_world.get_connections_by_type(ActiveConnection)
        if connection.active_dofs
    }
    _execute_on_the_server(
        robot=robot,
        node=SetSeedConfiguration(
            seed_configuration=JointState.from_mapping(start_configuration)
        ),
    )
    return robot


# %% pouring on the server


def test_a_cup_the_client_filled_pours_to_its_goal_on_the_server(
    tracy_in_pouring_pose: TracyTester,
) -> None:
    """
    A cup given its fill level in the client's world drains to the goal on the server,
    and the client's copy follows.
    """
    robot = tracy_in_pouring_pose
    pouring_setup = PouringSetup()
    world = robot.api.world
    left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    cup = pouring_setup.grasp_cup(
        world=world, left_tool_frame=left_tool_frame, name="grasped_cup"
    )
    pouring_task = PouringTask(container=cup, goal_value=0.8, fill_level_tolerance=0.05)
    keep_position = CartesianPosition(
        kinematic_chain=KinematicChain(root=world.root, tip=left_tool_frame),
        goal_point=Point3(reference_frame=left_tool_frame),
        weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
    )

    _execute_on_the_server(
        robot=robot, node=Parallel(nodes=[pouring_task, keep_position])
    )

    server_cup = robot.world.get_semantic_annotation_by_id(cup.id)
    assert server_cup.fill_level == pytest.approx(
        pouring_task.goal_value, abs=pouring_task.fill_level_tolerance
    )
    assert cup.fill_level == pytest.approx(server_cup.fill_level)


def test_a_transfer_the_client_coupled_fills_the_receiver_on_the_server(
    tracy_in_pouring_pose: TracyTester,
) -> None:
    """
    Two cups coupled in the client's world pour into each other on the server, the
    source losing what the receiver gains, and the client's copies follow.
    """
    robot = tracy_in_pouring_pose
    pouring_setup = PouringSetup()
    world = robot.api.world
    left_tool_frame = world.get_body_by_name("l_gripper_tool_frame")
    source_cup = pouring_setup.grasp_cup(
        world=world, left_tool_frame=left_tool_frame, name="source_cup"
    )
    receiving_cup = pouring_setup.place_receiving_cup(world=world)
    receiving_cup.receive_outflow_from(source=source_cup)
    transfer = TransferMotion.build(
        transfer_world=(world, source_cup, receiving_cup, left_tool_frame),
        setup=pouring_setup,
        executor=robot.api,
    )

    robot.api.compile(statechart=transfer.statechart)
    robot.api.execute()
    robot.wait_for_cycles()

    server_source = robot.world.get_semantic_annotation_by_id(source_cup.id)
    server_receiver = robot.world.get_semantic_annotation_by_id(receiving_cup.id)
    tolerance = transfer.transfer_task.fill_level_tolerance
    assert server_source.pours_into is server_receiver
    assert server_receiver.fill_level == pytest.approx(
        transfer.transfer_task.goal_value, abs=tolerance
    )
    assert 1.0 - server_source.fill_level == pytest.approx(
        server_receiver.fill_level, abs=tolerance
    )
    assert receiving_cup.fill_level == pytest.approx(server_receiver.fill_level)
    assert source_cup.fill_level == pytest.approx(server_source.fill_level)
