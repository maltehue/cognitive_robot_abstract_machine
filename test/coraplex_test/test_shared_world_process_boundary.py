"""
Two robots of one world, each moved by a giskard running in a process of its own.

Everything else about performing a plan is covered against a controller inside the
test's own interpreter. This is the only test where the world is served to real giskard
processes, and so the only one that exercises what the demo does: a giskard that
controls one robot of a world it fetched, a plan per robot performed at the same time,
and the joint positions those processes produce arriving back here.
"""

from __future__ import annotations

import importlib.util
import sys
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Dict, List
from uuid import UUID

import pytest
from rclpy.action import get_action_names_and_types
from rclpy.node import Node

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import ConcurrentPlans
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import StandaloneProcess
from giskardpy.middleware.ros2.robot_interface_config import OneRobotOfManyInterface
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

# %% the launcher under test

ROBOT_LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "coraplex"
    / "demos"
    / "coraplex_shared_world_demo"
    / "robot.py"
)
"""
The robot process the demo starts one of per robot.
"""

WORLD_OWNER = ROBOT_LAUNCHER.with_name("demo.py")
"""
The process that owns the world and starts the robot processes.
"""


def load_script(path: Path) -> ModuleType:
    """
    Read a demo script's definitions, so that the robot names, the node names, the
    action names and the commands are taken from the scripts under test rather than
    spelled again here.

    The demos are scripts rather than an importable package, which is why this goes
    through the file rather than through an import.

    :param path: The script to read.
    """
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    specification = importlib.util.spec_from_file_location(
        f"shared_world_demo_{path.stem}", path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_launcher_definitions() -> ModuleType:
    """
    The robot process's definitions.
    """
    return load_script(ROBOT_LAUNCHER)


DemoRobot = load_launcher_definitions().DemoRobot
"""
The robots the launcher can drive.
"""

ROBOTS = [DemoRobot.PR2, DemoRobot.STRETCH]
"""
The two robots of this world, one per process.
"""

WORLD_SYNC_QUEUE_DEPTH = 1000
"""
How many updates the synchronizer of the served world queues.
"""


# %% the world the processes share


@pytest.fixture
def world_with_both_robots() -> World:
    """
    A world holding nothing but the two robots, so that a fetch takes seconds.
    """
    try:
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=robot.annotation_type,
                    world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                        y=2.0 * position
                    ),
                )
                for position, robot in enumerate(ROBOTS)
            ],
        ).to_domain_object()
    except ParsingError as error:
        pytest.skip(f"Robot URDF not available: {error}")


@pytest.fixture
def served_world(world_with_both_robots: World, rclpy_node: Node) -> World:
    """
    The world of this process, published and offered to whoever fetches it.
    """
    synchronizer = WorldSynchronizer(
        _world=world_with_both_robots,
        node=rclpy_node,
        queue_depth=WORLD_SYNC_QUEUE_DEPTH,
    )
    fetch_server = FetchWorldServer(node=rclpy_node, world=world_with_both_robots)
    yield world_with_both_robots
    fetch_server.close()
    synchronizer.close()


def accepts_goals(node: Node, robot: StrEnum) -> bool:
    """
    Whether the giskard of the given robot takes goals yet.
    """
    return robot.command_action_name in [
        name.lstrip("/") for name, _ in get_action_names_and_types(node)
    ]


@pytest.fixture
def robot_processes(served_world: World, rclpy_node: Node) -> List[StandaloneProcess]:
    """
    One giskard process per robot, each fetching the served world.
    """
    processes = [
        StandaloneProcess(
            launcher_path=ROBOT_LAUNCHER,
            arguments=["--robot", robot.value],
            is_ready=partial(accepts_goals, rclpy_node, robot),
        )
        for robot in ROBOTS
    ]
    for process in processes:
        process.start()
    yield processes
    for process in processes:
        process.stop()


# %% what the robots did


def controlled_positions(world: World, robot: StrEnum) -> Dict[UUID, float]:
    """
    The position of every degree of freedom the given robot is moved through.
    """
    connections = OneRobotOfManyInterface(
        robot_type=robot.annotation_type
    ).connections_to_control(world)
    return {
        degree_of_freedom.id: world.state[degree_of_freedom.id].position
        for connection in connections
        for degree_of_freedom in connection.active_dofs
    }


def test_two_robots_move_through_two_giskards(
    served_world: World, rclpy_node: Node, robot_processes: List[StandaloneProcess]
):
    """
    Both robots park their arms at the same time, each through the giskard of its own
    process, and what they did arrives in the world this process owns.
    """
    contexts = {
        robot: Context(
            world=served_world,
            robot=served_world.get_semantic_annotations_by_type(robot.annotation_type)[
                0
            ],
            ros_node=rclpy_node,
            giskard_node_name=robot.giskard_node_name,
            evaluate_conditions=False,
        )
        for robot in ROBOTS
    }
    positions_before = {
        robot: controlled_positions(served_world, robot) for robot in ROBOTS
    }

    with real_robot:
        ConcurrentPlans(
            plans=[
                sequential([ParkArmsAction(Arms.BOTH)], context=contexts[robot]).plan
                for robot in ROBOTS
            ]
        ).perform()

    for robot in ROBOTS:
        assert controlled_positions(served_world, robot) != positions_before[robot]


# %% one command for the whole demo


def test_the_world_owner_starts_one_robot_process_per_robot():
    """
    The demo is started from ``demo.py`` alone: it starts the robot process of every
    robot it places, in this interpreter, as ``robot.py`` would be started by hand.
    """
    world_owner = load_script(WORLD_OWNER)

    assert world_owner.robot_process_command(DemoRobot.STRETCH) == [
        sys.executable,
        str(ROBOT_LAUNCHER),
        "--robot",
        str(DemoRobot.STRETCH),
    ]
