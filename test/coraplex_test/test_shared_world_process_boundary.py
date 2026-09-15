"""
Three robots of one world, each held by a giskard running in a process of its own.

Everything else about performing a plan is covered against a controller inside the
test's own interpreter. This is the only test where the world is served to real giskard
processes, and so the only one that exercises what the demo does: a giskard that holds
one robot of a world it fetched, a plan per commanded robot performed at the same time,
a robot whose state is published from outside instead, and the joint positions all of
them produce arriving back here.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from enum import StrEnum
from functools import partial
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Dict, List
from uuid import UUID

import psutil
import pytest
from rclpy.action import get_action_names_and_types
from rclpy.node import Node
from sensor_msgs.msg import JointState

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import ConcurrentPlans
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.testing import StandaloneProcess
from giskardpy.middleware.ros2.robot_interface_config import OneRobotOfManyInterface
from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerSettings,
)
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
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

MARKER_LAUNCHER = ROBOT_LAUNCHER.with_name("marker.py")
"""
The interactive marker the demo starts one of per commanded robot.
"""

PROCESS_LAUNCHER_STAND_IN = (
    Path(__file__).resolve().parents[1] / "dataset" / "process_launcher_stand_in.py"
)
"""
A process that runs a second one, the way the joint state publisher is run.
"""

STARTED_PROCESS_TIMEOUT = 10.0
"""
How long that process may take to report what it started.
"""

STARTED_PROCESS_POLL_INTERVAL = 0.1
"""
How often to look for that report.
"""


def wait_for_started_process_id(log_path: Path) -> int:
    """
    The id of the process the launcher stand-in started, read from its output.

    :param log_path: The file that output goes to.
    :raises TimeoutError: If nothing is reported in time.
    """
    deadline = time.monotonic() + STARTED_PROCESS_TIMEOUT
    while time.monotonic() < deadline:
        reported = log_path.read_text().split()
        if reported:
            return int(reported[0])
        time.sleep(STARTED_PROCESS_POLL_INTERVAL)
    raise TimeoutError(f"{PROCESS_LAUNCHER_STAND_IN.name} started nothing")


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
    name = f"shared_world_demo_{path.stem}"
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def load_launcher_definitions() -> ModuleType:
    """
    The robot process's definitions.
    """
    return load_script(ROBOT_LAUNCHER)


DemoRobot = load_launcher_definitions().DemoRobot
"""
The robots the launcher can hold.
"""

COMMANDED_ROBOTS = [DemoRobot.PR2, DemoRobot.STRETCH]
"""
The robots this process moves by performing a plan against their giskard.
"""

MIRRORED_ROBOT = DemoRobot.TIAGO
"""
The robot whose state is published from outside instead.
"""

ROBOTS = COMMANDED_ROBOTS + [MIRRORED_ROBOT]
"""
Every robot of this world, one process each.
"""

WORLD_SYNC_QUEUE_DEPTH = 1000
"""
How many updates the synchronizer of the served world queues.
"""

MIRRORED_JOINT_NAME = "torso_lift_joint"
"""
The joint reported for the mirrored robot, under the plain name its own description
gives it, which the PR2 of this world carries as well.
"""

MIRRORED_JOINT_POSITION = 0.25
"""
The height reported for that joint, within its limits.
"""

MIRROR_TIMEOUT = 15.0
"""
How long the reported position may take to arrive here.

It is written in the mirroring process's next idle cycle and travels back as a world
state update.
"""

MIRROR_POLL_INTERVAL = 0.2
"""
How often the reported position is published again and looked for.
"""


# %% the world the processes share


@pytest.fixture
def world_with_every_robot() -> World:
    """
    A world holding nothing but the robots, so that a fetch takes seconds.
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
def served_world(world_with_every_robot: World, rclpy_node: Node) -> World:
    """
    The world of this process, published and offered to whoever fetches it.
    """
    synchronizer = WorldSynchronizer(
        _world=world_with_every_robot,
        node=rclpy_node,
        queue_depth=WORLD_SYNC_QUEUE_DEPTH,
    )
    fetch_server = FetchWorldServer(node=rclpy_node, world=world_with_every_robot)
    yield world_with_every_robot
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


def mirrored_joint_position(world: World) -> float:
    """
    The position the mirrored robot's reported joint stands at in the given world.
    """
    robot = world.get_semantic_annotations_by_type(MIRRORED_ROBOT.annotation_type)[0]
    connection = world.get_connection_by_name(
        PrefixedName(MIRRORED_JOINT_NAME, robot.root.name.prefix)
    )
    return world.state[connection.raw_dof.id].position


def test_two_robots_move_through_two_giskards_while_a_third_is_mirrored(
    served_world: World, rclpy_node: Node, robot_processes: List[StandaloneProcess]
):
    """
    Both commanded robots park their arms at the same time, each through the giskard of
    its own process, while the third robot's joint states are published from here, and
    everything the three processes did arrives in the world this process owns.
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
        for robot in COMMANDED_ROBOTS
    }
    positions_before = {
        robot: controlled_positions(served_world, robot) for robot in COMMANDED_ROBOTS
    }
    assert mirrored_joint_position(served_world) != MIRRORED_JOINT_POSITION
    publisher = rclpy_node.create_publisher(
        JointState, MIRRORED_ROBOT.joint_states_topic, 1
    )
    reported_state = JointState()
    reported_state.name = [MIRRORED_JOINT_NAME]
    reported_state.position = [MIRRORED_JOINT_POSITION]

    with real_robot:
        ConcurrentPlans(
            plans=[
                sequential([ParkArmsAction(Arms.BOTH)], context=contexts[robot]).plan
                for robot in COMMANDED_ROBOTS
            ]
        ).perform()

    deadline = time.monotonic() + MIRROR_TIMEOUT
    while (
        mirrored_joint_position(served_world) != MIRRORED_JOINT_POSITION
        and time.monotonic() < deadline
    ):
        publisher.publish(reported_state)
        time.sleep(MIRROR_POLL_INTERVAL)
    rclpy_node.destroy_publisher(publisher)

    for robot in COMMANDED_ROBOTS:
        assert controlled_positions(served_world, robot) != positions_before[robot]
    assert mirrored_joint_position(served_world) == MIRRORED_JOINT_POSITION


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


def test_the_world_owner_starts_one_marker_per_robot():
    """
    The demo also starts the interactive marker of every robot it places, so that a
    robot can be dragged about while the plans run.
    """
    world_owner = load_script(WORLD_OWNER)

    assert world_owner.marker_process_command(DemoRobot.STRETCH) == [
        sys.executable,
        str(MARKER_LAUNCHER),
        "--robot",
        str(DemoRobot.STRETCH),
    ]


def test_the_world_owner_starts_a_joint_state_publisher_for_the_mirrored_robot():
    """
    The robot nobody commands is moved through a window of its own, which publishes its
    joint states under the namespace the topic names.
    """
    world_owner = load_script(WORLD_OWNER)
    description_file = CompositePathResolver().resolve(
        MIRRORED_ROBOT.annotation_type.get_ros_file_path()
    )

    assert world_owner.joint_state_publisher_command(MIRRORED_ROBOT) == [
        "ros2",
        "run",
        "joint_state_publisher_gui",
        "joint_state_publisher_gui",
        str(description_file),
        "--ros-args",
        "-r",
        f"__ns:={PurePosixPath(MIRRORED_ROBOT.joint_states_topic).parent}",
        "-r",
        f"__node:={MIRRORED_ROBOT.joint_state_publisher_node_name}",
    ]
    assert Path(description_file).is_file()


def test_stopping_a_started_process_stops_what_it_started(tmp_path: Path):
    """
    The joint state publisher window is started through ``ros2 run``, which runs it as a
    process of its own, so signalling only the process the demo started would leave the
    window behind.
    """
    world_owner = load_script(WORLD_OWNER)
    log_path = tmp_path / "launcher.log"
    process = world_owner.start_process(
        [sys.executable, str(PROCESS_LAUNCHER_STAND_IN)], log_path
    )
    started_process_id = wait_for_started_process_id(log_path)

    world_owner.stop_processes([process])

    assert not psutil.pid_exists(started_process_id)


# %% who is commanded and who is followed


def test_only_the_mirrored_robot_reports_its_joint_states():
    """
    A robot the plans command is moved by a goal rather than by what it reports, so it
    names no topic to be followed on.
    """
    assert MIRRORED_ROBOT.is_mirrored
    assert (
        MIRRORED_ROBOT.joint_states_topic == f"/{MIRRORED_ROBOT.lower()}/joint_states"
    )
    for robot in COMMANDED_ROBOTS:
        assert not robot.is_mirrored
        assert robot.joint_states_topic is None


def test_a_mirrored_robot_offers_no_handles(world_with_every_robot: World):
    """
    Dragging a handle would send a goal to a robot whose joints whoever publishes its
    states overwrites again, so a mirrored robot gets none.
    """
    assert MIRRORED_ROBOT.marker_chains(str(world_with_every_robot.root.name)) == []


# %% the handles the markers offer


def test_every_marker_chain_names_one_body_of_the_world(world_with_every_robot: World):
    """
    Every chain a marker offers a handle for names exactly one body of the shared world,
    where the plain link names of one robot are carried by the other as well.
    """
    world_root_name = str(world_with_every_robot.root.name)

    for robot in ROBOTS:
        for chain in robot.marker_chains(world_root_name):
            for link in [chain.root, chain.tip]:
                assert (
                    InteractiveMarkerSettings.link_named(
                        world_with_every_robot, link
                    ).name.name
                    == PrefixedName.from_string(link).name
                )


def test_a_markers_ros_arguments_name_its_chains_and_its_giskard():
    """
    The marker process hands its robot's chains, giskard and topic namespace to the
    marker node as parameter overrides, which is what a launch file would otherwise do.
    """
    marker = load_script(MARKER_LAUNCHER)

    assert marker.marker_ros_arguments(DemoRobot.STRETCH) == [
        "--ros-args",
        "-p",
        "root_links:=[apartment_root,stretch_description/base_link]",
        "-p",
        "tip_links:=[stretch_description/base_link,link_grasp_center]",
        "-p",
        "giskard_node_name:=giskard_stretch",
        "-p",
        "marker_namespace:=giskard_stretch/cartesian_goals",
    ]
