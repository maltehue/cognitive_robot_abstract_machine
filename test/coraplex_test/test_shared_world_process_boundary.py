"""
Two robots of one world, each held by a giskard running in a process of its own and each
moved by joint states published from outside.

This is the only test where the world is served to real giskard processes, and so the
only one that exercises what the demo does: a giskard that holds one robot of a world it
fetched, writes what that robot's joint state topic reports into it, and announces it
back to the process that owns the world.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import psutil
import pytest
from rclpy.action import get_action_names_and_types
from rclpy.node import Node
from sensor_msgs.msg import JointState

from coraplex.testing import StandaloneProcess
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import ParsingError
from semantic_digital_twin.robots.stretch import StretchJoint
from semantic_digital_twin.robots.tiago import TiagoJoint
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


DemoRobot = load_script(ROBOT_LAUNCHER).DemoRobot
"""
The robots the launcher can hold.
"""

ROBOTS = list(DemoRobot)
"""
Every robot of this world, one process each.
"""


# %% what is reported for the robots


@dataclass(frozen=True)
class ReportedJoint:
    """
    One joint position a robot reports on its joint state topic.
    """

    name: str
    """
    The joint's name in the robot's own description, without the prefix it carries in
    the world.
    """

    position: float
    """
    The position reported for it, within its limits.
    """


REPORTED_JOINTS: Dict[StrEnum, ReportedJoint] = {
    DemoRobot.STRETCH: ReportedJoint(name=StretchJoint.LIFT, position=0.5),
    DemoRobot.TIAGO: ReportedJoint(name=TiagoJoint.TORSO_LIFT, position=0.25),
}
"""
The joint reported for each robot.
"""

MIRROR_TIMEOUT = 15.0
"""
How long a reported position may take to arrive here.

It is written in the robot process's next idle cycle and travels back as a world state
update.
"""

MIRROR_POLL_INTERVAL = 0.2
"""
How often a reported position is published again and looked for.
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
    synchronizer = WorldSynchronizer(_world=world_with_every_robot, node=rclpy_node)
    fetch_server = FetchWorldServer(node=rclpy_node, world=world_with_every_robot)
    yield world_with_every_robot
    fetch_server.close()
    synchronizer.close()


def is_up(node: Node, robot: StrEnum) -> bool:
    """
    Whether the giskard of the given robot is up, which it shows by taking goals.
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
            is_ready=partial(is_up, rclpy_node, robot),
        )
        for robot in ROBOTS
    ]
    for process in processes:
        process.start()
    yield processes
    for process in processes:
        process.stop()


# %% what arrived from the robots


def reported_joint_position(world: World, robot: StrEnum) -> float:
    """
    The position the given robot's reported joint stands at in the given world.
    """
    annotation = world.get_semantic_annotations_by_type(robot.annotation_type)[0]
    connection = world.get_connection_by_name(
        PrefixedName(REPORTED_JOINTS[robot].name, annotation.root.name.prefix)
    )
    return world.state[connection.raw_dof.id].position


def reported_joint_arrived(world: World, robot: StrEnum) -> bool:
    """
    Whether the position reported for the given robot has reached the given world.
    """
    return reported_joint_position(world, robot) == REPORTED_JOINTS[robot].position


def test_every_robot_is_moved_through_its_own_giskard_by_published_joint_states(
    served_world: World, rclpy_node: Node, robot_processes: List[StandaloneProcess]
):
    """
    A joint state published for a robot is written into the shared world by that robot's
    own process and arrives in the world this process owns, for every robot.
    """
    for robot in ROBOTS:
        assert not reported_joint_arrived(served_world, robot)
    publishers = {
        robot: rclpy_node.create_publisher(JointState, robot.joint_states_topic, 1)
        for robot in ROBOTS
    }
    reported_states = {}
    for robot in ROBOTS:
        reported_state = JointState()
        reported_state.name = [REPORTED_JOINTS[robot].name]
        reported_state.position = [REPORTED_JOINTS[robot].position]
        reported_states[robot] = reported_state

    deadline = time.monotonic() + MIRROR_TIMEOUT
    while (
        not all(reported_joint_arrived(served_world, robot) for robot in ROBOTS)
        and time.monotonic() < deadline
    ):
        for robot in ROBOTS:
            publishers[robot].publish(reported_states[robot])
        time.sleep(MIRROR_POLL_INTERVAL)
    for publisher in publishers.values():
        rclpy_node.destroy_publisher(publisher)

    for robot in ROBOTS:
        assert reported_joint_position(served_world, robot) == (
            REPORTED_JOINTS[robot].position
        )


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


def test_the_world_owner_starts_a_joint_state_publisher_per_robot():
    """
    Every robot is moved through a window of its own, which publishes its joint states
    under the robot's namespace.
    """
    world_owner = load_script(WORLD_OWNER)
    description_file = CompositePathResolver().resolve(
        DemoRobot.TIAGO.annotation_type.get_ros_file_path()
    )

    assert world_owner.joint_state_publisher_command(DemoRobot.TIAGO) == [
        "ros2",
        "run",
        "joint_state_publisher_gui",
        "joint_state_publisher_gui",
        str(description_file),
        "--ros-args",
        "-r",
        f"__ns:={DemoRobot.TIAGO.namespace}",
        "-r",
        f"__node:={DemoRobot.TIAGO.joint_state_publisher_node_name}",
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


# %% every robot's topics live in its own namespace


def test_every_robot_reports_under_a_namespace_of_its_own():
    """
    Two robots' joint states must not share a topic, or one giskard would write the
    other robot's positions into its own.
    """
    topics = [robot.joint_states_topic for robot in ROBOTS]

    assert len(set(topics)) == len(ROBOTS)
    for robot in ROBOTS:
        assert robot.joint_states_topic.startswith(f"{robot.namespace}/")
