"""
A Stretch and a Tiago share one apartment, each held by a giskard process of its own and
each moved from a window of its own.

This process owns the world: it builds it, serves it, draws it and starts one
``robot.py`` and one joint state publisher window per robot::

    python demo.py

Every robot process fetches this world and holds its own robot in it, so the robots see
each other while each of them is moved by a source of its own. Moving a slider in a
robot's window publishes that robot's joint states; its giskard writes them into its
copy of the world and announces them, and this process draws what happened. Their output
goes to a log file each, whose place is printed, and every started process is stopped
with this one.

A robot's base stays where the world put it, because the window reports joint positions
and no base pose.

Watching it in RViz
-------------------

Set the fixed frame to ``apartment/apartment_root`` and add a ``MarkerArray`` display on
``/semworld/viz_marker``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rclpy.action import get_action_names_and_types
from typing_extensions import List

from giskardpy.middleware.ros2 import rospy
from robot import DemoRobot
from semantic_digital_twin.adapters.package_resolver import CompositePathResolver
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World

# %% where everything stands

APARTMENT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "worlds" / "apartment.urdf"
)
"""
The environment the robots share.
"""

STRETCH_START = (1.5, 1.2)
"""
Where the Stretch stands, in the corridor in front of the kitchen counter.
"""

TIAGO_START = (1.5, 3.8)
"""
Where the Tiago stands, further up the corridor.
"""

READY_TIMEOUT = 600.0
"""
How long to wait for the robot processes.

Each of them fetches an apartment of a few hundred bodies, and this process answers one
fetch after the other.
"""

READY_POLL_INTERVAL = 1.0
"""
How often to look whether a robot process is up.
"""

ROBOT_LAUNCHER = Path(__file__).with_name("robot.py")
"""
The script started once per robot.
"""

ROS_RUN_COMMAND = ["ros2", "run"]
"""
The command that runs one executable of an installed ROS package.
"""

JOINT_STATE_PUBLISHER_PACKAGE = "joint_state_publisher_gui"
"""
The package holding the window a robot is moved from.
"""

JOINT_STATE_PUBLISHER_EXECUTABLE = "joint_state_publisher_gui"
"""
The executable of that package, which takes the robot's description file and publishes
one slider per movable joint of it.
"""

ROS_ARGUMENTS_FLAG = "--ros-args"
"""
The flag that opens the part of a command line ROS reads itself.
"""

REMAP_FLAG = "-r"
"""
The flag that renames one node or topic of a started node.
"""

LOG_DIRECTORY = Path(tempfile.gettempdir()) / "coraplex_shared_world_demo"
"""
Where the started processes write their output, one file each.
"""

SHUTDOWN_TIMEOUT = 10.0
"""
How long a started process may take to end after it was interrupted, in seconds, before
it is killed.
"""

# %% the processes started per robot


def robot_process_command(robot: DemoRobot) -> List[str]:
    """
    The command that starts the process holding one robot, in this interpreter.

    :param robot: The robot the process holds.
    """
    return [sys.executable, str(ROBOT_LAUNCHER), "--robot", str(robot)]


def joint_state_publisher_command(robot: DemoRobot) -> List[str]:
    """
    The command that opens the window one robot's joint positions are reported from.

    The window reads the robot's own description file, so it offers a slider per movable
    joint of it, and publishes under the robot's namespace.

    :param robot: The robot whose joint states are reported.
    """
    return [
        *ROS_RUN_COMMAND,
        JOINT_STATE_PUBLISHER_PACKAGE,
        JOINT_STATE_PUBLISHER_EXECUTABLE,
        str(CompositePathResolver().resolve(robot.annotation_type.get_ros_file_path())),
        ROS_ARGUMENTS_FLAG,
        REMAP_FLAG,
        f"__ns:={robot.namespace}",
        REMAP_FLAG,
        f"__node:={robot.joint_state_publisher_node_name}",
    ]


def start_process(command: List[str], log_path: Path) -> subprocess.Popen:
    """
    Start a child process in a session of its own so that it can be stopped as a group,
    with its output in the given file.

    :param command: The command to run.
    :param log_path: The file its output goes to.
    """
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        command,
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def start_robot_process(robot: DemoRobot) -> subprocess.Popen:
    """
    Start the process holding one robot.

    :param robot: The robot the process holds.
    """
    log_path = LOG_DIRECTORY / f"{robot}.log"
    process = start_process(robot_process_command(robot), log_path)
    print(f"started the {robot} process, logging to {log_path}", flush=True)
    return process


def start_joint_state_publisher(robot: DemoRobot) -> subprocess.Popen:
    """
    Open the window one robot's joint positions are reported from.

    :param robot: The robot whose joint states are reported.
    """
    log_path = LOG_DIRECTORY / f"{robot}_joint_states.log"
    process = start_process(joint_state_publisher_command(robot), log_path)
    print(f"opened the {robot} joint states, logging to {log_path}", flush=True)
    return process


def signal_process_group(process: subprocess.Popen, sent_signal: int) -> None:
    """
    Send a signal to a started process and to everything it started itself.

    A process started here leads a session of its own, so the signal reaches the whole
    group rather than only the process this one holds; a launcher runs the program it
    was asked for as a process of its own, which would otherwise be left behind.

    :param process: The started process.
    :param sent_signal: The signal to send.
    """
    if process.poll() is not None:
        return
    os.killpg(os.getpgid(process.pid), sent_signal)


def stop_processes(processes: List[subprocess.Popen]) -> None:
    """
    Interrupt every started process, and everything it started, killing what does not
    end in :data:`SHUTDOWN_TIMEOUT`.

    :param processes: The processes to stop.
    """
    for process in processes:
        signal_process_group(process, signal.SIGINT)
    deadline = time.monotonic() + SHUTDOWN_TIMEOUT
    for process in processes:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(READY_POLL_INTERVAL)
        signal_process_group(process, signal.SIGKILL)
        process.wait()


# %% building and serving the world


def build_world() -> World:
    """
    The apartment with every robot standing in it.
    """
    return WorldSpecification.from_urdf(
        str(APARTMENT_PATH),
        robots=[
            RobotSpecification(
                semantic_annotation_type=DemoRobot.STRETCH.annotation_type,
                world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(
                    *STRETCH_START
                ),
            ),
            RobotSpecification(
                semantic_annotation_type=DemoRobot.TIAGO.annotation_type,
                world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(*TIAGO_START),
            ),
        ],
    ).to_domain_object()


def wait_until_ready(robots: List[DemoRobot]) -> None:
    """
    Block until every robot's giskard is up, which it shows by taking goals.

    :param robots: The robots whose processes to wait for.
    :raises TimeoutError: If one of them does not come up in time.
    """
    for robot in robots:
        started = time.monotonic()
        deadline = started + READY_TIMEOUT
        while time.monotonic() < deadline:
            action_names = [
                name.lstrip("/")
                for name, _ in get_action_names_and_types(rospy.get_node())
            ]
            if robot.command_action_name in action_names:
                print(
                    f"{robot} is ready after {time.monotonic() - started:.1f}s",
                    flush=True,
                )
                break
            time.sleep(READY_POLL_INTERVAL)
        else:
            raise TimeoutError(f"{robot} did not come up within {READY_TIMEOUT:.0f}s")


# %% the demo


def main() -> None:
    """
    Serve the world, wait for the robot processes, open a window per robot and keep
    serving the world until interrupted.

    The windows are opened once the robots hold the world, so that their start does not
    compete with the robots' fetches.
    """
    rospy.init_node("shared_world")
    world = build_world()
    print(f"built the apartment and its robots: {len(world.bodies)} bodies", flush=True)

    WorldSynchronizer(_world=world, node=rospy.get_node())
    FetchWorldServer(node=rospy.get_node(), world=world)
    TFPublisher.create_with_ignore_existing_tf(node=rospy.get_node(), world=world)
    VizMarkerPublisher(node=rospy.get_node(), _world=world)
    robots = list(DemoRobot)
    processes = [start_robot_process(robot) for robot in robots]
    try:
        wait_until_ready(robots)
        processes.extend(start_joint_state_publisher(robot) for robot in robots)
        print("serving the world until interrupted", flush=True)
        signal.pause()
    finally:
        stop_processes(processes)


if __name__ == "__main__":
    main()
