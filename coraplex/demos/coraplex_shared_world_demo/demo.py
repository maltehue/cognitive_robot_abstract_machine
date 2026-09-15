"""
A PR2, a Stretch and a Tiago share one apartment, each held by a giskard process of its
own.

This process owns the world: it builds it, serves it, draws it, starts one ``robot.py``
per robot and performs a plan per commanded robot on top of it::

    python demo.py

Every robot process fetches this world and holds its own robot in it, so the robots see
each other while each of them is moved by a source of its own. Their output goes to a log
file each, whose place is printed, and every started process is stopped with this one.

It shows the three ways a robot's state reaches this process:

- the PR2 and the Stretch are commanded by the plans below, one after the other first
  and then both at once,
- either of them can be dragged about by the handles of its ``marker.py``,
- the Tiago is moved by nobody here: a joint state publisher window opens for it, and
  its giskard writes whatever that window reports into this world. Its base stays where
  the world put it, because that window reports joint positions and no base pose.

Watching and dragging it in RViz
--------------------------------

Set the fixed frame to ``apartment/apartment_root`` and add

- a ``MarkerArray`` display on ``/semworld/viz_marker`` for the world itself,
- an ``InteractiveMarkers`` display per commanded robot, with update topic
  ``/giskard_pr2/cartesian_goals/update`` and ``/giskard_stretch/cartesian_goals/update``.

Dragging a handle and releasing it moves that robot through its own giskard, and this
process draws what happened. The Stretch's base is a differential drive and cannot be
dragged sideways: a lateral goal converges slowly or runs into the marker's timeout,
which ends it cleanly.

..note:: A marker goal reaches a robot's giskard directly and is therefore not
    serialized against the plans this process performs. Dragging a robot while its own
    plan runs leaves the two goals to giskard, and a model change during a marker motion
    would abort that motion; the plans below change no model.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

from rclpy.action import get_action_names_and_types
from typing_extensions import List

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import real_robot
from coraplex.plans.factories import sequential
from coraplex.plans.plan import ConcurrentPlans, Plan
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
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
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World

# %% where everything stands

APARTMENT_PATH = (
    Path(__file__).resolve().parents[2] / "resources" / "worlds" / "apartment.urdf"
)
"""
The environment the robots share.
"""

PR2_START = (1.5, 2.5)
"""
Where the PR2 stands, in the corridor in front of the kitchen counter.
"""

STRETCH_START = (1.5, 1.2)
"""
Where the Stretch stands, a good arm's length away from the PR2.
"""

TIAGO_START = (1.5, 3.8)
"""
Where the Tiago stands, further up the corridor and clear of the PR2's drive.
"""

PR2_NAVIGATION_TARGET = (2.0, 2.5)
"""
Where the PR2 drives while the Stretch parks its arm.
"""

WORLD_SYNC_QUEUE_DEPTH = 1000
"""
How many updates the synchronizer of this world queues.

Every robot process publishes on every control cycle, while a receiver applies one
update at a time and holds its world for a model change; the default queue of ten would
overflow and the lost update would never be sent again.
"""

READY_TIMEOUT = 600.0
"""
How long to wait for the robot processes.

Each of them fetches an apartment of a few hundred bodies, and this process answers one
fetch after the other.
"""

READY_POLL_INTERVAL = 1.0
"""
How often to look whether a robot process is accepting goals.
"""

ROBOT_LAUNCHER = Path(__file__).with_name("robot.py")
"""
The script started once per robot.
"""

MARKER_LAUNCHER = Path(__file__).with_name("marker.py")
"""
The interactive marker started once per commanded robot.
"""

ROS_RUN_COMMAND = ["ros2", "run"]
"""
The command that runs one executable of an installed ROS package.
"""

JOINT_STATE_PUBLISHER_PACKAGE = "joint_state_publisher_gui"
"""
The package holding the window a mirrored robot is moved from.
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


def marker_process_command(robot: DemoRobot) -> List[str]:
    """
    The command that starts the interactive marker of one robot, in this interpreter.

    :param robot: The robot the marker offers handles for.
    """
    return [sys.executable, str(MARKER_LAUNCHER), "--robot", str(robot)]


def joint_state_publisher_command(robot: DemoRobot) -> List[str]:
    """
    The command that opens the window one robot's joint positions are reported from.

    The window reads the robot's own description file, so it offers a slider per movable
    joint of it, and publishes under the namespace the robot's topic names.

    :param robot: The robot whose joint states are reported.
    """
    return [
        *ROS_RUN_COMMAND,
        JOINT_STATE_PUBLISHER_PACKAGE,
        JOINT_STATE_PUBLISHER_EXECUTABLE,
        str(CompositePathResolver().resolve(robot.annotation_type.get_ros_file_path())),
        ROS_ARGUMENTS_FLAG,
        REMAP_FLAG,
        f"__ns:={PurePosixPath(robot.joint_states_topic).parent}",
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


def start_marker_process(robot: DemoRobot) -> subprocess.Popen:
    """
    Start the interactive marker of one robot.

    :param robot: The robot the marker offers handles for.
    """
    log_path = LOG_DIRECTORY / f"{robot}_marker.log"
    process = start_process(marker_process_command(robot), log_path)
    print(f"started the {robot} marker, logging to {log_path}", flush=True)
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
                semantic_annotation_type=DemoRobot.PR2.annotation_type,
                world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(*PR2_START),
            ),
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
    Block until every robot's giskard takes goals.

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


# %% the plans


def park_arms(context: Context) -> Plan:
    """
    A plan that brings both arms of the context's robot into their park pose.
    """
    return sequential([ParkArmsAction(Arms.BOTH)], context=context).plan


def park_arms_and_drive(context: Context) -> Plan:
    """
    A plan that parks both arms of the context's robot and then drives it aside.

    The target keeps the height the robot's root stands at, because a base goal of a
    robot driven through giskard is a pose of that root rather than a point on the
    floor.
    """
    root_height = context.robot.root.global_pose.to_position().to_np()[2]
    return sequential(
        [
            ParkArmsAction(Arms.BOTH),
            NavigateAction(
                Pose.from_xyz_rpy(
                    *PR2_NAVIGATION_TARGET,
                    root_height,
                    reference_frame=context.world.root,
                )
            ),
        ],
        context=context,
    ).plan


# %% the demo


def main() -> None:
    """
    Serve the world, wait for the robot processes, offer a handle per commanded robot,
    open the window the mirrored one is moved from and perform a plan per commanded
    robot.

    The markers and the window are started once the robots hold the world, so that their
    fetch does not queue behind the robots' own.
    """
    rospy.init_node("shared_world")
    world = build_world()
    print(f"built the apartment and its robots: {len(world.bodies)} bodies", flush=True)

    WorldSynchronizer(
        _world=world, node=rospy.get_node(), queue_depth=WORLD_SYNC_QUEUE_DEPTH
    )
    FetchWorldServer(node=rospy.get_node(), world=world)
    TFPublisher.create_with_ignore_existing_tf(node=rospy.get_node(), world=world)
    VizMarkerPublisher(node=rospy.get_node(), _world=world)
    robots = [DemoRobot.PR2, DemoRobot.STRETCH, DemoRobot.TIAGO]
    commanded_robots = [robot for robot in robots if not robot.is_mirrored]
    processes = [start_robot_process(robot) for robot in robots]
    try:
        wait_until_ready(robots)
        for robot in commanded_robots:
            processes.append(start_marker_process(robot))
        for robot in robots:
            if robot.is_mirrored:
                processes.append(start_joint_state_publisher(robot))
        perform_the_plans(world, commanded_robots)
    finally:
        stop_processes(processes)


def perform_the_plans(world: World, robots: List[DemoRobot]) -> None:
    """
    Perform a plan per robot, then keep serving the world until interrupted.

    :param world: The served world.
    :param robots: The robots the plans command.
    """
    contexts = {
        robot: Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(robot.annotation_type)[0],
            ros_node=rospy.get_node(),
            giskard_node_name=robot.giskard_node_name,
            evaluate_conditions=False,
        )
        for robot in robots
    }

    with real_robot:
        started = time.monotonic()
        park_arms_and_drive(contexts[DemoRobot.PR2]).perform()
        park_arms(contexts[DemoRobot.STRETCH]).perform()
        print(
            f"one robot after the other took {time.monotonic() - started:.1f}s",
            flush=True,
        )

        started = time.monotonic()
        ConcurrentPlans(
            plans=[
                park_arms_and_drive(contexts[DemoRobot.PR2]),
                park_arms(contexts[DemoRobot.STRETCH]),
            ]
        ).perform()
        print(f"both robots at once took {time.monotonic() - started:.1f}s", flush=True)

    print("done; serving the world until interrupted", flush=True)
    signal.pause()


if __name__ == "__main__":
    main()
