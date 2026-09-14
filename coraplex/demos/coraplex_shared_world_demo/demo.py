"""
A PR2 and a Stretch share one apartment, each driven by a giskard process of its own.

This process owns the world: it builds it, serves it, draws it, and performs a plan per
robot on top of it. Start it first, then one ``robot.py`` per robot::

    python demo.py
    python robot.py --robot PR2
    python robot.py --robot Stretch

Every robot process fetches this world and controls its own robot in it, so the robots
see each other while each of them is moved by a controller of its own. The plans are
performed one after the other first and then all at once.
"""

from __future__ import annotations

import signal
import time
from pathlib import Path

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

# %% building and serving the world


def build_world() -> World:
    """
    The apartment with both robots standing in it.
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
                print(f"{robot} is ready after {time.monotonic() - started:.1f}s")
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
    Serve the world, wait for the robot processes and perform a plan per robot.
    """
    rospy.init_node("shared_world")
    world = build_world()
    print(f"built the apartment and its robots: {len(world.bodies)} bodies")

    WorldSynchronizer(
        _world=world, node=rospy.get_node(), queue_depth=WORLD_SYNC_QUEUE_DEPTH
    )
    FetchWorldServer(node=rospy.get_node(), world=world)
    TFPublisher.create_with_ignore_existing_tf(node=rospy.get_node(), world=world)
    VizMarkerPublisher(node=rospy.get_node(), _world=world)
    print("serving the world; start one robot.py per robot now")

    robots = [DemoRobot.PR2, DemoRobot.STRETCH]
    wait_until_ready(robots)

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
        print(f"one robot after the other took {time.monotonic() - started:.1f}s")

        started = time.monotonic()
        ConcurrentPlans(
            plans=[
                park_arms_and_drive(contexts[DemoRobot.PR2]),
                park_arms(contexts[DemoRobot.STRETCH]),
            ]
        ).perform()
        print(f"both robots at once took {time.monotonic() - started:.1f}s")

    print("done; serving the world until interrupted")
    signal.pause()


main()
