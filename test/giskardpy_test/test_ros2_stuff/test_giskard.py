"""
What a giskard makes of a world it shares with other processes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from giskardpy.data_types.exceptions import NoControlledJointsError
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    OneRobotOfManyInterface,
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.model.world_config import WorldConfig
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection

# %% a giskard over a world built elsewhere


@dataclass
class GivenWorld(WorldConfig):
    """
    Holds a world that was built before the giskard, and builds nothing itself.
    """

    def setup_world(self) -> None:
        return


def giskard_over(
    world: World,
    robot_interface_config: RobotInterfaceConfig,
    server_config: GiskardServerConfig,
) -> Giskard:
    """
    A giskard holding the given world, not yet set up.
    """
    return Giskard(
        world_config=GivenWorld(world=world),
        robot_interface_config=robot_interface_config,
        server_config=server_config,
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )


# %% the sanity check counts the controlled joints of every robot


def release_every_joint(world: World) -> None:
    """
    Leave no active connection of the given world flagged as controlled, as in a world
    whose robots are controlled by other processes.
    """
    for connection in world.get_connections_by_type(ActiveConnection):
        connection.has_hardware_interface = False


def test_a_giskard_holding_a_later_robot_of_the_world_passes_the_sanity_check(
    init_rospy, world_with_two_robots: World
):
    """
    A giskard that controls a robot other than the world's first one controls joints all
    the same, so the check must look at every robot rather than at the first.
    """
    release_every_joint(world_with_two_robots)
    giskard = giskard_over(
        world_with_two_robots,
        StandAloneRobotInterfaceConfig([]),
        GiskardServerConfig(),
    )
    later_robot = [robot for robot in giskard.robots if robot is not giskard.robot][0]
    interface = OneRobotOfManyInterface(robot_type=type(later_robot))
    for connection in interface.connections_to_control(world_with_two_robots):
        connection.has_hardware_interface = True

    giskard._controlled_joints_sanity_check()


def test_a_giskard_controlling_no_joint_of_any_robot_fails_the_sanity_check(
    init_rospy, world_with_two_robots: World
):
    release_every_joint(world_with_two_robots)
    giskard = giskard_over(
        world_with_two_robots,
        StandAloneRobotInterfaceConfig([]),
        GiskardServerConfig(),
    )

    with pytest.raises(NoControlledJointsError):
        giskard._controlled_joints_sanity_check()


# %% serving and drawing the world is left to its owner


def test_a_giskard_that_does_not_publish_its_world_serves_and_draws_nothing(
    init_rospy, mini_world: World
):
    """
    A giskard that fetched its world leaves serving and drawing it to the process it
    fetched from.
    """
    giskard = giskard_over(
        mini_world,
        StandAloneRobotInterfaceConfig([]),
        GiskardServerConfig(publishes_world=False),
    )

    giskard.setup_world_model_ros_interface()
    try:
        assert (
            giskard.world_fetcher,
            giskard.tf_publisher,
            giskard.viz_marker_publisher,
        ) == (None, None, None)
    finally:
        giskard.close_world_model_ros_interface()


def test_a_giskard_serves_and_draws_its_world_by_default(init_rospy, mini_world: World):
    giskard = giskard_over(
        mini_world, StandAloneRobotInterfaceConfig([]), GiskardServerConfig()
    )

    giskard.setup_world_model_ros_interface()
    try:
        assert isinstance(giskard.world_fetcher, FetchWorldServer)
        assert isinstance(giskard.tf_publisher, TFPublisher)
        assert isinstance(giskard.viz_marker_publisher, VizMarkerPublisher)
    finally:
        giskard.close_world_model_ros_interface()
