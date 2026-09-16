"""
What a giskard makes of a world it shares with other processes.
"""

from __future__ import annotations

import pytest

from giskardpy.data_types.exceptions import (
    NoControlledJointsError,
    RobotNotInWorldError,
)
from giskardpy.middleware.ros2.giskard import Giskard
from giskardpy.middleware.ros2.robot_interface_config import (
    RobotInterfaceConfig,
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.model.world_config import WorldConfig, WorldFromFetchService
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.robots.tiago import Tiago
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import ActiveConnection

# %% a giskard over a world built elsewhere


def giskard_over(
    world_config: WorldConfig,
    robot_interface_config: RobotInterfaceConfig,
    server_config: GiskardServerConfig,
) -> Giskard:
    """
    A giskard holding the given world config, not yet set up.
    """
    return Giskard(
        world_config=world_config,
        robot_interface_config=robot_interface_config,
        server_config=server_config,
        qp_controller_config=QPControllerConfig(target_frequency=50),
    )


def later_robot_of(world: World) -> AbstractRobot:
    """
    A robot of the given world that is not its first.
    """
    return world.get_semantic_annotations_by_type(AbstractRobot)[1]


# %% the giskard controls the robot its world config names


def test_the_giskards_robot_is_the_one_of_the_type_its_world_config_names(
    world_with_two_robots: World,
):
    later_robot = later_robot_of(world_with_two_robots)
    giskard = giskard_over(
        WorldFromFetchService(
            world=world_with_two_robots, robot_type=type(later_robot)
        ),
        StandAloneRobotInterfaceConfig([]),
        GiskardServerConfig(),
    )

    assert giskard.robot is later_robot


def test_a_world_config_naming_a_robot_that_is_not_there_says_so(
    world_with_two_robots: World,
):
    giskard = giskard_over(
        WorldFromFetchService(world=world_with_two_robots, robot_type=Tiago),
        StandAloneRobotInterfaceConfig([]),
        GiskardServerConfig(),
    )

    with pytest.raises(RobotNotInWorldError):
        giskard.robot


# %% the sanity check counts the joints of that robot


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
    the same, so the check must count that robot's joints rather than the first one's.
    """
    release_every_joint(world_with_two_robots)
    later_robot = later_robot_of(world_with_two_robots)
    for connection in later_robot.connections:
        if isinstance(connection, ActiveConnection):
            connection.has_hardware_interface = True
    giskard = giskard_over(
        WorldFromFetchService(
            world=world_with_two_robots, robot_type=type(later_robot)
        ),
        StandAloneRobotInterfaceConfig([]),
        GiskardServerConfig(),
    )

    giskard._controlled_joints_sanity_check()


def test_a_giskard_controlling_no_joint_of_its_robot_fails_the_sanity_check(
    init_rospy, world_with_two_robots: World
):
    release_every_joint(world_with_two_robots)
    giskard = giskard_over(
        WorldFromFetchService(world=world_with_two_robots),
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
        WorldFromFetchService(world=mini_world),
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
