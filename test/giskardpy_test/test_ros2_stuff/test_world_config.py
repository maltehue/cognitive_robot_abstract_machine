from typing import Iterator

import pytest
from rclpy.node import Node

from giskardpy.middleware.ros2 import rospy
from giskardpy.model.world_config import WorldFromFetchService
from semantic_digital_twin.adapters.ros.node_registry import ROSNodeRegistry
from semantic_digital_twin.adapters.ros.world_fetcher import FetchWorldServer
from semantic_digital_twin.world import World

# %% fixtures


@pytest.fixture()
def giskard_node() -> Iterator[Node]:
    """
    Start Giskard's ROS runtime for the test and stop it afterwards.
    """
    registry = ROSNodeRegistry()
    registry.clear()
    rospy.init_node("giskard_world_config_test")
    yield rospy.get_node()
    rospy.shutdown()
    registry.clear()


# %% fetching the world of another process


def test_the_fetched_world_holds_the_served_bodies(
    giskard_node: Node, mini_world: World
):
    server = FetchWorldServer(node=giskard_node, world=mini_world)
    config = WorldFromFetchService()

    config.setup_world()
    server.close()

    assert {body.id for body in config.world.bodies} == {
        body.id for body in mini_world.bodies
    }


def test_the_fetched_world_holds_the_served_connections(
    giskard_node: Node, mini_world: World
):
    server = FetchWorldServer(node=giskard_node, world=mini_world)
    config = WorldFromFetchService()

    config.setup_world()
    server.close()

    assert {connection.id for connection in config.world.connections} == {
        connection.id for connection in mini_world.connections
    }
