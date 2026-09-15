"""
What an interactive marker node reads before it offers a handle.

The settings decide which giskard the goals go to and under which topics the handles
appear, so several marker nodes can serve one world without sharing either.
"""

from __future__ import annotations

import pytest
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

from giskardpy.middleware.ros2.scripts.tools.interactive_marker import (
    InteractiveMarkerSettings,
    MarkerParameter,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.exceptions import DuplicateWorldEntityError
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% what the nodes under test are given

ROOT_LINKS = ["apartment_root", "base_footprint"]
"""
The roots the marker node is told to offer handles against.
"""

TIP_LINKS = ["base_footprint", "r_gripper_tool_frame"]
"""
The tips belonging to :data:`ROOT_LINKS`.
"""

GISKARD_NODE_NAME = "giskard_stretch"
"""
The giskard the overriding node sends its goals to.
"""

MARKER_NAMESPACE = "giskard_stretch/cartesian_goals"
"""
The topic namespace the overriding node's handles appear under.
"""


def link_parameters() -> list[Parameter]:
    """
    The two parameters every marker node needs.
    """
    return [
        Parameter(
            MarkerParameter.ROOT_LINKS.value, Parameter.Type.STRING_ARRAY, ROOT_LINKS
        ),
        Parameter(
            MarkerParameter.TIP_LINKS.value, Parameter.Type.STRING_ARRAY, TIP_LINKS
        ),
    ]


@pytest.fixture
def node_naming_its_giskard(rclpy_node: Node) -> Node:
    """
    A node that overrides all four parameters.
    """
    node = rclpy.create_node(
        "marker_naming_its_giskard",
        parameter_overrides=link_parameters()
        + [
            Parameter(
                MarkerParameter.GISKARD_NODE_NAME.value,
                Parameter.Type.STRING,
                GISKARD_NODE_NAME,
            ),
            Parameter(
                MarkerParameter.MARKER_NAMESPACE.value,
                Parameter.Type.STRING,
                MARKER_NAMESPACE,
            ),
        ],
    )
    yield node
    node.destroy_node()


@pytest.fixture
def node_naming_only_its_links(rclpy_node: Node) -> Node:
    """
    A node that leaves the giskard and the namespace to their defaults.
    """
    node = rclpy.create_node(
        "marker_naming_only_its_links", parameter_overrides=link_parameters()
    )
    yield node
    node.destroy_node()


# %% reading the settings off a node


def test_settings_are_read_from_the_nodes_parameters(node_naming_its_giskard: Node):
    """
    A marker node that names its giskard and its namespace serves that giskard under
    that namespace.
    """
    settings = InteractiveMarkerSettings.from_node(node_naming_its_giskard)

    assert settings.root_links == ROOT_LINKS
    assert settings.tip_links == TIP_LINKS
    assert settings.giskard_node_name == GISKARD_NODE_NAME
    assert settings.marker_namespace == MARKER_NAMESPACE


def test_a_lone_marker_serves_the_plain_giskard(node_naming_only_its_links: Node):
    """
    A marker node that names neither talks to the giskard called ``giskard`` under the
    namespace RViz's own recipes expect.
    """
    settings = InteractiveMarkerSettings.from_node(node_naming_only_its_links)

    assert settings.giskard_node_name == "giskard"
    assert settings.marker_namespace == "cartesian_goals"


# %% naming a link of a world holding several robots


@pytest.fixture
def world_with_two_base_links() -> World:
    """
    A world in which the plain name ``base_link`` means either of two bodies.
    """
    world = World()
    first = Body(name=PrefixedName(name="base_link", prefix="a"))
    second = Body(name=PrefixedName(name="base_link", prefix="b"))
    with world.modify_world():
        world.add_kinematic_structure_entity(first)
        world.add_kinematic_structure_entity(second)
        world.add_connection(FixedConnection(parent=first, child=second))
    return world


def test_a_prefix_picks_one_of_two_equally_named_links(
    world_with_two_base_links: World,
):
    """
    A link name carrying a prefix names exactly the body of that prefix.
    """
    link = InteractiveMarkerSettings.link_named(
        world_with_two_base_links, "a/base_link"
    )

    assert link.name == PrefixedName(name="base_link", prefix="a")


def test_an_ambiguous_link_name_is_refused(world_with_two_base_links: World):
    """
    A plain link name that several bodies carry is an error rather than a guess.
    """
    with pytest.raises(DuplicateWorldEntityError):
        InteractiveMarkerSettings.link_named(world_with_two_base_links, "base_link")
