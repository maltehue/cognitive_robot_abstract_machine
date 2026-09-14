import logging
from dataclasses import dataclass
from typing import Any

import pytest

import giskardpy.middleware.ros2.python_interface
from coraplex.datastructures.dataclasses import Context
from semantic_digital_twin.world import World

# %% debug validation


def test_debug_requires_a_ros_node(immutable_model_world):
    """
    Debug output is visualized over ROS, so a context constructed in debug mode without
    a node is rejected at construction rather than failing later during execution.
    """
    world, robot, _ = immutable_model_world

    with pytest.raises(ValueError):
        Context(world, robot, _debug=True)


def test_debug_raises_the_coraplex_log_level(immutable_model_world, rclpy_node):
    """
    Constructing a context in debug mode lowers the package's log level, so debug
    messages are emitted without the caller touching logging.
    """
    world, robot, _ = immutable_model_world
    coraplex_logger = logging.getLogger("coraplex")
    previous_level = coraplex_logger.level

    try:
        Context(world, robot, ros_node=rclpy_node, _debug=True)
        assert coraplex_logger.level == logging.DEBUG
    finally:
        coraplex_logger.setLevel(previous_level)


def test_default_context_logs_at_info(immutable_model_world):
    """
    Without debug mode the package logs at info level.
    """
    world, robot, _ = immutable_model_world
    coraplex_logger = logging.getLogger("coraplex")
    previous_level = coraplex_logger.level

    try:
        context = Context(world, robot)
        assert not context.debug
        assert coraplex_logger.level == logging.INFO
    finally:
        coraplex_logger.setLevel(previous_level)


# %% the giskard the context talks to


@dataclass
class WrapperRecordingItsArguments:
    """
    Stands in for the wrapper a context builds, recording what it was built with.
    """

    node_handle: Any
    """
    The node the context handed over.
    """

    giskard_node_name: str
    """
    The name of the giskard node the context handed over.
    """

    world: World
    """
    The world the context handed over.
    """


def test_the_context_talks_to_the_giskard_it_names(
    immutable_model_world, rclpy_node, monkeypatch
):
    """
    A process driving several robots reaches each of them through the giskard node of
    that robot, so the name belongs to the context rather than to the wrapper's default.
    """
    world, robot, _ = immutable_model_world
    monkeypatch.setattr(
        giskardpy.middleware.ros2.python_interface,
        "GiskardWrapper",
        WrapperRecordingItsArguments,
    )
    context = Context(
        world, robot, ros_node=rclpy_node, giskard_node_name="giskard_stretch"
    )

    wrapper = context.giskard_wrapper

    assert wrapper.giskard_node_name == context.giskard_node_name
    assert wrapper.world is world
    assert wrapper.node_handle is rclpy_node


def test_the_context_talks_to_a_giskard_named_giskard_by_default(immutable_model_world):
    world, robot, _ = immutable_model_world

    assert Context(world, robot).giskard_node_name == "giskard"
