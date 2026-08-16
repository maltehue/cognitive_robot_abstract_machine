"""
Watching an in-process run in RViz.

Attaches the tf and marker publishers to a world and spins a node for them in the
background, so an executor ticked in this process is visible live. In RViz: add a
MarkerArray display on ``/semworld/viz_marker`` with transient-local durability, and set
the fixed frame to the world root.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.world import World


@dataclass
class WorldVisualization:
    """
    The publishers and the node that make a world's run visible in RViz.
    """

    node: Node
    """The node the publishers live on, spun in a background thread."""

    tf_publisher: TFPublisher
    """
    Publishes the tf tree on every state change, which each control tick causes.
    """

    marker_publisher: VizMarkerPublisher
    """Publishes the world's bodies as markers, positioned through the tf tree."""

    _ros_executor: SingleThreadedExecutor = field(repr=False)
    """
    Spins the node so the publishers' messages leave the process.
    """

    @classmethod
    def attach(
        cls, world: World, node_name: str = "knowledge_servoing_visualization"
    ) -> WorldVisualization:
        """
        Attaches live visualization to a world.

        :param world: The world whose run should be visible.
        :param node_name: Name of the node the publishers live on.
        :return: The attached visualization; keep it referenced while the run lasts.
        """
        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node(node_name)
        ros_executor = SingleThreadedExecutor()
        ros_executor.add_node(node)
        threading.Thread(
            target=ros_executor.spin, daemon=True, name=f"{node_name}-spin"
        ).start()
        return cls(
            node=node,
            tf_publisher=TFPublisher(_world=world, node=node),
            marker_publisher=VizMarkerPublisher(_world=world, node=node),
            _ros_executor=ros_executor,
        )

    def close(self) -> None:
        """
        Stops spinning and destroys the node; the world outlives its visualization.
        """
        self._ros_executor.shutdown()
        self.node.destroy_node()
