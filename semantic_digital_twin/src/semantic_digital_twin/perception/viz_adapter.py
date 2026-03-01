from __future__ import annotations

from ..adapters.ros.tf_publisher import TFPublisher

"""ROS2 VizMarkerPublisher adapter for segment-based perception prototypes.

This module provides small utilities to visualize the fitted reference model
and the perception input using the Semantic Digital Twin's VizMarkerPublisher.

It converts 2D line segments into temporary Bodies with thin Box shapes inside
an ephemeral World and publishes them as MarkerArray messages. This keeps the
visualization consistent with the rest of the project.
"""

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.geometry import Box, Color, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)

from .line_segment_model import LineSegment2D


@dataclass
class SegmentVizConfig:
    """Styling configuration for rendering segments as thin boxes.

    Attributes define the thickness and color for a group of segments.
    """

    color: Color = field(default_factory=lambda: Color(0.1, 0.4, 0.8, 0.9))
    width_y: float = 0.02
    thickness_z: float = 0.01
    z_layer: float = 0.0
    name_prefix: str = "seg"


def _segment_midpoint_and_yaw(seg: LineSegment2D) -> tuple[float, float, float]:
    dx = float(seg.end.x) - float(seg.start.x)
    dy = float(seg.end.y) - float(seg.start.y)
    mx = 0.5 * (float(seg.start.x) + float(seg.end.x))
    my = 0.5 * (float(seg.start.y) + float(seg.end.y))
    yaw = float(np.arctan2(dy, dx))
    return mx, my, yaw


def _body_with_box(name: str, length: float, cfg: SegmentVizConfig) -> Body:
    box = Box(
        scale=Scale(x=length, y=cfg.width_y, z=cfg.thickness_z),
        color=cfg.color,
        # Origin at body origin; orientation handled by body pose via connection
        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ),
    )
    shapes = ShapeCollection([box])
    body = Body(name=PrefixedName(name), visual=shapes, collision=ShapeCollection())
    return body


def build_world_from_segments(
    segments: Iterable[LineSegment2D],
    *,
    cfg: Optional[SegmentVizConfig] = None,
    world_name: str = "segment_viz_world",
) -> World:
    """Create a transient World containing a root and one child Body per segment.

    Each segment is represented as a thin Box aligned with the segment.
    """

    cfg = cfg or SegmentVizConfig()
    world = World(name=world_name)

    # Root body at world origin
    root = Body(name=PrefixedName(f"{cfg.name_prefix}_root"))
    world.add_kinematic_structure_entity(root)

    for idx, seg in enumerate(segments):
        length = float(seg.length)
        child = _body_with_box(f"{cfg.name_prefix}_{idx}", length, cfg)
        world.add_kinematic_structure_entity(child)
        # Attach child to root with proper pose
        mx, my, yaw = _segment_midpoint_and_yaw(seg)
        T = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=mx, y=my, z=cfg.z_layer, roll=0.0, pitch=0.0, yaw=yaw
        )
        conn = Connection6DoF.create_with_dofs(
            world=world,
            parent=root,
            child=child,
            name=PrefixedName(f"{cfg.name_prefix}_conn_{idx}"),
        )
        world.add_connection(conn)
        conn.origin = T

    world.validate()
    return world


def publish_segments_with_viz_marker(
    node,
    segments: Iterable[LineSegment2D],
    *,
    topic_name: Optional[str] = None,
    cfg: Optional[SegmentVizConfig] = None,
) -> VizMarkerPublisher:
    """Publish a set of segments via VizMarkerPublisher as thin boxes.

    Returns the constructed VizMarkerPublisher instance. The publisher calls
    notify in its initializer, so markers should appear immediately in RViz.
    """

    world = build_world_from_segments(segments, cfg=cfg)
    topic = topic_name or "/semworld/segments_viz"
    viz = VizMarkerPublisher(_world=world, node=node, topic_name=topic)
    return viz


def publish_fitted_and_observed(
    node,
    *,
    fitted_segments: List[LineSegment2D],
    observed_segments: List[LineSegment2D],
    topic_name: Optional[str] = None,
) -> VizMarkerPublisher:
    """Publish both fitted model segments and observed segments using VizMarkerPublisher.

    Fitted segments and observations are rendered in different colors and
    separated by a small z-layer offset to avoid z-fighting.
    """

    cfg_model = SegmentVizConfig(
        color=Color(R=0.1, G=0.4, B=0.9, A=0.95),
        width_y=0.025,
        thickness_z=0.01,
        z_layer=0.01,
        name_prefix="model",
    )
    cfg_obs = SegmentVizConfig(
        color=Color(R=0.9, G=0.3, B=0.2, A=0.95),
        width_y=0.02,
        thickness_z=0.01,
        z_layer=0.0,
        name_prefix="obs",
    )

    # Build one world that contains both groups under a single root
    world = World(name="fitted_vs_observed")
    root = Body(name=PrefixedName("viz_root"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    def add_group(group: List[LineSegment2D], cfg: SegmentVizConfig) -> None:
        for idx, seg in enumerate(group):
            length = float(seg.length)
            child = _body_with_box(f"{cfg.name_prefix}_{idx}", length, cfg)
            world.add_kinematic_structure_entity(child)
            mx, my, yaw = _segment_midpoint_and_yaw(seg)
            T = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=mx, y=my, z=cfg.z_layer, roll=0.0, pitch=0.0, yaw=yaw
            )
            conn = Connection6DoF.create_with_dofs(
                world=world,
                parent=root,
                child=child,
                name=PrefixedName(f"{cfg.name_prefix}_conn_{idx}"),
            )
            world.add_connection(conn)
            conn.origin = T

    with world.modify_world():
        add_group(fitted_segments, cfg_model)
        add_group(observed_segments, cfg_obs)

    world.validate()
    topic = topic_name or "/semworld/fitted_vs_observed"
    tf_publisher = TFPublisher(node=node, _world=world)
    viz = VizMarkerPublisher(_world=world, node=node, topic_name=topic)
    return viz


__all__ = [
    "SegmentVizConfig",
    "build_world_from_segments",
    "publish_segments_with_viz_marker",
    "publish_fitted_and_observed",
]
