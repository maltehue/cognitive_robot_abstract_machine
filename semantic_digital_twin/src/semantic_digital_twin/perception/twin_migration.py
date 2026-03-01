from __future__ import annotations

"""Minimal prototype for migrating to Bodies and Connections.

This module introduces a very small adapter around the existing line-segment
prototype to express the model in terms of Semantic Digital Twin Bodies and an
implicit connection between them. It keeps the state minimal and 2D, with:

- body0: base pose variable (x, y, yaw)
- body1: pose derived from body0 by a relative angle and a vertical gap
- Each body carries a single 2D line-segment proxy of fixed length oriented
  along the body's +x axis in its local frame.

The inter-segment distance is measured strictly along the global y-axis with
zero x-offset between the nearest endpoints, as required by the current
prototype constraints.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types import Point3, Vector3

from .line_segment_model import LineSegment2D, _segment_angle, _vertical_endpoint_gap


@dataclass
class Pose2D:
    """Planar pose with translation and yaw.

    Represents the pose of a body in the world for the minimal 2D prototype.
    """

    x: float
    y: float
    yaw: float

    def rotate(self, vx: float, vy: float) -> Tuple[float, float]:
        """Rotate a 2D vector by this pose's yaw and return the new components."""

        c, s = float(np.cos(self.yaw)), float(np.sin(self.yaw))
        return c * vx - s * vy, s * vx + c * vy


@dataclass
class SegmentAttachment:
    """Associates a body with a fixed-length 2D line-segment proxy.

    The segment is oriented along the body's +x axis in its local frame. The
    segment's start point coincides with the body's local origin.
    """

    body: Body
    length: float


@dataclass
class TwinLineSegmentModel:
    """Reference model of two bodies connected by a relative angle and gap.

    This minimal prototype only supports two bodies with one segment each.
    """

    seg0: SegmentAttachment
    seg1: SegmentAttachment


@dataclass
class TwinState:
    """Numerical state for the twin model.

    - base: pose of body0 in the world.
    - relative_angle: orientation difference between body1 and body0 (added to yaw).
    - vertical_gap: distance between segment start points along global y, with
      zero x-offset (i.e., x1 == x0).
    """

    base: Pose2D
    relative_angle: float
    vertical_gap: float

    def body0_pose(self) -> Pose2D:
        return self.base

    def body1_pose(self) -> Pose2D:
        return Pose2D(
            x=self.base.x,
            y=self.base.y + float(max(0.0, self.vertical_gap)),
            yaw=self.base.yaw + self.relative_angle,
        )


def materialize_segments(
    model: TwinLineSegmentModel, state: TwinState
) -> List[LineSegment2D]:
    """Create global 2D segments for both bodies using the provided state.

    Each body's segment lies along its +x axis with the start point at the
    body's origin. Global positions are computed by rotating and translating
    by the body's planar pose.
    """

    def seg_from(body_pose: Pose2D, length: float) -> LineSegment2D:
        start = Point3(x=body_pose.x, y=body_pose.y, z=0.0)
        dx, dy = body_pose.rotate(length, 0.0)
        end = Point3(x=body_pose.x + dx, y=body_pose.y + dy, z=0.0)
        return LineSegment2D(start=start, end=end)

    p0 = state.body0_pose()
    p1 = state.body1_pose()
    return [
        seg_from(p0, model.seg0.length),
        seg_from(p1, model.seg1.length),
    ]


@dataclass
class TwinFitResult:
    """Result of fitting the twin model to two observed segments."""

    state: TwinState
    segments: List[LineSegment2D]
    residual: float


def fit_twin_model_to_segments(
    model: TwinLineSegmentModel, observed: List[LineSegment2D]
) -> TwinFitResult:
    """Estimate the minimal twin state from two observed segments.

    For the minimal 2D prototype:
    - Set base yaw and translation to align body0's segment with observed[0].
      The base origin is placed at observed[0].start and base yaw equals its
      segment angle.
    - Set relative_angle to match the observed relative angle between segments.
    - Set vertical_gap to the minimal vertical endpoint gap between the two
      observed segments (measured along global y), enforcing non-negativity.
    """

    assert len(observed) == 2, "Provide two observed segments"

    obs0, obs1 = observed
    yaw0 = _segment_angle(obs0)
    yaw1 = _segment_angle(obs1)
    rel = yaw1 - yaw0
    rel = (rel + np.pi) % (2 * np.pi) - np.pi
    gap = _vertical_endpoint_gap(obs0, obs1)

    base = Pose2D(x=float(obs0.start.x), y=float(obs0.start.y), yaw=float(yaw0))
    state = TwinState(
        base=base, relative_angle=float(rel), vertical_gap=float(max(0.0, gap))
    )

    segs = materialize_segments(model, state)

    # Residual: enforce fixed lengths by construction, compare relative angle and
    # vertical gap w.r.t. observations. Add small absolute angle term for seg0.
    rel_fit = _segment_angle(segs[1]) - _segment_angle(segs[0])
    rel_fit = (rel_fit + np.pi) % (2 * np.pi) - np.pi
    gap_fit = _vertical_endpoint_gap(segs[0], segs[1])

    def angdiff(a: float, b: float) -> float:
        return float((a - b + np.pi) % (2 * np.pi) - np.pi)

    res = (
        angdiff(rel_fit, rel) ** 2
        + (gap_fit - gap) ** 2
        + 0.1 * angdiff(_segment_angle(segs[0]), yaw0) ** 2
    )

    return TwinFitResult(state=state, segments=segs, residual=float(res))


__all__ = [
    "Pose2D",
    "SegmentAttachment",
    "TwinLineSegmentModel",
    "TwinState",
    "TwinFitResult",
    "materialize_segments",
    "fit_twin_model_to_segments",
]
