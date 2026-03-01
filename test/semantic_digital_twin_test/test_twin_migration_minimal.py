import numpy as np

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.perception import (
    LineSegment2D,
    Pose2D,
    SegmentAttachment,
    TwinLineSegmentModel,
    fit_twin_model_to_segments,
    publish_fitted_and_observed,
)
from semantic_digital_twin.perception.line_segment_model import (
    _segment_angle,
    _vertical_endpoint_gap,
)


def make_segment_from_xy(x0, y0, x1, y1) -> LineSegment2D:
    return LineSegment2D(
        start=Point3(x=x0, y=y0, z=0.0),
        end=Point3(x=x1, y=y1, z=0.0),
    )


def test_twin_migration_two_bodies_minimal(rclpy_node):
    # Create two minimal bodies (no world required for this prototype)
    node = rclpy_node
    b0 = Body(name=PrefixedName("body0"))
    b1 = Body(name=PrefixedName("body1"))

    # Fixed lengths for the two body-attached segments
    seg0 = SegmentAttachment(body=b0, length=2.0)
    seg1 = SegmentAttachment(body=b1, length=1.3)

    model = TwinLineSegmentModel(seg0=seg0, seg1=seg1)

    # Observed segments: arbitrary but consistent geometry
    observed = [
        make_segment_from_xy(0.2, 0.2, 1.0, 0.1),
        make_segment_from_xy(0.75, 0.4, 1.75, 0.6),
    ]

    result = fit_twin_model_to_segments(model, observed)
    segs = result.segments

    publish_fitted_and_observed(node, fitted_segments=segs, observed_segments=observed)
    # Lengths remain fixed by construction
    assert np.isclose(segs[0].length, seg0.length, atol=1e-9)
    assert np.isclose(segs[1].length, seg1.length, atol=1e-9)

    # Base pose aligns segment 0 to observed[0]
    assert np.isclose(result.state.base.x, observed[0].start.x, atol=1e-9)
    assert np.isclose(result.state.base.y, observed[0].start.y, atol=1e-9)
    assert np.isclose(result.state.base.yaw, _segment_angle(observed[0]), atol=1e-9)

    # Relative angle and vertical gap should match observations
    obs_rel = _segment_angle(observed[1]) - _segment_angle(observed[0])
    obs_rel = (obs_rel + np.pi) % (2 * np.pi) - np.pi
    fit_rel = _segment_angle(segs[1]) - _segment_angle(segs[0])
    fit_rel = (fit_rel + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(fit_rel, obs_rel, atol=5e-2)

    obs_gap = _vertical_endpoint_gap(observed[0], observed[1])
    fit_gap = _vertical_endpoint_gap(segs[0], segs[1])
    assert np.isclose(fit_gap, obs_gap, atol=5e-2)

    # Residual should be very small for this consistent setup
    assert result.residual < 1e-3
