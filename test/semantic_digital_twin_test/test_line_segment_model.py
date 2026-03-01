import numpy as np

from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.perception.line_segment_model import (
    Adjacency,
    LineSegment2D,
    LineSegmentObjectModel,
    fit_model_to_perception_data,
    _segment_angle,
    _vertical_endpoint_gap,
)


def make_segment_from_xy(x0, y0, x1, y1) -> LineSegment2D:
    return LineSegment2D(
        start=Point3(x=x0, y=y0, z=0.0),
        end=Point3(x=x1, y=y1, z=0.0),
    )


def test_two_segment_object():
    adjacency = Adjacency()
    adjacency.add_edge(0, 1)
    model = LineSegmentObjectModel(fixed_lengths=[2.0, 2.0], adjacency=adjacency)

    perceived_segments = [
        make_segment_from_xy(0.1, 0.1, 2.3, 0.2),
        make_segment_from_xy(0.75, 0.3, 1.75, 2.1),
    ]

    result = fit_model_to_perception_data(model, perceived_segments)
    fitted = result.model

    segs = fitted.compute_geometry()
    seg0, seg1 = segs[0], segs[1]

    assert np.isclose(seg0.length, 2.0, atol=1e-9)
    assert np.isclose(seg1.length, 2.0, atol=1e-9)

    obs_rel = _segment_angle(perceived_segments[1]) - _segment_angle(
        perceived_segments[0]
    )
    obs_rel = (obs_rel + np.pi) % (2 * np.pi) - np.pi
    fit_rel = _segment_angle(seg1) - _segment_angle(seg0)
    fit_rel = (fit_rel + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(fit_rel, obs_rel, atol=5e-2)

    obs_gap = _vertical_endpoint_gap(perceived_segments[0], perceived_segments[1])
    fit_gap = _vertical_endpoint_gap(seg0, seg1)
    assert np.isclose(fit_gap, obs_gap, atol=5e-2)

    assert result.residual < 1.0
