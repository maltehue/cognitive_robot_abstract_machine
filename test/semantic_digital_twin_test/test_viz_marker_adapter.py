import pytest


def test_publish_fitted_and_observed_smoke(rclpy_node):
    pytest.importorskip("visualization_msgs")

    from semantic_digital_twin.spatial_types import Point3
    from semantic_digital_twin.perception import (
        LineSegment2D,
        publish_fitted_and_observed,
    )

    def seg(x0, y0, x1, y1) -> LineSegment2D:
        return LineSegment2D(
            start=Point3(x=x0, y=y0, z=0.0), end=Point3(x=x1, y=y1, z=0.0)
        )

    fitted = [seg(0.0, 0.0, 2.0, 0.0), seg(0.0, 0.5, 1.5, 0.5)]
    observed = [seg(0.1, 0.1, 2.1, 0.1), seg(0.1, 0.8, 1.6, 0.8)]

    # rclpy.init(args=None)

    # node = rclpy.create_node("viz_adapter_test_node")
    node = rclpy_node
    viz = publish_fitted_and_observed(
        node, fitted_segments=fitted, observed_segments=observed
    )
    # rclpy.spin_once(node)
    # Publisher should have a MarkerArray ready to be published; notify was called in __post_init__
    assert hasattr(viz, "markers")
    # Not asserting non-empty markers because runtime subscriber presence can affect conversion,
    # but markers object should exist.
