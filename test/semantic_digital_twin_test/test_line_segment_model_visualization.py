import os
import pytest

from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.perception.line_segment_model import (
    Adjacency,
    LineSegment2D,
    LineSegmentObjectModel,
    visualize_model_fit,
)


def make_segment_from_xy(x0, y0, x1, y1) -> LineSegment2D:
    return LineSegment2D(
        start=Point3(x=x0, y=y0, z=0.0),
        end=Point3(x=x1, y=y1, z=0.0),
    )


def test_visualize_model_fit_creates_figure_and_file(tmp_path):
    # mpl = pytest.importorskip("matplotlib")  # skip test if matplotlib is not installed

    adjacency = Adjacency()
    adjacency.add_edge(0, 1)
    model = LineSegmentObjectModel(fixed_lengths=[2.0, 1.3], adjacency=adjacency)

    observed = [
        make_segment_from_xy(0.1, 0.2, 1.0, 0.1),
        make_segment_from_xy(0.75, 0.3, 1.75, 0.4),
    ]

    save_path = tmp_path / "viz.png"
    ax = visualize_model_fit(model, observed, save_path=str(save_path))
    print(save_path)
    # Axes should be valid and linked to a Figure
    assert hasattr(ax, "figure")
    assert ax.figure is not None

    # File should have been written
    assert os.path.exists(save_path)
