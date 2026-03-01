"""Perception-related minimal prototypes.

This package hosts simple, extensible components that connect the world
representation with perception models.
"""

from .line_segment_model import (
    ModelSpecificationError,
    VisualizationError,
    LineSegment2D,
    Adjacency,
    LineSegmentObjectModel,
    FitResult,
    fit_model_to_perception_data,
    plot_segments_2d,
    visualize_model_fit,
)
from .twin_migration import (
    Pose2D,
    SegmentAttachment,
    TwinLineSegmentModel,
    TwinState,
    TwinFitResult,
    materialize_segments,
    fit_twin_model_to_segments,
)
from .viz_adapter import (
    SegmentVizConfig,
    build_world_from_segments,
    publish_segments_with_viz_marker,
    publish_fitted_and_observed,
)

__all__ = [
    "ModelSpecificationError",
    "VisualizationError",
    "LineSegment2D",
    "Adjacency",
    "LineSegmentObjectModel",
    "FitResult",
    "fit_model_to_perception_data",
    "plot_segments_2d",
    "visualize_model_fit",
    "Pose2D",
    "SegmentAttachment",
    "TwinLineSegmentModel",
    "TwinState",
    "TwinFitResult",
    "materialize_segments",
    "fit_twin_model_to_segments",
    "SegmentVizConfig",
    "build_world_from_segments",
    "publish_segments_with_viz_marker",
    "publish_fitted_and_observed",
]
