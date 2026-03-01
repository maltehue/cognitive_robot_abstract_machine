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
]
