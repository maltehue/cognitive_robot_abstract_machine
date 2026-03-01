from __future__ import annotations

"""Minimal 2D line segment object model for perception prototypes.

This module provides a parametric object model that represents an object as a
set of 2D line segments with fixed lengths and free parameters for relative
angles and minimal endpoint distances (gaps) between adjacent segments.

The implementation is intentionally simple and designed to integrate with the
existing spatial types of the project.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from semantic_digital_twin.spatial_types import Point3, Vector3


class ModelSpecificationError(ValueError):
    """Raised when the model specification is inconsistent."""


class VisualizationError(RuntimeError):
    """Raised when visualization backends are unavailable.

    The minimal visualization utilities depend on Matplotlib. This exception
    is raised if the package is missing or cannot be imported at runtime.
    """


def _to_float(value: object) -> float:
    """Convert project scalar types or numpy scalars to a Python float.

    This utility ensures compatibility with symbolic or array-like scalar
    wrappers used in the project's spatial types.
    """

    return float(value)


@dataclass(frozen=True)
class LineSegment2D:
    """A 2D line segment using Point3 endpoints with z = 0.

    This keeps compatibility with the spatial type layer so the model can be
    used together with world representation utilities.
    """

    start: Point3
    end: Point3

    @property
    def length(self) -> float:
        """Return the Euclidean length of the segment."""

        displacement = self.end - self.start
        return _to_float(
            np.linalg.norm(
                [
                    _to_float(displacement.x),
                    _to_float(displacement.y),
                    _to_float(displacement.z),
                ]
            )
        )

    @property
    def direction(self) -> Vector3:
        """Return the unit direction vector from start to end in the plane."""

        displacement = self.end - self.start
        norm_value = _to_float(
            np.linalg.norm(
                [
                    _to_float(displacement.x),
                    _to_float(displacement.y),
                    _to_float(displacement.z),
                ]
            )
        )
        if norm_value == 0.0:
            return Vector3(x=1.0, y=0.0, z=0.0)
        return Vector3(
            x=displacement.x / norm_value, y=displacement.y / norm_value, z=0.0
        )

    def as_tuple(self) -> Tuple[Point3, Point3]:
        """Return the start and end points as a pair."""

        return self.start, self.end


@dataclass
class Adjacency:
    """Adjacency relation between segment indices.

    The map lists neighbor indices for each segment index. This is sufficient
    for small prototypes and can be extended for more advanced models.
    """

    neighbors: Dict[int, List[int]] = field(default_factory=dict)

    def add_edge(self, i: int, j: int) -> None:
        """Add an undirected adjacency edge between segments i and j."""

        self.neighbors.setdefault(i, []).append(j)
        self.neighbors.setdefault(j, []).append(i)

    def adjacent_pairs(self) -> Iterable[Tuple[int, int]]:
        """Iterate unique adjacent index pairs (i, j) with no duplicates."""

        seen: set[Tuple[int, int]] = set()
        for i, nbrs in self.neighbors.items():
            for j in nbrs:
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    yield key


@dataclass
class LineSegmentObjectModel:
    """Parametric object model of 2D line segments with fixed lengths.

    The model stores fixed lengths for all segments and free parameters for the
    angle and minimal endpoint distance (gap) between adjacent segments.
    """

    fixed_lengths: List[float]
    adjacency: Adjacency
    reference_frame: Optional[object] = None

    angles_rad: Dict[Tuple[int, int], float] = field(default_factory=dict)
    gaps: Dict[Tuple[int, int], float] = field(default_factory=dict)

    anchor_origin: Point3 = field(default_factory=lambda: Point3(x=0.0, y=0.0, z=0.0))

    def __post_init__(self) -> None:
        if not self.fixed_lengths:
            raise ModelSpecificationError("At least one segment is required")
        if any(length_value <= 0 for length_value in self.fixed_lengths):
            raise ModelSpecificationError("All fixed lengths must be positive")
        for i, j in self.adjacency.adjacent_pairs():
            key = (i, j) if i < j else (j, i)
            self.angles_rad.setdefault(key, 0.0)
            self.gaps.setdefault(key, 0.0)

    @property
    def segment_count(self) -> int:
        """Return the number of segments in the model."""

        return len(self.fixed_lengths)

    def set_pair_params(self, i: int, j: int, angle_rad: float, gap: float) -> None:
        """Set the angle and gap for an adjacent pair.

        The order of indices is normalized internally.
        """

        key = (i, j) if i < j else (j, i)
        self.angles_rad[key] = float(angle_rad)
        self.gaps[key] = float(max(0.0, gap))

    def get_pair_params(self, i: int, j: int) -> Tuple[float, float]:
        """Get the angle and gap for an adjacent pair."""

        key = (i, j) if i < j else (j, i)
        return self.angles_rad[key], self.gaps[key]

    def compute_geometry(self) -> List[LineSegment2D]:
        """Compute the segment positions and orientations from parameters.

        The construction is deterministic and starts from segment index 0 placed
        along the positive x-axis. Remaining segments are expanded using a
        breadth-first strategy based on the adjacency structure.
        """

        created_segments: List[Optional[LineSegment2D]] = [None] * self.segment_count  # type: ignore[assignment]

        length_zero = self.fixed_lengths[0]
        start_zero = Point3(
            x=self.anchor_origin.x,
            y=self.anchor_origin.y,
            z=0.0,
            reference_frame=self.reference_frame,
        )
        end_zero = Point3(
            x=start_zero.x + length_zero,
            y=start_zero.y,
            z=0.0,
            reference_frame=self.reference_frame,
        )
        created_segments[0] = LineSegment2D(start=start_zero, end=end_zero)

        placed_indices = {0}
        open_list = [0]

        def unit_from_angle(angle: float) -> Vector3:
            return Vector3(x=float(np.cos(angle)), y=float(np.sin(angle)), z=0.0)

        while open_list:
            current = open_list.pop(0)
            current_segment = created_segments[current]
            assert current_segment is not None
            current_direction = current_segment.direction
            current_angle = _to_float(
                np.arctan2(
                    _to_float(current_direction.y), _to_float(current_direction.x)
                )
            )

            for neighbor in self.adjacency.neighbors.get(current, []):
                if neighbor in placed_indices:
                    continue
                key = (current, neighbor) if current < neighbor else (neighbor, current)
                relative_angle, gap_value = self.angles_rad[key], self.gaps[key]

                absolute_angle = current_angle + relative_angle
                neighbor_direction = unit_from_angle(absolute_angle)
                neighbor_length = self.fixed_lengths[neighbor]

                # Enforce constraint: distance between lines measured along y-axis,
                # with zero x-offset between the nearest endpoints.
                anchor = current_segment.start
                neighbor_start = Point3(
                    x=anchor.x,
                    y=anchor.y + float(max(0.0, gap_value)),
                    z=0.0,
                    reference_frame=self.reference_frame,
                )
                neighbor_end = Point3(
                    x=neighbor_start.x + neighbor_direction.x * neighbor_length,
                    y=neighbor_start.y + neighbor_direction.y * neighbor_length,
                    z=0.0,
                    reference_frame=self.reference_frame,
                )

                created_segments[neighbor] = LineSegment2D(
                    start=neighbor_start, end=neighbor_end
                )
                placed_indices.add(neighbor)
                open_list.append(neighbor)

        y_offset = 0.0
        for index in range(self.segment_count):
            if created_segments[index] is None:
                length_value = self.fixed_lengths[index]
                start_point = Point3(
                    x=0.0, y=y_offset, z=0.0, reference_frame=self.reference_frame
                )
                end_point = Point3(
                    x=length_value,
                    y=y_offset,
                    z=0.0,
                    reference_frame=self.reference_frame,
                )
                created_segments[index] = LineSegment2D(
                    start=start_point, end=end_point
                )
                y_offset += 2.0

        return [segment for segment in created_segments if segment is not None]  # type: ignore[return-value]

    def adjacent_pairs(self) -> Iterable[Tuple[int, int]]:
        """Return the unique adjacent segment index pairs."""

        return self.adjacency.adjacent_pairs()


@dataclass
class FitResult:
    """Container for the fitted model instance and a residual score."""

    model: LineSegmentObjectModel
    residual: float


def _segment_angle(segment: LineSegment2D) -> float:
    direction = segment.direction
    return _to_float(np.arctan2(_to_float(direction.y), _to_float(direction.x)))


def _angle_diff(angle_a: float, angle_b: float) -> float:
    difference = (angle_a - angle_b + np.pi) % (2 * np.pi) - np.pi
    return float(difference)


def _closest_endpoint_distance(a: LineSegment2D, b: LineSegment2D) -> float:
    points_a = [a.start, a.end]
    points_b = [b.start, b.end]
    distances: List[float] = []
    for point_a in points_a:
        for point_b in points_b:
            delta = point_b - point_a
            distances.append(
                _to_float(
                    np.linalg.norm(
                        [
                            _to_float(delta.x),
                            _to_float(delta.y),
                        ]
                    )
                )
            )
    return float(min(distances))


def _vertical_endpoint_gap(a: LineSegment2D, b: LineSegment2D) -> float:
    """Return minimal absolute vertical gap between endpoints of two segments.

    The gap is computed as the minimum over all endpoint pairs of |ya - yb|,
    ignoring x differences to satisfy the constraint that inter-line distance
    is measured along the y-axis.
    """

    points_a = [a.start, a.end]
    points_b = [b.start, b.end]
    gaps: List[float] = []
    for pa in points_a:
        for pb in points_b:
            gaps.append(abs(_to_float(pa.y) - _to_float(pb.y)))
    return float(min(gaps))


def fit_model_to_perception_data(
    model: LineSegmentObjectModel, observed_segments: List[LineSegment2D]
) -> FitResult:
    """Fit free parameters of the model to two observed segments.

    The function assumes two segments and assigns observed relative angle and
    minimal endpoint distance to the model parameters, then evaluates a simple
    least-squares residual.
    """

    assert model.segment_count == 2, "This minimal prototype expects two segments"
    assert len(observed_segments) == 2, "Provide two observed segments"

    observed_segment_zero, observed_segment_one = observed_segments
    angle_observed_zero = _segment_angle(observed_segment_zero)
    angle_observed_one = _segment_angle(observed_segment_one)
    relative_angle_observed = angle_observed_one - angle_observed_zero
    relative_angle_observed = (relative_angle_observed + np.pi) % (2 * np.pi) - np.pi
    # Use vertical endpoint gap per constraint
    gap_observed = _vertical_endpoint_gap(observed_segment_zero, observed_segment_one)

    fitted_model = LineSegmentObjectModel(
        fixed_lengths=list(model.fixed_lengths),
        adjacency=model.adjacency,
        reference_frame=model.reference_frame,
    )
    fitted_model.set_pair_params(
        0, 1, angle_rad=relative_angle_observed, gap=gap_observed
    )

    geometry = fitted_model.compute_geometry()
    segment_zero, segment_one = geometry[0], geometry[1]

    length_cost = (segment_zero.length - fitted_model.fixed_lengths[0]) ** 2 + (
        segment_one.length - fitted_model.fixed_lengths[1]
    ) ** 2

    absolute_angle_cost = (
        _angle_diff(_segment_angle(segment_zero), angle_observed_zero) ** 2
    )

    relative_angle_cost = (
        _angle_diff(
            _segment_angle(segment_one) - _segment_angle(segment_zero),
            relative_angle_observed,
        )
        ** 2
    )

    # Evaluate gap in the model using vertical metric
    gap_model = _vertical_endpoint_gap(segment_zero, segment_one)
    gap_cost = (gap_model - gap_observed) ** 2

    residual_value = float(
        length_cost + relative_angle_cost + gap_cost + 0.1 * absolute_angle_cost
    )

    return FitResult(model=fitted_model, residual=residual_value)


def plot_segments_2d(
    segments: List[LineSegment2D],
    ax: Optional["Axes"] = None,
    *,
    color: str = "C0",
    label_prefix: str = "Segment",
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha: float = 1.0,
) -> "Axes":
    """Plot a list of 2D line segments.

    Creates a Matplotlib axes if none is provided. Returns the axes used.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised by importorskip in tests
        raise VisualizationError("Matplotlib is required for plotting") from exc

    if ax is None:
        _, ax = plt.subplots()

    for idx, seg in enumerate(segments):
        x_vals = [_to_float(seg.start.x), _to_float(seg.end.x)]
        y_vals = [_to_float(seg.start.y), _to_float(seg.end.y)]
        ax.plot(
            x_vals,
            y_vals,
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            label=f"{label_prefix} {idx}",
        )
        ax.scatter(x_vals, y_vals, color=color, s=10, alpha=alpha)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", alpha=0.4)
    return ax


def visualize_model_fit(
    model: LineSegmentObjectModel,
    observed_segments: List[LineSegment2D],
    ax: Optional["Axes"] = None,
    *,
    save_path: Optional[str] = None,
) -> "Axes":
    """Visualize observed segments and a fitted model geometry.

    The function attempts to fit the model to the observed segments using the
    minimal two‑segment prototype. If fitting is not applicable (e.g., the
    model is not two segments long), it falls back to plotting the model's
    current geometry. If ``save_path`` is provided, the figure is saved there.
    """

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised by importorskip in tests
        raise VisualizationError("Matplotlib is required for plotting") from exc

    # Try to obtain a fitted model. Fall back to current model geometry if
    # assumptions do not hold.
    try:
        fit = fit_model_to_perception_data(model, observed_segments)
        fitted_segments = fit.model.compute_geometry()

        # Additionally align the entire fitted geometry to the first observed
        # segment via a rigid 2D transform (rotation + translation). This
        # places the object's pose into the observation frame, avoiding the
        # artificial horizontal anchor of segment 0.
        obs0 = observed_segments[0]
        mod0 = fitted_segments[0]

        angle_obs0 = _segment_angle(obs0)
        angle_mod0 = _segment_angle(mod0)
        dtheta = angle_obs0 - angle_mod0

        def _transform_point(
            p: Point3,
            anchor_from: Point3,
            anchor_to: Point3,
            cos_t: float,
            sin_t: float,
        ) -> Point3:
            dx = _to_float(p.x) - _to_float(anchor_from.x)
            dy = _to_float(p.y) - _to_float(anchor_from.y)
            rx = cos_t * dx - sin_t * dy
            ry = sin_t * dx + cos_t * dy
            return Point3(
                x=_to_float(anchor_to.x) + rx,
                y=_to_float(anchor_to.y) + ry,
                z=0.0,
                reference_frame=model.reference_frame,
            )

        c = float(np.cos(dtheta))
        s = float(np.sin(dtheta))
        anchor_from = mod0.start
        anchor_to = obs0.start

        aligned_segments: List[LineSegment2D] = []
        for seg in fitted_segments:
            s_new = _transform_point(seg.start, anchor_from, anchor_to, c, s)
            e_new = _transform_point(seg.end, anchor_from, anchor_to, c, s)
            aligned_segments.append(LineSegment2D(start=s_new, end=e_new))

        model_segments = aligned_segments
        title_suffix = " (fitted, aligned)"
    except AssertionError:
        model_segments = model.compute_geometry()
        title_suffix = ""

    ax = plot_segments_2d(
        model_segments,
        ax=ax,
        color="C0",
        label_prefix="Model",
        linestyle="-",
        linewidth=2.5,
    )
    plot_segments_2d(
        observed_segments,
        ax=ax,
        color="C1",
        label_prefix="Observed",
        linestyle="--",
        linewidth=2.0,
    )

    ax.legend(loc="best")
    ax.set_title(f"Observed vs. Model Segments{title_suffix}")

    if save_path is not None:
        fig = ax.figure
        fig.savefig(save_path, bbox_inches="tight")

    return ax


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
    "_segment_angle",
    "_closest_endpoint_distance",
    "_vertical_endpoint_gap",
]
