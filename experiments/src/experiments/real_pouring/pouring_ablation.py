"""Forecasts with fixed model parameters and no future measurement updates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math

import numpy

from experiments.pouring_model_verification.hypothesis_bank import (
    DrainGeometry,
    ModelPoint,
    PARAMETER_ORDER,
)
from experiments.pouring_model_verification.recovery import DrainCorrectionValues

# %% frozen dynamics


class ForecastModel(StrEnum):
    """Independent alternatives evaluated from the same observed state."""

    ADAPTED = "adapted"
    """The effective parameters actually written into the controller."""
    INITIAL = "initial"
    """The unadapted parameters selected at startup."""
    RECENT_RATE = "recent_rate"
    """Positive wall-fill slope over the preceding second."""
    NO_FLOW = "no_flow"
    """No additional material arrives."""
    FASTER = "faster"
    """Adapted parameters with doubled drain rate."""
    PACING_ONLY = "pacing_only"
    """The predictive fill constraint is disabled; all other tasks remain active."""


@dataclass(frozen=True)
class TiltTrajectory:
    """Recorded motion supplied as an exogenous input to a forecast."""

    times: numpy.ndarray
    """Strictly increasing timestamps in seconds."""
    tilts: numpy.ndarray
    """Source angles from upright in radians."""

    def __post_init__(self) -> None:
        if len(self.times) < 2 or self.times.shape != self.tilts.shape:
            raise ValueError("A trajectory needs matching time and tilt samples.")
        if not numpy.all(numpy.diff(self.times) > 0):
            raise ValueError("Trajectory times must increase strictly.")


@dataclass(frozen=True)
class FrozenForecast:
    """Integrate source depletion while keeping the initial model fixed."""

    geometry: DrainGeometry
    """Source geometry and the conversion from fill to material amount."""
    point: ModelPoint
    """Effective flow parameters at the forecast origin."""
    source_grains: float
    """Available source contents at the forecast origin."""
    grains_per_receiver: float
    """Material amount corresponding to one full receiver."""
    integration_step: float = 0.02
    """Maximum duration of a depletion integration step, in seconds."""

    def rate(self, tilt: float, source_grains: float) -> float:
        """Return outgoing material per second at the specified state."""
        parameters = numpy.array(
            [[self.point.value_of(parameter) for parameter in PARAMETER_ORDER]]
        )
        head = self.geometry.head(tilt, numpy.array([source_grains]), parameters)
        return float(self.geometry.outflow(head, parameters)[0])

    def predict(
        self, trajectory: TiltTrajectory, start: float, duration: float
    ) -> float:
        """Predict received fill increment assuming all outflow reaches the receiver."""
        end = start + duration
        if duration <= 0 or start < trajectory.times[0] or end > trajectory.times[-1]:
            raise ValueError("Forecast interval must lie inside recorded motion.")
        remaining = max(0.0, self.source_grains)
        count = math.ceil(duration / self.integration_step)
        step = duration / count
        for index in range(count):
            at = start + index * step
            tilt = float(numpy.interp(at, trajectory.times, trajectory.tilts))
            initial_rate = self.rate(tilt, remaining)
            midpoint = max(0.0, remaining - initial_rate * step / 2)
            midpoint_tilt = float(
                numpy.interp(at + step / 2, trajectory.times, trajectory.tilts)
            )
            remaining -= min(remaining, self.rate(midpoint_tilt, midpoint) * step)
        return (max(0.0, self.source_grains) - remaining) / self.grains_per_receiver

    @staticmethod
    def corrected(base: ModelPoint, corrections: DrainCorrectionValues) -> ModelPoint:
        """Recover the effective deterministic model used by the live controller."""
        return ModelPoint(
            repose_angle=base.repose_angle + corrections.repose_correction,
            discharge_coefficient=base.discharge_coefficient
            * math.exp(corrections.log_discharge_scale),
            outflow_rate_constant=base.outflow_rate_constant
            * math.exp(corrections.log_drain_scale),
        )
