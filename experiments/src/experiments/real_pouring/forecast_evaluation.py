"""Paired forecast errors against future camera readings and final weighings."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy

from experiments.pouring_model_verification.hypothesis_bank import (
    DrainGeometry,
    ModelPoint,
)
from experiments.real_pouring.pour_replay import ReplayStep, CUP_SIZE, GRAINS_PER_FILL
from experiments.real_pouring.pouring_ablation import (
    ForecastModel,
    FrozenForecast,
    TiltTrajectory,
)

# %% observations reserved for scoring


@dataclass
class MissingSourceSnapshotError(ValueError):
    """Raised when a replay lacks the state required to initialize a forecast."""


@dataclass(frozen=True)
class ObservedIncrement:
    """A future camera increment with consistent measurement provenance."""

    fill: float
    """Signed change in measured receiver fill."""
    calibrated: bool
    """Whether both endpoints and intervening observations are calibrated."""


@dataclass(frozen=True)
class ForecastError:
    """A paired future increment and its frozen prediction."""

    time: float
    """Forecast origin relative to the recorded pour start."""
    horizon: float
    """Duration with no measurement updates."""
    delay: float
    """Assumed camera lag used only to align evaluation targets."""
    model: ForecastModel
    """Forecast alternative."""
    observed: float
    """Future camera fill increment, including negative measurement noise."""
    predicted: float
    """Forecast fill increment."""
    flowing: bool
    """Whether a stream was observed at the forecast origin."""
    calibrated: bool
    """Whether the complete target interval used calibrated wall perception."""


@dataclass(frozen=True)
class EndpointForecast:
    """An independent final weighing compared with an uncorrected rollout."""

    time: float
    """Origin of the rollout to the recording's end."""
    horizon: float
    """Remaining recorded duration."""
    model: ForecastModel
    """Frozen model alternative."""
    predicted: float
    """Initial posterior receiver fill plus predicted future transfer."""
    weighed: float
    """Independently weighed final receiver fill."""


@dataclass(frozen=True)
class RecordingForecasts:
    """All scoring samples for one independent recorded pour."""

    recording: str
    """Original recording identity."""
    validation: bool
    """Whether excluded from numerical perception calibration."""
    errors: list[ForecastError]
    """Paired future-camera comparisons."""
    endpoints: list[EndpointForecast]
    """Rollouts compared with independent final weighing."""


@dataclass
class ForecastEvaluation:
    """Evaluate effective controller parameters on fixed recorded motion."""

    steps: list[ReplayStep]
    """Causal replay snapshots, including subsequent scoring observations."""
    grains_per_receiver: float
    """Conversion from model grains to receiver fill."""
    stride: int = 10
    """Spacing between forecast origins in replay samples."""

    @property
    def trajectory(self) -> TiltTrajectory:
        """Return recorded tilt, without passing future observations to the model."""
        return TiltTrajectory(
            numpy.array([step.time for step in self.steps]),
            numpy.array([step.tilt for step in self.steps]),
        )

    def forecast(self, index: int, model: ForecastModel) -> FrozenForecast:
        """Freeze the deployed deterministic model and current source estimate."""
        step = self.steps[index]
        if step.bank.source_grains is None:
            raise MissingSourceSnapshotError()
        base = ModelPoint(0.0, 0.2, 0.8)
        point = (
            base
            if model is ForecastModel.INITIAL
            else FrozenForecast.corrected(base, step.bank.corrections)
        )
        if model is ForecastModel.FASTER:
            point = replace(
                point, outflow_rate_constant=2 * point.outflow_rate_constant
            )
        return FrozenForecast(
            DrainGeometry(CUP_SIZE.z, CUP_SIZE.x / 2, GRAINS_PER_FILL, CUP_SIZE.x / 2),
            point,
            min(GRAINS_PER_FILL, max(0.0, step.bank.source_grains)),
            self.grains_per_receiver,
        )

    def recent_rate(self, index: int) -> float | None:
        """Estimate a nonnegative slope from observations available at the origin."""
        end = self.steps[index].time
        previous = [
            step for step in self.steps[: index + 1] if end - 1.0 <= step.time <= end
        ]
        if len(previous) < 3 or previous[-1].time - previous[0].time < 0.8:
            return None
        if any(step.reading.heap_fill is None for step in previous):
            return None
        if any(
            step.reading.wall_calibrated != previous[0].reading.wall_calibrated
            for step in previous
        ):
            return None
        times = numpy.array([step.time - end for step in previous])
        fills = numpy.array([step.reading.heap_fill for step in previous])
        return max(0.0, float(numpy.polyfit(times, fills, 1)[0]))

    def target(self, start: float, end: float) -> ObservedIncrement | None:
        """Score only complete intervals without perception fallback transitions."""
        times = self.trajectory.times
        if start < times[0] or end > times[-1]:
            return None
        first = max(0, int(numpy.searchsorted(times, start, side="right")) - 1)
        last = min(len(times) - 1, int(numpy.searchsorted(times, end)))
        selected = self.steps[first : last + 1]
        if any(step.reading.heap_fill is None for step in selected):
            return None
        calibrated = selected[0].reading.wall_calibrated
        if any(step.reading.wall_calibrated != calibrated for step in selected):
            return None
        fills = numpy.array([step.reading.heap_fill for step in selected])
        selected_times = times[first : last + 1]
        return ObservedIncrement(
            float(
                numpy.interp(end, selected_times, fills)
                - numpy.interp(start, selected_times, fills)
            ),
            calibrated,
        )

    def evaluate(
        self,
        horizons: tuple[float, ...] = (0.5, 1.5, 3.0),
        delays: tuple[float, ...] = (0.0, 0.33),
    ) -> list[ForecastError]:
        """Compare paired forecasts, retaining dry and flowing origins separately."""
        trajectory = self.trajectory
        results = []
        for index in range(0, len(self.steps), self.stride):
            origin = self.steps[index]
            recent_rate = self.recent_rate(index)
            if recent_rate is None:
                continue
            for horizon in horizons:
                if origin.time + horizon > trajectory.times[-1]:
                    continue
                predictions = {
                    model: self.forecast(index, model).predict(
                        trajectory, origin.time, horizon
                    )
                    for model in (
                        ForecastModel.ADAPTED,
                        ForecastModel.INITIAL,
                        ForecastModel.FASTER,
                    )
                }
                predictions[ForecastModel.RECENT_RATE] = min(
                    recent_rate * horizon,
                    self.forecast(index, ForecastModel.ADAPTED).source_grains
                    / self.grains_per_receiver,
                )
                predictions[ForecastModel.NO_FLOW] = 0.0
                for delay in delays:
                    target = self.target(
                        origin.time + delay, origin.time + horizon + delay
                    )
                    if target is None:
                        continue
                    results.extend(
                        ForecastError(
                            origin.time,
                            horizon,
                            delay,
                            model,
                            target.fill,
                            predicted,
                            origin.reading.stream_seen,
                            target.calibrated,
                        )
                        for model, predicted in predictions.items()
                    )
        return results

    def endpoints(self, weighed: float) -> list[EndpointForecast]:
        """Roll out from fixed early, middle and late recorded motion timestamps."""
        trajectory = self.trajectory
        results = []
        for share in (0.25, 0.5, 0.75):
            index = int(share * (len(self.steps) - 1))
            origin = self.steps[index]
            horizon = float(trajectory.times[-1] - origin.time)
            initial = origin.bank.received_grains / self.grains_per_receiver
            for model in (ForecastModel.ADAPTED, ForecastModel.INITIAL):
                increment = self.forecast(index, model).predict(
                    trajectory, origin.time, horizon
                )
                results.append(
                    EndpointForecast(
                        origin.time, horizon, model, initial + increment, weighed
                    )
                )
        return results
