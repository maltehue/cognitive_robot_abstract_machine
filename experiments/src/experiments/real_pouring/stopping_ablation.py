"""Stopping influence in paired production-controller snapshots."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy

from experiments.real_pouring.command_evaluation import CommandComparison
from experiments.real_pouring.pouring_ablation import ForecastModel

# %% stopping metrics


@dataclass
class UnpairedCommandsError(ValueError):
    """Raised when alternatives do not refer to identical recorded states."""


@dataclass(frozen=True)
class StoppingSummary:
    """Influence of one model relative to the same tasks without the fill objective."""

    model: ForecastModel
    """Model whose commands are summarized."""
    states: int
    """Number of paired snapshots."""
    suppressed_forward: int
    """Snapshots where forward pacing becomes a hold or reversal."""
    reversals: int
    """Snapshots with a negative mean planned tilt speed beyond the threshold."""
    first_sustained_reversal: float | None
    """First of two successive negative plans, or no such pair."""
    mean_plan_difference: float
    """Mean absolute planned-speed difference from pacing, in radians per second."""
    mean_command_difference: float
    """Mean absolute next-command difference from pacing, in radians per second."""


@dataclass
class StoppingAnalysis:
    """Summarize the selected model on exactly paired recorded states."""

    rows: list[CommandComparison]
    """Saved model alternatives for one recording and forecast window."""
    threshold: float = math.radians(0.1)
    """Speed deadband separating forward, hold and reversal decisions."""

    @staticmethod
    def mean_speed(row: CommandComparison) -> float:
        """Return mean planned tilt speed over the optimized motion horizon."""
        return float(numpy.mean(row.planned_tilt_velocities))

    def select(self, model: ForecastModel) -> dict[float, CommandComparison]:
        """Index one alternative by its unique snapshot timestamps."""
        selected = [row for row in self.rows if row.model is model]
        result = {row.time: row for row in selected}
        if len(result) != len(selected):
            raise UnpairedCommandsError()
        return result

    @staticmethod
    def check_pair(first: CommandComparison, second: CommandComparison) -> None:
        """Reject state changes that would confound a model or horizon comparison."""
        if (
            first.time,
            first.tilt,
            first.reported_fill,
            first.source_fill,
            first.joint_age,
        ) != (
            second.time,
            second.tilt,
            second.reported_fill,
            second.source_fill,
            second.joint_age,
        ):
            raise UnpairedCommandsError()

    def summarize(self, model: ForecastModel) -> StoppingSummary:
        """Count model-induced suppression of forward pacing and sustained reversal."""
        selected = self.select(model)
        reference = self.select(ForecastModel.PACING_ONLY)
        if not selected or selected.keys() != reference.keys():
            raise UnpairedCommandsError()
        times = sorted(selected)
        for time in times:
            self.check_pair(selected[time], reference[time])
        speeds = numpy.array([self.mean_speed(selected[time]) for time in times])
        pacing = numpy.array([self.mean_speed(reference[time]) for time in times])
        reversal = speeds < -self.threshold
        first_reversal = next(
            (
                times[index]
                for index in range(len(times) - 1)
                if reversal[index] and reversal[index + 1]
            ),
            None,
        )
        return StoppingSummary(
            model,
            len(times),
            int(numpy.sum((pacing > self.threshold) & (speeds <= self.threshold))),
            int(numpy.sum(reversal)),
            first_reversal,
            float(numpy.mean(numpy.abs(speeds - pacing))),
            float(
                numpy.mean(
                    [
                        abs(
                            selected[time].tilt_velocity - reference[time].tilt_velocity
                        )
                        for time in times
                    ]
                )
            ),
        )
