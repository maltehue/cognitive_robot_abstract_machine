"""
Replaying the pouring equation along a tilt that was recorded, so its prediction can be
set against what the grains did along the same tilt, for any parameters, without
running the physics again.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field

import numpy
import pandas
from typing_extensions import List, Sequence

from experiments.pouring_model_verification.parameters import ModelParameters
from experiments.pouring_model_verification.records import Sample
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    VariableParameters,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    SymbolicFillContext,
)

# %% one trajectory to replay along

INTEGRATION_STEP = 0.02
"""
The longest step the replay integrates the drain over, in seconds; a small share of the
shortest time constant the swept parameters give, about half a second.
"""

ONSET_THRESHOLD = 0.002
"""
How far a fill has to fall below where it started before the contents count as flowing:
a fifth of a percent of the cup, well below one grain.
"""


@dataclass
class RecordedPour:
    """
    A tilt that was recorded, with the fill level the grains showed along it.
    """

    time: numpy.ndarray
    """
    When each sample was taken, in seconds.
    """

    tilt: numpy.ndarray
    """
    The cup's tilt at each sample, in radians.
    """

    measured_fill: numpy.ndarray
    """
    The source's fill level the grains showed at each sample.
    """

    @classmethod
    def from_samples(cls, samples: pandas.DataFrame) -> RecordedPour:
        """
        :param samples: A trial's samples, restricted to the stretch to replay.
        :return: That stretch as a pour to replay along.
        """
        return cls(
            time=samples[Sample.TIME].to_numpy(dtype=float),
            tilt=samples[Sample.TILT].to_numpy(dtype=float),
            measured_fill=samples[Sample.MEASURED_SOURCE_FILL].to_numpy(dtype=float),
        )

    def onset_tilt(self, fill: numpy.ndarray) -> float:
        """
        :param fill: A fill level along this pour, measured or predicted.
        :return: The tilt at which that fill first fell below where it started, or
            ``nan`` if it never did.
        """
        started_to_flow = numpy.flatnonzero(fill < fill[0] - ONSET_THRESHOLD)
        if not len(started_to_flow):
            return float("nan")
        return float(self.tilt[started_to_flow[0]])


# %% the drain, compiled once per parameter set


@dataclass
class FillModelReplay:
    """
    The source's drain, evaluated with one set of parameters, integrated along recorded
    tilts.
    """

    equation: ArticulatedPouringEquation
    """
    The drain as the trial built it from the cup's geometry.
    """

    parameters: ModelParameters
    """
    The parameters the drain is evaluated with.
    """

    _velocity: CompiledFunction = field(init=False, repr=False)
    """
    The fill velocity as a function of tilt and fill.
    """

    def __post_init__(self) -> None:
        tilt, fill = FloatVariable("replay_tilt"), FloatVariable("replay_fill")
        drain = self.parameters.applied_to(self.equation)
        self._velocity = drain.symbolic_velocity(
            SymbolicFillContext(tilt, fill)
        ).compile(VariableParameters.from_lists([tilt, fill]))

    def velocity(self, tilt: float, fill: float) -> float:
        """
        :return: The fill velocity the drain predicts at that tilt and fill, per second.
        """
        return float(self._velocity(numpy.array([tilt, fill]))[0])

    def predict(self, pour: RecordedPour) -> numpy.ndarray:
        """
        Integrate the drain along a recorded tilt from the fill the grains started at.

        :param pour: The recorded pour.
        :return: The predicted fill at every sample of it.
        """
        predicted = numpy.empty_like(pour.measured_fill)
        fill = predicted[0] = pour.measured_fill[0]
        for index in range(1, len(pour.time)):
            interval = pour.time[index] - pour.time[index - 1]
            steps = max(1, int(numpy.ceil(interval / INTEGRATION_STEP)))
            step = interval / steps
            for substep in range(steps):
                share = (substep + 0.5) / steps
                tilt = (1 - share) * pour.tilt[index - 1] + share * pour.tilt[index]
                fill = min(1.0, max(0.0, fill + step * self.velocity(tilt, fill)))
            predicted[index] = fill
        return predicted

    def residuals(self, pours: Sequence[RecordedPour]) -> numpy.ndarray:
        """
        :param pours: The recorded pours to compare against.
        :return: Predicted minus measured fill at every sample of every pour.
        """
        if not pours:
            return numpy.empty(0)
        return numpy.concatenate(
            [self.predict(pour) - pour.measured_fill for pour in pours]
        )

    def root_mean_square_error(self, pours: Sequence[RecordedPour]) -> float:
        """
        :param pours: The recorded pours to compare against.
        :return: The root mean square fill error over every sample of every pour, or
            ``nan`` for no pours.
        """
        residuals = self.residuals(pours)
        if not len(residuals):
            return float("nan")
        return float(numpy.sqrt(numpy.mean(residuals**2)))


def root_mean_square_errors(
    equation: ArticulatedPouringEquation,
    parameters: ModelParameters,
    pour_groups: List[Sequence[RecordedPour]],
) -> List[float]:
    """
    :param equation: The drain as built from the cup's geometry.
    :param parameters: The parameters to evaluate it with.
    :param pour_groups: Sets of recorded pours, each scored on its own.
    :return: The root mean square fill error on each set, compiling the drain once.
    """
    replay = FillModelReplay(equation=equation, parameters=parameters)
    return [replay.root_mean_square_error(pours) for pours in pour_groups]
