"""
A black-box model of how fast grains arrive in the receiver for a given cup tilt,
identified from recorded fill readings alone: no repose angle, drain constant or
discharge coefficient.

It is what a Smith predictor needs to see through a late sensor without the pouring
model: three numbers fitted to the pours a PID controller was tuned on.
"""

# %% imports

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy
from scipy.optimize import least_squares
from typing_extensions import Any, Dict, List

from experiments.pouring_model_verification.records import (
    Phase,
    Sample,
    TrialRecord,
)

# %% the model


class InflowModelKey(StrEnum):
    """
    A key of an inflow model written to JSON.
    """

    GAIN = "gain"
    ONSET_TILT = "onset_tilt"
    TIME_CONSTANT = "time_constant"


@dataclass(frozen=True)
class FirstOrderInflowModel:
    """
    Grains arrive at a rate that follows, with a first-order lag, a steady rate rising
    linearly with the tilt past an onset.
    """

    gain: float
    """
    Steady inflow per radian of tilt past the onset, in grains per second.
    """

    onset_tilt: float
    """
    The tilt below which nothing arrives, in radians.
    """

    time_constant: float
    """
    How quickly the inflow follows a change of tilt, in seconds.
    """

    def steady_rate(self, tilt: float) -> float:
        """
        :param tilt: A tilt held long enough, in radians.
        :return: The inflow it settles at, in grains per second.
        """
        return self.gain * max(0.0, tilt - self.onset_tilt)

    def step(self, rate: float, tilt: float, duration: float) -> float:
        """
        :param rate: The inflow now, in grains per second.
        :param tilt: The tilt held over the step, in radians.
        :param duration: How long the step lasts, in seconds.
        :return: The inflow at the end of the step.
        """
        approach = 1.0 - math.exp(-duration / self.time_constant)
        return rate + approach * (self.steady_rate(tilt) - rate)

    def simulate(self, times: numpy.ndarray, tilts: numpy.ndarray) -> numpy.ndarray:
        """
        :param times: Increasing times, in seconds.
        :param tilts: The tilt at each time, in radians.
        :return: The grains arrived by each time, starting from none and no inflow.
        """
        fill = numpy.zeros(len(times))
        rate = 0.0
        for index in range(1, len(times)):
            duration = times[index] - times[index - 1]
            rate = self.step(rate, tilts[index - 1], duration)
            fill[index] = fill[index - 1] + rate * duration
        return fill

    def to_json(self) -> Dict[str, Any]:
        return {
            InflowModelKey.GAIN: self.gain,
            InflowModelKey.ONSET_TILT: self.onset_tilt,
            InflowModelKey.TIME_CONSTANT: self.time_constant,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> FirstOrderInflowModel:
        return cls(
            gain=data[InflowModelKey.GAIN],
            onset_tilt=data[InflowModelKey.ONSET_TILT],
            time_constant=data[InflowModelKey.TIME_CONSTANT],
        )


# %% identifying it from readings

IDENTIFIED_PHASES = (Phase.PRE_TILT, Phase.TRANSFER, Phase.RETURN)
"""
The parts of a recorded pour the model is fitted to: from the settled cup until it
stands upright again.
"""

INITIAL_GUESS = FirstOrderInflowModel(gain=20.0, onset_tilt=1.0, time_constant=0.5)
"""
Where the fit starts.
"""


@dataclass(frozen=True)
class RecordedInflow:
    """
    The tilt and the committed grains of one recorded pour, as the fit reads them.
    """

    times: numpy.ndarray
    """
    Sample times, in seconds.
    """

    tilts: numpy.ndarray
    """
    The cup's tilt at each sample, in radians.
    """

    grains: numpy.ndarray
    """
    The grains in the receiver and on their way to it at each sample.
    """

    @classmethod
    def from_record(cls, record: TrialRecord) -> RecordedInflow:
        """
        :param record: A recorded pour.
        :return: Its tilt and committed grains, with the committed fill converted to
            grains through the landed grains the same record counted.
        """
        samples = record.samples
        landed = samples[samples[Sample.GRAINS_IN_RECEIVER] > 0]
        grains_per_share = float(
            (
                landed[Sample.GRAINS_IN_RECEIVER]
                / landed[Sample.MEASURED_RECEIVER_FILL]
            ).median()
        )
        poured = samples[samples[Sample.PHASE].isin(IDENTIFIED_PHASES)]
        return cls(
            times=poured[Sample.TIME].to_numpy(dtype=float),
            tilts=poured[Sample.TILT].to_numpy(dtype=float),
            grains=poured[Sample.COMMITTED_RECEIVER_FILL].to_numpy(dtype=float)
            * grains_per_share,
        )

    def residuals(self, model: FirstOrderInflowModel) -> numpy.ndarray:
        """
        :return: The model's grains minus the recorded ones, at every sample.
        """
        return self.grains[0] + model.simulate(self.times, self.tilts) - self.grains


def identify_inflow_model(records: List[TrialRecord]) -> FirstOrderInflowModel:
    """
    :param records: Recorded pours of one cup that grains arrived in.
    :return: The first-order model whose grains match theirs best, in the least-squares
        sense over every sample.
    """
    pours = [
        RecordedInflow.from_record(record)
        for record in records
        if (record.samples[Sample.GRAINS_IN_RECEIVER] > 0).any()
    ]

    def residuals(values: numpy.ndarray) -> numpy.ndarray:
        model = FirstOrderInflowModel(*values)
        return numpy.concatenate([pour.residuals(model) for pour in pours])

    fit = least_squares(
        residuals,
        x0=[INITIAL_GUESS.gain, INITIAL_GUESS.onset_tilt, INITIAL_GUESS.time_constant],
        bounds=([0.0, 0.0, 0.02], [500.0, math.pi / 2, 10.0]),
    )
    return FirstOrderInflowModel(*(float(value) for value in fit.x))


# %% keeping one per cup


@dataclass(frozen=True)
class InflowModels:
    """
    The inflow model each source cup's Smith predictor runs with.
    """

    models: Dict[str, FirstOrderInflowModel]
    """
    The identified models, keyed by source cup.
    """

    def of(self, source: str) -> FirstOrderInflowModel:
        """
        :param source: A source cup.
        :return: Its identified inflow model.
        :raises MissingInflowModelError: If none was identified for that cup.
        """
        if source not in self.models:
            raise MissingInflowModelError(source=source, known=sorted(self.models))
        return self.models[source]

    def write(self, path: Path) -> None:
        """
        :param path: The file to write the models to.
        """
        path.write_text(
            json.dumps(
                {source: model.to_json() for source, model in self.models.items()},
                indent=2,
            )
        )

    @classmethod
    def read(cls, path: Path) -> InflowModels:
        """
        :param path: A file written by :meth:`write`.
        :return: The models it holds.
        :raises InflowModelsNotIdentifiedError: If the file does not exist yet.
        """
        if not path.exists():
            raise InflowModelsNotIdentifiedError(path=path)
        return cls(
            models={
                source: FirstOrderInflowModel.from_json(data)
                for source, data in json.loads(path.read_text()).items()
            }
        )


@dataclass
class MissingInflowModelError(Exception):
    """
    A Smith predictor was asked for a source cup no inflow model was identified for.
    """

    source: str
    """
    The source cup that has no model.
    """

    known: List[str]
    """
    The source cups that do.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no inflow model for source cup {self.source!r} (identified: "
            f"{self.known}); run the identification first"
        )


@dataclass
class InflowModelsNotIdentifiedError(Exception):
    """
    A study needing inflow models was started before they were identified.
    """

    path: Path
    """
    Where the models were looked for.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no inflow models at {self.path}; run the identification first"
        )
