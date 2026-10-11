"""
Predicting how soon a pour can tell a wrong model from the right one, from the Fisher
information its readings carry about the parameter that was set wrong.

Along a recorded pour, each reading's prediction from the state one reading before is
differentiated by the wrong parameters, between the value the model was given and the
true one. The information the readings carry adds up over the pour; the predicted time
to tell the two apart is when half the squared error, weighed by that information, has
grown past the log odds the bank needs to commit.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy
from typing_extensions import Dict, List, Optional, Sequence

from experiments.pouring_model_verification.bank_replay import LoggedPour
from experiments.pouring_model_verification.hypothesis_bank import (
    PARAMETER_ORDER,
    BankSettings,
    FreeParameter,
    ModelPoint,
    StreamGeometry,
)
from experiments.pouring_model_verification.records import Sample
from experiments.pouring_model_verification.sensor import SensorSet
from experiments.pouring_model_verification.transfer_trial import BankReadingKey

# %% the information of one pour

PATH_POINTS = (0.0, 0.5, 1.0)
"""
Where between the given and the true parameters the information is evaluated, as
shares of the way: averaging over the path keeps a large error from being judged by
the slope at one end alone.
"""

STEP = {
    FreeParameter.REPOSE_ANGLE: 1e-3,
    FreeParameter.DISCHARGE_COEFFICIENT: 1e-3,
    FreeParameter.OUTFLOW_RATE_CONSTANT: 1e-3,
}
"""
The step each parameter's derivative is taken over, relative for the coefficient and
the rate, in radians for the repose angle.
"""


@dataclass(frozen=True)
class FisherPrediction:
    """
    How soon a pour's readings tell the model as given from the true one.
    """

    times: numpy.ndarray
    """
    The time of each reading since the transfer started, in seconds.
    """

    separation: numpy.ndarray
    """
    Half the squared error weighed by the information gathered up to each reading, in
    nats: the expected log likelihood ratio of the true model over the given one.
    """

    needed: float
    """
    The log odds the bank needs to commit, in nats.
    """

    lag: float
    """
    How late the receiver's sensor reports what the readings show, in seconds.
    """

    @property
    def time_to_commit(self) -> Optional[float]:
        """
        :return: When the readings carry enough to commit, plus the sensor's lag, or
            ``None`` if they never do within the pour.
        """
        enough = numpy.flatnonzero(self.separation >= self.needed)
        if not len(enough):
            return None
        return float(self.times[enough[0]]) + self.lag


def _point_on_path(given: ModelPoint, true: ModelPoint, share: float) -> ModelPoint:
    """
    :return: The parameters a share of the way from the given to the true ones: the
        repose angle in step, the coefficient and the rate by their ratio.
    """
    return ModelPoint(
        repose_angle=given.repose_angle
        + share * (true.repose_angle - given.repose_angle),
        discharge_coefficient=given.discharge_coefficient
        * (true.discharge_coefficient / given.discharge_coefficient) ** share,
        outflow_rate_constant=given.outflow_rate_constant
        * (true.outflow_rate_constant / given.outflow_rate_constant) ** share,
    )


def _nudged(point: ModelPoint, parameter: FreeParameter) -> ModelPoint:
    """
    :return: The point with one parameter moved by its step.
    """
    value = point.value_of(parameter)
    if parameter is FreeParameter.REPOSE_ANGLE:
        return point.with_values({parameter: value + STEP[parameter]})
    return point.with_values({parameter: value * (1 + STEP[parameter])})


def _error(given: ModelPoint, true: ModelPoint, parameter: FreeParameter) -> float:
    """
    :return: How far the given value lies from the true one, in the units the step is
        taken in: radians, or the log of the ratio.
    """
    if parameter is FreeParameter.REPOSE_ANGLE:
        return true.repose_angle - given.repose_angle
    return math.log(true.value_of(parameter) / given.value_of(parameter))


def fisher_prediction(
    pour: LoggedPour,
    sensors: SensorSet,
    settings: BankSettings,
    wrong: Sequence[FreeParameter],
    hypotheses: int,
) -> FisherPrediction:
    """
    :param pour: A recorded pour with the stream logged at each reading.
    :param sensors: The sensors whose readings carry the information.
    :param settings: The bank's settings: its noise floors and threshold.
    :param wrong: The parameters set wrong.
    :param hypotheses: How many hypotheses the bank weighs, which sets its prior odds.
    :return: The information the pour's readings carry about the wrong parameters.
    """
    geometry = pour.geometry()
    given, true = pour.given, pour.true_point
    receiver, wrist, camera = (
        sensors.receiver.model,
        sensors.wrist.model,
        sensors.camera.model,
    )
    transfer = pour.transfer
    true_states = {
        round(float(sample[Sample.TIME]), 3): sample
        for _, sample in transfer.iterrows()
    }
    source_scale = pour.source_grains_per_fill
    start = float(transfer[Sample.TIME].iloc[0])
    errors = numpy.array([_error(given, true, parameter) for parameter in wrong])
    floor = settings.reading_floor
    times, separation = [], []
    information = numpy.zeros((len(wrong), len(wrong)))
    previous = None
    logged = pour.logged_streams
    for reading in pour.readings(sensors, seed=0):
        sample = true_states[round(reading.time, 3)]
        grains = float(sample[Sample.MEASURED_SOURCE_FILL] * source_scale)
        crossed = logged.get(round(reading.time, 3), {}).get(
            BankReadingKey.TRUE_CROSSING
        )
        if previous is not None:
            interval = reading.time - previous[0]
            information += _reading_information(
                geometry,
                [_point_on_path(given, true, share) for share in PATH_POINTS],
                wrong,
                previous[1],
                previous[2],
                interval,
                reading.stream,
                spreads=(
                    math.hypot(receiver.noise, floor),
                    math.hypot(wrist.noise, floor),
                    (
                        None
                        if camera is None
                        else math.hypot(camera.noise, settings.crossing_floor)
                    ),
                ),
                rate_noise=settings.rate_noise,
                camera_seen=(
                    0.0 if camera is None or crossed is None else 1.0 - camera.dropout
                ),
                spread=settings.stream_spread,
            )
            times.append(reading.time - start)
            separation.append(0.5 * float(errors @ information @ errors))
        previous = (reading.time, grains, reading.tilt)
    threshold = settings.commit_threshold
    return FisherPrediction(
        times=numpy.array(times),
        separation=numpy.array(separation),
        needed=math.log(threshold / (1 - threshold)) + math.log(hypotheses - 1),
        lag=receiver.delay,
    )


def _reading_information(
    geometry,
    points: List[ModelPoint],
    wrong: Sequence[FreeParameter],
    grains: float,
    tilt: float,
    interval: float,
    stream: StreamGeometry,
    spreads: tuple,
    rate_noise: float,
    camera_seen: float,
    spread: float,
) -> numpy.ndarray:
    """
    :return: The Fisher information one reading carries about the wrong parameters,
        averaged over the points along the path from the given to the true ones.
    """
    receiver_spread, source_spread, crossing_spread = spreads
    total = numpy.zeros((len(wrong), len(wrong)))
    for point in points:
        base = _predicted(geometry, point, grains, tilt, interval, stream, spread)
        gradients = numpy.array(
            [
                (
                    _predicted(
                        geometry,
                        _nudged(point, parameter),
                        grains,
                        tilt,
                        interval,
                        stream,
                        spread,
                    )
                    - base
                )
                / STEP[parameter]
                for parameter in wrong
            ]
        )
        drained = base[0]
        variances = numpy.array(
            [
                source_spread**2 + (rate_noise * drained) ** 2,
                receiver_spread**2 + (rate_noise * drained) ** 2,
                (crossing_spread or 1.0) ** 2,
            ]
        )
        weights = numpy.array(
            [1.0, 1.0, 0.0 if crossing_spread is None else camera_seen]
        )
        total += (gradients * weights / variances) @ gradients.T
    return total / len(points)


def _predicted(
    geometry,
    point: ModelPoint,
    grains: float,
    tilt: float,
    interval: float,
    stream: StreamGeometry,
    spread: float,
) -> numpy.ndarray:
    """
    :return: What a model predicts one reading on from a known state: the grains the
        source loses, the grains the receiver gains, and where the stream crosses.
    """
    parameters = numpy.array(
        [[point.value_of(parameter) for parameter in PARAMETER_ORDER]]
    )
    parameters[:, 3] = numpy.nan
    head = geometry.head(tilt, numpy.array([grains]), parameters)
    drained = min(grains, float(geometry.outflow(head, parameters)[0]) * interval)
    speed = geometry.exit_speed(head, parameters)
    inside = float(stream.inside_share(speed, spread)[0])
    return numpy.array([drained, drained * inside, float(stream.distance(speed)[0])])
