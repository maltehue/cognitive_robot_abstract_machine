"""
Whether the hypothesis a pour was explained by also predicts what the pour showed.

A hypothesis can win the bank's weighing while getting a feature of the pour wrong: a
restricted outflow is explained by a later onset, although the first grain left at the
usual tilt. Comparing the explanation's predictions with what was read, feature by
feature, exposes such an explanation before it is written back.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum

import numpy
from typing_extensions import Dict, Optional, Sequence, Tuple

from experiments.pouring_model_verification.hypothesis_bank import (
    INTEGRATION_STEP,
    BankSettings,
    DrainGeometry,
    Reading,
    SensorKnowledge,
)

# %% what a pour shows

EARLY_GRAINS = 10.0
"""
How many grains the early outflow rate is measured over.
"""

PREDICTIVE_SHARE = 0.95
"""
The share of the predictions an observed feature has to fall within.
"""


class Feature(StrEnum):
    """
    A feature of a pour an explanation is checked against.
    """

    FIRST_TILT = "first_tilt"
    """
    Where the first grain left.
    """

    EARLY_RATE = "early_rate"
    """
    How fast the first grains left.
    """

    @property
    def unseen(self) -> float:
        """
        :return: The value standing for a pour that never showed the feature: a first
            grain infinitely late, an early outflow of none.
        """
        match self:
            case Feature.FIRST_TILT:
                return math.inf
            case Feature.EARLY_RATE:
                return 0.0


@dataclass(frozen=True)
class PredictiveInterval:
    """
    The range an explanation expects a feature in.
    """

    low: float
    """
    The lower end.
    """

    high: float
    """
    The upper end.
    """

    def holds(self, value: float) -> bool:
        """
        :return: Whether a value lies within the range.
        """
        return self.low <= value <= self.high


@dataclass(frozen=True)
class PourFeatures:
    """
    The features of a pour an explanation is checked against.
    """

    first_tilt: Optional[float]
    """
    The tilt at the first reading that showed grains leaving the source, in radians, or
    ``None`` if none did.
    """

    early_rate: Optional[float]
    """
    How fast the first :data:`EARLY_GRAINS` grains left, in grains per second, or
    ``None`` if fewer left.
    """

    approach_rate: Optional[float] = None
    """
    How fast the cup turned on its way to the first grain, in radians per second, or
    ``None`` if no grain left.
    """

    def value_of(self, feature: Feature) -> float:
        """
        :return: The value this pour shows for a feature, or the feature's stand-in for
            a pour that never showed it.
        """
        match feature:
            case Feature.FIRST_TILT:
                value = self.first_tilt
            case Feature.EARLY_RATE:
                value = self.early_rate
        return feature.unseen if value is None else value

    @classmethod
    def of(cls, readings: Sequence[Reading], threshold: float) -> PourFeatures:
        """
        :param readings: The pour's readings, in order.
        :param threshold: How many grains the source has to have lost before a reading
            counts as showing them leave: above the wrist's noise.
        :return: The features the readings show.
        """
        times = numpy.array([reading.time for reading in readings])
        tilts = numpy.array([reading.tilt for reading in readings])
        lost = numpy.array(
            [readings[0].source_grains - reading.source_grains for reading in readings]
        )
        return cls.from_series(times, tilts, lost, threshold)

    @classmethod
    def from_series(
        cls,
        times: numpy.ndarray,
        tilts: numpy.ndarray,
        lost: numpy.ndarray,
        threshold: float,
    ) -> PourFeatures:
        """
        :param times: When each reading was taken, in seconds.
        :param tilts: The tilt at each reading, in radians.
        :param lost: How many grains the source had lost by each reading.
        :param threshold: How many lost grains count as grains leaving.
        :return: The features of that series.
        """
        leaving = numpy.nonzero(lost >= threshold)[0]
        if not len(leaving):
            return cls(first_tilt=None, early_rate=None)
        first = leaving[0]
        approach_rate = (
            float((tilts[first] - tilts[0]) / (times[first] - times[0]))
            if times[first] > times[0]
            else None
        )
        early = numpy.nonzero(lost >= threshold + EARLY_GRAINS)[0]
        if not len(early) or times[early[0]] <= times[first]:
            return cls(
                first_tilt=float(tilts[first]),
                early_rate=None,
                approach_rate=approach_rate,
            )
        return cls(
            first_tilt=float(tilts[first]),
            early_rate=EARLY_GRAINS / float(times[early[0]] - times[first]),
            approach_rate=approach_rate,
        )


# %% what the model never captures


@dataclass(frozen=True)
class RateResiduals:
    """
    How far a feature landed from its prediction on pours with nothing set wrong, kept
    apart by how fast the cup turned towards its first grain: a faster turn carries the
    cup further past the onset before a reading shows the grains.
    """

    edges: Tuple[float, ...]
    """
    The tilt rates separating one bin from the next, ascending, in radians per second.
    """

    samples: Tuple[Tuple[float, ...], ...]
    """
    The residuals of each bin, one bin more than there are edges.
    """

    def for_rate(self, rate: Optional[float]) -> numpy.ndarray:
        """
        :param rate: The tilt rate of the pour judged, or ``None`` if unknown.
        :return: The residuals of the bin that rate falls in; for an unknown rate, all
            of them.
        """
        if rate is None:
            return numpy.concatenate([numpy.array(bin_) for bin_ in self.samples])
        return numpy.array(self.samples[int(numpy.searchsorted(self.edges, rate))])

    def to_json(self) -> Dict[str, object]:
        return {"edges": list(self.edges), "samples": [list(b) for b in self.samples]}

    @classmethod
    def from_json(cls, data: Dict[str, object]) -> RateResiduals:
        return cls(
            edges=tuple(data["edges"]),
            samples=tuple(tuple(bin_) for bin_ in data["samples"]),
        )


# %% the verdict on an explanation


@dataclass(frozen=True)
class Consistency:
    """
    How an explanation's predictions compare with what the pour showed.
    """

    observed: PourFeatures
    """
    What the pour showed.
    """

    intervals: Dict[Feature, PredictiveInterval]
    """
    The predictive interval of each feature.
    """

    medians: Dict[Feature, float]
    """
    The explanation's median prediction of each feature, before any residual is added.
    """

    positions: Dict[Feature, float]
    """
    The share of the predictions, residuals added, below the observed value of each
    feature, ties counting half: near zero or one where the observation is unlikely.
    """

    @property
    def outside(self) -> Dict[Feature, bool]:
        """
        :return: For each feature, whether the observation lies outside its predictive
            interval.
        """
        return {
            feature: not interval.holds(self.observed.value_of(feature))
            for feature, interval in self.intervals.items()
        }

    @property
    def consistent(self) -> bool:
        """
        :return: Whether every feature lies within its predictive interval.
        """
        return not any(self.outside.values())

    def to_json(self) -> Dict[str, object]:
        return {
            "first_tilt": self.observed.first_tilt,
            "early_rate": self.observed.early_rate,
            "intervals": {
                feature: [interval.low, interval.high]
                for feature, interval in self.intervals.items()
            },
            "medians": self.medians,
            "positions": self.positions,
            "outside": self.outside,
            "consistent": self.consistent,
        }


@dataclass
class ConsistencyCheck:
    """
    Predicts the features of a pour from an explanation's posterior, along the tilt the
    pour really took and read the way the wrist reads it, and compares them with what
    was read.
    """

    geometry: DrainGeometry
    """
    The source cup as the model sees it.
    """

    sensors: SensorKnowledge
    """
    How the wrist reads the source.
    """

    settings: BankSettings
    """
    The bank's settings, whose noise on the drain the predictions carry.
    """

    seed: int
    """
    Where the predictions' randomness is drawn from.
    """

    draws: int = 400
    """
    How many pours the prediction simulates.
    """

    residuals: Dict[Feature, RateResiduals] = field(default_factory=dict)
    """
    How far each feature landed from the median prediction on pours with nothing set
    wrong, by tilt rate, which the prediction adds to cover what the model never
    captures, such as the burst before the trickle; none where not calibrated.
    """

    predictive_share: float = PREDICTIVE_SHARE
    """
    The share of the predictions an observed feature has to fall within.
    """

    random: numpy.random.Generator = field(init=False)
    """
    The predictions' randomness.
    """

    def __post_init__(self) -> None:
        self.random = numpy.random.default_rng([self.seed, 7])

    @property
    def threshold(self) -> float:
        """
        :return: How many lost grains count as grains leaving: twice the wrist's noise,
            and at least one grain.
        """
        return max(1.0, 2.0 * (self.sensors.source_noise or 0.0))

    def verdict(
        self,
        parameters: numpy.ndarray,
        log_weights: numpy.ndarray,
        readings: Sequence[Reading],
    ) -> Consistency:
        """
        :param parameters: The explanation's particles, one row of parameters each.
        :param log_weights: The particles' log weights.
        :param readings: The pour's readings, in order.
        :return: How the explanation's predictions compare with the readings.
        """
        observed = PourFeatures.of(readings, self.threshold)
        weights = numpy.exp(log_weights - log_weights.max())
        chosen = self.random.choice(
            len(parameters), size=self.draws, p=weights / weights.sum()
        )
        predicted = self._simulate(parameters[chosen], readings)
        intervals = {}
        tail = (1.0 - self.predictive_share) / 2 * 100
        medians, positions = {}, {}
        for feature in Feature:
            values = numpy.array([features.value_of(feature) for features in predicted])
            medians[feature] = float(numpy.median(values))
            residuals = self.residuals.get(feature)
            if residuals is not None:
                drawn = residuals.for_rate(observed.approach_rate)
                if len(drawn):
                    values = values + self.random.choice(drawn, size=len(values))
            value = observed.value_of(feature)
            positions[feature] = float(
                numpy.mean(values < value) + numpy.mean(values == value) / 2
            )
            intervals[feature] = PredictiveInterval(
                low=float(numpy.percentile(values, tail, method="lower")),
                high=float(numpy.percentile(values, 100 - tail, method="higher")),
            )
        return Consistency(
            observed=observed,
            intervals=intervals,
            medians=medians,
            positions=positions,
        )

    def _simulate(self, parameters: numpy.ndarray, readings: Sequence[Reading]) -> list:
        """
        :return: The features of one simulated pour per row of parameters, drained along
            the readings' tilt with the bank's noise on the drain and read with the
            wrist's noise and delay.
        """
        times = numpy.array([reading.time for reading in readings])
        tilts = numpy.array([reading.tilt for reading in readings])
        count = len(parameters)
        source = numpy.full(count, float(readings[0].source_grains))
        states = [source.copy()]
        noise = self.settings.rate_noise
        for index in range(1, len(readings)):
            interval = times[index] - times[index - 1]
            steps = max(1, math.ceil(interval / INTEGRATION_STEP))
            drained = numpy.zeros(count)
            remaining = source.copy()
            for step_index in range(steps):
                share = (step_index + 0.5) / steps
                tilt = (1 - share) * tilts[index - 1] + share * tilts[index]
                head = self.geometry.head(tilt, remaining, parameters)
                flow = numpy.minimum(
                    remaining,
                    self.geometry.outflow(head, parameters) * interval / steps,
                )
                remaining = remaining - flow
                drained = drained + flow
            factor = numpy.exp(self.random.normal(-(noise**2) / 2, noise, count))
            flowing = self.random.random(count) >= self.settings.stall_probability
            source = source - numpy.minimum(source, drained * factor * flowing)
            states.append(source.copy())
        states = numpy.array(states)
        delay = self.sensors.source_delay or 0.0
        read = numpy.array(
            [numpy.interp(times - delay, times, states[:, row]) for row in range(count)]
        )
        read = read + self.random.normal(
            0.0, self.sensors.source_noise or 0.0, read.shape
        )
        start = float(readings[0].source_grains)
        return [
            PourFeatures.from_series(times, tilts, start - read[row], self.threshold)
            for row in range(count)
        ]
