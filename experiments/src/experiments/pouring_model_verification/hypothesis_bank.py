"""
Telling online which concept of the pouring model is wrong, from one pour.

A bank of hypotheses runs alongside the pour, each a version of the model in which a
named concept is left free: where the contents start to flow, where the stream lands,
both, a second lip at the slot edge of a spouted cup, or none of them with the noise
inflated to catch what no concept covers. Each hypothesis tracks the fill levels and its
free parameters with a particle filter; every reading updates how probable each
hypothesis is, and a small chance of switching keeps every hypothesis alive.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy
from scipy.special import logsumexp, ndtr
from typing_extensions import Any, Dict, List, Optional, Tuple

from semantic_digital_twin.physics.equations.pouring_equations import (
    MINIMUM_POUR_HEAD,
    STANDARD_GRAVITY,
)

# %% the concepts a hypothesis can leave free


class FreeParameter(StrEnum):
    """
    A parameter of the pouring model a hypothesis estimates from the pour.
    """

    REPOSE_ANGLE = "repose_angle"
    """
    The angle the contents hold before they flow, in radians: where the pour starts.
    """

    DISCHARGE_COEFFICIENT = "discharge_coefficient"
    """
    The factor on the exit speed: where the stream lands.
    """

    OUTFLOW_RATE_CONSTANT = "outflow_rate_constant"
    """
    How fast a head above the lip drains the source, in 1/s.
    """

    SLOT_HEIGHT = "slot_height"
    """
    How high above the cup's floor the slot edge of a spouted cup stands, in metres.
    """


PARAMETER_ORDER = (
    FreeParameter.REPOSE_ANGLE,
    FreeParameter.DISCHARGE_COEFFICIENT,
    FreeParameter.OUTFLOW_RATE_CONSTANT,
    FreeParameter.SLOT_HEIGHT,
)
"""
The column each parameter takes in a filter's particles.
"""


class Hypothesis(StrEnum):
    """
    A version of the pouring model the bank weighs against the pour.
    """

    NOMINAL = "nominal"
    """
    The model as given, with nothing left free.
    """

    ONSET = "onset"
    """
    The contents start to flow at another tilt: the repose angle is free.
    """

    AIM = "aim"
    """
    The stream lands elsewhere: the discharge coefficient is free.
    """

    COUPLED = "coupled"
    """
    Both at once: the repose angle and the discharge coefficient are free together.
    """

    SECOND_LIP = "second_lip"
    """
    A spouted cup also pours over the edge of the slot in its wall, whichever of the
    two lips stands lower at the tilt: the slot edge's height and the repose angle are
    free.
    """

    CATCH_ALL = "catch_all"
    """
    The model as given, with the noise inflated: what wins when no concept explains the
    pour.
    """

    OUTFLOW_CAPACITY = "outflow_capacity"
    """
    Less, or more, can pass the lip than the model says: the drain rate is free, as a
    concept of its own apart from where the pour starts.
    """

    @property
    def label(self) -> str:
        """
        :return: The short name the report uses, ``H0`` to ``H6``.
        """
        return f"H{list(Hypothesis).index(self)}"

    def free_parameters(self, frees_rate: bool) -> Tuple[FreeParameter, ...]:
        """
        :param frees_rate: Whether every concept hypothesis also frees the drain rate.
        :return: The parameters this hypothesis estimates.
        """
        match self:
            case Hypothesis.NOMINAL | Hypothesis.CATCH_ALL:
                return ()
            case Hypothesis.ONSET:
                free = (FreeParameter.REPOSE_ANGLE,)
            case Hypothesis.AIM:
                free = (FreeParameter.DISCHARGE_COEFFICIENT,)
            case Hypothesis.COUPLED:
                free = (
                    FreeParameter.REPOSE_ANGLE,
                    FreeParameter.DISCHARGE_COEFFICIENT,
                )
            case Hypothesis.SECOND_LIP:
                free = (FreeParameter.SLOT_HEIGHT, FreeParameter.REPOSE_ANGLE)
            case Hypothesis.OUTFLOW_CAPACITY:
                return (FreeParameter.OUTFLOW_RATE_CONSTANT,)
        if frees_rate:
            return free + (FreeParameter.OUTFLOW_RATE_CONSTANT,)
        return free

    @property
    def concepts(self) -> frozenset[FreeParameter]:
        """
        :return: The concepts of the model this hypothesis holds to be wrong: the
            parameters it frees of its own, not the drain rate every hypothesis may also
            be given free.
        """
        return frozenset(self.free_parameters(frees_rate=False))

    @property
    def opens_slot(self) -> bool:
        """
        :return: Whether this hypothesis pours over the slot edge as well.
        """
        return self is Hypothesis.SECOND_LIP

    @property
    def adapts_onset(self) -> bool:
        """
        :return: Whether this hypothesis keeps the onset as the twin has it, which with
            an onset spread estimates each pour's repose angle within that spread.
        """
        return FreeParameter.REPOSE_ANGLE not in self.concepts

    @property
    def adapts_drain_rate(self) -> bool:
        """
        :return: Whether this hypothesis keeps the drain rate as the twin has it, which
            with a drain rate spread estimates each pour's rate within that spread.
        """
        return FreeParameter.OUTFLOW_RATE_CONSTANT not in self.concepts

    @property
    def revises(self) -> bool:
        """
        :return: Whether committing to this hypothesis changes the controller's model.
        """
        return self not in (Hypothesis.NOMINAL, Hypothesis.CATCH_ALL)


# %% the model's parameters at one point


@dataclass(frozen=True)
class ModelPoint:
    """
    The values the pouring model is evaluated with.
    """

    repose_angle: float
    """
    Angle the contents hold before they flow, in radians.
    """

    discharge_coefficient: float
    """
    Factor on the Torricelli exit speed.
    """

    outflow_rate_constant: float
    """
    Outflow rate constant of the drain, in 1/s.
    """

    slot_height: Optional[float] = None
    """
    Height of the slot edge above the cup's floor, in metres, or ``None`` for a model
    that pours over its one lip only.
    """

    def value_of(self, parameter: FreeParameter) -> float:
        """
        :param parameter: The parameter to read.
        :return: Its value; ``nan`` for the slot height of a model without a slot.
        """
        match parameter:
            case FreeParameter.REPOSE_ANGLE:
                return self.repose_angle
            case FreeParameter.DISCHARGE_COEFFICIENT:
                return self.discharge_coefficient
            case FreeParameter.OUTFLOW_RATE_CONSTANT:
                return self.outflow_rate_constant
            case FreeParameter.SLOT_HEIGHT:
                return math.nan if self.slot_height is None else self.slot_height

    def with_values(self, values: Dict[FreeParameter, float]) -> ModelPoint:
        """
        :param values: New values for some parameters.
        :return: A copy of this point with those parameters changed.
        """
        changes = {parameter.value: value for parameter, value in values.items()}
        return replace(self, **changes)

    def to_json(self) -> Dict[str, Any]:
        return {
            parameter.value: self.value_of(parameter) for parameter in FreeParameter
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> ModelPoint:
        slot = data[FreeParameter.SLOT_HEIGHT]
        return cls(
            repose_angle=data[FreeParameter.REPOSE_ANGLE],
            discharge_coefficient=data[FreeParameter.DISCHARGE_COEFFICIENT],
            outflow_rate_constant=data[FreeParameter.OUTFLOW_RATE_CONSTANT],
            slot_height=None if slot is None or math.isnan(slot) else slot,
        )


# %% the bounds the twin puts on each parameter


REPOSE_ANGLE_BOUNDS = (0.0, math.radians(65.0))
"""
The repose angles the bank considers, in radians: from a free liquid surface to the
steepest heap a dry granular material stands in.
"""

DISCHARGE_COEFFICIENT_BOUNDS = (0.03, 1.0)
"""
The discharge coefficients the bank considers: a stream cannot leave faster than a free
fall from the head gives it.
"""

RATE_FACTOR_BOUNDS = (0.1, 10.0)
"""
How far the bank lets the drain rate depart from the model's, as factors on it.
"""


@dataclass(frozen=True)
class ParameterBounds:
    """
    The physically plausible range of each free parameter, as the twin knows it.
    """

    container_height: float
    """
    Height of the cup, which bounds the slot edge, in metres.
    """

    rate: float
    """
    The model's drain rate, which the rate's bounds are factors on, in 1/s.
    """

    def of(self, parameter: FreeParameter) -> Tuple[float, float]:
        """
        :param parameter: A free parameter.
        :return: Its lowest and highest plausible value.
        """
        match parameter:
            case FreeParameter.REPOSE_ANGLE:
                return REPOSE_ANGLE_BOUNDS
            case FreeParameter.DISCHARGE_COEFFICIENT:
                return DISCHARGE_COEFFICIENT_BOUNDS
            case FreeParameter.OUTFLOW_RATE_CONSTANT:
                return (
                    self.rate * RATE_FACTOR_BOUNDS[0],
                    self.rate * RATE_FACTOR_BOUNDS[1],
                )
            case FreeParameter.SLOT_HEIGHT:
                return (0.0, self.container_height)

    @staticmethod
    def is_logarithmic(parameter: FreeParameter) -> bool:
        """
        :return: Whether the parameter is a factor, drawn and walked on a log scale.
        """
        return parameter in (
            FreeParameter.DISCHARGE_COEFFICIENT,
            FreeParameter.OUTFLOW_RATE_CONSTANT,
        )


# %% the drain, for many particles at once


@dataclass(frozen=True)
class DrainGeometry:
    """
    The source cup as the pouring model sees it, with the grains as its unit.
    """

    container_height: float
    """
    Height of the outlet above the cup's floor, in metres.
    """

    lip_offset: float
    """
    Horizontal distance of the outlet from the tilt axis, in metres.
    """

    grains_per_fill: float
    """
    How many grains fill the cup to the top.
    """

    slot_offset: float
    """
    Horizontal distance of the slot edge from the tilt axis, in metres.
    """

    @staticmethod
    def _lip_head(
        offset: numpy.ndarray | float,
        height: numpy.ndarray | float,
        tilt: float,
        surface: numpy.ndarray,
        repose_angle: numpy.ndarray,
    ) -> numpy.ndarray:
        """
        :return: How far the contents' surface stands above one lip, in metres.
        """
        dry = height - surface
        reach = numpy.sqrt(dry**2 + offset**2)
        onset = numpy.arctan2(dry, offset) + repose_angle
        return numpy.maximum(0.0, reach * numpy.sin(tilt - onset))

    def head(
        self, tilt: float, source_grains: numpy.ndarray, parameters: numpy.ndarray
    ) -> numpy.ndarray:
        """
        :param tilt: The cup's tilt, in radians.
        :param source_grains: The grains in the source, one per particle.
        :param parameters: The particles' parameters, one row each, in
            :data:`PARAMETER_ORDER`.
        :return: The head above whichever lip stands lower, in metres.
        """
        surface = source_grains / self.grains_per_fill * self.container_height
        repose = parameters[:, 0]
        outlet = self._lip_head(
            self.lip_offset, self.container_height, tilt, surface, repose
        )
        slot = parameters[:, 3]
        has_slot = ~numpy.isnan(slot)
        if not has_slot.any():
            return outlet
        over_slot = self._lip_head(
            self.slot_offset, numpy.where(has_slot, slot, 0.0), tilt, surface, repose
        )
        return numpy.where(has_slot, numpy.maximum(outlet, over_slot), outlet)

    def outflow(self, head: numpy.ndarray, parameters: numpy.ndarray) -> numpy.ndarray:
        """
        :param head: The head above the lip, in metres.
        :param parameters: The particles' parameters.
        :return: How many grains leave the source per second.
        """
        return parameters[:, 2] * head / self.container_height * self.grains_per_fill

    @staticmethod
    def exit_speed(head: numpy.ndarray, parameters: numpy.ndarray) -> numpy.ndarray:
        """
        :return: How fast the stream leaves the lip, in metres per second.
        """
        return parameters[:, 1] * numpy.sqrt(
            2.0 * STANDARD_GRAVITY * numpy.maximum(head, MINIMUM_POUR_HEAD)
        )

    def onset_tilt(self, source_grains: float, point: ModelPoint) -> float:
        """
        :param source_grains: The grains in the source.
        :param point: The model's parameters.
        :return: The tilt at which the model says these grains start to flow.
        """
        surface = source_grains / self.grains_per_fill * self.container_height
        dry = self.container_height - surface
        onset = math.atan2(dry, self.lip_offset) + point.repose_angle
        if point.slot_height is None:
            return onset
        slot_dry = point.slot_height - surface
        return min(onset, math.atan2(slot_dry, self.slot_offset) + point.repose_angle)


# %% where the stream lands


@dataclass(frozen=True)
class StreamGeometry:
    """
    How the stream leaves the lip now, and where the receiver's opening is from there.
    """

    horizontal_share: float
    """
    The share of the exit speed that is horizontal.
    """

    vertical_share: float
    """
    The share of the exit speed that is upwards.
    """

    drop: float
    """
    How far the lip stands above the receiver's opening, in metres.
    """

    centre_distance: float
    """
    How far the receiver's centre lies from the lip, along the stream's horizontal
    direction, in metres.
    """

    opening_radius: float
    """
    The receiver opening's radius, in metres.
    """

    def distance(self, speed: numpy.ndarray) -> numpy.ndarray:
        """
        :param speed: Exit speeds, in metres per second.
        :return: How far from the lip, horizontally, each stream crosses the opening.
        """
        upwards = speed * self.vertical_share
        flight = (
            upwards + numpy.sqrt(upwards**2 + 2 * STANDARD_GRAVITY * self.drop)
        ) / STANDARD_GRAVITY
        return speed * self.horizontal_share * flight

    def inside_share(self, speed: numpy.ndarray, spread: float) -> numpy.ndarray:
        """
        :param speed: Exit speeds, in metres per second.
        :param spread: How widely the grains of one stream scatter, in metres.
        :return: The share of each stream that lands inside the opening.
        """
        offset = self.distance(speed) - self.centre_distance
        radius = self.opening_radius
        return ndtr((radius - offset) / spread) - ndtr((-radius - offset) / spread)

    def to_json(self) -> Dict[str, float]:
        return {
            "horizontal_share": self.horizontal_share,
            "vertical_share": self.vertical_share,
            "drop": self.drop,
            "centre_distance": self.centre_distance,
            "opening_radius": self.opening_radius,
        }


# %% what the bank is told


@dataclass(frozen=True)
class Reading:
    """
    One reading of every sensor, with the tilt it was taken at.
    """

    time: float
    """
    When the reading was taken, in seconds.
    """

    tilt: float
    """
    The cup's tilt then, in radians.
    """

    receiver_grains: Optional[float]
    """
    What the receiver's sensor reported, in grains, or ``None`` with nothing under the
    receiver.
    """

    source_grains: Optional[float]
    """
    What the wrist reported the source to hold, in grains, or ``None`` without one.
    """

    crossing_distance: Optional[float] = None
    """
    How far from the lip the camera saw the stream cross the opening plane, in metres,
    or ``None`` if it saw no stream or there is no camera.
    """

    stream: Optional[StreamGeometry] = None
    """
    How the stream leaves the lip now, or ``None`` where it is not known.
    """

    stream_seen: Optional[bool] = None
    """
    Whether the camera saw a stream cross the opening plane, or ``None`` where a lost
    stream tells nothing.
    """

    counted_grains: Optional[float] = None
    """
    How many grains the camera has counted crossing the opening plane since the pour
    began, or ``None`` if it counts none.
    """


@dataclass(frozen=True)
class SensorKnowledge:
    """
    What the bank knows about the sensors it reads: how noisy and how late each is.
    """

    receiver_noise: Optional[float]
    """
    Standard deviation of the receiver's reading, in grains, or ``None`` with nothing
    under the receiver.
    """

    receiver_delay: float
    """
    How late the receiver's reading arrives, in seconds.
    """

    counts_falling_grains: bool
    """
    Whether the receiver's reading counts every grain still falling above its opening,
    as a camera watching the stream would, rather than only those that have landed.
    """

    fall_time: float
    """
    How long a grain falls from the lip to the receiver's opening, in seconds.
    """

    source_noise: Optional[float]
    """
    Standard deviation of the wrist's reading, in grains, or ``None`` without a wrist.
    """

    source_delay: float
    """
    How late the wrist's reading arrives, in seconds.
    """

    crossing_noise: Optional[float]
    """
    Standard deviation of where the camera sees the stream cross, in metres, or ``None``
    without a camera.
    """

    stream_dropout: Optional[float] = None
    """
    The share of readings in which the camera misses a stream that is there, or
    ``None`` if a reading without a stream is not weighed at all.
    """

    count_noise: Optional[float] = None
    """
    Relative standard deviation of the grains the camera counts crossing over a pour,
    or ``None`` if it counts none.
    """

    def to_json(self) -> Dict[str, Any]:
        return dict(self.__dict__)


STREAM_FALSE_SIGHTING = 0.01
"""
The share of readings in which a camera sees a stream where none falls.
"""

COUNT_FLOOR = 2.0
"""
The least standard deviation, in grains, a camera's count of crossed grains is trusted
to.
"""

# %% the twin's values as they vary from pour to pour

INTERVAL_SPREADS = 1.96
"""
How many spreads either side of the centre the twin's 95 % interval of a value reaches.
"""

OUTSIDE_SHARE = 0.95
"""
The share of a pour's posterior that has to lie outside the twin's interval before the
twin's centre counts as wrong.
"""


@dataclass(frozen=True)
class ParameterSpread:
    """
    A value of the twin as the pours scatter it while the cup turns: a centre and a
    spread, each pour drawing its own value from them.
    """

    centre: float
    """
    The value the pours scatter around.
    """

    spread: float
    """
    How widely they scatter, as a standard deviation on the scale the value scatters on.
    """

    def _scaled(self, values: numpy.ndarray | float) -> numpy.ndarray | float:
        """
        :return: Values on the scale they scatter on.
        """
        return values

    def _unscaled(self, values: numpy.ndarray | float) -> numpy.ndarray | float:
        """
        :return: Values on the scale they scatter on, back on their own.
        """
        return values

    @property
    def interval(self) -> Tuple[float, float]:
        """
        :return: The range 95 % of the pours' values fall in.
        """
        half = INTERVAL_SPREADS * self.spread
        centre = self._scaled(self.centre)
        return float(self._unscaled(centre - half)), float(
            self._unscaled(centre + half)
        )

    def draw(self, random: numpy.random.Generator, count: int) -> numpy.ndarray:
        """
        :param random: Where the draws come from.
        :param count: How many pours to draw.
        :return: One value per pour.
        """
        return self._unscaled(
            random.normal(self._scaled(self.centre), self.spread, count)
        )

    def share_inside(self, values: numpy.ndarray, weights: numpy.ndarray) -> float:
        """
        :param values: Particles of a pour's value.
        :param weights: Their weights.
        :return: The share of the weight lying within the interval.
        """
        low, high = self.interval
        inside = (values >= low) & (values <= high)
        return float(numpy.sum(weights[inside]) / numpy.sum(weights))

    def side_of(self, value: float) -> int:
        """
        :param value: A pour's value.
        :return: ``1`` above the interval, ``-1`` below it, ``0`` within it.
        """
        low, high = self.interval
        return 1 if value > high else -1 if value < low else 0

    def to_json(self) -> Dict[str, float]:
        return {"centre": self.centre, "spread": self.spread}

    @classmethod
    def from_json(cls, data: Dict[str, float]) -> ParameterSpread:
        return cls(centre=data["centre"], spread=data["spread"])


@dataclass(frozen=True)
class OnsetSpread(ParameterSpread):
    """
    The twin's repose angle while the cup turns, in radians, scattering evenly either
    side of its centre.
    """


@dataclass(frozen=True)
class DrainRateSpread(ParameterSpread):
    """
    The twin's drain rate while the cup turns, in 1/s, scattering by factors: its spread
    is a standard deviation of the rate's logarithm.
    """

    def _scaled(self, values: numpy.ndarray | float) -> numpy.ndarray | float:
        return numpy.log(values)

    def _unscaled(self, values: numpy.ndarray | float) -> numpy.ndarray | float:
        return numpy.exp(values)


# %% the settings of the bank


@dataclass(frozen=True)
class BankSettings:
    """
    The tunable settings of the hypothesis bank and the controller's coupling to it.
    """

    label: str = "bank"
    """
    A short name of this set of settings, usable in a file name.
    """

    particles: int = 200
    """
    How many particles each hypothesis's filter carries.
    """

    rate_noise: float = 0.5
    """
    Standard deviation of the log of the factor the drain departs from its model by,
    drawn afresh at each reading.
    """

    stall_probability: float = 0.2
    """
    The chance that the grains hold still for one reading where the model says they
    flow: granular flow starts late and comes in avalanches.
    """

    reading_floor: float = 1.0
    """
    The least standard deviation a grain reading is trusted to, in grains: the grains
    flow in clumps the model smooths over.
    """

    crossing_floor: float = 0.005
    """
    The least standard deviation a crossing is trusted to, in metres.
    """

    stream_spread: float = 0.01
    """
    How widely the grains of one stream scatter around where it lands, in metres.
    """

    repose_walk: float = 0.005
    """
    How far the repose angle may drift between readings, in radians.
    """

    factor_walk: float = 0.01
    """
    How far the log of the discharge coefficient or the drain rate may drift between
    readings.
    """

    slot_walk: float = 0.0005
    """
    How far the slot edge's height may drift between readings, in metres.
    """

    switching_probability: float = 0.01
    """
    The chance per reading that the pour moves from one hypothesis to another.
    """

    catch_all_inflation: float = 3.0
    """
    How much more noise the catch-all hypothesis assumes, on the drain and on every
    reading.
    """

    catch_all_leak: float = 2.0
    """
    How many grains per square root of a second the catch-all lets flow where the model
    says none does.
    """

    commit_threshold: float = 0.9
    """
    How probable a hypothesis has to be before the controller switches to it.
    """

    confirmations: int = 3
    """
    How many readings in a row have to call for the same hypothesis before the
    controller switches to it.
    """

    trickle: float = 2.0
    """
    The least flow the controller's model expects at the tilt, in grains per second:
    its onset is held low enough that it does, so it always has a gradient to tilt
    along, and little enough that the controller still tilts on where no grains flow.
    """

    onset_quantile: float = 0.1
    """
    The quantile of the repose angle and the slot height the controller runs on: low,
    so the model expects flow just past where none has been seen yet and still has a
    gradient to tilt along, rather than a mean that may sit beyond the tilt.
    """

    margin_factor: float = 1.0
    """
    How many standard deviations of the predicted final fill the controller stops short
    of the goal by.
    """

    frees_rate: bool = False
    """
    Whether every concept hypothesis also leaves the drain rate free.
    """

    adapts_onset: bool = False
    """
    Whether the model as given carries the twin's spread of the repose angle, estimating
    each pour's angle within it, and the onset counts as wrong only outside it.
    """

    outflow_capacity: bool = False
    """
    Whether the twin has the concept of how much can pass the lip, weighed as a
    hypothesis of its own.
    """

    checks_consistency: bool = False
    """
    Whether the explanation a pour ends on is checked against what the pour showed.
    """

    static_onset: bool = False
    """
    Whether a hypothesis holding the onset wrong treats the pour's repose angle as
    unknown but constant within the pour, rather than letting it drift between
    readings.
    """

    adapts_drain_rate: bool = False
    """
    Whether every hypothesis that keeps the drain rate carries the twin's spread of it,
    estimating each pour's rate within it, and the drain counts as wrong only outside
    it.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            name: value
            for name, value in self.__dict__.items()
            if not name.startswith("_")
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> BankSettings:
        return cls(**data)


RESAMPLING_SHARE = 0.5
"""
The effective share of particles below which a filter resamples.
"""

KERNEL_SHRINKAGE = 0.98
"""
How strongly resampled parameters are pulled towards their mean before they are
scattered again, so the resampled cloud keeps its spread.
"""

INTEGRATION_STEP = 0.025
"""
The longest step the drain is integrated over between readings, in seconds.
"""

HISTORY = 1.5
"""
How far back each filter keeps its states, to answer late readings, in seconds.
"""

MARGIN_HORIZON = 0.5
"""
How far ahead the spread of the final fill is predicted, in seconds: about how long the
grains already leaving keep arriving once the cup stops.
"""


# %% one hypothesis, followed by a particle filter


@dataclass
class HypothesisFilter:
    """
    A particle filter over the fill levels and the free parameters of one hypothesis.
    """

    hypothesis: Hypothesis
    """
    The version of the model the filter follows.
    """

    geometry: DrainGeometry
    """
    The source cup as the model sees it.
    """

    base: ModelPoint
    """
    The model's parameters, which the free ones start around and the others keep.
    """

    sensors: SensorKnowledge
    """
    What the filter knows about the readings.
    """

    settings: BankSettings
    """
    The bank's settings.
    """

    random: numpy.random.Generator
    """
    Where the filter's randomness is drawn from.
    """

    onset: Optional[OnsetSpread] = None
    """
    How the repose angle scatters from pour to pour, which a hypothesis holding the
    model as given draws each pour's angle from, or ``None`` for one fixed angle.
    """

    drain_rate: Optional[DrainRateSpread] = None
    """
    How the drain rate scatters from pour to pour, which a hypothesis keeping the drain
    rate draws each pour's rate from, or ``None`` for one fixed rate.
    """

    parameters: numpy.ndarray = field(init=False)
    """
    One row of parameters per particle, in :data:`PARAMETER_ORDER`.
    """

    source: numpy.ndarray = field(init=False)
    """
    The grains in the source, per particle.
    """

    committed: numpy.ndarray = field(init=False)
    """
    The grains that left the source bound for the receiver, per particle.
    """

    spilled: numpy.ndarray = field(init=False)
    """
    The grains that left the source bound for elsewhere, per particle.
    """

    log_weights: numpy.ndarray = field(init=False)
    """
    The particles' normalised log weights.
    """

    clipped: int = field(init=False, default=0)
    """
    How many times a parameter of a particle was held at its bound.
    """

    walked: int = field(init=False, default=0)
    """
    How many parameter steps the particles have taken.
    """

    _times: List[float] = field(init=False, default_factory=list)
    """
    When each kept state was reached, oldest first, in seconds.
    """

    _sources: List[numpy.ndarray] = field(init=False, default_factory=list)
    """
    The grains in the source at each kept time.
    """

    _committed: List[numpy.ndarray] = field(init=False, default_factory=list)
    """
    The grains bound for the receiver at each kept time.
    """

    _spilled: List[numpy.ndarray] = field(init=False, default_factory=list)
    """
    The grains bound for elsewhere at each kept time.
    """

    _last_tilt: Optional[float] = field(init=False, default=None)
    """
    The tilt of the previous reading.
    """

    _last_head: numpy.ndarray = field(init=False)
    """
    The head each particle had at the previous reading, in metres.
    """

    @property
    def free(self) -> Tuple[FreeParameter, ...]:
        """
        :return: The parameters this filter estimates: with a spread of the twin's
            repose angle or drain rate, a hypothesis keeping that value estimates the
            pour's own too.
        """
        free = self.hypothesis.free_parameters(self.settings.frees_rate)
        if self.adapts_onset:
            free = (FreeParameter.REPOSE_ANGLE,) + free
        if self.adapts_drain_rate and FreeParameter.OUTFLOW_RATE_CONSTANT not in free:
            free = free + (FreeParameter.OUTFLOW_RATE_CONSTANT,)
        return free

    @property
    def keeps_onset_static(self) -> bool:
        """
        :return: Whether the pour's repose angle stays where it was drawn: drawn from
            the twin's spread, or held wrong by the onset or coupled hypothesis under a
            static onset.
        """
        return self.adapts_onset or (
            self.settings.static_onset
            and self.hypothesis in (Hypothesis.ONSET, Hypothesis.COUPLED)
        )

    @property
    def adapts_onset(self) -> bool:
        """
        :return: Whether this filter draws the pour's repose angle from the twin's
            spread rather than from its bounds.
        """
        return self.onset is not None and self.hypothesis.adapts_onset

    @property
    def adapts_drain_rate(self) -> bool:
        """
        :return: Whether this filter draws the pour's drain rate from the twin's spread
            rather than from its bounds.
        """
        return self.drain_rate is not None and self.hypothesis.adapts_drain_rate

    def _spread_of(self, parameter: FreeParameter) -> Optional[ParameterSpread]:
        """
        :return: The twin's spread this filter draws the pour's value of a parameter
            from, or ``None`` where it draws from the bounds.
        """
        match parameter:
            case FreeParameter.REPOSE_ANGLE if self.adapts_onset:
                return self.onset
            case FreeParameter.OUTFLOW_RATE_CONSTANT if self.adapts_drain_rate:
                return self.drain_rate
        return None

    @property
    def bounds(self) -> ParameterBounds:
        """
        :return: The bounds on its parameters.
        """
        return ParameterBounds(
            container_height=self.geometry.container_height,
            rate=self.base.outflow_rate_constant,
        )

    @property
    def inflation(self) -> float:
        """
        :return: How much the filter inflates the noise it assumes.
        """
        if self.hypothesis is Hypothesis.CATCH_ALL:
            return self.settings.catch_all_inflation
        return 1.0

    def start(self, reading: Reading) -> None:
        """
        Spread the particles over the prior, starting from the first reading.

        :param reading: The first reading.
        """
        count = self.settings.particles
        point = self.base
        if self.hypothesis.opens_slot and point.slot_height is None:
            point = replace(point, slot_height=self.geometry.container_height)
        self.parameters = numpy.tile(
            [point.value_of(parameter) for parameter in PARAMETER_ORDER], (count, 1)
        )
        for parameter in self.free:
            low, high = self.bounds.of(parameter)
            column = PARAMETER_ORDER.index(parameter)
            spread = self._spread_of(parameter)
            if spread is not None:
                draws = numpy.clip(spread.draw(self.random, count), low, high)
            elif ParameterBounds.is_logarithmic(parameter):
                draws = numpy.exp(
                    self.random.uniform(math.log(low), math.log(high), count)
                )
            else:
                draws = self.random.uniform(low, high, count)
            self.parameters[:, column] = draws
        source = (
            reading.source_grains
            if reading.source_grains is not None
            else self.geometry.grains_per_fill
        )
        spread = math.hypot(
            self.sensors.source_noise or 0.0, self.settings.reading_floor
        )
        self.source = numpy.maximum(0.0, source + self.random.normal(0, spread, count))
        received = reading.receiver_grains
        if received is None:
            received = 0.0
        self.committed = numpy.full(count, received)
        self.spilled = numpy.zeros(count)
        self.log_weights = numpy.full(count, -math.log(count))
        self._last_tilt = reading.tilt
        self._last_head = numpy.zeros(count)
        self._last_rate = numpy.zeros(count)
        self._last_interval = 0.0
        self._loaded = self.source.copy()
        self._remember(reading.time)

    def _remember(self, at: float) -> None:
        """
        Keep the states reached now, dropping those too old to answer any reading.
        """
        self._times.append(at)
        self._sources.append(self.source.copy())
        self._committed.append(self.committed.copy())
        self._spilled.append(self.spilled.copy())
        while self._times and self._times[0] < at - HISTORY:
            self._times.pop(0)
            self._sources.pop(0)
            self._committed.pop(0)
            self._spilled.pop(0)

    def _state_at(self, states: List[numpy.ndarray], at: float) -> numpy.ndarray:
        """
        :param states: Kept states, one per kept time.
        :param at: A past time, in seconds.
        :return: The states at that time, interpolated between the kept ones.
        """
        times = self._times
        if at <= times[0]:
            return states[0]
        if at >= times[-1]:
            return states[-1]
        index = int(numpy.searchsorted(times, at))
        share = (at - times[index - 1]) / (times[index] - times[index - 1])
        return (1 - share) * states[index - 1] + share * states[index]

    def predict(self, reading: Reading) -> None:
        """
        Carry every particle from the previous reading to this one along the tilt.

        :param reading: The reading to carry the particles to.
        """
        interval = reading.time - self._times[-1]
        if interval <= 0.0:
            return
        self._walk()
        steps = max(1, math.ceil(interval / INTEGRATION_STEP))
        step = interval / steps
        source = self.source.copy()
        drained = numpy.zeros_like(source)
        head = self._last_head
        for index in range(steps):
            share = (index + 0.5) / steps
            tilt = (1 - share) * self._last_tilt + share * reading.tilt
            head = self.geometry.head(tilt, source, self.parameters)
            flow = numpy.minimum(
                source, self.geometry.outflow(head, self.parameters) * step
            )
            source = source - flow
            drained = drained + flow
        noise = self.settings.rate_noise * self.inflation
        factor = numpy.exp(self.random.normal(-(noise**2) / 2, noise, len(source)))
        flowing = self.random.random(len(source)) >= self.settings.stall_probability
        outflow = drained * factor * flowing
        if self.hypothesis is Hypothesis.CATCH_ALL:
            outflow = outflow + numpy.abs(
                self.random.normal(
                    0, self.settings.catch_all_leak * math.sqrt(interval), len(source)
                )
            )
        outflow = numpy.minimum(self.source, outflow)
        inside = (
            numpy.ones_like(outflow)
            if reading.stream is None
            else reading.stream.inside_share(
                self.geometry.exit_speed(head, self.parameters),
                self.settings.stream_spread,
            )
        )
        self._last_rate = outflow / interval
        self._last_interval = interval
        self.source = self.source - outflow
        self.committed = self.committed + outflow * inside
        self.spilled = self.spilled + outflow * (1 - inside)
        self._last_tilt = reading.tilt
        self._last_head = head
        self._remember(reading.time)

    def _walk(self) -> None:
        """
        Let each free parameter drift a little, held within its bounds. A value drawn
        from the twin's spread is the pour's own and stays put.
        """
        for parameter in self.free:
            if self._spread_of(parameter) is not None or (
                parameter is FreeParameter.REPOSE_ANGLE and self.keeps_onset_static
            ):
                continue
            column = PARAMETER_ORDER.index(parameter)
            low, high = self.bounds.of(parameter)
            values = self.parameters[:, column]
            match parameter:
                case FreeParameter.REPOSE_ANGLE:
                    values = values + self.random.normal(
                        0, self.settings.repose_walk, len(values)
                    )
                case FreeParameter.SLOT_HEIGHT:
                    values = values + self.random.normal(
                        0, self.settings.slot_walk, len(values)
                    )
                case _:
                    values = values * numpy.exp(
                        self.random.normal(0, self.settings.factor_walk, len(values))
                    )
            self.clipped += int(numpy.sum((values < low) | (values > high)))
            self.walked += len(values)
            self.parameters[:, column] = numpy.clip(values, low, high)

    def receiver_reading_at(self, at: float) -> numpy.ndarray:
        """
        :param at: The time of a reading, in seconds.
        :return: What each particle says the receiver's sensor reports then: the grains
            landed, or for a sensor that counts falling grains also every grain still
            above the opening, whichever cup it falls towards.
        """
        sensors = self.sensors
        seen = at - sensors.receiver_delay
        if not sensors.counts_falling_grains:
            return self._state_at(self._committed, seen - sensors.fall_time)
        falling_elsewhere = self._state_at(self._spilled, seen) - self._state_at(
            self._spilled, seen - sensors.fall_time
        )
        return self._state_at(self._committed, seen) + falling_elsewhere

    def log_likelihoods(self, reading: Reading) -> numpy.ndarray:
        """
        :param reading: The reading just taken.
        :return: The log likelihood of it under each particle.
        """
        floor, inflation = self.settings.reading_floor, self.inflation
        result = numpy.zeros(len(self.source))
        if (
            reading.receiver_grains is not None
            and self.sensors.receiver_noise is not None
        ):
            receiver = self.receiver_reading_at(reading.time)
            spread = inflation * math.hypot(self.sensors.receiver_noise, floor)
            result = result + _log_normal(reading.receiver_grains, receiver, spread)
        if reading.source_grains is not None and self.sensors.source_noise is not None:
            source = self._state_at(
                self._sources, reading.time - self.sensors.source_delay
            )
            spread = inflation * math.hypot(self.sensors.source_noise, floor)
            result = result + _log_normal(reading.source_grains, source, spread)
        if (
            reading.crossing_distance is not None
            and reading.stream is not None
            and self.sensors.crossing_noise is not None
        ):
            predicted = reading.stream.distance(
                self.geometry.exit_speed(self._last_head, self.parameters)
            )
            spread = inflation * math.hypot(
                self.sensors.crossing_noise, self.settings.crossing_floor
            )
            result = result + _log_normal(reading.crossing_distance, predicted, spread)
        return result + self._stream_log_likelihoods(reading)

    def _stream_log_likelihoods(self, reading: Reading) -> numpy.ndarray:
        """
        :param reading: The reading just taken.
        :return: The log likelihood under each particle of whether the camera saw a
            stream, and of how many grains it has counted crossing, missing those that
            fall while it loses the stream.
        """
        result = numpy.zeros(len(self.source))
        dropout = self.sensors.stream_dropout
        if reading.stream_seen is not None and dropout is not None:
            some_grain = 1.0 - numpy.exp(-self._last_rate * self._last_interval)
            seen = (
                STREAM_FALSE_SIGHTING
                + (1.0 - STREAM_FALSE_SIGHTING) * (1.0 - dropout) * some_grain
            )
            result = result + numpy.log(seen if reading.stream_seen else 1.0 - seen)
        if reading.counted_grains is not None and self.sensors.count_noise is not None:
            left = self._loaded - self._state_at(
                self._sources, reading.time - self.sensors.fall_time
            )
            counted = (1.0 - (dropout or 0.0)) * left
            spread = self.inflation * numpy.hypot(
                self.sensors.count_noise * counted, COUNT_FLOOR
            )
            result = result + _log_normal(reading.counted_grains, counted, spread)
        return result

    def update(self, reading: Reading) -> float:
        """
        Predict to the reading, weigh the particles by it, and resample if they have
        thinned out.

        :param reading: The reading just taken.
        :return: The log likelihood of the reading under this hypothesis.
        """
        self.predict(reading)
        likelihoods = self.log_likelihoods(reading)
        joint = self.log_weights + likelihoods
        evidence = float(logsumexp(joint))
        self.log_weights = joint - evidence
        weights = numpy.exp(self.log_weights)
        if 1.0 / numpy.sum(weights**2) < RESAMPLING_SHARE * len(weights):
            self._resample(weights)
        return evidence

    def _resample(self, weights: numpy.ndarray) -> None:
        """
        Draw the particles anew in proportion to their weights, and scatter the free
        parameters so the cloud keeps its spread.
        """
        count = len(weights)
        positions = (self.random.random() + numpy.arange(count)) / count
        chosen = numpy.minimum(
            numpy.searchsorted(numpy.cumsum(weights), positions), count - 1
        )
        means = numpy.average(self.parameters, axis=0, weights=weights)
        spreads = numpy.sqrt(
            numpy.average((self.parameters - means) ** 2, axis=0, weights=weights)
        )
        self.parameters = self.parameters[chosen]
        self.source = self.source[chosen]
        self.committed = self.committed[chosen]
        self.spilled = self.spilled[chosen]
        self._last_head = self._last_head[chosen]
        self._sources = [states[chosen] for states in self._sources]
        self._committed = [states[chosen] for states in self._committed]
        self._spilled = [states[chosen] for states in self._spilled]
        self.log_weights = numpy.full(count, -math.log(count))
        bandwidth = math.sqrt(1 - KERNEL_SHRINKAGE**2)
        for parameter in self.free:
            column = PARAMETER_ORDER.index(parameter)
            low, high = self.bounds.of(parameter)
            values = (
                KERNEL_SHRINKAGE * self.parameters[:, column]
                + (1 - KERNEL_SHRINKAGE) * means[column]
                + bandwidth * spreads[column] * self.random.normal(0, 1, count)
            )
            self.parameters[:, column] = numpy.clip(values, low, high)

    @property
    def weights(self) -> numpy.ndarray:
        """
        :return: The particles' normalised weights.
        """
        return numpy.exp(self.log_weights)

    def mean(self) -> ModelPoint:
        """
        :return: The posterior mean of the model's parameters.
        """
        means = numpy.average(self.parameters, axis=0, weights=self.weights)
        values = {
            parameter: float(means[PARAMETER_ORDER.index(parameter)])
            for parameter in PARAMETER_ORDER
        }
        slot = values.pop(FreeParameter.SLOT_HEIGHT)
        return replace(
            self.base.with_values(values),
            slot_height=None if math.isnan(slot) else slot,
        )

    def quantile(self, parameter: FreeParameter, share: float) -> float:
        """
        :param parameter: A parameter.
        :param share: The share of the posterior below the value asked for.
        :return: That quantile of the posterior.
        """
        column = self.parameters[:, PARAMETER_ORDER.index(parameter)]
        order = numpy.argsort(column)
        cumulative = numpy.cumsum(self.weights[order])
        index = min(int(numpy.searchsorted(cumulative, share)), len(column) - 1)
        return float(column[order][index])

    def controller_point(self, tilt: float, trickle: float) -> ModelPoint:
        """
        :param tilt: The cup's tilt now, in radians.
        :param trickle: The least flow the model is held to expect at the tilt, in
            grains per second; nothing leaves the model free to say the flow has
            stopped.
        :return: The parameters a controller runs on under this hypothesis: the
            posterior mean, with the concepts that set where flow starts at a low
            quantile, and the onset held no further than a margin below the tilt.
        """
        point = self.mean().with_values(
            {
                parameter: self.quantile(parameter, self.settings.onset_quantile)
                for parameter in self.free
                if parameter in (FreeParameter.REPOSE_ANGLE, FreeParameter.SLOT_HEIGHT)
            }
        )
        if FreeParameter.REPOSE_ANGLE not in self.free or trickle <= 0.0:
            return point
        grains = float(numpy.average(self.source, weights=self.weights))
        geometry = self.geometry
        geometric_onset = geometry.onset_tilt(grains, point) - point.repose_angle
        surface = grains / geometry.grains_per_fill * geometry.container_height
        reach = math.hypot(geometry.container_height - surface, geometry.lip_offset)
        trickle_head = (
            trickle
            * geometry.container_height
            / (point.outflow_rate_constant * geometry.grains_per_fill)
        )
        if trickle_head >= reach:
            return point
        highest = tilt - math.asin(trickle_head / reach) - geometric_onset
        low, _ = self.bounds.of(FreeParameter.REPOSE_ANGLE)
        return point.with_values(
            {FreeParameter.REPOSE_ANGLE: max(low, min(point.repose_angle, highest))}
        )

    def spread(self, parameter: FreeParameter) -> float:
        """
        :return: The posterior standard deviation of one parameter.
        """
        column = self.parameters[:, PARAMETER_ORDER.index(parameter)]
        mean = numpy.average(column, weights=self.weights)
        return float(
            math.sqrt(numpy.average((column - mean) ** 2, weights=self.weights))
        )

    def bound_share(self, parameter: FreeParameter) -> float:
        """
        :return: The posterior mass held at a bound of one parameter.
        """
        low, high = self.bounds.of(parameter)
        column = self.parameters[:, PARAMETER_ORDER.index(parameter)]
        tolerance = 1e-3 * (high - low)
        at_bound = (column <= low + tolerance) | (column >= high - tolerance)
        return float(numpy.sum(self.weights * at_bound))

    @property
    def source_grains(self) -> float:
        """
        :return: The posterior mean of the grains still in the source.
        """
        return float(numpy.average(self.source, weights=self.weights))

    @property
    def received_grains(self) -> float:
        """
        :return: The posterior mean of the grains that have reached the receiver or are
            falling into it; without a receiver reading, counted from this pour's start.
        """
        return float(numpy.average(self.committed, weights=self.weights))

    def final_fill_spread(self, tilt: float) -> float:
        """
        :param tilt: The cup's tilt now, in radians.
        :return: The standard deviation, over the particles, of the grains the receiver
            is predicted to hold once what is leaving now has arrived.
        """
        head = self.geometry.head(tilt, self.source, self.parameters)
        final = self.committed + MARGIN_HORIZON * self.geometry.outflow(
            head, self.parameters
        )
        mean = numpy.average(final, weights=self.weights)
        return float(
            math.sqrt(numpy.average((final - mean) ** 2, weights=self.weights))
        )


def _log_normal(
    observed: float, predicted: numpy.ndarray, spread: float | numpy.ndarray
) -> numpy.ndarray:
    """
    :return: The log density of a normal distribution around each prediction, with one
        spread for all or one per prediction.
    """
    return -0.5 * ((observed - predicted) / spread) ** 2 - numpy.log(
        spread * math.sqrt(2 * math.pi)
    )


# %% the bank of hypotheses


@dataclass
class HypothesisBank:
    """
    Every hypothesis the twin has for the cup, weighed against each reading.
    """

    hypotheses: Tuple[Hypothesis, ...]
    """
    The hypotheses the bank weighs.
    """

    geometry: DrainGeometry
    """
    The source cup as the model sees it.
    """

    base: ModelPoint
    """
    The model as given.
    """

    sensors: SensorKnowledge
    """
    What the bank knows about the readings.
    """

    settings: BankSettings
    """
    The bank's settings.
    """

    seed: int
    """
    Where the filters draw their randomness from.
    """

    onset: Optional[OnsetSpread] = None
    """
    How the repose angle scatters from pour to pour, or ``None`` for one fixed angle.
    """

    drain_rate: Optional[DrainRateSpread] = None
    """
    How the drain rate scatters from pour to pour, or ``None`` for one fixed rate.
    """

    filters: Dict[Hypothesis, HypothesisFilter] = field(init=False)
    """
    One filter per hypothesis.
    """

    probabilities: Dict[Hypothesis, float] = field(init=False)
    """
    How probable each hypothesis is, given the readings so far.
    """

    started: bool = field(init=False, default=False)
    """
    Whether the first reading has been taken.
    """

    def __post_init__(self) -> None:
        self.filters = {
            hypothesis: HypothesisFilter(
                hypothesis=hypothesis,
                geometry=self.geometry,
                base=self.base,
                sensors=self.sensors,
                settings=self.settings,
                random=numpy.random.default_rng([self.seed, index]),
                onset=self.onset,
                drain_rate=self.drain_rate,
            )
            for index, hypothesis in enumerate(self.hypotheses)
        }
        share = 1.0 / len(self.hypotheses)
        self.probabilities = {hypothesis: share for hypothesis in self.hypotheses}

    def observe(self, reading: Reading) -> None:
        """
        Update every hypothesis and how probable it is with one reading.

        :param reading: The reading just taken.
        """
        if not self.started:
            for hypothesis_filter in self.filters.values():
                hypothesis_filter.start(reading)
            self.started = True
            return
        count = len(self.hypotheses)
        switching = self.settings.switching_probability if count > 1 else 0.0
        stay = 1.0 - switching
        move = switching / max(1, count - 1)
        log_prior = {
            hypothesis: math.log(
                max(
                    1e-300,
                    stay * self.probabilities[hypothesis]
                    + move * (1.0 - self.probabilities[hypothesis]),
                )
            )
            for hypothesis in self.hypotheses
        }
        log_posterior = {
            hypothesis: log_prior[hypothesis] + self.filters[hypothesis].update(reading)
            for hypothesis in self.hypotheses
        }
        total = float(logsumexp(list(log_posterior.values())))
        self.probabilities = {
            hypothesis: math.exp(value - total)
            for hypothesis, value in log_posterior.items()
        }

    def spread_of(self, parameter: FreeParameter) -> Optional[ParameterSpread]:
        """
        :return: The twin's spread of a parameter, or ``None`` where it has none.
        """
        match parameter:
            case FreeParameter.REPOSE_ANGLE:
                return self.onset
            case FreeParameter.OUTFLOW_RATE_CONSTANT:
                return self.drain_rate
        return None

    def share_inside(self, parameter: FreeParameter) -> Optional[float]:
        """
        :param parameter: A parameter the twin has a spread of.
        :return: The share of the pour's posterior of that parameter within the twin's
            interval, read off the most probable hypothesis that holds it wrong, or
            ``None`` without a spread or such a hypothesis.
        """
        spread = self.spread_of(parameter)
        candidates = [
            hypothesis
            for hypothesis in self.hypotheses
            if parameter in hypothesis.concepts
        ]
        if spread is None or not candidates or not self.started:
            return None
        chosen = max(candidates, key=self.probabilities.get)
        hypothesis_filter = self.filters[chosen]
        column = PARAMETER_ORDER.index(parameter)
        return spread.share_inside(
            hypothesis_filter.parameters[:, column], hypothesis_filter.weights
        )

    @property
    def most_probable(self) -> Hypothesis:
        """
        :return: The hypothesis the readings favour most now.
        """
        return max(self.probabilities, key=self.probabilities.get)


# %% the controller's commitment to one hypothesis


@dataclass(frozen=True)
class Commit:
    """
    The controller switching its model to a hypothesis.
    """

    time: float
    """
    When it switched, in seconds.
    """

    hypothesis: Hypothesis
    """
    The hypothesis switched to.
    """

    previous: Hypothesis
    """
    The hypothesis switched from.
    """

    probability: float
    """
    How probable the hypothesis was when switched to.
    """

    point: ModelPoint
    """
    The parameters the controller took on.
    """

    estimate: ModelPoint
    """
    The hypothesis's posterior mean when the controller switched to it.
    """

    poured_share: float
    """
    The share of the goal the receiver held when the controller switched.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            "time": self.time,
            "hypothesis": self.hypothesis,
            "previous": self.previous,
            "probability": self.probability,
            "point": self.point.to_json(),
            "estimate": self.estimate.to_json(),
            "poured_share": self.poured_share,
        }


@dataclass
class ModelSwitch:
    """
    Which hypothesis the controller runs on: the nominal one until a concept of the
    model is probably wrong, then the simplest hypothesis that frees every concept that
    probably is.

    A concept is probably wrong when the hypotheses that free it are together probable
    enough. Weighing concepts rather than single hypotheses keeps a hypothesis that frees
    one more parameter than the data can tell from sharing the probability of the one
    that frees fewer and holding both under the threshold.
    """

    threshold: float
    """
    How probable a concept has to be wrong before the controller switches.
    """

    confirmations: int = 1
    """
    How many readings in a row have to call for the same hypothesis before the
    controller switches to it.
    """

    active: Hypothesis = Hypothesis.NOMINAL
    """
    The hypothesis the controller runs on now.
    """

    _pending: Optional[Hypothesis] = field(init=False, default=None)
    """
    The hypothesis the latest readings call for, while it waits to be confirmed.
    """

    _calls: int = field(init=False, default=0)
    """
    How many readings in a row have called for it.
    """

    @staticmethod
    def concept_probabilities(
        probabilities: Dict[Hypothesis, float],
    ) -> Dict[FreeParameter, float]:
        """
        :param probabilities: How probable each hypothesis is now.
        :return: How probable each concept is wrong: the summed probability of the
            hypotheses that free it.
        """
        concepts: Dict[FreeParameter, float] = {}
        for hypothesis, probability in probabilities.items():
            for concept in hypothesis.concepts:
                concepts[concept] = concepts.get(concept, 0.0) + probability
        return concepts

    def target(
        self,
        probabilities: Dict[Hypothesis, float],
        vetoed: frozenset[FreeParameter] = frozenset(),
    ) -> Tuple[Optional[Hypothesis], float]:
        """
        A concept the controller already treats as wrong stays wrong until it is
        probably right, so the controller does not flip between hypotheses on the
        margin of the threshold.

        :param probabilities: How probable each hypothesis is now.
        :param vetoed: Concepts that may not be newly held wrong, however probable.
        :return: The hypothesis the readings call for, or ``None`` if they call for
            none, with how probable the concepts it frees are wrong.
        """
        catch_all = probabilities.get(Hypothesis.CATCH_ALL, 0.0)
        if catch_all >= self.threshold or (
            self.active is Hypothesis.CATCH_ALL and catch_all > 1.0 - self.threshold
        ):
            return Hypothesis.CATCH_ALL, catch_all
        concepts = self.concept_probabilities(probabilities)
        kept = {
            concept
            for concept in self.active.concepts
            if concepts.get(concept, 0.0) > 1.0 - self.threshold
        }
        wrong = kept | {
            concept
            for concept, probability in concepts.items()
            if probability >= self.threshold and concept not in vetoed
        }
        if wrong:
            covering = [
                hypothesis
                for hypothesis in probabilities
                if hypothesis.concepts >= wrong
            ]
            if covering:
                chosen = min(
                    covering,
                    key=lambda hypothesis: (
                        len(hypothesis.concepts),
                        -probabilities[hypothesis],
                    ),
                )
                return chosen, min(concepts[concept] for concept in wrong)
        if probabilities.get(Hypothesis.NOMINAL, 0.0) >= self.threshold:
            return Hypothesis.NOMINAL, probabilities[Hypothesis.NOMINAL]
        return None, 0.0

    def decide(
        self,
        probabilities: Dict[Hypothesis, float],
        vetoed: frozenset[FreeParameter] = frozenset(),
    ) -> Optional[Hypothesis]:
        """
        :param probabilities: How probable each hypothesis is now.
        :param vetoed: Concepts that may not be newly held wrong, however probable.
        :return: The hypothesis to switch to, or ``None`` to stay.
        """
        chosen, _ = self.target(probabilities, vetoed)
        if chosen is None or chosen is self.active:
            self._pending, self._calls = None, 0
            return None
        if chosen is self._pending:
            self._calls += 1
        else:
            self._pending, self._calls = chosen, 1
        if self._calls < self.confirmations:
            return None
        self.active, self._pending, self._calls = chosen, None, 0
        return chosen
