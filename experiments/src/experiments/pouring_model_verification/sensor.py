"""
What a fill-level sensor reports of what is really there.

The closed-loop study measures the grains perfectly by default: it counts them as the
simulation sees them, including those still falling into the receiver, and hands the
count over at once. A sensor on a real robot is noisy, late, and a scale under the
receiver sees only what has landed. This module models those three departures, so a
controller can be run on what such a sensor would report.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy
from typing_extensions import List, Optional, Tuple

# %% what a sensor gets wrong


@dataclass(frozen=True)
class SensorModel:
    """
    How a fill-level sensor departs from the true fill level.
    """

    noise: float
    """
    Standard deviation of the white noise on each reading, in grains.
    """

    delay: float
    """
    How long a reading takes to reach the controller, in seconds.
    """

    counts_falling_grains: bool = True
    """
    Whether the receiver's reading includes the grains already falling into it, as a
    camera watching the stream would; a scale under the receiver sees only what has
    landed.
    """

    @property
    def is_ideal(self) -> bool:
        """
        :return: Whether this sensor reports the true fill level at once.
        """
        return self.noise == 0.0 and self.delay == 0.0 and self.counts_falling_grains


class SensorCondition(StrEnum):
    """
    A sensor the closed loop is run with.
    """

    IDEAL = "ideal"
    """
    Every grain counted the moment it leaves the source, without noise.
    """

    NOISY = "noisy"
    """
    Two grains of noise on each reading, about 5 % of the goal.
    """

    VERY_NOISY = "very_noisy"
    """
    Five grains of noise on each reading, about 12 % of the goal.
    """

    DELAYED = "delayed"
    """
    Readings arrive 0.2 s late.
    """

    VERY_DELAYED = "very_delayed"
    """
    Readings arrive 0.5 s late.
    """

    LANDED_ONLY = "landed_only"
    """
    The receiver is weighed: grains count once they have landed, not while they fall.
    """

    SCALE = "scale"
    """
    A realistic scale: landed grains only, two grains of noise, 0.2 s late.
    """

    @property
    def model(self) -> SensorModel:
        """
        :return: How this sensor departs from the true fill level.
        """
        match self:
            case SensorCondition.IDEAL:
                return SensorModel(noise=0.0, delay=0.0)
            case SensorCondition.NOISY:
                return SensorModel(noise=2.0, delay=0.0)
            case SensorCondition.VERY_NOISY:
                return SensorModel(noise=5.0, delay=0.0)
            case SensorCondition.DELAYED:
                return SensorModel(noise=0.0, delay=0.2)
            case SensorCondition.VERY_DELAYED:
                return SensorModel(noise=0.0, delay=0.5)
            case SensorCondition.LANDED_ONLY:
                return SensorModel(noise=0.0, delay=0.0, counts_falling_grains=False)
            case SensorCondition.SCALE:
                return SensorModel(noise=2.0, delay=0.2, counts_falling_grains=False)


WRIST_NOISE = 3.0
"""
Standard deviation of a wrist force-torque sensor's reading of the source, in grains:

noisier than a scale on the table, since the arm moves and shakes it.
"""

WRIST_DELAY = 0.1
"""
How late a wrist force-torque sensor's reading arrives, in seconds.
"""


class WristSensor(StrEnum):
    """
    A force-torque sensor in the wrist, weighing the source cup the hand holds.
    """

    IDEAL = "ideal"
    """
    Every grain in the source weighed at once, without noise.
    """

    REALISTIC = "realistic"
    """
    The source weighed with the wrist's noise and delay.
    """

    @property
    def model(self) -> SensorModel:
        """
        :return: How this sensor departs from the source's true content.
        """
        match self:
            case WristSensor.IDEAL:
                return SensorModel(noise=0.0, delay=0.0)
            case WristSensor.REALISTIC:
                return SensorModel(noise=WRIST_NOISE, delay=WRIST_DELAY)


# %% one measured quantity, as the sensor reports it

DELAY_TOLERANCE = 1e-6
"""
How much earlier than one delay ago a reading may be taken and still count as that old,
in seconds, so readings taken on a fixed period are not lost to rounding.
"""


@dataclass
class SensorChannel:
    """
    One fill level as a sensor reports it: late by the sensor's delay, scattered by its
    noise, and never outside the container.
    """

    model: SensorModel
    """
    How the sensor departs from the true fill level.
    """

    grain_share: float
    """
    The share of the container's capacity one grain takes up, converting the noise from
    grains into a fill level.
    """

    random: numpy.random.Generator
    """
    Where the noise is drawn from.
    """

    _readings: List[Tuple[float, float]] = field(
        init=False, default_factory=list, repr=False
    )
    """
    Every true fill level read so far, with the time it was read, oldest first.
    """

    def read(self, value: float, at: float) -> float:
        """
        :param value: The true fill level now.
        :param at: The time now, in seconds.
        :return: What the sensor reports now: the latest true level at least one delay
            old, or the first one before a delay has passed, plus noise.
        """
        self._readings.append((at, value))
        delayed = self._readings[0][1]
        for taken_at, reading in self._readings:
            if taken_at > at - self.model.delay + DELAY_TOLERANCE:
                break
            delayed = reading
        noise = self.random.normal(0.0, self.model.noise * self.grain_share)
        return min(1.0, max(0.0, delayed + noise))


# %% a camera watching the stream


CAMERA_NOISE = 0.01
"""
Standard deviation of where a noisy camera sees the stream cross the receiver's opening
plane, in metres.
"""

CAMERA_DROPOUT = 0.3
"""
The share of readings a noisy camera loses the stream in.
"""

POLE_CAMERA_NOISE = 0.015
"""
Standard deviation of where the pole camera above Tracy's table saw the stream cross the
receiver's opening, from one reading to the next, in metres: 0.8-1.9 cm per image axis
in the five demo pours of 2026-10-12, the larger along the fall.
"""

POLE_CAMERA_DROPOUT = 0.05
"""
The share of readings the pole camera lost the stream in during those pours: 0-6.5 %.
"""

POLE_CAMERA_DELAY = 0.11
"""
How late the pole camera's images arrived after they were taken, in seconds.
"""

POLE_CAMERA_COUNT_NOISE = 0.065
"""
Relative standard deviation of how many grains the pole camera counts crossing the
receiver's opening over a pour, from the stream's white area: as the white area of the
grains on the black tray scattered around their weighed mass over the five ramps.
"""

ROUGH_COUNT_NOISE = 0.2
"""
Relative standard deviation of a camera's count of crossed grains three times as rough
as the tray's, for a stream that is harder to segment than grains lying still.
"""


@dataclass(frozen=True)
class CameraModel:
    """
    How a camera watching the stream departs from where it really crosses.
    """

    noise: float
    """
    Standard deviation of the crossing it reports, in metres.
    """

    dropout: float
    """
    The share of readings in which it sees no stream.
    """

    delay: float = 0.0
    """
    How late what it sees reaches the bank, in seconds.
    """

    count_noise: Optional[float] = None
    """
    Relative standard deviation of how many grains it counts crossing the opening plane
    over a pour, or ``None`` if it reports only where they cross.
    """

    def grains_crossed(self, counted: float) -> float:
        """
        :param counted: How many grains the camera counted crossing the opening plane.
        :return: How many crossed, crediting those that fell while it lost the stream.
        """
        return counted / (1.0 - self.dropout)


class CameraCondition(StrEnum):
    """
    A camera watching where the stream crosses the receiver's opening plane.
    """

    IDEAL = "ideal"
    """
    Every crossing seen where it is.
    """

    NOISY = "noisy"
    """
    Crossings scattered by a centimetre, and the stream lost in some readings.
    """

    POLE = "pole"
    """
    As the pole camera above Tracy's table saw the stream in the real pours: scattered
    by 1.5 cm, lost in some readings, and late.
    """

    POLE_COUNTING_ROUGH = "pole_counting_rough"
    """
    The pole camera, counting the crossed grains three times as roughly.
    """

    POLE_COUNTING = "pole_counting"
    """
    The pole camera, also counting how many grains have crossed the receiver's opening
    plane.
    """

    NONE = "none"
    """
    No camera: only the weights of the two cups, as for a clear liquid.
    """

    @property
    def model(self) -> Optional[CameraModel]:
        """
        :return: How this camera departs from the true crossing, or ``None`` without
            one.
        """
        match self:
            case CameraCondition.IDEAL:
                return CameraModel(noise=0.0, dropout=0.0)
            case CameraCondition.NOISY:
                return CameraModel(noise=CAMERA_NOISE, dropout=CAMERA_DROPOUT)
            case CameraCondition.POLE:
                return CameraModel(
                    noise=POLE_CAMERA_NOISE,
                    dropout=POLE_CAMERA_DROPOUT,
                    delay=POLE_CAMERA_DELAY,
                )
            case CameraCondition.POLE_COUNTING:
                return CameraModel(
                    noise=POLE_CAMERA_NOISE,
                    dropout=POLE_CAMERA_DROPOUT,
                    delay=POLE_CAMERA_DELAY,
                    count_noise=POLE_CAMERA_COUNT_NOISE,
                )
            case CameraCondition.POLE_COUNTING_ROUGH:
                return CameraModel(
                    noise=POLE_CAMERA_NOISE,
                    dropout=POLE_CAMERA_DROPOUT,
                    delay=POLE_CAMERA_DELAY,
                    count_noise=ROUGH_COUNT_NOISE,
                )
            case CameraCondition.NONE:
                return None


class SensorSet(StrEnum):
    """
    Every sensor a pour that diagnoses its own model reads, as one choice.
    """

    IDEAL = "ideal"
    """
    The receiver, the wrist and the camera, all ideal.
    """

    REALISTIC = "realistic"
    """
    A realistic scale and wrist, with an ideal camera.
    """

    NOISY_CAMERA = "noisy_camera"
    """
    A realistic scale and wrist, with a noisy camera.
    """

    NO_CAMERA = "no_camera"
    """
    A realistic scale and wrist, and no camera.
    """

    WRIST_CAMERA = "wrist_camera"
    """
    A realistic wrist and a noisy camera, with nothing under the receiver: what an
    everyday robot carries.
    """

    WRIST_POLE_CAMERA = "wrist_pole_camera"
    """
    A realistic wrist and the pole camera of the real pours, with nothing under the
    receiver.
    """

    POLE_CAMERA_ROUGH = "pole_camera_rough"
    """
    The rough counting pole camera alone.
    """

    POLE_CAMERA = "pole_camera"
    """
    The counting pole camera alone: nothing weighs either cup once the pour has started.
    """

    @property
    def receiver(self) -> Optional[SensorCondition]:
        """
        :return: What weighs the receiver, or ``None`` if nothing does.
        """
        match self:
            case SensorSet.IDEAL:
                return SensorCondition.IDEAL
            case (
                SensorSet.WRIST_CAMERA
                | SensorSet.WRIST_POLE_CAMERA
                | SensorSet.POLE_CAMERA
                | SensorSet.POLE_CAMERA_ROUGH
            ):
                return None
            case _:
                return SensorCondition.SCALE

    @property
    def wrist(self) -> Optional[WristSensor]:
        """
        :return: What weighs the source at the wrist, or ``None`` if nothing does.
        """
        if self in (SensorSet.POLE_CAMERA, SensorSet.POLE_CAMERA_ROUGH):
            return None
        if self is SensorSet.IDEAL:
            return WristSensor.IDEAL
        return WristSensor.REALISTIC

    @property
    def camera(self) -> CameraCondition:
        """
        :return: What watches the stream.
        """
        match self:
            case SensorSet.IDEAL | SensorSet.REALISTIC:
                return CameraCondition.IDEAL
            case SensorSet.NOISY_CAMERA | SensorSet.WRIST_CAMERA:
                return CameraCondition.NOISY
            case SensorSet.WRIST_POLE_CAMERA:
                return CameraCondition.POLE
            case SensorSet.POLE_CAMERA:
                return CameraCondition.POLE_COUNTING
            case SensorSet.POLE_CAMERA_ROUGH:
                return CameraCondition.POLE_COUNTING_ROUGH
            case SensorSet.NO_CAMERA:
                return CameraCondition.NONE


@dataclass
class CrossingLog:
    """
    Every grain a camera saw cross the receiver's opening plane, with when it saw it.
    """

    times: List[float] = field(default_factory=list)
    """
    When each crossing was seen, in seconds, oldest first.
    """

    crossings: List[numpy.ndarray] = field(default_factory=list)
    """
    Where each grain seen crossing stood, horizontally, in metres.
    """

    looks: List[float] = field(default_factory=list)
    """
    When the camera looked, in seconds, oldest first, whether it saw grains or not.
    """

    def look(self, at: float) -> None:
        """
        :param at: The time the camera looks, in seconds.
        """
        self.looks.append(at)

    def record(self, at: float, crossing: numpy.ndarray) -> None:
        """
        :param at: The time the grain was seen, in seconds.
        :param crossing: Where it stood, horizontally, in metres.
        """
        self.times.append(at)
        self.crossings.append(crossing)

    def between(
        self, since: float, until: Optional[float] = None
    ) -> List[numpy.ndarray]:
        """
        :param since: The earliest time to report, in seconds.
        :param until: The latest time to report, or ``None`` for up to now; a late
            camera reports nothing seen after one delay ago.
        :return: The crossings seen in that time, oldest first.
        """
        return [
            crossing
            for at, crossing in zip(self.times, self.crossings)
            if at >= since and (until is None or at <= until)
        ]

    def grains_per_look(self, since: float, until: Optional[float] = None) -> float:
        """
        :param since: The earliest time to count, in seconds.
        :param until: The latest time to count, or ``None`` for up to now.
        :return: How many grains the camera saw near the opening plane per look in that
            time, or nothing if it did not look.
        """
        looks = [
            at for at in self.looks if at >= since and (until is None or at <= until)
        ]
        if not looks:
            return 0.0
        return len(self.between(since, until)) / len(looks)
