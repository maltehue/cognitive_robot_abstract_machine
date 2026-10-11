"""
A model-free baseline for the closed loop: a PID controller that tilts the cup by how
many grains the receiver is still missing, knowing nothing of how grains leave a cup.

It replaces only the fill row of the transfer. Where the cup goes is decided either by
the same aiming tasks the model-based controller uses, or by holding the cup still over
the receiver, so a comparison can separate what the pouring model contributes to the
fill from what it contributes to the aim.
"""

# %% imports

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path

import numpy

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import FloatVariable, Scalar
from typing_extensions import Any, Dict, List, Optional, Tuple

from experiments.pouring_model_verification.inflow_model import (
    FirstOrderInflowModel,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    NodeArtifacts,
    Task,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% the gains


class PidGainKey(StrEnum):
    """
    A key of a set of gains written to JSON.
    """

    PROPORTIONAL = "proportional"
    INTEGRAL = "integral"
    DERIVATIVE = "derivative"
    START_TILT = "start_tilt"


@dataclass(frozen=True)
class PidGains:
    """
    How strongly the controller tilts the cup for the grains the receiver is missing.
    """

    proportional: float
    """
    Tilt per missing grain, in radians.
    """

    integral: float
    """
    Tilt per missing grain and second, in radians.
    """

    derivative: float
    """
    Tilt taken back per grain arriving each second, in radians.
    """

    start_tilt: float
    """
    The tilt the cup is carried to before the controller takes over, and that its
    command is added to, in radians; tuned like the gains rather than taken from a
    model.
    """

    @property
    def label(self) -> str:
        """
        :return: A short identifier of these gains, usable in a file name.
        """
        return (
            f"kp{self.proportional:g}_ki{self.integral:g}_kd{self.derivative:g}"
            f"_start{self.start_tilt:g}"
        )

    def scaled(self, factor: float) -> PidGains:
        """
        :param factor: What the three gains are multiplied by.
        :return: These gains scaled, from the same start tilt.
        """
        return replace(
            self,
            proportional=self.proportional * factor,
            integral=self.integral * factor,
            derivative=self.derivative * factor,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            PidGainKey.PROPORTIONAL: self.proportional,
            PidGainKey.INTEGRAL: self.integral,
            PidGainKey.DERIVATIVE: self.derivative,
            PidGainKey.START_TILT: self.start_tilt,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> PidGains:
        return cls(
            proportional=data[PidGainKey.PROPORTIONAL],
            integral=data[PidGainKey.INTEGRAL],
            derivative=data[PidGainKey.DERIVATIVE],
            start_tilt=data[PidGainKey.START_TILT],
        )


# %% what the controller reads


class PidVariant(StrEnum):
    """
    How the PID baseline makes up for a reading that arrives late.
    """

    PLAIN = "plain"
    """
    It does not: it acts and stops on the reading as it arrives.
    """

    STOP_EARLY = "stop_early"
    """
    It acts on the reading, but stops once the reading projected over the sensor's lag
    by its recent rate reaches the goal.
    """

    SMITH = "smith"
    """
    A Smith predictor: it acts and stops on the reading corrected by what an inflow
    model identified from readings says has arrived since the reading was taken.
    """


@dataclass
class ReadingCompensation(ABC):
    """
    What the PID controller makes of each reading before it acts on it and before it
    decides whether to stop.
    """

    @abstractmethod
    def observe(self, measured: float, at: float, tilt: float) -> None:
        """
        :param measured: How many grains the receiver is read to hold.
        :param at: When it was read, in seconds.
        :param tilt: How far the cup is tilted now, in radians.
        """

    @abstractmethod
    def control_reading(self, measured: float) -> float:
        """
        :param measured: The reading just observed, in grains.
        :return: The grains the controller's terms act on.
        """

    @abstractmethod
    def stop_reading(self, measured: float) -> float:
        """
        :param measured: The reading just observed, in grains.
        :return: The grains the stop rule compares with the goal.
        """


@dataclass
class NoCompensation(ReadingCompensation):
    """
    The reading is taken as it is.
    """

    def observe(self, measured: float, at: float, tilt: float) -> None:
        pass

    def control_reading(self, measured: float) -> float:
        return measured

    def stop_reading(self, measured: float) -> float:
        return measured


RATE_WINDOW = 0.5
"""
How far back the readings reach that the inflow rate is fitted to, in seconds.
"""


@dataclass
class RateProjection(ReadingCompensation):
    """
    The reading the stop rule sees is projected forward by the recent inflow rate over
    the sensor's lag, so the pour stops while the grains still unseen are on their way.
    """

    horizon: float
    """
    How long a grain takes to show in a reading, in seconds.
    """

    window: float = RATE_WINDOW
    """
    How far back the rate is fitted, in seconds.
    """

    _readings: List[Tuple[float, float]] = field(
        init=False, default_factory=list, repr=False
    )
    """
    The readings within the window, with their times, oldest first.
    """

    def observe(self, measured: float, at: float, tilt: float) -> None:
        self._readings.append((at, measured))
        self._readings = [
            reading
            for reading in self._readings
            if reading[0] >= at - self.window - 1e-9
        ]

    @property
    def rate(self) -> float:
        """
        :return: The least-squares slope of the readings in the window, in grains per
            second; nothing before two readings.
        """
        if len(self._readings) < 2:
            return 0.0
        times, readings = zip(*self._readings)
        slope, _ = numpy.polyfit(times, readings, 1)
        return float(slope)

    def control_reading(self, measured: float) -> float:
        return measured

    def stop_reading(self, measured: float) -> float:
        return measured + max(0.0, self.rate) * self.horizon


@dataclass
class SmithPrediction(ReadingCompensation):
    """
    A Smith predictor: the reading plus what the inflow model says arrived over the
    sensor's lag, driven by the cup's own tilt. With an exact model this is the fill the
    reading will show one lag from now.
    """

    model: FirstOrderInflowModel
    """
    The inflow model, identified from readings.
    """

    horizon: float
    """
    How long a grain takes to show in a reading, in seconds.
    """

    _history: List[Tuple[float, float]] = field(
        init=False, default_factory=list, repr=False
    )
    """
    The model's grains at each observation, with its time, oldest first.
    """

    _rate: float = field(init=False, default=0.0, repr=False)
    """
    The model's inflow now, in grains per second.
    """

    _tilt: float = field(init=False, default=0.0, repr=False)
    """
    The tilt at the previous observation, in radians.
    """

    def observe(self, measured: float, at: float, tilt: float) -> None:
        if not self._history:
            self._history.append((at, 0.0))
            self._tilt = tilt
            return
        previous_time, previous_grains = self._history[-1]
        duration = at - previous_time
        self._rate = self.model.step(self._rate, self._tilt, duration)
        self._history.append((at, previous_grains + self._rate * duration))
        self._tilt = tilt

    @property
    def arrived_over_horizon(self) -> float:
        """
        :return: The grains the model says arrived over the last horizon.
        """
        now, grains_now = self._history[-1]
        earlier = self._history[0][1]
        for taken_at, grains in self._history:
            if taken_at > now - self.horizon + 1e-9:
                break
            earlier = grains
        return grains_now - earlier

    def control_reading(self, measured: float) -> float:
        return measured + self.arrived_over_horizon

    def stop_reading(self, measured: float) -> float:
        return self.control_reading(measured)


# %% the controller


@dataclass
class PidFillController:
    """
    Turns each reading of the receiver's fill into the tilt the cup should be at.

    The command is the start tilt plus a proportional, an integral and a derivative term
    on the grains still missing, held between upright and the largest tilt. The
    derivative acts on the reading rather than on the error, and the integral stops
    growing while the command is held at a limit it is pushing against. Once a reading
    reaches the goal the cup is turned upright and stays so, as a pour stops once its
    target is met. A compensation may correct each reading for the sensor's lag first.
    """

    gains: PidGains
    """
    How strongly the cup is tilted for the missing grains.
    """

    goal: float
    """
    How many grains the receiver is to hold.
    """

    tolerance: float
    """
    How many grains short of the goal still count as reached.
    """

    maximum_tilt: float
    """
    The largest tilt the controller commands, in radians.
    """

    compensation: ReadingCompensation = field(default_factory=NoCompensation)
    """
    What the controller makes of each reading before acting and before stopping.
    """

    setpoint: float = field(init=False)
    """
    The tilt last commanded, in radians.
    """

    reached: bool = field(init=False, default=False)
    """
    Whether a reading has reached the goal; once it has, it stays reached.
    """

    _integral: float = field(init=False, default=0.0, repr=False)
    """
    The missing grains summed over time, in grain-seconds.
    """

    _previous_reading: Optional[float] = field(init=False, default=None, repr=False)
    """
    The previous reading, in grains.
    """

    _previous_time: Optional[float] = field(init=False, default=None, repr=False)
    """
    When the previous reading was taken, in seconds.
    """

    def __post_init__(self) -> None:
        self.setpoint = self.gains.start_tilt

    def update(self, measured: float, at: float, tilt: float = 0.0) -> float:
        """
        :param measured: How many grains the receiver is read to hold.
        :param at: When it was read, in seconds.
        :param tilt: How far the cup is tilted now, in radians.
        :return: The tilt the cup should now be at, in radians: upright from the first
            reading that reaches the goal on.
        """
        self.compensation.observe(measured, at, tilt)
        if self.goal - self.compensation.stop_reading(measured) <= self.tolerance:
            self.reached = True
        measured = self.compensation.control_reading(measured)
        missing = self.goal - measured
        if self.reached:
            self.setpoint = self.minimum_tilt
            return self.setpoint
        elapsed = 0.0 if self._previous_time is None else at - self._previous_time
        arriving = (
            0.0 if elapsed <= 0.0 else (measured - self._previous_reading) / elapsed
        )
        self._previous_reading, self._previous_time = measured, at

        integral = self._integral + missing * elapsed
        command = self._command(missing, integral, arriving)
        pushes_past_largest = command >= self.maximum_tilt and missing > 0.0
        pushes_past_upright = command <= self.minimum_tilt and missing < 0.0
        if not (pushes_past_largest or pushes_past_upright):
            self._integral = integral
        self.setpoint = min(
            self.maximum_tilt,
            max(self.minimum_tilt, self._command(missing, self._integral, arriving)),
        )
        return self.setpoint

    @property
    def minimum_tilt(self) -> float:
        """
        :return: The smallest tilt the controller commands: upright.
        """
        return 0.0

    def _command(self, missing: float, integral: float, arriving: float) -> float:
        """
        :return: The unbounded command for these terms, in radians.
        """
        return (
            self.gains.start_tilt
            + self.gains.proportional * missing
            + self.gains.integral * integral
            - self.gains.derivative * arriving
        )


# %% steering the tilt in the controller


def signed_tilt_about_pour_axis(
    root_T_tool: HomogeneousTransformationMatrix, root_R_carry: RotationMatrix
) -> Scalar:
    """
    :param root_T_tool: Where the hand is.
    :param root_R_carry: The orientation the hand carries the cup upright in, whose
        ``z`` axis is the axis the cup is poured about.
    :return: How far the hand is turned about that axis away from the carry
        orientation, signed, in radians.
    """
    carry_R_tool = root_R_carry.T @ root_T_tool.to_rotation_matrix()
    return sm.atan2(carry_R_tool[1, 0], carry_R_tool[0, 0])


@dataclass(eq=False, repr=False)
class TiltTowards(Task):
    """
    Turns the hand about the pour axis towards a tilt that is read anew every cycle, so
    a controller outside the motion can steer it while it runs.
    """

    root_link: Body = field(kw_only=True)
    """
    The frame the carry orientation is given in.
    """

    tip_link: Body = field(kw_only=True)
    """
    The hand holding the cup.
    """

    root_R_carry: RotationMatrix = field(kw_only=True)
    """
    The orientation the hand carries the cup upright in.
    """

    goal: FloatVariable = field(kw_only=True)
    """
    The tilt to turn towards, in radians.
    """

    reference_velocity: float = field(default=0.3, kw_only=True)
    """
    How fast the tilt is asked to change, in radians per second.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    How expensive it is to miss the tilt.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :param context: The build context.
        :return: One row driving the tilt to the goal.
        """
        artifacts = NodeArtifacts()
        tilt = signed_tilt_about_pour_axis(
            context.world.compose_forward_kinematics_expression(
                self.root_link, self.tip_link
            ),
            self.root_R_carry,
        )
        artifacts.constraints.add_equality_constraint(
            task_expression=tilt,
            equality_bound=self.goal - tilt,
            quadratic_weight=self.weight,
            reference_velocity=self.reference_velocity,
            name=f"{self.name}_tilt",
        )
        return artifacts


@dataclass(eq=False, repr=False)
class FillReached(MotionStatechartNode):
    """
    Turns true once the controller steering the tilt has read the goal reached.
    """

    controller: PidFillController = field(kw_only=True)
    """
    The controller whose reading decides.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self.controller.reached:
            return ObservationStateValues.TRUE
        return None


# %% tuning the gains

SOFTER_GAINS = 0.5
"""
What the tuned gains are scaled by to run the baseline softer than tuned.
"""

STIFFER_GAINS = 2.0
"""
What the tuned gains are scaled by to run the baseline stiffer than tuned.
"""

TUNING_PROPORTIONAL = (0.005, 0.01, 0.02)
"""
The proportional gains tried while tuning, in radians per grain.
"""

TUNING_INTEGRAL = (0.0, 0.002, 0.005)
"""
The integral gains tried while tuning, in radians per grain-second.
"""

TUNING_DERIVATIVE = (0.0, 0.01, 0.03)
"""
The derivative gains tried while tuning, in radian-seconds per grain.
"""

TUNING_START_TILT = (0.5, 0.9)
"""
The start tilts tried while tuning, in radians.
"""


def tuning_grid() -> List[PidGains]:
    """
    :return: Every combination of the gains and start tilts tried while tuning.
    """
    return [
        PidGains(
            proportional=proportional,
            integral=integral,
            derivative=derivative,
            start_tilt=start_tilt,
        )
        for proportional in TUNING_PROPORTIONAL
        for integral in TUNING_INTEGRAL
        for derivative in TUNING_DERIVATIVE
        for start_tilt in TUNING_START_TILT
    ]


@dataclass(frozen=True)
class TunedPidGains:
    """
    The gains each source cup's PID baseline runs with: those that poured best while
    tuning on that cup.
    """

    gains: Dict[str, PidGains]
    """
    The tuned gains, keyed by source cup.
    """

    def of(self, source: str) -> PidGains:
        """
        :param source: A source cup.
        :return: Its tuned gains.
        :raises MissingTunedGainsError: If that cup was not tuned.
        """
        if source not in self.gains:
            raise MissingTunedGainsError(source=source, known=sorted(self.gains))
        return self.gains[source]

    def write(self, path: Path) -> None:
        """
        :param path: The file to write the gains to.
        """
        path.write_text(
            json.dumps(
                {source: gains.to_json() for source, gains in self.gains.items()},
                indent=2,
            )
        )

    @classmethod
    def read(cls, path: Path) -> TunedPidGains:
        """
        :param path: A file written by :meth:`write`.
        :return: The gains it holds.
        :raises PidGainsNotTunedError: If the file does not exist yet.
        """
        if not path.exists():
            raise PidGainsNotTunedError(path=path)
        return cls(
            gains={
                source: PidGains.from_json(data)
                for source, data in json.loads(path.read_text()).items()
            }
        )


@dataclass
class MissingTunedGainsError(Exception):
    """
    A PID trial was asked for a source cup the PID baseline was not tuned on.
    """

    source: str
    """
    The source cup that has no tuned gains.
    """

    known: List[str]
    """
    The source cups that do.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no tuned PID gains for source cup {self.source!r} (tuned: {self.known}); "
            f"record the PID tuning study and run the tuning first"
        )


@dataclass
class PidGainsNotTunedError(Exception):
    """
    The robustness study was started before the PID baseline was tuned.
    """

    path: Path
    """
    Where the tuned gains were looked for.
    """

    def __post_init__(self) -> None:
        super().__init__(
            f"no tuned PID gains at {self.path}; record the PID tuning study and run "
            f"the tuning first"
        )
