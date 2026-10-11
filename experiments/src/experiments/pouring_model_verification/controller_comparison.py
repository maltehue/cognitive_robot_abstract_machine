"""
What the robustness study says: how the model-predictive controller compares with the
model-free PID baseline, and how both hold up when the fill levels reach them through an
imperfect sensor.

Every pour is scored by the grains it put in the wrong place: those it missed the goal
by, either way, plus those it spilled. A pour counts as successful when both stay within
the transfer's fill tolerance.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import pandas
from typing_extensions import Any, Dict, List, Optional

from experiments.pouring_model_verification.parameters import BASELINE, Perturbation
from experiments.pouring_model_verification.pid import (
    SOFTER_GAINS,
    STIFFER_GAINS,
    PidGains,
    PidVariant,
    TunedPidGains,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    FILL_TOLERANCE_GRAINS,
    GOAL_GRAINS,
    FillController,
    LoopMode,
)

# %% scoring a pour


def misplaced_grains(delivered: float, spilled: float) -> float:
    """
    :param delivered: How many grains ended in the receiver.
    :param spilled: How many ended neither in the receiver nor in the source.
    :return: How many grains the pour put in the wrong place: what it missed the goal
        by, either way, plus what it spilled.
    """
    return abs(delivered - GOAL_GRAINS) + spilled


def succeeded(delivered: float, spilled: float) -> bool:
    """
    :param delivered: How many grains ended in the receiver.
    :param spilled: How many ended neither in the receiver nor in the source.
    :return: Whether the pour met the goal and spilled no more than the fill tolerance.
    """
    return (
        abs(delivered - GOAL_GRAINS) <= FILL_TOLERANCE_GRAINS
        and spilled <= FILL_TOLERANCE_GRAINS
    )


class ComparisonColumn(StrEnum):
    """
    A column of a table the comparison produces.
    """

    SOURCE = "source"
    TARGET = "target"
    RECEIVER = "receiver"
    ARM = "arm"
    SENSOR = "sensor"
    SEED = "seed"
    GAINS = "gains"
    RULE = "rule"
    DELIVERED = "delivered"
    SPILLED = "spilled"
    MISPLACED = "misplaced"
    SUCCEEDED = "succeeded"
    DURATION = "duration"
    PEAK_TILT = "peak_tilt"
    ENDING = "ending"
    MEAN = "mean"
    STANDARD_DEVIATION = "standard_deviation"
    SUCCESS_RATE = "success_rate"
    RUNS = "runs"
    TIME = "time"
    TILT = "tilt"
    GRAINS_IN_RECEIVER = "grains_in_receiver"


def _scored(outcome: Dict[str, Any]) -> Dict[ComparisonColumn, Any]:
    """
    :param outcome: A transfer trial's outcome.
    :return: What the comparison reads of it.
    """
    delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
    return {
        ComparisonColumn.DELIVERED: delivered,
        ComparisonColumn.SPILLED: spilled,
        ComparisonColumn.MISPLACED: misplaced_grains(delivered, spilled),
        ComparisonColumn.SUCCEEDED: succeeded(delivered, spilled),
        ComparisonColumn.DURATION: outcome[Outcome.TRANSFER_DURATION],
        ComparisonColumn.PEAK_TILT: outcome[Outcome.PEAK_TILT],
        ComparisonColumn.ENDING: outcome[Outcome.ENDING],
    }


def _gains_of(record: TrialRecord) -> Optional[PidGains]:
    """
    :return: The PID gains a trial ran with, or ``None`` for a model-predictive one.
    """
    gains = record.description.get(TrialKey.GAINS)
    return None if gains is None else PidGains.from_json(gains)


def _controller_of(record: TrialRecord) -> FillController:
    """
    :return: The controller a trial ran; trials recorded before there was a choice ran
        the model-predictive one.
    """
    return FillController(
        record.description.get(TrialKey.CONTROLLER, FillController.MODEL_PREDICTIVE)
    )


def _sensor_of(record: TrialRecord) -> SensorCondition:
    """
    :return: The sensor a trial ran with; trials recorded before there was a choice ran
        with an ideal one.
    """
    return SensorCondition(
        record.description.get(TrialKey.SENSOR, SensorCondition.IDEAL)
    )


# %% tuning the PID baseline


@dataclass
class PidTuning:
    """
    The PID baseline's tuning trials, and the gains that poured best on each cup.
    """

    records: List[TrialRecord]
    """
    Every tuning trial.
    """

    @classmethod
    def from_directory(cls, directory: Path) -> PidTuning:
        """
        :param directory: A directory holding one directory per tuning trial.
        :return: The tuning those trials record.
        """
        return cls(records=TrialRecord.read_all(directory))

    @cached_property
    def summary(self) -> pandas.DataFrame:
        """
        :return: One row per tuning trial: the cup, the gains and how the pour scored.
        """
        return pandas.DataFrame(
            [
                {
                    ComparisonColumn.SOURCE: record.description[TrialKey.SOURCE],
                    ComparisonColumn.GAINS: _gains_of(record).label,
                    ComparisonColumn.SEED: record.description[TrialKey.SEED],
                    **_scored(record.description[TrialKey.OUTCOME]),
                }
                for record in self.records
            ]
        )

    @cached_property
    def table(self) -> pandas.DataFrame:
        """
        :return: One row per cup and gains, averaged over the seeds, fewest misplaced
            grains first.
        """
        return (
            self.summary.groupby(
                [ComparisonColumn.SOURCE, ComparisonColumn.GAINS], as_index=False
            )
            .agg(
                **{
                    ComparisonColumn.MISPLACED: (ComparisonColumn.MISPLACED, "mean"),
                    ComparisonColumn.DELIVERED: (ComparisonColumn.DELIVERED, "mean"),
                    ComparisonColumn.SPILLED: (ComparisonColumn.SPILLED, "mean"),
                    ComparisonColumn.SUCCESS_RATE: (
                        ComparisonColumn.SUCCEEDED,
                        "mean",
                    ),
                    ComparisonColumn.DURATION: (ComparisonColumn.DURATION, "mean"),
                    ComparisonColumn.RUNS: (ComparisonColumn.SEED, "count"),
                }
            )
            .sort_values(
                [
                    ComparisonColumn.SOURCE,
                    ComparisonColumn.MISPLACED,
                    ComparisonColumn.DURATION,
                ]
            )
        )

    def best(self) -> TunedPidGains:
        """
        :return: For each cup, the gains that misplaced the fewest grains on average,
            the quicker pour among equals.
        """
        gains_by_label = {gains.label: gains for gains in map(_gains_of, self.records)}
        best_rows = self.table.groupby(ComparisonColumn.SOURCE).head(1)
        return TunedPidGains(
            gains={
                row[ComparisonColumn.SOURCE]: gains_by_label[
                    row[ComparisonColumn.GAINS]
                ]
                for _, row in best_rows.iterrows()
            }
        )


# %% the controllers compared


class ControllerArm(StrEnum):
    """
    One controller of the comparison, as it was set up.
    """

    MODEL_FILL_FEEDBACK = "model_fill_feedback"
    """
    The model-predictive controller at the fitted parameters, with fill feedback.
    """

    MODEL_CALIBRATION = "model_calibration"
    """
    As :attr:`MODEL_FILL_FEEDBACK`, with the drain calibrated to the measured inflow.
    """

    PID = "pid"
    """
    The PID baseline at its tuned gains, aimed by the model's aiming tasks.
    """

    PID_SOFTER = "pid_softer"
    """
    The PID baseline with its gains halved.
    """

    PID_STIFFER = "pid_stiffer"
    """
    The PID baseline with its gains doubled.
    """

    PID_FIXED_POSE = "pid_fixed_pose"
    """
    The PID baseline at its tuned gains, holding the cup still over the receiver: no
    model at all.
    """

    PID_STOP_EARLY = "pid_stop_early"
    """
    The PID baseline that stops once its reading, projected over the sensor's lag by
    its recent rate, reaches the goal.
    """

    PID_SMITH = "pid_smith"
    """
    The PID baseline as a Smith predictor, with an inflow model identified from
    readings.
    """

    PID_STOP_EARLY_FIXED_POSE = "pid_stop_early_fixed_pose"
    """
    :attr:`PID_STOP_EARLY`, holding the cup still over the receiver.
    """

    PID_SMITH_FIXED_POSE = "pid_smith_fixed_pose"
    """
    :attr:`PID_SMITH`, holding the cup still over the receiver.
    """


AIMED_ARMS: Dict[PidVariant, ControllerArm] = {
    PidVariant.PLAIN: ControllerArm.PID,
    PidVariant.STOP_EARLY: ControllerArm.PID_STOP_EARLY,
    PidVariant.SMITH: ControllerArm.PID_SMITH,
}
"""
The arm of each PID variant at its tuned gains, aimed by the model's aiming tasks.
"""

HELD_STILL_ARMS: Dict[PidVariant, ControllerArm] = {
    PidVariant.PLAIN: ControllerArm.PID_FIXED_POSE,
    PidVariant.STOP_EARLY: ControllerArm.PID_STOP_EARLY_FIXED_POSE,
    PidVariant.SMITH: ControllerArm.PID_SMITH_FIXED_POSE,
}
"""
The arm of each PID variant holding the cup still over the receiver.
"""


@dataclass
class ControllerComparison:
    """
    The model-predictive controller and the PID baseline, poured in every scene under
    every sensor.
    """

    records: List[TrialRecord]
    """
    The trials of the studies compared: the robustness study and those after it.
    """

    tuned: TunedPidGains
    """
    The gains each cup's PID baseline was tuned to.
    """

    def arm_of(self, record: TrialRecord) -> Optional[ControllerArm]:
        """
        :param record: A transfer trial.
        :return: The controller it ran as part of the comparison, or ``None`` for a
            trial that is no part of it.
        """
        controller = _controller_of(record)
        if controller is FillController.MODEL_PREDICTIVE:
            return self._model_arm_of(record)
        variant = PidVariant(
            record.description.get(TrialKey.PID_VARIANT, PidVariant.PLAIN)
        )
        if controller is FillController.PID_FIXED_POSE:
            return HELD_STILL_ARMS[variant]
        tuned = self.tuned.of(record.description[TrialKey.SOURCE])
        gains = _gains_of(record)
        if variant is not PidVariant.PLAIN:
            return AIMED_ARMS[variant] if gains == tuned else None
        return {
            tuned: ControllerArm.PID,
            tuned.scaled(SOFTER_GAINS): ControllerArm.PID_SOFTER,
            tuned.scaled(STIFFER_GAINS): ControllerArm.PID_STIFFER,
        }.get(gains)

    @staticmethod
    def _model_arm_of(record: TrialRecord) -> Optional[ControllerArm]:
        """
        :return: The model-predictive arm a trial ran, or ``None`` if it ran away from
            the fitted parameters or without feedback.
        """
        perturbation = Perturbation.from_json(record.description[TrialKey.PERTURBATION])
        if perturbation.label != BASELINE.label:
            return None
        return {
            LoopMode.FILL_FEEDBACK: ControllerArm.MODEL_FILL_FEEDBACK,
            LoopMode.FILL_FEEDBACK_AND_CALIBRATION: ControllerArm.MODEL_CALIBRATION,
        }.get(LoopMode(record.description[TrialKey.LOOP_MODE]))

    @cached_property
    def compared(self) -> List[TrialRecord]:
        """
        :return: Every recorded trial that is part of the comparison.
        """
        return [record for record in self.records if self.arm_of(record) is not None]

    @cached_property
    def summary(self) -> pandas.DataFrame:
        """
        :return: One row per compared trial: scene, controller, sensor and score.
        """
        return pandas.DataFrame(
            [
                {
                    ComparisonColumn.SOURCE: record.description[TrialKey.SOURCE],
                    ComparisonColumn.RECEIVER: record.description[TrialKey.RECEIVER],
                    ComparisonColumn.ARM: self.arm_of(record),
                    ComparisonColumn.SENSOR: _sensor_of(record),
                    ComparisonColumn.SEED: record.description[TrialKey.SEED],
                    **_scored(record.description[TrialKey.OUTCOME]),
                }
                for record in self.compared
            ]
        )

    @cached_property
    def aggregate(self) -> pandas.DataFrame:
        """
        :return: One row per scene, controller and sensor, over the seeds.
        """
        return self.summary.groupby(
            [
                ComparisonColumn.SOURCE,
                ComparisonColumn.RECEIVER,
                ComparisonColumn.ARM,
                ComparisonColumn.SENSOR,
            ],
            as_index=False,
        ).agg(
            **{
                ComparisonColumn.DELIVERED: (ComparisonColumn.DELIVERED, "mean"),
                ComparisonColumn.STANDARD_DEVIATION: (
                    ComparisonColumn.DELIVERED,
                    "std",
                ),
                ComparisonColumn.SPILLED: (ComparisonColumn.SPILLED, "mean"),
                ComparisonColumn.MISPLACED: (ComparisonColumn.MISPLACED, "mean"),
                ComparisonColumn.SUCCESS_RATE: (ComparisonColumn.SUCCEEDED, "mean"),
                ComparisonColumn.DURATION: (ComparisonColumn.DURATION, "mean"),
                ComparisonColumn.RUNS: (ComparisonColumn.SEED, "count"),
            }
        )

    @cached_property
    def traces(self) -> pandas.DataFrame:
        """
        :return: The tilt and the grains in the receiver over time, from the cup
            settling until it is upright again, for every compared trial.
        """
        frames = []
        for record in self.compared:
            samples = record.samples
            poured = samples[
                samples[Sample.PHASE].isin(
                    [Phase.PRE_TILT, Phase.TRANSFER, Phase.RETURN]
                )
            ]
            frames.append(
                pandas.DataFrame(
                    {
                        ComparisonColumn.SOURCE: record.description[TrialKey.SOURCE],
                        ComparisonColumn.RECEIVER: record.description[
                            TrialKey.RECEIVER
                        ],
                        ComparisonColumn.ARM: self.arm_of(record),
                        ComparisonColumn.SENSOR: _sensor_of(record),
                        ComparisonColumn.SEED: record.description[TrialKey.SEED],
                        ComparisonColumn.TIME: poured[Sample.TIME].to_numpy(),
                        ComparisonColumn.TILT: poured[Sample.TILT].to_numpy(),
                        ComparisonColumn.GRAINS_IN_RECEIVER: poured[
                            Sample.GRAINS_IN_RECEIVER
                        ].to_numpy(),
                    }
                )
            )
        return pandas.concat(frames, ignore_index=True)

    @property
    def sources(self) -> List[SourceCup]:
        """
        :return: The source cups compared, in enum order.
        """
        present = set(self.summary[ComparisonColumn.SOURCE])
        return [source for source in SourceCup if source in present]
