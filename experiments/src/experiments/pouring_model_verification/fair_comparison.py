"""
The model with fill feedback against the fairly tuned PID baselines, on the held-out
seeds, scene by scene and sensor by sensor.

Which PID meets the model under a sensor is decided by the tuning alone, never by the
held-out pours: the PID tuned for the ideal sensor under a sensor without lag, and the
variant that tuned best for the realistic scale under one with lag.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import pandas
from typing_extensions import Dict, List, Sequence, Tuple

from experiments.pouring_model_verification.controller_comparison import (
    ComparisonColumn,
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.fair_tuning import (
    PidSelection,
    has_lag,
)
from experiments.pouring_model_verification.parameters import BASELINE, Perturbation
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.records import (
    Outcome,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import ReceiverCup, SourceCup
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    FillController,
    LoopMode,
)

# %% how two sets of pours compare


@dataclass(frozen=True)
class SeedSpread:
    """
    How a controller's pours of one scene and sensor spread over the seeds.
    """

    mean: float
    """
    Mean grains misplaced.
    """

    minimum: float
    """
    Fewest grains misplaced in one pour.
    """

    maximum: float
    """
    Most grains misplaced in one pour.
    """

    within_tolerance: int
    """
    How many pours ended within tolerance.
    """

    runs: int
    """
    How many pours there were.
    """

    @classmethod
    def of(cls, misplaced: Sequence[float], succeeded: Sequence[bool]) -> SeedSpread:
        """
        :param misplaced: Each pour's grains misplaced.
        :param succeeded: Whether each pour ended within tolerance.
        :return: Their spread.
        """
        return cls(
            mean=sum(misplaced) / len(misplaced),
            minimum=min(misplaced),
            maximum=max(misplaced),
            within_tolerance=sum(succeeded),
            runs=len(misplaced),
        )

    def covers(self, value: float) -> bool:
        """
        :return: Whether a value lies within the range of these pours.
        """
        return self.minimum <= value <= self.maximum


class Verdict(StrEnum):
    """
    How the PID compares with the model on one scene and sensor.
    """

    PID_BETTER = "pid_better"
    MODEL_BETTER = "model_better"
    WITHIN_NOISE = "within_noise"

    @classmethod
    def between(cls, model: SeedSpread, pid: SeedSpread) -> Verdict:
        """
        :return: Within noise when either mean lies within the other's range over the
            seeds; otherwise whichever misplaced fewer grains on average.
        """
        if pid.covers(model.mean) or model.covers(pid.mean):
            return cls.WITHIN_NOISE
        return cls.PID_BETTER if pid.mean < model.mean else cls.MODEL_BETTER


class FairColumn(StrEnum):
    """
    A column of the fair comparison's table.
    """

    SOURCE = "source"
    RECEIVER = "receiver"
    SENSOR = "sensor"
    PID_VARIANT = "pid_variant"
    PID_TUNED_FOR = "pid_tuned_for"
    MODEL_MEAN = "model_mean"
    MODEL_MINIMUM = "model_minimum"
    MODEL_MAXIMUM = "model_maximum"
    MODEL_WITHIN_TOLERANCE = "model_within_tolerance"
    PID_MEAN = "pid_mean"
    PID_MINIMUM = "pid_minimum"
    PID_MAXIMUM = "pid_maximum"
    PID_WITHIN_TOLERANCE = "pid_within_tolerance"
    RUNS = "runs"
    MARGIN = "margin"
    VERDICT = "verdict"


# %% reading the pours


def _spread(records: List[TrialRecord]) -> SeedSpread:
    """
    :return: The spread of some pours' grains misplaced.
    """
    outcomes = [record.description[TrialKey.OUTCOME] for record in records]
    return SeedSpread.of(
        misplaced=[
            misplaced_grains(outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED])
            for outcome in outcomes
        ],
        succeeded=[
            succeeded(outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED])
            for outcome in outcomes
        ],
    )


def _is_model_with_fill_feedback(record: TrialRecord) -> bool:
    """
    :return: Whether a trial ran the model at its fitted parameters with fill feedback.
    """
    description = record.description
    return (
        description.get(TrialKey.CONTROLLER, FillController.MODEL_PREDICTIVE)
        == FillController.MODEL_PREDICTIVE
        and description[TrialKey.LOOP_MODE] == LoopMode.FILL_FEEDBACK
        and Perturbation.from_json(description[TrialKey.PERTURBATION]).label
        == BASELINE.label
    )


# %% the comparison


@dataclass
class FairComparison:
    """
    The model with fill feedback and the fairly tuned PID, on the held-out seeds.
    """

    model: List[TrialRecord]
    """
    Pours of the model, of which those at the fitted parameters with fill feedback are
    compared.
    """

    pid: List[TrialRecord]
    """
    The rescoring pours of the tuned PIDs.
    """

    selection: PidSelection
    """
    Which tuned PID meets the model under each sensor.
    """

    def _pid_pours(
        self, source: str, receiver: str, sensor: SensorCondition
    ) -> List[TrialRecord]:
        variant, gains = self.selection.for_sensor(SourceCup(source), sensor)
        return [
            record
            for record in self.pid
            if record.description[TrialKey.SOURCE] == source
            and record.description[TrialKey.RECEIVER] == receiver
            and record.description[TrialKey.SENSOR] == sensor
            and record.description[TrialKey.PID_VARIANT] == variant
            and PidGains.from_json(record.description[TrialKey.GAINS]) == gains
        ]

    @cached_property
    def table(self) -> pandas.DataFrame:
        """
        :return: One row per scene and sensor with pours of both: each side's spread,
            the margin (PID minus model, in mean grains misplaced) and the verdict.
        """
        models: Dict[Tuple[str, str, str], List[TrialRecord]] = {}
        for record in filter(_is_model_with_fill_feedback, self.model):
            description = record.description
            key = (
                description[TrialKey.SOURCE],
                description[TrialKey.RECEIVER],
                description.get(TrialKey.SENSOR, SensorCondition.IDEAL),
            )
            models.setdefault(key, []).append(record)
        rows = []
        for (source, receiver, sensor), model_pours in models.items():
            sensor = SensorCondition(sensor)
            pid_pours = self._pid_pours(source, receiver, sensor)
            if not pid_pours:
                continue
            model, pid = _spread(model_pours), _spread(pid_pours)
            variant, _ = self.selection.for_sensor(SourceCup(source), sensor)
            rows.append(
                {
                    FairColumn.SOURCE: source,
                    FairColumn.RECEIVER: receiver,
                    FairColumn.SENSOR: sensor,
                    FairColumn.PID_VARIANT: variant,
                    FairColumn.PID_TUNED_FOR: (
                        SensorCondition.SCALE
                        if has_lag(sensor)
                        else SensorCondition.IDEAL
                    ),
                    FairColumn.MODEL_MEAN: model.mean,
                    FairColumn.MODEL_MINIMUM: model.minimum,
                    FairColumn.MODEL_MAXIMUM: model.maximum,
                    FairColumn.MODEL_WITHIN_TOLERANCE: model.within_tolerance,
                    FairColumn.PID_MEAN: pid.mean,
                    FairColumn.PID_MINIMUM: pid.minimum,
                    FairColumn.PID_MAXIMUM: pid.maximum,
                    FairColumn.PID_WITHIN_TOLERANCE: pid.within_tolerance,
                    FairColumn.RUNS: pid.runs,
                    FairColumn.MARGIN: pid.mean - model.mean,
                    FairColumn.VERDICT: Verdict.between(model, pid),
                }
            )
        return pandas.DataFrame(rows)
