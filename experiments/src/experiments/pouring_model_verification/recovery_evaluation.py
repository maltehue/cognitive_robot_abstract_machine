"""
Scoring the recovery study: one row per pour with how it ended and, for a pour with the
hypothesis bank, what the bank committed to, when, and how close it came to the truth.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

import numpy
import pandas
from typing_extensions import Any, Dict, List, Optional

from experiments.pouring_model_verification.controller_comparison import (
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.hypothesis_bank import (
    FreeParameter,
    Hypothesis,
    ModelPoint,
)
from experiments.pouring_model_verification.onset_rule import OnsetResponse
from experiments.pouring_model_verification.onset_study import (
    OnsetCondition,
    SelectsPid,
    pid_records_of,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    Baselines,
    ModelParameter,
    Perturbation,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.recovery import injected_hypothesis
from experiments.pouring_model_verification.recovery_study import (
    COMPARISON_SENSORS,
    RecoveryCondition,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition, SensorSet
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    FillController,
    LoopMode,
    MotionEnding,
)

# %% the arms and columns


class RecoveryArm(StrEnum):
    """
    What steered a pour of the recovery study.
    """

    MODEL_ALONE = "model_alone"
    RULE = "rule"
    BANK = "bank"
    ORACLE = "oracle"
    BEST_PID = "best_pid"


class RecoveryColumn(StrEnum):
    """
    A column of the recovery study's rows.
    """

    SOURCE = "source"
    RECEIVER = "receiver"
    PERTURBATION = "perturbation"
    FORCED_LIP = "forced_lip"
    FAULT = "fault"
    SENSORS = "sensors"
    ARM = "arm"
    SETTINGS = "settings"
    SEED = "seed"
    SEQUENCE_POUR = "sequence_pour"
    DELIVERED = "delivered"
    SPILLED = "spilled"
    MISPLACED = "misplaced"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    ENDING = "ending"
    DURATION = "duration"
    TRUTH = "truth"
    COMMITTED = "committed"
    FINAL = "final"
    CORRECT = "correct"
    COMMIT_TIME = "commit_time"
    POURED_SHARE_AT_COMMIT = "poured_share_at_commit"
    LOST_BEFORE_COMMIT = "lost_before_commit"
    REPOSE_ERROR = "repose_error"
    DISCHARGE_RATIO = "discharge_ratio"
    RATE_RATIO = "rate_ratio"
    SLOT_HEIGHT = "slot_height"
    REVISES = "revises"
    CATCH_ALL = "catch_all"
    SWITCHES = "switches"
    TOO_LATE = "too_late"
    WRITES_BACK = "writes_back"
    BOUND_SHARE = "bound_share"
    TILT_STEP_AT_SWITCH = "tilt_step_at_switch"
    TILT_STEP_ELSEWHERE = "tilt_step_elsewhere"
    LIP_ACTIVE = "lip_active"


TOO_LATE_SHARE = 0.5
"""
The share of the goal after which a commit comes too late to recover the pour.
"""

SWITCH_WINDOW = 0.5
"""
How long after a switch the tilt is watched for a jump, in seconds.
"""

SENSOR_OF_SET = {
    SensorSet.IDEAL: SensorCondition.IDEAL,
    SensorSet.REALISTIC: SensorCondition.SCALE,
    SensorSet.NOISY_CAMERA: SensorCondition.SCALE,
    SensorSet.NO_CAMERA: SensorCondition.SCALE,
}
"""
The receiver's sensor the arms without a bank ran with, for each of the bank's sensor
sets.
"""

TROUGH_SHARE = 0.5
"""
How many grains, as a share of one, have to stand in the spout before any leaves the cup
for the slot edge to count as a lip of its own.
"""

TROUGH_HOLD = 1.0
"""
How long grains have to stand in the spout before any leaves the cup, in seconds, for
the slot edge to count as a lip of its own.
"""

# %% what the simulation says of the spout's slot edge


def slot_edge_acts_as_lip(samples: pandas.DataFrame) -> Optional[bool]:
    """
    :param samples: A pour's samples.
    :return: Whether the grains crossed the slot edge and stood in the trough for a
        while before any left the cup, so the slot edge, not the outlet, is where the
        pour started; ``None`` for a cup without a spout.
    """
    if Sample.GRAINS_IN_SPOUT not in samples:
        return None
    transfer = samples[samples[Sample.PHASE] == Phase.TRANSFER]
    if not len(transfer) or not transfer[Sample.GRAINS_IN_SPOUT].any():
        return False
    total = transfer[Sample.GRAINS_IN_SOURCE].iloc[0]
    left = transfer[transfer[Sample.GRAINS_IN_SOURCE] < total]
    first_left = left[Sample.TIME].iloc[0] if len(left) else math.inf
    in_trough = transfer[transfer[Sample.GRAINS_IN_SPOUT] >= TROUGH_SHARE]
    if not len(in_trough):
        return False
    return bool(first_left - in_trough[Sample.TIME].iloc[0] >= TROUGH_HOLD)


# %% one row per pour


def _condition_key(description: Dict[str, Any]) -> tuple:
    return (
        description[TrialKey.SOURCE],
        description[TrialKey.RECEIVER],
        Perturbation.from_json(description[TrialKey.PERTURBATION]).label,
        bool(description.get(TrialKey.FORCES_SECOND_LIP) or False),
        description.get(TrialKey.FAULT),
    )


@dataclass
class RecoveryEvaluation:
    """
    Every pour of the recovery study, scored, with the bank's commits judged against the
    error set into the model.
    """

    records: List[TrialRecord]
    """
    The pours of every arm but the PID.
    """

    conditions: List[RecoveryCondition]
    """
    The conditions evaluated.
    """

    baselines: Baselines
    """
    Each cup's fitted parameters: the truth a parameter's estimate is set against.
    """

    pid_records: List[TrialRecord]
    """
    The fairly tuned PIDs' pours.
    """

    selection: Optional[SelectsPid]
    """
    Which PID meets the model under each sensor, or ``None`` to leave the PID out.
    """

    @cached_property
    def rows(self) -> pandas.DataFrame:
        """
        :return: One row per pour and condition: the arms without a bank once per
            receiver sensor, the oracle and the PID standing for every model of their
            scene.
        """
        wanted = {
            (
                condition.scene.source,
                condition.scene.receiver,
                condition.perturbation.label,
                condition.forces_second_lip,
                condition.fault,
            ): condition
            for condition in self.conditions
        }
        rows = []
        for record in self.records:
            description = record.description
            if description[TrialKey.LOOP_MODE] != LoopMode.FILL_FEEDBACK:
                continue
            if (
                description.get(TrialKey.CONTROLLER, FillController.MODEL_PREDICTIVE)
                != FillController.MODEL_PREDICTIVE
            ):
                continue
            arm = self._arm_of(description)
            if arm is None:
                continue
            if arm is RecoveryArm.ORACLE:
                for condition in self._oracle_conditions(description, wanted):
                    rows.append(self._row(record, condition, arm))
                continue
            condition = wanted.get(_condition_key(description))
            if condition is None:
                continue
            if (
                arm is RecoveryArm.MODEL_ALONE
                and condition.perturbation == BASELINE
                and condition.scene.source is not SourceCup.SPOUTED_CUP
            ):
                rows.append(self._row(record, condition, RecoveryArm.ORACLE))
            rows.append(self._row(record, condition, arm))
        rows += self._pid_rows()
        return pandas.DataFrame(rows)

    @staticmethod
    def _arm_of(description: Dict[str, Any]) -> Optional[RecoveryArm]:
        """
        :return: The arm a pour belongs to, or ``None`` for one of no arm.
        """
        if description.get(TrialKey.BANK) is not None:
            return RecoveryArm.BANK
        rule = description.get(TrialKey.ONSET_RULE)
        if rule is not None:
            if rule["response"] == OnsetResponse.CONCEPT_AWARE_REVISION:
                return RecoveryArm.RULE
            return None
        if "second_lip_fit" in description[TrialKey.NAME]:
            return RecoveryArm.ORACLE
        if description.get(TrialKey.SEQUENCE) is not None:
            return None
        return RecoveryArm.MODEL_ALONE

    def _oracle_conditions(
        self, description: Dict[str, Any], wanted: Dict[tuple, RecoveryCondition]
    ) -> List[RecoveryCondition]:
        """
        :return: The conditions an offline-fitted oracle pour stands for: the spouted
            cup at its fit in its scene.
        """
        return [
            condition
            for key, condition in wanted.items()
            if key[0] == description[TrialKey.SOURCE]
            and key[1] == description[TrialKey.RECEIVER]
            and condition.perturbation == BASELINE
            and condition.fault == description.get(TrialKey.FAULT)
            and not condition.forces_second_lip
        ]

    def _pid_rows(self) -> List[Dict[str, Any]]:
        """
        :return: The best PID's pours, once per condition and receiver sensor.
        """
        if self.selection is None:
            return []
        rows = []
        seen = set()
        for condition in self.conditions:
            if condition.fault is not None or condition.forces_second_lip:
                continue
            key = (condition.scene, condition.perturbation)
            if key in seen:
                continue
            seen.add(key)
            for sensor in COMPARISON_SENSORS:
                onset_condition = OnsetCondition(
                    scene=condition.scene,
                    perturbation=condition.perturbation,
                    sensor=sensor,
                )
                for record in pid_records_of(
                    onset_condition, self.pid_records, self.selection
                ):
                    rows.append(self._row(record, condition, RecoveryArm.BEST_PID))
        return rows

    def _row(
        self, record: TrialRecord, condition: RecoveryCondition, arm: RecoveryArm
    ) -> Dict[str, Any]:
        """
        :return: What the evaluation reads of one pour under one condition and arm.
        """
        description = record.description
        outcome = description[TrialKey.OUTCOME]
        delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
        sensors = description.get(TrialKey.SENSOR_SET) or description.get(
            TrialKey.SENSOR, SensorCondition.IDEAL
        )
        bank = description.get(TrialKey.BANK) or {}
        sequence = description.get(TrialKey.SEQUENCE) or {}
        row = {
            RecoveryColumn.SOURCE: condition.scene.source,
            RecoveryColumn.RECEIVER: condition.scene.receiver,
            RecoveryColumn.PERTURBATION: condition.perturbation.label,
            RecoveryColumn.FORCED_LIP: condition.forces_second_lip,
            RecoveryColumn.FAULT: condition.fault,
            RecoveryColumn.SENSORS: sensors,
            RecoveryColumn.ARM: arm,
            RecoveryColumn.SETTINGS: bank.get("label"),
            RecoveryColumn.SEED: description[TrialKey.SEED],
            RecoveryColumn.SEQUENCE_POUR: sequence.get("pour"),
            RecoveryColumn.DELIVERED: delivered,
            RecoveryColumn.SPILLED: spilled,
            RecoveryColumn.MISPLACED: misplaced_grains(delivered, spilled),
            RecoveryColumn.SUCCEEDED: succeeded(delivered, spilled),
            RecoveryColumn.TIMED_OUT: outcome[Outcome.ENDING] == MotionEnding.TIMED_OUT,
            RecoveryColumn.ENDING: outcome[Outcome.ENDING],
            RecoveryColumn.DURATION: outcome[Outcome.TRANSFER_DURATION],
        }
        if arm is RecoveryArm.BANK:
            row.update(self._bank_columns(record, condition))
        return row

    def _bank_columns(
        self, record: TrialRecord, condition: RecoveryCondition
    ) -> Dict[str, Any]:
        """
        :return: What the bank committed to and how well, for one pour with a bank.
        """
        description = record.description
        samples = record.samples
        transfer = samples[samples[Sample.PHASE] == Phase.TRANSFER]
        start = float(transfer[Sample.TIME].iloc[0])
        commits = description[TrialKey.COMMITS]
        first = next(
            (
                commit
                for commit in commits
                if commit["hypothesis"] != Hypothesis.NOMINAL
            ),
            None,
        )
        lip = slot_edge_acts_as_lip(samples)
        truth = injected_hypothesis(condition.perturbation)
        if condition.fault is not None:
            truth = Hypothesis.CATCH_ALL
        elif truth is Hypothesis.NOMINAL and lip:
            truth = Hypothesis.SECOND_LIP
        committed = None if first is None else Hypothesis(first["hypothesis"])
        succeeded_now = succeeded(
            description[TrialKey.OUTCOME][Outcome.DELIVERED],
            description[TrialKey.OUTCOME][Outcome.SPILLED],
        )
        share = None if first is None else first["poured_share"]
        too_late = (share is None or share > TOO_LATE_SHARE) and not succeeded_now
        row = {
            RecoveryColumn.TRUTH: None if truth is None else truth.label,
            RecoveryColumn.COMMITTED: None if committed is None else committed.label,
            RecoveryColumn.FINAL: Hypothesis(
                commits[-1]["hypothesis"] if commits else Hypothesis.NOMINAL
            ).label,
            RecoveryColumn.CORRECT: (
                None if truth is None else (committed or Hypothesis.NOMINAL) is truth
            ),
            RecoveryColumn.COMMIT_TIME: (
                None if first is None else first["time"] - start
            ),
            RecoveryColumn.POURED_SHARE_AT_COMMIT: share,
            RecoveryColumn.LOST_BEFORE_COMMIT: (
                None if first is None else self._lost_by(record, first["time"])
            ),
            RecoveryColumn.REVISES: committed is not None and committed.revises,
            RecoveryColumn.CATCH_ALL: any(
                commit["hypothesis"] == Hypothesis.CATCH_ALL for commit in commits
            ),
            RecoveryColumn.SWITCHES: len(commits),
            RecoveryColumn.TOO_LATE: too_late,
            RecoveryColumn.WRITES_BACK: description.get(TrialKey.WRITES_BACK),
            RecoveryColumn.BOUND_SHARE: self._bound_share(description, committed),
            RecoveryColumn.LIP_ACTIVE: lip,
        }
        row.update(self._parameter_errors(condition, first))
        row.update(self._tilt_steps(transfer, [commit["time"] for commit in commits]))
        return row

    @staticmethod
    def _lost_by(record: TrialRecord, at: float) -> float:
        """
        :return: How many grains had landed outside both cups by a time.
        """
        samples = record.samples
        before = samples[samples[Sample.TIME] <= at + 1e-9]
        if not len(before):
            return 0.0
        landed = samples[samples[Sample.GRAINS_IN_RECEIVER] > 0]
        scale = (
            landed[Sample.GRAINS_IN_RECEIVER].iloc[-1]
            / landed[Sample.MEASURED_RECEIVER_FILL].iloc[-1]
            if len(landed)
            else 0.0
        )
        last = before.iloc[-1]
        total = record.description[TrialKey.OUTCOME][Outcome.TOTAL_GRAINS]
        return max(
            0.0,
            total
            - last[Sample.GRAINS_IN_SOURCE]
            - last[Sample.COMMITTED_RECEIVER_FILL] * scale,
        )

    def _parameter_errors(
        self, condition: RecoveryCondition, first: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        :return: How far the committed parameters lay from the cup's fit: the repose
            angle in degrees, the coefficient and the rate as ratios.
        """
        if first is None:
            return {}
        point = ModelPoint.from_json(first.get("estimate", first["point"]))
        truth = self.baselines.of(condition.scene.source)
        return {
            RecoveryColumn.REPOSE_ERROR: math.degrees(
                point.repose_angle - truth.repose_angle
            ),
            RecoveryColumn.DISCHARGE_RATIO: point.discharge_coefficient
            / truth.discharge_coefficient,
            RecoveryColumn.RATE_RATIO: point.outflow_rate_constant
            / truth.outflow_rate_constant,
            RecoveryColumn.SLOT_HEIGHT: point.slot_height,
        }

    @staticmethod
    def _bound_share(
        description: Dict[str, Any], committed: Optional[Hypothesis]
    ) -> Optional[float]:
        """
        :return: The posterior mass the committed hypothesis held at a bound of any of
            its parameters at the end of the pour.
        """
        if committed is None or not committed.revises:
            return None
        shares = description[TrialKey.BANK_SUMMARY][committed]["bound_share"]
        return max(shares.values(), default=0.0)

    @staticmethod
    def _tilt_steps(
        transfer: pandas.DataFrame, switches: List[float]
    ) -> Dict[str, Any]:
        """
        :return: The largest tilt change between two samples within a short window
            after any switch, and the largest elsewhere in the transfer, in radians.
        """
        times = transfer[Sample.TIME].to_numpy(dtype=float)
        steps = numpy.abs(numpy.diff(transfer[Sample.TILT].to_numpy(dtype=float)))
        if not len(steps):
            return {}
        near = numpy.zeros(len(steps), dtype=bool)
        for at in switches:
            near |= (times[1:] >= at) & (times[1:] <= at + SWITCH_WINDOW)
        return {
            RecoveryColumn.TILT_STEP_AT_SWITCH: (
                float(steps[near].max()) if near.any() else None
            ),
            RecoveryColumn.TILT_STEP_ELSEWHERE: (
                float(steps[~near].max()) if (~near).any() else None
            ),
        }
