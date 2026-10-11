"""
Choosing what to revise: the model with the concept-aware revision, which weighs the
source to tell a stall the onset explains from one the aim explains, against the model
alone, the revision that only knows the onset, and the best PID, where the aim fails
(the spouted cup into the narrow receiver, and a discharge coefficient set wrong) and
where the onset does (a repose angle set too low).
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property

import numpy
import pandas
from typing_extensions import Any, Dict, FrozenSet, List, Optional

from experiments.pouring_model_verification.concept_revision import (
    DiagnosisKey,
    RevisedConcept,
)
from experiments.pouring_model_verification.controller_comparison import (
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.onset_rule import (
    ARRIVAL,
    OnsetResponse,
    OnsetRule,
)
from experiments.pouring_model_verification.onset_study import (
    EARLY_ONSET_PERTURBATIONS,
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
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition, WristSensor
from experiments.pouring_model_verification.transfer_trial import (
    POUR_START_TILT_MARGIN,
    FillController,
    LoopMode,
    MotionEnding,
    TransferTrial,
)

# %% the conditions

DIAGNOSIS_SENSORS: Dict[SensorCondition, WristSensor] = {
    SensorCondition.IDEAL: WristSensor.IDEAL,
    SensorCondition.SCALE: WristSensor.REALISTIC,
}
"""
The receiver's sensor of each run, with the wrist sensor weighing the source beside it:

both ideal, or both realistic.
"""

WRONG_DISCHARGE_PERTURBATIONS = (
    Perturbation(parameter=ModelParameter.DISCHARGE_COEFFICIENT, factor=0.33),
    Perturbation(parameter=ModelParameter.DISCHARGE_COEFFICIENT, factor=2.0),
)
"""
The models told the stream leaves slower or faster than it does, so they aim it wrong.
"""

AIM_FAILS_AT_FIT = Scene(source=SourceCup.SPOUTED_CUP, receiver=ReceiverCup.NARROW)
"""
The scene in which the aim fails at the cup's own fit.
"""

EARLY_ONSET_RECEIVER = ReceiverCup.WIDE
"""
The receiver the early-onset models pour into.
"""


def diagnosis_conditions() -> List[OnsetCondition]:
    """
    :return: The spouted cup at its fit into the narrow receiver, a wrong discharge
        coefficient in every scene, and a repose angle set too low for both cups into
        the wide receiver, each with both sensor pairs.
    """
    situations = (
        [(AIM_FAILS_AT_FIT, BASELINE)]
        + [
            (Scene(source=source, receiver=receiver), perturbation)
            for source in SourceCup
            for receiver in ReceiverCup
            for perturbation in WRONG_DISCHARGE_PERTURBATIONS
        ]
        + [
            (Scene(source=source, receiver=EARLY_ONSET_RECEIVER), perturbation)
            for source in SourceCup
            for perturbation in EARLY_ONSET_PERTURBATIONS
        ]
    )
    return [
        OnsetCondition(scene=scene, perturbation=perturbation, sensor=sensor)
        for scene, perturbation in situations
        for sensor in DIAGNOSIS_SENSORS
    ]


def diagnosis_trials(
    seeds: List[int],
    baselines: Baselines,
    revision: OnsetRule,
    lowers: bool,
    recorded: FrozenSet[str],
) -> List[TransferTrial]:
    """
    Under every condition: the model alone from the usual start margin, the revision
    that knows only the onset and the concept-aware revision from the model's onset.

    :param seeds: The held-out microstates.
    :param baselines: Each cup's fitted parameters.
    :param revision: The tuned revision, whose patience both revisions use.
    :param lowers: Whether the revisions also lower the repose angle.
    :param recorded: The names of pours an earlier study already recorded.
    :return: The pours not recorded yet.
    """
    onset_only = replace(revision, lowers=lowers)
    concept_aware = replace(onset_only, response=OnsetResponse.CONCEPT_AWARE_REVISION)
    trials = []
    for condition in diagnosis_conditions():
        common = dict(
            scene=condition.scene,
            baseline=baselines.of(condition.scene.source),
            perturbation=condition.perturbation,
            loop_mode=LoopMode.FILL_FEEDBACK,
            sensor=condition.sensor,
        )
        for seed in seeds:
            trials += [
                TransferTrial(**common, seed=seed),
                TransferTrial(
                    **common, seed=seed, onset_rule=onset_only, start_margin=0.0
                ),
                TransferTrial(
                    **common,
                    seed=seed,
                    onset_rule=concept_aware,
                    start_margin=0.0,
                    wrist_sensor=DIAGNOSIS_SENSORS[condition.sensor],
                ),
            ]
    return [trial for trial in trials if trial.name not in recorded]


# %% which concept was the right one


ARRIVING_SHARE = 0.5
"""
The share of the grains the source lost that has to reach the receiver over the same
time for a stall to count as grains still arriving, not a miss.
"""


def right_concept(
    samples: pandas.DataFrame, at: float, patience: float
) -> Optional[RevisedConcept]:
    """
    :param samples: A pour's samples.
    :param at: When a stall was diagnosed, in seconds.
    :param patience: How long the receiver had gained nothing, in seconds.
    :return: The concept the stall was truly due to, by the grains the simulation
        counted over the patience: the onset if the source lost less than a grain, the
        aim if it lost grains the receiver did not get, and ``None`` if they did reach
        the receiver, so nothing stalled but the reading.
    """
    times = samples[Sample.TIME].to_numpy(dtype=float)
    source = numpy.interp(
        [at - patience, at],
        times,
        samples[Sample.GRAINS_IN_SOURCE].to_numpy(dtype=float),
    )
    receiver = numpy.interp(
        [at - patience, at],
        times,
        samples[Sample.GRAINS_IN_RECEIVER].to_numpy(dtype=float),
    )
    lost, gained = source[0] - source[1], receiver[1] - receiver[0]
    if lost < ARRIVAL:
        return RevisedConcept.REPOSE_ANGLE
    if gained >= ARRIVING_SHARE * lost:
        return None
    return RevisedConcept.DISCHARGE_COEFFICIENT


# %% reading the pours


class DiagnosisArm(StrEnum):
    """
    One way the study pours.
    """

    MODEL_ALONE = "model_alone"
    """
    The model with fill feedback, from the usual start margin.
    """

    ONSET_REVISION = "onset_revision"
    """
    The revision that blames every stall on the onset.
    """

    CONCEPT_AWARE_REVISION = "concept_aware_revision"
    """
    The revision that weighs the source to choose the concept it blames.
    """

    BEST_PID = "best_pid"
    """
    The fairly tuned PID chosen for the sensor.
    """


class DiagnosisColumn(StrEnum):
    """
    A column of the study's tables.
    """

    SOURCE = "source"
    RECEIVER = "receiver"
    PERTURBATION = "perturbation"
    SENSOR = "sensor"
    ARM = "arm"
    SEED = "seed"
    DELIVERED = "delivered"
    SPILLED = "spilled"
    MISPLACED = "misplaced"
    MISPLACED_LOW = "misplaced_low"
    MISPLACED_HIGH = "misplaced_high"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    REPOSE_REVISIONS = "repose_revisions"
    DISCHARGE_REVISIONS = "discharge_revisions"
    UNSEEN_MISSES = "unseen_misses"
    RIGHT_CONCEPT = "right_concept"
    FALSE_STALLS = "false_stalls"
    DIAGNOSES = "diagnoses"
    FINAL_DISCHARGE_COEFFICIENT = "final_discharge_coefficient"
    RUNS = "runs"


def _arm_of(description: Dict[str, Any], revision: OnsetRule) -> Optional[DiagnosisArm]:
    """
    :return: The arm a model pour ran, or ``None`` for a pour of no arm.
    """
    rule_data = description.get(TrialKey.ONSET_RULE)
    margin = description.get(TrialKey.START_MARGIN, POUR_START_TILT_MARGIN)
    if rule_data is None:
        return DiagnosisArm.MODEL_ALONE if margin == POUR_START_TILT_MARGIN else None
    rule = OnsetRule.from_json(rule_data)
    if (
        margin != 0.0
        or rule.patience != revision.patience
        or rule.lowers != revision.lowers
    ):
        return None
    match rule.response:
        case OnsetResponse.REVISION:
            return DiagnosisArm.ONSET_REVISION
        case OnsetResponse.CONCEPT_AWARE_REVISION:
            return DiagnosisArm.CONCEPT_AWARE_REVISION
    return None


def _condition_of(description: Dict[str, Any]) -> Optional[OnsetCondition]:
    """
    :return: The condition a pour ran under, or ``None`` if it is none of the study's.
    """
    condition = OnsetCondition(
        scene=Scene(
            source=SourceCup(description[TrialKey.SOURCE]),
            receiver=ReceiverCup(description[TrialKey.RECEIVER]),
        ),
        perturbation=Perturbation.from_json(description[TrialKey.PERTURBATION]),
        sensor=SensorCondition(description.get(TrialKey.SENSOR, SensorCondition.IDEAL)),
    )
    return condition if condition in diagnosis_conditions() else None


@dataclass
class DiagnosisEvaluation:
    """
    Every arm under every condition, on the held-out seeds.
    """

    records: List[TrialRecord]
    """
    Pours of the model; those under no condition or on no arm of the study are left out.
    """

    pid_records: List[TrialRecord]
    """
    The fairly tuned PIDs' held-out pours.
    """

    revision: OnsetRule
    """
    The tuned revision, whose patience the revisions ran with and which says whether
    they also lowered the repose angle.
    """

    selection: Optional[SelectsPid]
    """
    Which PID meets the model under each sensor, or ``None`` to leave the PID out.
    """

    @cached_property
    def rows(self) -> pandas.DataFrame:
        """
        :return: One row per pour and condition. A PID knows no model, so its pours of
            a scene and sensor stand for every model of that scene.
        """
        rows = []
        for record in self.records:
            description = record.description
            if description[TrialKey.LOOP_MODE] != LoopMode.FILL_FEEDBACK:
                continue
            if description.get(
                TrialKey.CONTROLLER, FillController.MODEL_PREDICTIVE
            ) != (FillController.MODEL_PREDICTIVE):
                continue
            condition = _condition_of(description)
            arm = _arm_of(description, self.revision)
            if condition is None or arm is None:
                continue
            if arm is DiagnosisArm.CONCEPT_AWARE_REVISION and (
                description[TrialKey.WRIST_SENSOR]
                != DIAGNOSIS_SENSORS[condition.sensor]
            ):
                continue
            rows.append(self._row(record, condition, arm))
        if self.selection is not None:
            for condition in diagnosis_conditions():
                for record in pid_records_of(
                    condition, self.pid_records, self.selection
                ):
                    rows.append(self._row(record, condition, DiagnosisArm.BEST_PID))
        return pandas.DataFrame(rows)

    def _row(
        self, record: TrialRecord, condition: OnsetCondition, arm: DiagnosisArm
    ) -> Dict[DiagnosisColumn, Any]:
        """
        :return: What the evaluation reads of one pour.
        """
        description = record.description
        outcome = description[TrialKey.OUTCOME]
        delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
        diagnoses = description.get(TrialKey.DIAGNOSES) or []
        transfer = record.samples[record.samples[Sample.PHASE] == Phase.TRANSFER]
        revised = [
            diagnosis
            for diagnosis in diagnoses
            if diagnosis[DiagnosisKey.NEW_VALUE] != diagnosis[DiagnosisKey.OLD_VALUE]
        ]
        concepts = [diagnosis[DiagnosisKey.CONCEPT] for diagnosis in diagnoses]
        return {
            DiagnosisColumn.SOURCE: condition.scene.source,
            DiagnosisColumn.RECEIVER: condition.scene.receiver,
            DiagnosisColumn.PERTURBATION: condition.perturbation.label,
            DiagnosisColumn.SENSOR: condition.sensor,
            DiagnosisColumn.ARM: arm,
            DiagnosisColumn.SEED: description[TrialKey.SEED],
            DiagnosisColumn.DELIVERED: delivered,
            DiagnosisColumn.SPILLED: spilled,
            DiagnosisColumn.MISPLACED: misplaced_grains(delivered, spilled),
            DiagnosisColumn.SUCCEEDED: succeeded(delivered, spilled),
            DiagnosisColumn.TIMED_OUT: outcome[Outcome.ENDING]
            == MotionEnding.TIMED_OUT,
            DiagnosisColumn.REPOSE_REVISIONS: concepts.count(
                RevisedConcept.REPOSE_ANGLE
            ),
            DiagnosisColumn.DISCHARGE_REVISIONS: sum(
                diagnosis[DiagnosisKey.CONCEPT] == RevisedConcept.DISCHARGE_COEFFICIENT
                for diagnosis in revised
            ),
            DiagnosisColumn.UNSEEN_MISSES: len(diagnoses) - len(revised),
            DiagnosisColumn.RIGHT_CONCEPT: sum(
                right_concept(
                    transfer, diagnosis[DiagnosisKey.TIME], self.revision.patience
                )
                == diagnosis[DiagnosisKey.CONCEPT]
                for diagnosis in diagnoses
            ),
            DiagnosisColumn.FALSE_STALLS: sum(
                right_concept(
                    transfer, diagnosis[DiagnosisKey.TIME], self.revision.patience
                )
                is None
                for diagnosis in diagnoses
            ),
            DiagnosisColumn.DIAGNOSES: len(diagnoses),
            DiagnosisColumn.FINAL_DISCHARGE_COEFFICIENT: description.get(
                TrialKey.DISCHARGE_COEFFICIENT_AT_END
            ),
        }

    @cached_property
    def summary(self) -> pandas.DataFrame:
        """
        :return: One row per condition and arm: the mean and range of the grains
            misplaced, the mean spill, the runs within tolerance and timed out, and
            the diagnoses by concept and how many were right.
        """
        column = DiagnosisColumn
        keys = [
            column.SOURCE,
            column.RECEIVER,
            column.PERTURBATION,
            column.SENSOR,
            column.ARM,
        ]
        return self.rows.groupby(keys, as_index=False, sort=False).agg(
            **{
                column.MISPLACED: (column.MISPLACED, "mean"),
                column.MISPLACED_LOW: (column.MISPLACED, "min"),
                column.MISPLACED_HIGH: (column.MISPLACED, "max"),
                column.SPILLED: (column.SPILLED, "mean"),
                column.SUCCEEDED: (column.SUCCEEDED, "sum"),
                column.TIMED_OUT: (column.TIMED_OUT, "sum"),
                column.RUNS: (column.MISPLACED, "count"),
                column.REPOSE_REVISIONS: (column.REPOSE_REVISIONS, "sum"),
                column.DISCHARGE_REVISIONS: (column.DISCHARGE_REVISIONS, "sum"),
                column.UNSEEN_MISSES: (column.UNSEEN_MISSES, "sum"),
                column.RIGHT_CONCEPT: (column.RIGHT_CONCEPT, "sum"),
                column.FALSE_STALLS: (column.FALSE_STALLS, "sum"),
                column.DIAGNOSES: (column.DIAGNOSES, "sum"),
            }
        )
