"""
Starting the pour: the onset rule's settings tuned once on tuning seeds, then the model
with each response evaluated on the held-out seeds, where the model alone stalls (the
plain cup at its fit, and either cup told the contents start sooner than they do) and
where it does not (the spouted cup at its fit, to check the rule does no harm).
"""

# %% imports

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import pandas
from typing_extensions import Any, Dict, List, Optional, Protocol, Tuple

from experiments.pouring_model_verification.controller_comparison import (
    ComparisonColumn,
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.onset_rule import (
    FILL_ROW_WEIGHT,
    OVERRIDE_WEIGHT,
    OnsetResponse,
    OnsetRule,
    RevisionKey,
)
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    Baselines,
    ModelParameter,
    Perturbation,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    POUR_START_TILT_MARGIN,
    FillController,
    LoopMode,
    MotionEnding,
    TransferTrial,
)

# %% tuning the rule

ONSET_TUNING_SEEDS = (10, 11, 12)
"""
The microstates the rule is tuned on: new to every held-out comparison.
"""

ONSET_TUNING_SCENE = Scene(source=SourceCup.CUP, receiver=ReceiverCup.WIDE)
"""
The scene the rule is tuned in: the one where the model alone stalls at its fit.
"""

PATIENCES = (1.0, 2.0, 4.0)
"""
How long nothing may arrive before a stall is called, in seconds.
"""

OVERRIDE_VELOCITIES = (0.05, 0.1, 0.2)
"""
How fast the override tilts the cup, in radians per second.
"""

OVERRIDE_WEIGHTS = (OVERRIDE_WEIGHT, FILL_ROW_WEIGHT)
"""
How much the override counts: below the fill row, or as much as it.
"""


def onset_tuning_rules() -> List[OnsetRule]:
    """
    :return: Every revision patience, and every override patience, velocity and weight.
    """
    revisions = [
        OnsetRule(response=OnsetResponse.REVISION, patience=patience)
        for patience in PATIENCES
    ]
    overrides = [
        OnsetRule(
            response=OnsetResponse.OVERRIDE,
            patience=patience,
            override_velocity=velocity,
            override_weight=weight,
        )
        for patience, velocity, weight in itertools.product(
            PATIENCES, OVERRIDE_VELOCITIES, OVERRIDE_WEIGHTS
        )
    ]
    return revisions + overrides


def _onset_trial(
    scene: Scene,
    baselines: Baselines,
    perturbation: Perturbation,
    sensor: SensorCondition,
    seed: int,
    rule: Optional[OnsetRule],
    start_margin: float,
) -> TransferTrial:
    """
    :return: One pour of the model with fill feedback, with or without the rule.
    """
    return TransferTrial(
        scene=scene,
        baseline=baselines.of(scene.source),
        perturbation=perturbation,
        loop_mode=LoopMode.FILL_FEEDBACK,
        seed=seed,
        sensor=sensor,
        onset_rule=rule,
        start_margin=start_margin,
    )


def onset_tuning_trials(baselines: Baselines) -> List[TransferTrial]:
    """
    :param baselines: Each cup's fitted parameters.
    :return: Every rule poured on every tuning seed, from the model's onset itself.
    """
    return [
        _onset_trial(
            ONSET_TUNING_SCENE,
            baselines,
            BASELINE,
            SensorCondition.IDEAL,
            seed,
            rule,
            start_margin=0.0,
        )
        for rule in onset_tuning_rules()
        for seed in ONSET_TUNING_SEEDS
    ]


class TunedOnsetRuleKey(StrEnum):
    """
    A key of the file of tuned rules.
    """

    OVERRIDE = "override"
    REVISION = "revision"
    WITH_START_MARGIN = "with_start_margin"


@dataclass(frozen=True)
class TunedOnsetRules:
    """
    The rule each response is evaluated with, and the one kept with the start margin.
    """

    override: OnsetRule
    """
    The override that misplaced fewest grains while tuning.
    """

    revision: OnsetRule
    """
    The revision that misplaced fewest grains while tuning.
    """

    with_start_margin: OnsetRule
    """
    Whichever of the two misplaced fewer, also run from the usual start margin.
    """

    def write(self, path: Path) -> None:
        """
        :param path: The file to write the rules to.
        """
        path.write_text(
            json.dumps(
                {
                    TunedOnsetRuleKey.OVERRIDE: self.override.to_json(),
                    TunedOnsetRuleKey.REVISION: self.revision.to_json(),
                    TunedOnsetRuleKey.WITH_START_MARGIN: self.with_start_margin.to_json(),
                },
                indent=2,
            )
        )

    @classmethod
    def read(cls, path: Path) -> TunedOnsetRules:
        """
        :param path: A file written by :meth:`write`.
        :return: The rules it holds.
        """
        data = json.loads(path.read_text())
        return cls(
            override=OnsetRule.from_json(data[TunedOnsetRuleKey.OVERRIDE]),
            revision=OnsetRule.from_json(data[TunedOnsetRuleKey.REVISION]),
            with_start_margin=OnsetRule.from_json(
                data[TunedOnsetRuleKey.WITH_START_MARGIN]
            ),
        )


@dataclass
class OnsetTuning:
    """
    The rule's tuning pours, and the rules that poured best.
    """

    records: List[TrialRecord]
    """
    Every tuning pour.
    """

    @cached_property
    def table(self) -> pandas.DataFrame:
        """
        :return: One row per rule: mean grains misplaced, runs within tolerance and
            pours, fewest misplaced first, most within tolerance among equals.
        """
        rows = []
        for record in self.records:
            outcome = record.description[TrialKey.OUTCOME]
            delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
            rows.append(
                {
                    ComparisonColumn.RULE: OnsetRule.from_json(
                        record.description[TrialKey.ONSET_RULE]
                    ),
                    ComparisonColumn.MISPLACED: misplaced_grains(delivered, spilled),
                    ComparisonColumn.SUCCEEDED: succeeded(delivered, spilled),
                    ComparisonColumn.SEED: record.description[TrialKey.SEED],
                }
            )
        return (
            pandas.DataFrame(rows)
            .groupby(ComparisonColumn.RULE, as_index=False, sort=False)
            .agg(
                **{
                    ComparisonColumn.MISPLACED: (ComparisonColumn.MISPLACED, "mean"),
                    ComparisonColumn.SUCCESS_RATE: (ComparisonColumn.SUCCEEDED, "sum"),
                    ComparisonColumn.RUNS: (ComparisonColumn.SEED, "count"),
                }
            )
            .sort_values(
                [ComparisonColumn.MISPLACED, ComparisonColumn.SUCCESS_RATE],
                ascending=[True, False],
                kind="stable",
            )
        )

    def best(self, response: OnsetResponse) -> OnsetRule:
        """
        :return: The rule of a response that misplaced fewest grains.
        """
        rules = self.table[ComparisonColumn.RULE]
        return next(rule for rule in rules if rule.response is response)

    def tuned(self) -> TunedOnsetRules:
        """
        :return: Each response's best rule, and the better of the two for the start
            margin.
        """
        return TunedOnsetRules(
            override=self.best(OnsetResponse.OVERRIDE),
            revision=self.best(OnsetResponse.REVISION),
            with_start_margin=self.table[ComparisonColumn.RULE].iloc[0],
        )


# %% evaluating it

ONSET_SENSORS = (SensorCondition.IDEAL, SensorCondition.SCALE)
"""
The sensors the rule is evaluated with at the fitted parameters.
"""

EARLY_ONSET_PERTURBATIONS = (
    Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.0),
    Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5),
)
"""
The models told the contents start pouring sooner than they do, under which the model
alone stalls.
"""

EARLY_ONSET_RECEIVER = ReceiverCup.WIDE
"""
The receiver the early-onset models pour into.
"""


@dataclass(frozen=True)
class OnsetCondition:
    """
    One scene, model and sensor the rule is evaluated under.
    """

    scene: Scene
    """
    Which cup pours into which receiver.
    """

    perturbation: Perturbation
    """
    How the model's parameters differ from the cup's fit.
    """

    sensor: SensorCondition
    """
    What reports the fill levels.
    """


def onset_conditions() -> List[OnsetCondition]:
    """
    :return: Both cups at their fit in both receivers with both sensors, and both cups
        told the contents start sooner, into the wide receiver with an ideal sensor.
    """
    at_fit = [
        OnsetCondition(
            scene=Scene(source=source, receiver=receiver),
            perturbation=BASELINE,
            sensor=sensor,
        )
        for source in SourceCup
        for receiver in ReceiverCup
        for sensor in ONSET_SENSORS
    ]
    early = [
        OnsetCondition(
            scene=Scene(source=source, receiver=EARLY_ONSET_RECEIVER),
            perturbation=perturbation,
            sensor=SensorCondition.IDEAL,
        )
        for source in SourceCup
        for perturbation in EARLY_ONSET_PERTURBATIONS
    ]
    return at_fit + early


def onset_evaluation_trials(
    seeds: List[int], baselines: Baselines, tuned: TunedOnsetRules
) -> List[TransferTrial]:
    """
    Under every condition: the override and the revision from the model's onset itself,
    the better rule from the usual start margin, and the model alone where no earlier
    study ran it on these seeds.

    :param seeds: The held-out microstates.
    :param baselines: Each cup's fitted parameters.
    :param tuned: The tuned rules.
    :return: The trials of the onset study.
    """
    arms = [
        (tuned.override, 0.0),
        (tuned.revision, 0.0),
        (tuned.with_start_margin, POUR_START_TILT_MARGIN),
    ]
    trials = []
    for condition in onset_conditions():
        model_alone = (
            [(None, POUR_START_TILT_MARGIN)]
            if condition.perturbation.parameter is not None
            else []
        )
        for rule, margin in arms + model_alone:
            for seed in seeds:
                trials.append(
                    _onset_trial(
                        condition.scene,
                        baselines,
                        condition.perturbation,
                        condition.sensor,
                        seed,
                        rule,
                        margin,
                    )
                )
    return trials


# %% reading the held-out pours


class OnsetArm(StrEnum):
    """
    One way the onset study starts the pour.
    """

    MODEL_ALONE = "model_alone"
    """
    The model with fill feedback, from the usual start margin, without the rule.
    """

    OVERRIDE = "override"
    """
    The tuned override, from the model's onset itself.
    """

    REVISION = "revision"
    """
    The tuned revision, from the model's onset itself.
    """

    START_MARGIN_AND_RULE = "start_margin_and_rule"
    """
    The better of the two, from the usual start margin.
    """

    BEST_PID = "best_pid"
    """
    The fairly tuned PID chosen for the sensor, which knows no onset at all.
    """

    @classmethod
    def of(cls, record: TrialRecord, tuned: TunedOnsetRules) -> Optional[OnsetArm]:
        """
        :param record: A pour of the model with fill feedback.
        :param tuned: The tuned rules.
        :return: The arm it ran, or ``None`` for a rule that was not tuned.
        """
        description = record.description
        rule_data = description.get(TrialKey.ONSET_RULE)
        margin = description.get(TrialKey.START_MARGIN, POUR_START_TILT_MARGIN)
        if rule_data is None:
            return cls.MODEL_ALONE if margin == POUR_START_TILT_MARGIN else None
        rule = OnsetRule.from_json(rule_data)
        if margin == POUR_START_TILT_MARGIN:
            return (
                cls.START_MARGIN_AND_RULE if rule == tuned.with_start_margin else None
            )
        if rule == tuned.override:
            return cls.OVERRIDE
        if rule == tuned.revision:
            return cls.REVISION
        return None


def final_repose_angle(record: TrialRecord) -> Optional[float]:
    """
    :param record: A pour of the model with the revision.
    :return: The repose angle its last revision left the model with, in radians, or
        ``None`` if it was never revised.
    """
    revisions = record.description.get(TrialKey.REVISIONS) or []
    if not revisions:
        return None
    return revisions[-1][RevisionKey.NEW_REPOSE_ANGLE]


# %% the evaluation


class OnsetColumn(StrEnum):
    """
    A column of the onset study's tables.
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
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    DURATION = "duration"
    FINAL_REPOSE_ANGLE = "final_repose_angle"
    STALLS = "stalls"


class SelectsPid(Protocol):
    """
    Decides which tuned PID meets the model under a sensor.
    """

    def for_sensor(
        self, source: SourceCup, sensor: SensorCondition
    ) -> Tuple[PidVariant, PidGains]:
        """
        :return: The variant and gains of the PID chosen for the sensor.
        """


def pid_records_of(
    condition: OnsetCondition, records: List[TrialRecord], selection: SelectsPid
) -> List[TrialRecord]:
    """
    :param condition: A scene, model and sensor.
    :param records: The fairly tuned PIDs' held-out pours.
    :param selection: Which PID meets the model under each sensor.
    :return: The pours of the PID chosen for the condition's sensor in its scene.
    """
    variant, gains = selection.for_sensor(condition.scene.source, condition.sensor)
    return [
        record
        for record in records
        if record.description[TrialKey.SOURCE] == condition.scene.source
        and record.description[TrialKey.RECEIVER] == condition.scene.receiver
        and record.description[TrialKey.SENSOR] == condition.sensor
        and record.description[TrialKey.PID_VARIANT] == variant
        and PidGains.from_json(record.description[TrialKey.GAINS]) == gains
    ]


def _condition_of(record: TrialRecord) -> Optional[OnsetCondition]:
    """
    :return: The evaluation condition a model pour ran under, or ``None`` if it is not
        one of them.
    """
    description = record.description
    condition = OnsetCondition(
        scene=Scene(
            source=SourceCup(description[TrialKey.SOURCE]),
            receiver=ReceiverCup(description[TrialKey.RECEIVER]),
        ),
        perturbation=Perturbation.from_json(description[TrialKey.PERTURBATION]),
        sensor=SensorCondition(description.get(TrialKey.SENSOR, SensorCondition.IDEAL)),
    )
    return condition if condition in onset_conditions() else None


def _row(
    record: TrialRecord, condition: OnsetCondition, arm: OnsetArm
) -> Dict[OnsetColumn, Any]:
    """
    :return: What the evaluation reads of one pour.
    """
    description = record.description
    outcome = description[TrialKey.OUTCOME]
    delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
    return {
        OnsetColumn.SOURCE: condition.scene.source,
        OnsetColumn.RECEIVER: condition.scene.receiver,
        OnsetColumn.PERTURBATION: condition.perturbation.label,
        OnsetColumn.SENSOR: condition.sensor,
        OnsetColumn.ARM: arm,
        OnsetColumn.SEED: description[TrialKey.SEED],
        OnsetColumn.DELIVERED: delivered,
        OnsetColumn.SPILLED: spilled,
        OnsetColumn.MISPLACED: misplaced_grains(delivered, spilled),
        OnsetColumn.SUCCEEDED: succeeded(delivered, spilled),
        OnsetColumn.TIMED_OUT: outcome[Outcome.ENDING] == MotionEnding.TIMED_OUT,
        OnsetColumn.DURATION: outcome[Outcome.TRANSFER_DURATION],
        OnsetColumn.FINAL_REPOSE_ANGLE: final_repose_angle(record),
        OnsetColumn.STALLS: len(description.get(TrialKey.STALLS) or []),
    }


@dataclass
class OnsetEvaluation:
    """
    Every arm of the onset study under every condition, on the held-out seeds.
    """

    records: List[TrialRecord]
    """
    Pours of the model, with and without the rule; those under no condition of the
    study, or on no arm of it, are left out.
    """

    pid_records: List[TrialRecord]
    """
    The fairly tuned PIDs' held-out pours.
    """

    tuned: TunedOnsetRules
    """
    The tuned rules.
    """

    selection: SelectsPid
    """
    Which PID meets the model under each sensor.
    """

    @cached_property
    def summary(self) -> pandas.DataFrame:
        """
        :return: One row per pour and condition. A PID knows no repose angle, so its
            pours of a scene and sensor stand for every model of that scene.
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
            condition, arm = _condition_of(record), OnsetArm.of(record, self.tuned)
            if condition is not None and arm is not None:
                rows.append(_row(record, condition, arm))
        for condition in onset_conditions():
            for record in pid_records_of(condition, self.pid_records, self.selection):
                rows.append(_row(record, condition, OnsetArm.BEST_PID))
        return pandas.DataFrame(rows)
