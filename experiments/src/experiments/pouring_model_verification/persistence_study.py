"""
Keeping what was learned: sequences of pours of one cup, each refilled and settled anew,
in which a pour starts either from the model's own repose angle or from the one the
previous pour's revisions left it with, and the revision either only raises it or also
lowers it.
"""

# %% imports

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property

import pandas
from typing_extensions import Any, Dict, List, Optional

from experiments.pouring_model_verification.controller_comparison import (
    misplaced_grains,
    succeeded,
)
from experiments.pouring_model_verification.onset_rule import (
    OnsetRule,
    RevisionCause,
    RevisionKey,
)
from experiments.pouring_model_verification.onset_study import (
    EARLY_ONSET_PERTURBATIONS,
    OnsetCondition,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    Baselines,
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
    LoopMode,
    MotionEnding,
    SequencePosition,
    TransferTrial,
)

# %% the sequences

SEQUENCE_LENGTH = 5
"""
How many pours a sequence has.
"""

POUR_SEED_STRIDE = 50
"""
How far apart the microstates of successive pours of a sequence lie, so no pour after
the first reuses a microstate any tuning or held-out pour starts from.
"""

PERSISTENCE_SENSORS = (SensorCondition.IDEAL, SensorCondition.SCALE)
"""
The sensors the sequences are poured with.
"""


class Persistence(StrEnum):
    """
    What a pour of a sequence keeps of the pours before it.
    """

    NONE = "no_persistence"
    """
    Every pour starts from the model's own repose angle.
    """

    LAST_VALUE = "last_value"
    """
    Every pour after the first starts from the repose angle the previous one ended with.
    """

    TWO_SIDED = "two_sided"
    """
    As :attr:`LAST_VALUE`, with a revision that also lowers the repose angle when grains
    arrive while the model predicts none.
    """

    @property
    def keeps(self) -> bool:
        """
        :return: Whether a pour starts from the repose angle the previous one ended
            with.
        """
        return self is not Persistence.NONE

    @property
    def lowers(self) -> bool:
        """
        :return: Whether the revision also lowers the repose angle.
        """
        return self is Persistence.TWO_SIDED


def persistence_conditions() -> List[OnsetCondition]:
    """
    :return: The plain cup at its fit and told the contents start sooner, into either
        receiver, and the spouted cup at its fit into the wide one, each with an ideal
        sensor and with a scale.
    """
    plain = [
        OnsetCondition(
            scene=Scene(source=SourceCup.CUP, receiver=receiver),
            perturbation=perturbation,
            sensor=sensor,
        )
        for receiver in ReceiverCup
        for perturbation in (BASELINE,) + EARLY_ONSET_PERTURBATIONS
        for sensor in PERSISTENCE_SENSORS
    ]
    spouted = [
        OnsetCondition(
            scene=Scene(source=SourceCup.SPOUTED_CUP, receiver=ReceiverCup.WIDE),
            perturbation=BASELINE,
            sensor=sensor,
        )
        for sensor in PERSISTENCE_SENSORS
    ]
    return plain + spouted


@dataclass(frozen=True)
class PourSequence:
    """
    One cup poured :data:`SEQUENCE_LENGTH` times under one condition, refilled before
    every pour.
    """

    condition: OnsetCondition
    """
    The scene, model and sensor every pour runs under.
    """

    persistence: Persistence
    """
    What each pour keeps of the ones before it.
    """

    sequence_seed: int
    """
    The microstate the first pour starts from.
    """

    revision: OnsetRule
    """
    The tuned revision every pour runs with.
    """

    def pour_seed(self, pour: int) -> int:
        """
        :param pour: Which pour, counted from one.
        :return: The microstate that pour starts from.
        """
        return self.sequence_seed + POUR_SEED_STRIDE * (pour - 1)

    def trial(
        self, pour: int, baselines: Baselines, kept: Optional[float]
    ) -> TransferTrial:
        """
        :param pour: Which pour, counted from one.
        :param baselines: Each cup's fitted parameters.
        :param kept: The repose angle the previous pour ended with, or ``None`` to
            start from the model's own.
        :return: The pour, from the model's onset itself.
        """
        condition = self.condition
        return TransferTrial(
            scene=condition.scene,
            baseline=baselines.of(condition.scene.source),
            perturbation=condition.perturbation,
            loop_mode=LoopMode.FILL_FEEDBACK,
            seed=self.pour_seed(pour),
            sensor=condition.sensor,
            onset_rule=replace(self.revision, lowers=self.persistence.lowers),
            start_margin=0.0,
            sequence=SequencePosition(
                pour=pour, sequence_seed=self.sequence_seed, kept_repose_angle=kept
            ),
        )

    def ready_trials(
        self, baselines: Baselines, records: Dict[str, TrialRecord]
    ) -> List[TransferTrial]:
        """
        :param baselines: Each cup's fitted parameters.
        :param records: The pours recorded so far, by name.
        :return: The pours not recorded yet whose start is known: every one of a
            sequence that keeps nothing, else the first one not recorded.
        """
        ready, kept = [], None
        for pour in range(1, SEQUENCE_LENGTH + 1):
            trial = self.trial(pour, baselines, kept)
            record = records.get(trial.name)
            if record is None:
                ready.append(trial)
                if self.persistence.keeps:
                    break
                continue
            if self.persistence.keeps:
                kept = record.description[TrialKey.REPOSE_ANGLE_AT_END]
        return ready


def pour_sequences(seeds: List[int], revision: OnsetRule) -> List[PourSequence]:
    """
    :param seeds: The microstates the sequences' first pours start from.
    :param revision: The tuned revision.
    :return: Every kind of sequence under every condition from every seed.
    """
    return [
        PourSequence(
            condition=condition,
            persistence=persistence,
            sequence_seed=seed,
            revision=revision,
        )
        for condition in persistence_conditions()
        for persistence in Persistence
        for seed in seeds
    ]


def ready_trials(
    sequences: List[PourSequence],
    baselines: Baselines,
    records: Dict[str, TrialRecord],
) -> List[TransferTrial]:
    """
    :param sequences: The sequences to pour.
    :param baselines: Each cup's fitted parameters.
    :param records: The pours recorded so far, by name.
    :return: Every pour that can run now, once, however many sequences share it.
    """
    trials = {
        trial.name: trial
        for sequence in sequences
        for trial in sequence.ready_trials(baselines, records)
    }
    return list(trials.values())


# %% reading them


class PersistenceColumn(StrEnum):
    """
    A column of the persistence study's tables.
    """

    SOURCE = "source"
    RECEIVER = "receiver"
    PERTURBATION = "perturbation"
    SENSOR = "sensor"
    PERSISTENCE = "persistence"
    SEQUENCE_SEED = "sequence_seed"
    POUR = "pour"
    DELIVERED = "delivered"
    SPILLED = "spilled"
    MISPLACED = "misplaced"
    MISPLACED_LOW = "misplaced_low"
    MISPLACED_HIGH = "misplaced_high"
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed_out"
    DURATION = "duration"
    RAISED = "raised"
    LOWERED_AT_STALL = "lowered_at_stall"
    LOWERED_ON_ARRIVAL = "lowered_on_arrival"
    REPOSE_ANGLE_AT_START = "repose_angle_at_start"
    REPOSE_ANGLE_AT_END = "repose_angle_at_end"
    RUNS = "runs"


def _persistences_of(rule: OnsetRule, position: SequencePosition) -> List[Persistence]:
    """
    :return: The kinds of sequence a pour belongs to: a first pour that does not lower
        is the first pour of both sequences that only raise.
    """
    if rule.lowers:
        return [Persistence.TWO_SIDED]
    if position.kept_repose_angle is not None:
        return [Persistence.LAST_VALUE]
    if position.pour == 1:
        return [Persistence.NONE, Persistence.LAST_VALUE]
    return [Persistence.NONE]


CONDITION_COLUMNS = [
    PersistenceColumn.SOURCE,
    PersistenceColumn.RECEIVER,
    PersistenceColumn.PERTURBATION,
    PersistenceColumn.SENSOR,
    PersistenceColumn.PERSISTENCE,
]
"""
The columns that name one kind of sequence under one condition.
"""


@dataclass
class PersistenceEvaluation:
    """
    Every pour of every sequence, and how each pour index fared.
    """

    records: List[TrialRecord]
    """
    The pours of the sequences.
    """

    @cached_property
    def rows(self) -> pandas.DataFrame:
        """
        :return: One row per pour and kind of sequence it belongs to, with its repose
            angles in degrees.
        """
        rows = []
        for record in self.records:
            description = record.description
            position = SequencePosition.from_json(description[TrialKey.SEQUENCE])
            rule = OnsetRule.from_json(description[TrialKey.ONSET_RULE])
            for persistence in _persistences_of(rule, position):
                rows.append(self._row(description, position, persistence))
        return pandas.DataFrame(rows)

    @staticmethod
    def _row(
        description: Dict[str, Any],
        position: SequencePosition,
        persistence: Persistence,
    ) -> Dict[PersistenceColumn, Any]:
        """
        :return: What the evaluation reads of one pour.
        """
        outcome = description[TrialKey.OUTCOME]
        delivered, spilled = outcome[Outcome.DELIVERED], outcome[Outcome.SPILLED]
        revisions = description[TrialKey.REVISIONS]
        on_arrival = [
            revision
            for revision in revisions
            if revision.get(RevisionKey.CAUSE, RevisionCause.STALL)
            == RevisionCause.ARRIVAL
        ]
        raised = sum(
            revision[RevisionKey.NEW_REPOSE_ANGLE]
            > revision[RevisionKey.OLD_REPOSE_ANGLE]
            for revision in revisions
        )
        return {
            PersistenceColumn.SOURCE: description[TrialKey.SOURCE],
            PersistenceColumn.RECEIVER: description[TrialKey.RECEIVER],
            PersistenceColumn.PERTURBATION: Perturbation.from_json(
                description[TrialKey.PERTURBATION]
            ).label,
            PersistenceColumn.SENSOR: description[TrialKey.SENSOR],
            PersistenceColumn.PERSISTENCE: persistence,
            PersistenceColumn.SEQUENCE_SEED: position.sequence_seed,
            PersistenceColumn.POUR: position.pour,
            PersistenceColumn.DELIVERED: delivered,
            PersistenceColumn.SPILLED: spilled,
            PersistenceColumn.MISPLACED: misplaced_grains(delivered, spilled),
            PersistenceColumn.SUCCEEDED: succeeded(delivered, spilled),
            PersistenceColumn.TIMED_OUT: outcome[Outcome.ENDING]
            == MotionEnding.TIMED_OUT,
            PersistenceColumn.DURATION: outcome[Outcome.TRANSFER_DURATION],
            PersistenceColumn.RAISED: raised,
            PersistenceColumn.LOWERED_AT_STALL: len(revisions)
            - raised
            - len(on_arrival),
            PersistenceColumn.LOWERED_ON_ARRIVAL: len(on_arrival),
            PersistenceColumn.REPOSE_ANGLE_AT_START: math.degrees(
                description[TrialKey.REPOSE_ANGLE_AT_START]
            ),
            PersistenceColumn.REPOSE_ANGLE_AT_END: math.degrees(
                description[TrialKey.REPOSE_ANGLE_AT_END]
            ),
        }

    @cached_property
    def summary(self) -> pandas.DataFrame:
        """
        :return: One row per kind of sequence, condition and pour index: the mean and
            range of the grains misplaced, the runs within tolerance, the revisions in
            each direction, the duration and the repose angles.
        """
        column = PersistenceColumn
        return (
            self.rows.groupby(CONDITION_COLUMNS + [column.POUR], as_index=False)
            .agg(
                **{
                    column.MISPLACED: (column.MISPLACED, "mean"),
                    column.MISPLACED_LOW: (column.MISPLACED, "min"),
                    column.MISPLACED_HIGH: (column.MISPLACED, "max"),
                    column.SPILLED: (column.SPILLED, "mean"),
                    column.SUCCEEDED: (column.SUCCEEDED, "sum"),
                    column.TIMED_OUT: (column.TIMED_OUT, "sum"),
                    column.RUNS: (column.MISPLACED, "count"),
                    column.RAISED: (column.RAISED, "mean"),
                    column.LOWERED_AT_STALL: (column.LOWERED_AT_STALL, "mean"),
                    column.LOWERED_ON_ARRIVAL: (column.LOWERED_ON_ARRIVAL, "mean"),
                    column.DURATION: (column.DURATION, "mean"),
                    column.REPOSE_ANGLE_AT_START: (
                        column.REPOSE_ANGLE_AT_START,
                        "mean",
                    ),
                    column.REPOSE_ANGLE_AT_END: (column.REPOSE_ANGLE_AT_END, "mean"),
                }
            )
            .sort_values(CONDITION_COLUMNS + [column.POUR], kind="stable")
        )
