import pandas

from experiments.pouring_model_verification.fair_tuning import FAIR_TUNING_SEEDS
from experiments.pouring_model_verification.onset_rule import (
    OnsetResponse,
    OnsetRule,
    RevisionCause,
    RevisionKey,
)
from experiments.pouring_model_verification.onset_study import (
    ONSET_TUNING_SEEDS,
    OnsetCondition,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    DEMO_PARAMETERS,
    Baselines,
)
from experiments.pouring_model_verification.persistence_study import (
    SEQUENCE_LENGTH,
    Persistence,
    PersistenceColumn,
    PersistenceEvaluation,
    PourSequence,
    persistence_conditions,
    pour_sequences,
    ready_trials,
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
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    LoopMode,
    MotionEnding,
)

BASELINES = Baselines(parameters={source: DEMO_PARAMETERS for source in SourceCup})
REVISION = OnsetRule(response=OnsetResponse.REVISION, patience=2.0)
CONDITION = OnsetCondition(
    scene=Scene(source=SourceCup.CUP, receiver=ReceiverCup.WIDE),
    perturbation=BASELINE,
    sensor=SensorCondition.IDEAL,
)
HELD_OUT_SEEDS = (4, 5, 6, 7)


def _sequence(persistence: Persistence, seed: int = 4) -> PourSequence:
    return PourSequence(
        condition=CONDITION,
        persistence=persistence,
        sequence_seed=seed,
        revision=REVISION,
    )


def _finished(trial, end_angle: float) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.NAME: trial.name,
            TrialKey.REPOSE_ANGLE_AT_END: end_angle,
        },
        samples=pandas.DataFrame({Sample.PHASE: [Phase.TRANSFER]}),
    )


# %% which pours are run


def test_the_plain_cup_runs_in_both_receivers_and_the_spouted_cup_in_the_wide_one():
    scenes = {condition.scene for condition in persistence_conditions()}
    assert scenes == {
        Scene(source=SourceCup.CUP, receiver=ReceiverCup.WIDE),
        Scene(source=SourceCup.CUP, receiver=ReceiverCup.NARROW),
        Scene(source=SourceCup.SPOUTED_CUP, receiver=ReceiverCup.WIDE),
    }


def test_no_pour_of_a_sequence_reuses_a_tuning_or_another_held_out_microstate():
    seeds = [
        sequence.pour_seed(pour)
        for sequence in pour_sequences(list(HELD_OUT_SEEDS), REVISION)
        for pour in range(2, SEQUENCE_LENGTH + 1)
    ]
    assert not set(seeds) & (
        set(HELD_OUT_SEEDS) | set(FAIR_TUNING_SEEDS) | set(ONSET_TUNING_SEEDS)
    )


def test_keeping_the_last_value_and_starting_afresh_share_their_first_pour():
    fresh = _sequence(Persistence.NONE).trial(1, BASELINES, kept=None)
    kept = _sequence(Persistence.LAST_VALUE).trial(1, BASELINES, kept=None)
    assert fresh.name == kept.name


def test_the_two_sided_sequence_pours_with_a_rule_that_also_lowers():
    trial = _sequence(Persistence.TWO_SIDED).trial(1, BASELINES, kept=None)
    assert trial.onset_rule.lowers


def test_every_pour_of_a_fresh_sequence_is_ready_at_once():
    trials = ready_trials([_sequence(Persistence.NONE)], BASELINES, records={})
    assert [trial.sequence.pour for trial in trials] == list(
        range(1, SEQUENCE_LENGTH + 1)
    )


def test_a_kept_sequence_waits_for_its_previous_pour():
    trials = ready_trials([_sequence(Persistence.LAST_VALUE)], BASELINES, records={})
    assert [trial.sequence.pour for trial in trials] == [1]


def test_a_kept_pour_starts_from_the_repose_angle_the_previous_pour_ended_with():
    sequence = _sequence(Persistence.LAST_VALUE)
    first = sequence.trial(1, BASELINES, kept=None)
    ended_with = 0.95
    (second,) = ready_trials(
        [sequence], BASELINES, records={first.name: _finished(first, ended_with)}
    )
    assert (second.sequence.pour, second.sequence.kept_repose_angle) == (
        2,
        ended_with,
    )


def test_a_pour_shared_by_two_sequences_is_run_once():
    trials = ready_trials(
        [_sequence(Persistence.NONE), _sequence(Persistence.LAST_VALUE)],
        BASELINES,
        records={},
    )
    assert len({trial.name for trial in trials}) == len(trials)


# %% reading the sequences


def _pour(sequence: PourSequence, pour: int, misplaced: float, kept=None, revisions=()):
    trial = sequence.trial(pour, BASELINES, kept=kept)
    return TrialRecord(
        description={
            TrialKey.NAME: trial.name,
            TrialKey.SOURCE: SourceCup.CUP,
            TrialKey.RECEIVER: ReceiverCup.WIDE,
            TrialKey.PERTURBATION: BASELINE.to_json(),
            TrialKey.SENSOR: SensorCondition.IDEAL,
            TrialKey.LOOP_MODE: LoopMode.FILL_FEEDBACK,
            TrialKey.SEED: trial.seed,
            TrialKey.ONSET_RULE: trial.onset_rule.to_json(),
            TrialKey.SEQUENCE: trial.sequence.to_json(),
            TrialKey.REVISIONS: list(revisions),
            TrialKey.REPOSE_ANGLE_AT_START: 0.7,
            TrialKey.REPOSE_ANGLE_AT_END: 0.8,
            TrialKey.OUTCOME: {
                Outcome.GOAL_GRAINS: GOAL_GRAINS,
                Outcome.DELIVERED: GOAL_GRAINS + misplaced,
                Outcome.SPILLED: 0,
                Outcome.ENDING: MotionEnding.REACHED_GOAL,
                Outcome.TRANSFER_DURATION: 5.0,
            },
        },
        samples=pandas.DataFrame({Sample.PHASE: [Phase.TRANSFER]}),
    )


def test_a_shared_first_pour_counts_for_both_sequences_that_share_it():
    record = _pour(_sequence(Persistence.NONE), 1, misplaced=2.0)
    rows = PersistenceEvaluation(records=[record]).rows
    assert set(rows[PersistenceColumn.PERSISTENCE]) == {
        Persistence.NONE,
        Persistence.LAST_VALUE,
    }


def test_a_kept_pour_counts_only_for_keeping_the_last_value():
    record = _pour(_sequence(Persistence.LAST_VALUE), 2, misplaced=2.0, kept=0.8)
    rows = PersistenceEvaluation(records=[record]).rows
    assert list(rows[PersistenceColumn.PERSISTENCE]) == [Persistence.LAST_VALUE]


def test_revisions_are_counted_by_cause_and_direction():
    raised = {RevisionKey.OLD_REPOSE_ANGLE: 0.7, RevisionKey.NEW_REPOSE_ANGLE: 0.8}
    stall_lowered = {
        RevisionKey.OLD_REPOSE_ANGLE: 0.8,
        RevisionKey.NEW_REPOSE_ANGLE: 0.75,
        RevisionKey.CAUSE: RevisionCause.STALL,
    }
    arrival_lowered = {
        RevisionKey.OLD_REPOSE_ANGLE: 0.75,
        RevisionKey.NEW_REPOSE_ANGLE: 0.7,
        RevisionKey.CAUSE: RevisionCause.ARRIVAL,
    }
    record = _pour(
        _sequence(Persistence.TWO_SIDED),
        1,
        misplaced=2.0,
        revisions=[raised, raised, stall_lowered, arrival_lowered],
    )
    row = PersistenceEvaluation(records=[record]).rows.iloc[0]
    assert (
        row[PersistenceColumn.RAISED],
        row[PersistenceColumn.LOWERED_AT_STALL],
        row[PersistenceColumn.LOWERED_ON_ARRIVAL],
    ) == (2, 1, 1)


def test_the_summary_has_one_row_per_pour_of_each_sequence_kind():
    sequence = _sequence(Persistence.NONE)
    records = [
        _pour(sequence, pour, misplaced=float(pour))
        for pour in range(1, SEQUENCE_LENGTH + 1)
    ]
    summary = PersistenceEvaluation(records=records).summary
    fresh = summary[summary[PersistenceColumn.PERSISTENCE] == Persistence.NONE]
    assert list(fresh[PersistenceColumn.POUR]) == list(range(1, SEQUENCE_LENGTH + 1))
