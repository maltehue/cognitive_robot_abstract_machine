import pandas
import pytest

from experiments.pouring_model_verification.concept_revision import (
    Diagnosis,
    RevisedConcept,
    StallEvidence,
)
from experiments.pouring_model_verification.diagnosis_study import (
    DIAGNOSIS_SENSORS,
    DiagnosisArm,
    DiagnosisColumn,
    DiagnosisEvaluation,
    diagnosis_conditions,
    diagnosis_trials,
    right_concept,
)
from experiments.pouring_model_verification.onset_rule import (
    OnsetResponse,
    OnsetRule,
)
from experiments.pouring_model_verification.onset_study import OnsetCondition
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    DEMO_PARAMETERS,
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
    GOAL_GRAINS,
    LoopMode,
    MotionEnding,
    POUR_START_TILT_MARGIN,
)

BASELINES = Baselines(parameters={source: DEMO_PARAMETERS for source in SourceCup})
REVISION = OnsetRule(response=OnsetResponse.REVISION, patience=1.0)
SPOUT_NARROW = Scene(source=SourceCup.SPOUTED_CUP, receiver=ReceiverCup.NARROW)


# %% which pours are run


def test_the_discharge_coefficient_is_set_wrong_in_every_scene():
    scenes = {
        condition.scene
        for condition in diagnosis_conditions()
        if condition.perturbation.parameter is ModelParameter.DISCHARGE_COEFFICIENT
    }
    assert scenes == {
        Scene(source=source, receiver=receiver)
        for source in SourceCup
        for receiver in ReceiverCup
    }


def test_the_spouted_cup_at_its_fit_pours_only_into_the_narrow_receiver():
    at_fit = {
        condition.scene
        for condition in diagnosis_conditions()
        if condition.perturbation == BASELINE
    }
    assert at_fit == {SPOUT_NARROW}


def test_every_condition_runs_with_both_sensor_pairs():
    sensors = {condition.sensor for condition in diagnosis_conditions()}
    assert sensors == set(DIAGNOSIS_SENSORS)


def _trials(recorded=frozenset()):
    return diagnosis_trials(
        seeds=[4],
        baselines=BASELINES,
        revision=REVISION,
        lowers=False,
        recorded=recorded,
    )


def test_the_concept_aware_pour_weighs_the_source_with_the_matching_wrist():
    wrists = {
        (trial.sensor, trial.wrist_sensor)
        for trial in _trials()
        if trial.onset_rule is not None
        and trial.onset_rule.response is OnsetResponse.CONCEPT_AWARE_REVISION
    }
    assert wrists == {
        (sensor, DIAGNOSIS_SENSORS[sensor]) for sensor in DIAGNOSIS_SENSORS
    }


def test_a_pour_recorded_by_an_earlier_study_is_not_run_again():
    first = _trials()[0]
    assert first.name not in {trial.name for trial in _trials({first.name})}


def test_each_arm_runs_once_per_condition_and_seed():
    assert len(_trials()) == 3 * len(diagnosis_conditions())


# %% which concept was the right one


def _samples(source_grains, receiver_grains=None):
    receiver_grains = receiver_grains or [0] * len(source_grains)
    return pandas.DataFrame(
        {
            Sample.TIME: [0.1 * step for step in range(len(source_grains))],
            Sample.PHASE: Phase.TRANSFER,
            Sample.GRAINS_IN_SOURCE: source_grains,
            Sample.GRAINS_IN_RECEIVER: receiver_grains,
        }
    )


def test_a_stall_while_the_source_lost_grains_the_receiver_did_not_get_was_a_miss():
    samples = _samples([100 - 2 * step for step in range(30)])
    assert right_concept(samples, at=2.5, patience=1.0) is (
        RevisedConcept.DISCHARGE_COEFFICIENT
    )


def test_a_stall_while_the_source_truly_held_its_grains_was_the_onset():
    samples = _samples([100] * 30)
    assert right_concept(samples, at=2.5, patience=1.0) is RevisedConcept.REPOSE_ANGLE


def test_a_stall_while_the_grains_truly_arrived_was_no_stall_at_all():
    samples = _samples(
        [100 - 2 * step for step in range(30)], [2 * step for step in range(30)]
    )
    assert right_concept(samples, at=2.5, patience=1.0) is None


# %% the table


def _record(arm_rule, wrist, diagnoses, perturbation=BASELINE, misplaced=2.0):
    return TrialRecord(
        description={
            TrialKey.NAME: "pour",
            TrialKey.SOURCE: SPOUT_NARROW.source,
            TrialKey.RECEIVER: SPOUT_NARROW.receiver,
            TrialKey.PERTURBATION: perturbation.to_json(),
            TrialKey.SENSOR: SensorCondition.IDEAL,
            TrialKey.LOOP_MODE: LoopMode.FILL_FEEDBACK,
            TrialKey.SEED: 4,
            TrialKey.ONSET_RULE: None if arm_rule is None else arm_rule.to_json(),
            TrialKey.START_MARGIN: POUR_START_TILT_MARGIN if arm_rule is None else 0.0,
            TrialKey.WRIST_SENSOR: wrist,
            TrialKey.DIAGNOSES: [diagnosis.to_json() for diagnosis in diagnoses],
            TrialKey.OUTCOME: {
                Outcome.GOAL_GRAINS: GOAL_GRAINS,
                Outcome.DELIVERED: GOAL_GRAINS + misplaced,
                Outcome.SPILLED: 0,
                Outcome.ENDING: MotionEnding.REACHED_GOAL,
                Outcome.TRANSFER_DURATION: 5.0,
            },
        },
        samples=_samples([100 - 2 * step for step in range(60)]),
    )


def _diagnosis(concept, at=2.5):
    return Diagnosis(
        time=at,
        concept=concept,
        evidence=StallEvidence.SOURCE_LOSING,
        source_loss=10.0,
        required_loss=1.0,
        old_value=0.3,
        new_value=0.4,
    )


CONCEPT_AWARE = OnsetRule(response=OnsetResponse.CONCEPT_AWARE_REVISION, patience=1.0)


def test_each_pour_is_counted_for_the_arm_it_ran():
    records = [
        _record(None, None, []),
        _record(REVISION, None, []),
        _record(CONCEPT_AWARE, WristSensor.IDEAL, []),
    ]
    evaluation = DiagnosisEvaluation(
        records=records, pid_records=[], revision=REVISION, selection=None
    )
    assert list(evaluation.rows[DiagnosisColumn.ARM]) == [
        DiagnosisArm.MODEL_ALONE,
        DiagnosisArm.ONSET_REVISION,
        DiagnosisArm.CONCEPT_AWARE_REVISION,
    ]


def test_diagnoses_are_counted_by_concept_and_by_whether_they_were_right():
    record = _record(
        CONCEPT_AWARE,
        WristSensor.IDEAL,
        [
            _diagnosis(RevisedConcept.DISCHARGE_COEFFICIENT),
            _diagnosis(RevisedConcept.REPOSE_ANGLE),
        ],
    )
    evaluation = DiagnosisEvaluation(
        records=[record], pid_records=[], revision=REVISION, selection=None
    )
    row = evaluation.rows.iloc[0]
    assert (
        row[DiagnosisColumn.DISCHARGE_REVISIONS],
        row[DiagnosisColumn.REPOSE_REVISIONS],
        row[DiagnosisColumn.RIGHT_CONCEPT],
        row[DiagnosisColumn.FALSE_STALLS],
    ) == (1, 1, 1, 0)
