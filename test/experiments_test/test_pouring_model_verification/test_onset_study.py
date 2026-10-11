import pandas

from experiments.pouring_model_verification.onset_rule import (
    FILL_ROW_WEIGHT,
    OnsetResponse,
    OnsetRule,
)
from experiments.pouring_model_verification.onset_study import (
    ONSET_TUNING_SEEDS,
    EARLY_ONSET_PERTURBATIONS,
    OnsetArm,
    OnsetColumn,
    OnsetEvaluation,
    OnsetTuning,
    final_repose_angle,
    TunedOnsetRules,
    onset_evaluation_trials,
    onset_tuning_rules,
    onset_tuning_trials,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE as BASELINE_PERTURBATION,
    DEMO_PARAMETERS,
    Baselines,
    ModelParameter,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import ReceiverCup, SourceCup
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    FillController,
    LoopMode,
    POUR_START_TILT_MARGIN,
    MotionEnding,
)

BASELINES = Baselines(parameters={source: DEMO_PARAMETERS for source in SourceCup})
REVISION = OnsetRule(response=OnsetResponse.REVISION, patience=2.0)
OVERRIDE = OnsetRule(
    response=OnsetResponse.OVERRIDE,
    patience=1.0,
    override_velocity=0.1,
    override_weight=FILL_ROW_WEIGHT,
)
TUNED = TunedOnsetRules(
    override=OVERRIDE, revision=REVISION, with_start_margin=REVISION
)


def _record(rule: OnsetRule, seed: int, delivered: float) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.SOURCE: SourceCup.CUP,
            TrialKey.SEED: seed,
            TrialKey.ONSET_RULE: rule.to_json(),
            TrialKey.OUTCOME: {
                Outcome.GOAL_GRAINS: GOAL_GRAINS,
                Outcome.DELIVERED: delivered,
                Outcome.SPILLED: 0,
                Outcome.ENDING: MotionEnding.REACHED_GOAL,
            },
        },
        samples=pandas.DataFrame({Sample.PHASE: [Phase.TRANSFER]}),
    )


# %% tuning


def test_tuning_pours_every_rule_on_every_tuning_seed_without_a_start_margin():
    trials = onset_tuning_trials(BASELINES)
    assert len(trials) == len(onset_tuning_rules()) * len(ONSET_TUNING_SEEDS)
    assert {
        (trial.start_margin, trial.scene.source, trial.sensor) for trial in trials
    } == {(0.0, SourceCup.CUP, SensorCondition.IDEAL)}


def test_each_response_keeps_its_rule_that_misplaced_fewest_grains():
    slower = OnsetRule(response=OnsetResponse.REVISION, patience=4.0)
    records = [
        _record(rule, seed, GOAL_GRAINS + misplaced)
        for rule, misplaced in ((REVISION, 2), (slower, 9), (OVERRIDE, 5))
        for seed in ONSET_TUNING_SEEDS
    ]
    tuned = OnsetTuning(records=records).tuned()
    assert (tuned.revision, tuned.override, tuned.with_start_margin) == (
        REVISION,
        OVERRIDE,
        REVISION,
    )


def test_tuned_rules_survive_a_json_round_trip(tmp_path):
    TUNED.write(tmp_path / "rules.json")
    assert TunedOnsetRules.read(tmp_path / "rules.json") == TUNED


# %% evaluation


def _evaluation():
    return onset_evaluation_trials(seeds=[4], baselines=BASELINES, tuned=TUNED)


def test_every_evaluation_trial_has_a_record_of_its_own():
    trials = _evaluation()
    assert len({trial.name for trial in trials}) == len(trials)


def test_both_responses_run_without_the_start_margin_and_one_with_it():
    arms = {(trial.onset_rule, trial.start_margin) for trial in _evaluation()}
    assert arms == {
        (OVERRIDE, 0.0),
        (REVISION, 0.0),
        (REVISION, POUR_START_TILT_MARGIN),
        (None, POUR_START_TILT_MARGIN),
    }


def test_the_model_alone_runs_only_where_no_earlier_study_ran_it_on_these_seeds():
    alone = [trial for trial in _evaluation() if trial.onset_rule is None]
    assert {
        (trial.perturbation.parameter, trial.perturbation.factor) for trial in alone
    } == {(ModelParameter.REPOSE_ANGLE, 0.0), (ModelParameter.REPOSE_ANGLE, 0.5)}


def test_the_spouted_cup_at_its_fit_is_checked_for_harm_in_both_receivers():
    spouted = {
        (trial.scene.receiver, trial.sensor)
        for trial in _evaluation()
        if trial.scene.source is SourceCup.SPOUTED_CUP
        and trial.perturbation.parameter is None
    }
    assert spouted == {
        (receiver, sensor)
        for receiver in ReceiverCup
        for sensor in (SensorCondition.IDEAL, SensorCondition.SCALE)
    }


# %% which arm a held-out pour belongs to


def _held_out(rule, margin, perturbation=None) -> TrialRecord:
    record = (
        _record(rule, 4, GOAL_GRAINS)
        if rule is not None
        else _record(REVISION, 4, GOAL_GRAINS)
    )
    description = {
        **record.description,
        TrialKey.ONSET_RULE: None if rule is None else rule.to_json(),
        TrialKey.START_MARGIN: margin,
    }
    if perturbation is not None:
        description[TrialKey.PERTURBATION] = perturbation.to_json()
    return TrialRecord(description=description, samples=record.samples)


def test_each_held_out_pour_is_counted_for_the_arm_it_ran():
    assert [
        OnsetArm.of(_held_out(rule, margin), TUNED)
        for rule, margin in (
            (None, POUR_START_TILT_MARGIN),
            (OVERRIDE, 0.0),
            (REVISION, 0.0),
            (REVISION, POUR_START_TILT_MARGIN),
        )
    ] == [
        OnsetArm.MODEL_ALONE,
        OnsetArm.OVERRIDE,
        OnsetArm.REVISION,
        OnsetArm.START_MARGIN_AND_RULE,
    ]


def test_a_pour_with_a_rule_that_was_not_tuned_is_no_arm():
    untuned = OnsetRule(response=OnsetResponse.REVISION, patience=4.0)
    assert OnsetArm.of(_held_out(untuned, 0.0), TUNED) is None


def test_the_last_revision_is_the_repose_angle_the_pour_ended_with():
    first, last = 0.8, 0.9
    record = _held_out(REVISION, 0.0)
    record.description[TrialKey.REVISIONS] = [
        {"time": 1.0, "tilt": 1.2, "old_repose_angle": 0.7, "new_repose_angle": first},
        {"time": 3.0, "tilt": 1.3, "old_repose_angle": first, "new_repose_angle": last},
    ]
    assert final_repose_angle(record) == last


# %% the evaluation's rows


def _pour(
    arm_rule,
    margin,
    sensor,
    seed,
    delivered,
    perturbation=BASELINE_PERTURBATION,
    controller=None,
    gains=None,
):
    record = _held_out(arm_rule, margin, perturbation)
    description = {
        **record.description,
        TrialKey.RECEIVER: ReceiverCup.WIDE,
        TrialKey.SENSOR: sensor,
        TrialKey.SEED: seed,
        TrialKey.LOOP_MODE: LoopMode.FILL_FEEDBACK,
        TrialKey.OUTCOME: {
            Outcome.GOAL_GRAINS: GOAL_GRAINS,
            Outcome.DELIVERED: delivered,
            Outcome.SPILLED: 0,
            Outcome.ENDING: MotionEnding.REACHED_GOAL,
            Outcome.TRANSFER_DURATION: 5.0,
        },
    }
    if controller is not None:
        description[TrialKey.CONTROLLER] = controller
        description[TrialKey.GAINS] = gains.to_json()
        description[TrialKey.PID_VARIANT] = PidVariant.PLAIN
    return TrialRecord(description=description, samples=record.samples)


class SelectionOfOnePid:
    def __init__(self, gains):
        self.gains = gains

    def for_sensor(self, source, sensor):
        return PidVariant.PLAIN, self.gains


def test_the_pid_poured_at_a_scene_stands_for_every_model_of_that_scene():
    gains = PidGains(proportional=0.01, integral=0.005, derivative=0.0, start_tilt=0.3)
    early = EARLY_ONSET_PERTURBATIONS[0]
    evaluation = OnsetEvaluation(
        records=[
            _pour(
                REVISION, 0.0, SensorCondition.IDEAL, 4, GOAL_GRAINS, perturbation=early
            )
        ],
        pid_records=[
            _pour(
                None,
                POUR_START_TILT_MARGIN,
                SensorCondition.IDEAL,
                4,
                GOAL_GRAINS + 3,
                controller=FillController.PID,
                gains=gains,
            )
        ],
        tuned=TUNED,
        selection=SelectionOfOnePid(gains),
    )
    summary = evaluation.summary
    pid = summary[summary[OnsetColumn.ARM] == OnsetArm.BEST_PID]
    assert set(pid[OnsetColumn.PERTURBATION]) == {BASELINE_PERTURBATION.label} | {
        perturbation.label for perturbation in EARLY_ONSET_PERTURBATIONS
    }
