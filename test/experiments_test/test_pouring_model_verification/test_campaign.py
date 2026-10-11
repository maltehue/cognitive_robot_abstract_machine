from experiments.pouring_model_verification.campaign import (
    FIRST_ROBUSTNESS_SEED,
    COMPENSATED_VARIANTS,
    delay_compensation_trials,
    PID_GAIN_FACTORS,
    pid_tuning_trials,
    robustness_seeds,
    robustness_trials,
)
from experiments.pouring_model_verification.parameters import (
    DEMO_PARAMETERS,
    Baselines,
)
from experiments.pouring_model_verification.inflow_model import (
    FirstOrderInflowModel,
    InflowModels,
)
from experiments.pouring_model_verification.pid import (
    PidVariant,
    PidGains,
    TunedPidGains,
    tuning_grid,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    SourceCup,
    all_scenes,
)
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    FillController,
    LoopMode,
)

BASELINES = Baselines(parameters={source: DEMO_PARAMETERS for source in SourceCup})
GAINS = PidGains(proportional=0.01, integral=0.002, derivative=0.01, start_tilt=0.9)
TUNED = TunedPidGains(gains={source: GAINS for source in SourceCup})
SEEDS = 2

# %% tuning


def test_tuning_tries_every_grid_point_on_every_seed_of_every_cup_into_the_wide_receiver():
    trials = pid_tuning_trials(
        seeds=SEEDS, sources=list(SourceCup), baselines=BASELINES
    )
    assert len(trials) == len(tuning_grid()) * SEEDS * len(SourceCup)
    assert {trial.scene.receiver for trial in trials} == {ReceiverCup.WIDE}
    assert {(trial.controller, trial.sensor) for trial in trials} == {
        (FillController.PID, SensorCondition.IDEAL)
    }


# %% robustness


def _robustness():
    return robustness_trials(
        seeds=robustness_seeds(SEEDS),
        scenes=all_scenes(),
        baselines=BASELINES,
        tuned=TUNED,
    )


def test_every_robustness_trial_has_a_record_of_its_own():
    trials = _robustness()
    assert len({trial.name for trial in trials}) == len(trials)


def test_the_robustness_study_runs_on_seeds_no_earlier_study_ran_on():
    assert {trial.seed for trial in _robustness()} == set(
        range(FIRST_ROBUSTNESS_SEED, FIRST_ROBUSTNESS_SEED + SEEDS)
    )


def test_the_model_predictive_controller_meets_every_sensor_in_both_feedback_modes():
    combinations = {
        (trial.loop_mode, trial.sensor)
        for trial in _robustness()
        if trial.controller is FillController.MODEL_PREDICTIVE
    }
    assert combinations == {
        (loop_mode, sensor)
        for loop_mode in (
            LoopMode.FILL_FEEDBACK,
            LoopMode.FILL_FEEDBACK_AND_CALIBRATION,
        )
        for sensor in SensorCondition
    }


def test_the_tuned_pid_controller_meets_every_sensor():
    sensors = {
        trial.sensor
        for trial in _robustness()
        if trial.controller is FillController.PID and trial.gains == GAINS
    }
    assert sensors == set(SensorCondition)


def test_the_pid_controller_is_also_run_with_its_gains_scaled_and_held_still():
    ideal_pid = {
        (trial.controller, trial.gains)
        for trial in _robustness()
        if not trial.controller.uses_pouring_model
        and trial.sensor is SensorCondition.IDEAL
    }
    assert ideal_pid == {
        (FillController.PID, GAINS.scaled(factor)) for factor in PID_GAIN_FACTORS
    } | {(FillController.PID, GAINS), (FillController.PID_FIXED_POSE, GAINS)}


# %% delay compensation

INFLOW = InflowModels(
    models={
        source: FirstOrderInflowModel(gain=20.0, onset_tilt=0.8, time_constant=0.4)
        for source in SourceCup
    }
)


def _compensated():
    return delay_compensation_trials(
        seeds=robustness_seeds(SEEDS),
        scenes=all_scenes(),
        baselines=BASELINES,
        tuned=TUNED,
        inflow_models=INFLOW,
    )


def test_every_compensated_pid_meets_every_sensor_in_every_scene():
    aimed = {
        (trial.pid_variant, trial.sensor, trial.scene, trial.seed)
        for trial in _compensated()
        if trial.controller is FillController.PID
    }
    assert aimed == {
        (variant, sensor, scene, seed)
        for variant in COMPENSATED_VARIANTS
        for sensor in SensorCondition
        for scene in all_scenes()
        for seed in robustness_seeds(SEEDS)
    }


def test_every_pid_is_also_held_still_once_per_cup_behind_a_scale():
    held = {
        (trial.pid_variant, trial.scene.source, trial.scene.receiver, trial.sensor)
        for trial in _compensated()
        if trial.controller is FillController.PID_FIXED_POSE
    }
    assert held == {
        (variant, source, ReceiverCup.WIDE, SensorCondition.SCALE)
        for variant in PidVariant
        for source in SourceCup
    }


def test_every_compensated_trial_has_a_record_of_its_own():
    trials = _compensated()
    assert len({trial.name for trial in trials}) == len(trials)
