import pandas
import pytest

from experiments.pouring_model_verification.fair_tuning import (
    FAIR_TUNING_SEEDS,
    SHORTLIST_SIZE,
    FairlyTunedGains,
    FairTuning,
    PidSelection,
    has_lag,
    TuningGrid,
    TuningScores,
    TuningTarget,
    fair_rescoring_trials,
    tuning_targets,
)
from experiments.pouring_model_verification.inflow_model import (
    FirstOrderInflowModel,
    InflowModels,
)
from experiments.pouring_model_verification.parameters import (
    DEMO_PARAMETERS,
    Baselines,
)
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    SourceCup,
    all_scenes,
)
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    FillController,
    MotionEnding,
)

GRID = TuningGrid(
    proportional=(0.005, 0.01),
    integral=(0.002, 0.005),
    derivative=(0.0, 0.02),
    start_tilt=(0.3, 0.6),
)
BASELINES = Baselines(parameters={source: DEMO_PARAMETERS for source in SourceCup})
INFLOW = InflowModels(
    models={
        source: FirstOrderInflowModel(gain=20.0, onset_tilt=0.8, time_constant=0.4)
        for source in SourceCup
    }
)
TARGET = TuningTarget(
    source=SourceCup.CUP, sensor=SensorCondition.SCALE, variant=PidVariant.SMITH
)


def _tuning(grid: TuningGrid = GRID) -> FairTuning:
    return FairTuning(
        grid=grid,
        targets=[TARGET],
        baselines=BASELINES,
        inflow_models=INFLOW,
    )


def _record(
    gains: PidGains, seed: int, delivered: float, target: TuningTarget = TARGET
) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.SOURCE: target.source,
            TrialKey.RECEIVER: ReceiverCup.WIDE,
            TrialKey.SEED: seed,
            TrialKey.CONTROLLER: FillController.PID,
            TrialKey.SENSOR: target.sensor,
            TrialKey.PID_VARIANT: target.variant,
            TrialKey.GAINS: gains.to_json(),
            TrialKey.OUTCOME: {
                Outcome.GOAL_GRAINS: GOAL_GRAINS,
                Outcome.DELIVERED: delivered,
                Outcome.SPILLED: 0,
                Outcome.TRANSFER_DURATION: 5.0,
                Outcome.PEAK_TILT: 1.0,
                Outcome.ENDING: MotionEnding.REACHED_GOAL,
            },
        },
        samples=pandas.DataFrame({Sample.PHASE: [Phase.TRANSFER]}),
    )


# %% what is tuned


def test_the_ideal_sensor_is_tuned_once_per_cup_and_the_scale_once_per_variant():
    targets = tuning_targets(list(SourceCup))
    assert set(targets) == {
        TuningTarget(
            source=source, sensor=SensorCondition.IDEAL, variant=PidVariant.PLAIN
        )
        for source in SourceCup
    } | {
        TuningTarget(source=source, sensor=SensorCondition.SCALE, variant=variant)
        for source in SourceCup
        for variant in PidVariant
    }


def test_the_tuning_seeds_are_new_to_every_earlier_study():
    assert set(FAIR_TUNING_SEEDS).isdisjoint(range(0, 8))
    assert len(FAIR_TUNING_SEEDS) >= 6


# %% the grid


def test_every_grid_point_is_screened_once():
    trials = _tuning().screening_trials()
    assert len(trials) == len(GRID.points())
    assert {trial.seed for trial in trials} == {FAIR_TUNING_SEEDS[0]}


def test_a_winner_on_the_grid_edge_is_extended_outward_along_that_edge():
    corner = PidGains(
        proportional=0.005, integral=0.005, derivative=0.02, start_tilt=0.6
    )
    assert set(GRID.outward_neighbours(corner, tried=GRID.points())) == {
        PidGains(proportional=0.0025, integral=0.005, derivative=0.02, start_tilt=0.6),
        PidGains(proportional=0.005, integral=0.01, derivative=0.02, start_tilt=0.6),
        PidGains(proportional=0.005, integral=0.005, derivative=0.04, start_tilt=0.6),
        PidGains(proportional=0.005, integral=0.005, derivative=0.02, start_tilt=0.9),
    }


def test_no_derivative_and_upright_have_nothing_further_out():
    corner = PidGains(proportional=0.01, integral=0.002, derivative=0.0, start_tilt=0.0)
    grid = TuningGrid(
        proportional=(0.005, 0.01, 0.02),
        integral=(0.002, 0.005),
        derivative=(0.0, 0.02),
        start_tilt=(0.0, 0.3),
    )
    assert set(grid.outward_neighbours(corner, tried=grid.points())) == {
        PidGains(proportional=0.01, integral=0.001, derivative=0.0, start_tilt=0.0)
    }


# %% scoring


def test_the_shortlist_takes_the_best_screened_settings_onto_every_further_seed():
    points = GRID.points()
    records = [
        _record(gains, FAIR_TUNING_SEEDS[0], GOAL_GRAINS + index)
        for index, gains in enumerate(points)
    ]
    trials = _tuning().shortlist_trials(TuningScores(records=records))
    assert {trial.gains for trial in trials} == set(points[:SHORTLIST_SIZE])
    assert {trial.seed for trial in trials} == set(FAIR_TUNING_SEEDS[1:])


def test_the_winner_misplaces_fewest_grains_over_all_its_pours():
    lucky, steady = GRID.points()[:2]
    records = (
        [_record(lucky, FAIR_TUNING_SEEDS[0], GOAL_GRAINS)]
        + [_record(lucky, seed, GOAL_GRAINS + 20) for seed in FAIR_TUNING_SEEDS[1:]]
        + [_record(steady, seed, GOAL_GRAINS + 3) for seed in FAIR_TUNING_SEEDS]
    )
    assert TuningScores(records=records).best(TARGET) == steady


def test_equal_misplaced_grains_are_split_by_runs_within_tolerance():
    spread, even = GRID.points()[:2]
    records = [
        _record(spread, seed, GOAL_GRAINS + (0 if index % 2 else 10))
        for index, seed in enumerate(FAIR_TUNING_SEEDS)
    ] + [_record(even, seed, GOAL_GRAINS + 5) for seed in FAIR_TUNING_SEEDS]
    assert TuningScores(records=records).best(TARGET) == even


def test_only_settings_poured_on_every_tuning_seed_can_win():
    screened_only, full = GRID.points()[:2]
    records = [_record(screened_only, FAIR_TUNING_SEEDS[0], GOAL_GRAINS)] + [
        _record(full, seed, GOAL_GRAINS + 4) for seed in FAIR_TUNING_SEEDS
    ]
    assert TuningScores(records=records).best(TARGET) == full


def test_the_tuning_cost_counts_every_pour_of_the_target():
    records = [
        _record(gains, FAIR_TUNING_SEEDS[0], GOAL_GRAINS) for gains in GRID.points()
    ]
    assert TuningScores(records=records).pours(TARGET) == len(GRID.points())


# %% rescoring


def test_fairly_tuned_gains_survive_a_json_round_trip(tmp_path):
    tuned = FairlyTunedGains(gains={TARGET: GRID.points()[0]}, pours={TARGET: 99})
    tuned.write(tmp_path / "gains.json")
    assert FairlyTunedGains.read(tmp_path / "gains.json") == tuned


def _winners_and_scores():
    ideal_gains, plain_scale, smith_scale, stop_scale = GRID.points()[:4]
    targets = {
        TuningTarget(SourceCup.CUP, SensorCondition.IDEAL, PidVariant.PLAIN): (
            ideal_gains,
            5,
        ),
        TuningTarget(SourceCup.CUP, SensorCondition.SCALE, PidVariant.PLAIN): (
            plain_scale,
            20,
        ),
        TuningTarget(SourceCup.CUP, SensorCondition.SCALE, PidVariant.SMITH): (
            smith_scale,
            2,
        ),
        TuningTarget(SourceCup.CUP, SensorCondition.SCALE, PidVariant.STOP_EARLY): (
            stop_scale,
            9,
        ),
    }
    records = [
        _record(gains, seed, GOAL_GRAINS + misplaced, target)
        for target, (gains, misplaced) in targets.items()
        for seed in FAIR_TUNING_SEEDS
    ]
    tuned = FairlyTunedGains(
        gains={target: gains for target, (gains, _) in targets.items()},
        pours={target: len(FAIR_TUNING_SEEDS) for target in targets},
    )
    return tuned, TuningScores(records=records), ideal_gains, smith_scale


def test_under_a_lagging_sensor_the_scale_variant_that_tuned_best_is_chosen():
    tuned, scores, _, smith_scale = _winners_and_scores()
    selection = PidSelection(tuned=tuned, scores=scores)
    assert selection.scale_tuned(SourceCup.CUP) == (PidVariant.SMITH, smith_scale)


def test_rescoring_runs_the_ideal_tuned_pid_everywhere_and_the_chosen_scale_pid_under_lag():
    tuned, scores, ideal_gains, smith_scale = _winners_and_scores()
    scenes = [scene for scene in all_scenes() if scene.source is SourceCup.CUP]
    trials = fair_rescoring_trials(
        seeds=[4],
        scenes=scenes,
        baselines=BASELINES,
        selection=PidSelection(tuned=tuned, scores=scores),
        inflow_models=INFLOW,
    )
    assert {
        (trial.scene, trial.sensor, trial.pid_variant, trial.gains) for trial in trials
    } == {
        (scene, sensor, PidVariant.PLAIN, ideal_gains)
        for scene in scenes
        for sensor in SensorCondition
    } | {
        (scene, sensor, PidVariant.SMITH, smith_scale)
        for scene in scenes
        for sensor in SensorCondition
        if has_lag(sensor)
    }
