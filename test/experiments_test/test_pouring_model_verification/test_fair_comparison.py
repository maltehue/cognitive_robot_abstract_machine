import pandas
import pytest

from experiments.pouring_model_verification.fair_comparison import (
    FairComparison,
    SeedSpread,
    Verdict,
)
from experiments.pouring_model_verification.fair_tuning import (
    FAIR_TUNING_SEEDS,
    FairlyTunedGains,
    PidSelection,
    TuningScores,
    TuningTarget,
)
from experiments.pouring_model_verification.parameters import BASELINE
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.scene import ReceiverCup, SourceCup
from experiments.pouring_model_verification.sensor import SensorCondition
from experiments.pouring_model_verification.transfer_trial import (
    GOAL_GRAINS,
    FillController,
    LoopMode,
    MotionEnding,
)

IDEAL_GAINS = PidGains(
    proportional=0.01, integral=0.005, derivative=0.0, start_tilt=0.6
)
PLAIN_SCALE = PidGains(
    proportional=0.005, integral=0.005, derivative=0.0, start_tilt=0.6
)
SMITH_SCALE = PidGains(
    proportional=0.02, integral=0.002, derivative=0.0, start_tilt=0.3
)


def _target(sensor: SensorCondition, variant: PidVariant) -> TuningTarget:
    return TuningTarget(source=SourceCup.CUP, sensor=sensor, variant=variant)


TUNED = FairlyTunedGains(
    gains={
        _target(SensorCondition.IDEAL, PidVariant.PLAIN): IDEAL_GAINS,
        _target(SensorCondition.SCALE, PidVariant.PLAIN): PLAIN_SCALE,
        _target(SensorCondition.SCALE, PidVariant.SMITH): SMITH_SCALE,
    },
    pours={},
)


def _record(
    delivered: float,
    sensor: SensorCondition,
    seed: int,
    controller: FillController,
    gains: PidGains | None = None,
    variant: PidVariant = PidVariant.PLAIN,
) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.SOURCE: SourceCup.CUP,
            TrialKey.RECEIVER: ReceiverCup.WIDE,
            TrialKey.SEED: seed,
            TrialKey.LOOP_MODE: LoopMode.FILL_FEEDBACK,
            TrialKey.PERTURBATION: BASELINE.to_json(),
            TrialKey.CONTROLLER: controller,
            TrialKey.SENSOR: sensor,
            TrialKey.PID_VARIANT: variant,
            TrialKey.GAINS: None if gains is None else gains.to_json(),
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


# %% verdicts


def test_means_inside_each_others_seed_range_are_within_noise():
    model = SeedSpread.of(misplaced=[4.0, 10.0, 12.0], succeeded=[True, False, False])
    pid = SeedSpread.of(misplaced=[2.0, 9.0, 30.0], succeeded=[True, False, False])
    assert Verdict.between(model, pid) is Verdict.WITHIN_NOISE


def test_a_pid_whose_whole_range_lies_below_the_models_mean_beats_it():
    model = SeedSpread.of(misplaced=[10.0, 12.0, 14.0], succeeded=[False] * 3)
    pid = SeedSpread.of(misplaced=[1.0, 2.0, 3.0], succeeded=[True] * 3)
    assert Verdict.between(model, pid) is Verdict.PID_BETTER


def test_a_pid_whose_whole_range_lies_above_the_models_mean_loses():
    model = SeedSpread.of(misplaced=[1.0, 2.0, 3.0], succeeded=[True] * 3)
    pid = SeedSpread.of(misplaced=[10.0, 12.0, 14.0], succeeded=[False] * 3)
    assert Verdict.between(model, pid) is Verdict.MODEL_BETTER


# %% which PID meets the model


def _scores() -> TuningScores:
    def tuning(gains, variant, delivered):
        record = _record(
            delivered, SensorCondition.SCALE, 0, FillController.PID, gains, variant
        )
        return [
            TrialRecord(
                description={**record.description, TrialKey.SEED: seed},
                samples=record.samples,
            )
            for seed in FAIR_TUNING_SEEDS
        ]

    return TuningScores(
        records=tuning(PLAIN_SCALE, PidVariant.PLAIN, GOAL_GRAINS + 9)
        + tuning(SMITH_SCALE, PidVariant.SMITH, GOAL_GRAINS + 2)
    )


def test_under_a_lagging_sensor_the_pid_is_the_variant_that_tuned_best_for_the_scale():
    comparison = FairComparison(
        model=[], pid=[], selection=PidSelection(tuned=TUNED, scores=_scores())
    )
    assert comparison.selection.for_sensor(
        SourceCup.CUP, SensorCondition.VERY_DELAYED
    ) == (
        PidVariant.SMITH,
        SMITH_SCALE,
    )


def test_without_lag_the_pid_is_the_one_tuned_for_the_ideal_sensor():
    comparison = FairComparison(
        model=[], pid=[], selection=PidSelection(tuned=TUNED, scores=_scores())
    )
    assert comparison.selection.for_sensor(SourceCup.CUP, SensorCondition.NOISY) == (
        PidVariant.PLAIN,
        IDEAL_GAINS,
    )


def test_each_scene_and_sensor_compares_the_model_with_its_pid():
    model = [
        _record(
            GOAL_GRAINS + 10 + seed,
            SensorCondition.SCALE,
            seed,
            FillController.MODEL_PREDICTIVE,
        )
        for seed in range(4)
    ]
    pid = [
        _record(
            GOAL_GRAINS + seed,
            SensorCondition.SCALE,
            seed,
            FillController.PID,
            SMITH_SCALE,
            PidVariant.SMITH,
        )
        for seed in range(4)
    ] + [
        _record(
            GOAL_GRAINS + 40,
            SensorCondition.SCALE,
            seed,
            FillController.PID,
            PLAIN_SCALE,
            PidVariant.PLAIN,
        )
        for seed in range(4)
    ]
    comparison = FairComparison(
        model=model, pid=pid, selection=PidSelection(tuned=TUNED, scores=_scores())
    )
    row = comparison.table.iloc[0]
    assert (row["pid_mean"], row["model_mean"], row["verdict"]) == (
        pytest.approx(1.5),
        pytest.approx(11.5),
        Verdict.PID_BETTER,
    )
