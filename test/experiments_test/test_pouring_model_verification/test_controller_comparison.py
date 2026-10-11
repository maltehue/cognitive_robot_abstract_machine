import pandas
import pytest

from experiments.pouring_model_verification.controller_comparison import (
    ComparisonColumn,
    ControllerArm,
    ControllerComparison,
    PidTuning,
    misplaced_grains,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    Perturbation,
    ModelParameter,
)
from experiments.pouring_model_verification.pid import (
    PidGains,
    PidVariant,
    TunedPidGains,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    Study,
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

TUNED = PidGains(proportional=0.01, integral=0.002, derivative=0.01, start_tilt=0.9)
WORSE = PidGains(proportional=0.02, integral=0.0, derivative=0.0, start_tilt=0.5)


def _record(
    delivered: float,
    spilled: float = 0.0,
    source: SourceCup = SourceCup.CUP,
    seed: int = 0,
    loop_mode: LoopMode = LoopMode.FILL_FEEDBACK,
    perturbation: Perturbation = BASELINE,
    controller: FillController | None = None,
    sensor: SensorCondition | None = None,
    gains: PidGains | None = None,
    variant: PidVariant | None = None,
) -> TrialRecord:
    description = {
        TrialKey.STUDY: Study.TRANSFER,
        TrialKey.SOURCE: source,
        TrialKey.RECEIVER: ReceiverCup.WIDE,
        TrialKey.SEED: seed,
        TrialKey.LOOP_MODE: loop_mode,
        TrialKey.PERTURBATION: perturbation.to_json(),
        TrialKey.OUTCOME: {
            Outcome.GOAL_GRAINS: GOAL_GRAINS,
            Outcome.DELIVERED: delivered,
            Outcome.SPILLED: spilled,
            Outcome.TRANSFER_DURATION: 5.0,
            Outcome.PEAK_TILT: 1.0,
            Outcome.ENDING: MotionEnding.REACHED_GOAL,
        },
    }
    if controller is not None:
        description[TrialKey.CONTROLLER] = controller
        description[TrialKey.SENSOR] = sensor
        description[TrialKey.GAINS] = None if gains is None else gains.to_json()
    if variant is not None:
        description[TrialKey.PID_VARIANT] = variant
    samples = pandas.DataFrame(
        {Sample.TIME: [0.0], Sample.PHASE: [Phase.TRANSFER], Sample.TILT: [1.0]}
    )
    return TrialRecord(description=description, samples=samples)


# %% how a pour is scored


def test_a_pour_misplaces_what_it_misses_the_goal_by_and_what_it_spills():
    assert misplaced_grains(delivered=GOAL_GRAINS - 3.0, spilled=4.0) == pytest.approx(
        7.0
    )
    assert misplaced_grains(delivered=GOAL_GRAINS + 3.0, spilled=0.0) == pytest.approx(
        3.0
    )


# %% tuning


def test_tuning_keeps_each_cups_gains_that_misplaced_fewest_grains_over_the_seeds():
    records = [
        _record(GOAL_GRAINS + 1, seed=0, controller=FillController.PID, gains=TUNED),
        _record(GOAL_GRAINS - 2, seed=1, controller=FillController.PID, gains=TUNED),
        _record(GOAL_GRAINS, seed=0, controller=FillController.PID, gains=WORSE),
        _record(GOAL_GRAINS + 9, seed=1, controller=FillController.PID, gains=WORSE),
        _record(
            GOAL_GRAINS,
            source=SourceCup.SPOUTED_CUP,
            controller=FillController.PID,
            gains=WORSE,
        ),
    ]
    best = PidTuning(records=records).best()
    assert best.of(SourceCup.CUP) == TUNED
    assert best.of(SourceCup.SPOUTED_CUP) == WORSE


# %% which controller a trial ran


def _arm_of(record: TrialRecord) -> ControllerArm:
    comparison = ControllerComparison(
        records=[record],
        tuned=TunedPidGains(gains={SourceCup.CUP.value: TUNED}),
    )
    return comparison.summary[ComparisonColumn.ARM].iloc[0]


@pytest.mark.parametrize(
    "record, arm",
    [
        (
            _record(
                40,
                controller=FillController.MODEL_PREDICTIVE,
                sensor=SensorCondition.SCALE,
            ),
            ControllerArm.MODEL_FILL_FEEDBACK,
        ),
        (
            _record(
                40,
                loop_mode=LoopMode.FILL_FEEDBACK_AND_CALIBRATION,
                controller=FillController.MODEL_PREDICTIVE,
                sensor=SensorCondition.NOISY,
            ),
            ControllerArm.MODEL_CALIBRATION,
        ),
        (
            _record(
                40,
                controller=FillController.PID,
                sensor=SensorCondition.IDEAL,
                gains=TUNED,
            ),
            ControllerArm.PID,
        ),
        (
            _record(
                40,
                controller=FillController.PID,
                sensor=SensorCondition.IDEAL,
                gains=TUNED.scaled(0.5),
            ),
            ControllerArm.PID_SOFTER,
        ),
        (
            _record(
                40,
                controller=FillController.PID,
                sensor=SensorCondition.IDEAL,
                gains=TUNED.scaled(2.0),
            ),
            ControllerArm.PID_STIFFER,
        ),
        (
            _record(
                40,
                controller=FillController.PID_FIXED_POSE,
                sensor=SensorCondition.IDEAL,
                gains=TUNED,
            ),
            ControllerArm.PID_FIXED_POSE,
        ),
        (
            _record(
                40,
                controller=FillController.PID,
                sensor=SensorCondition.SCALE,
                gains=TUNED,
                variant=PidVariant.STOP_EARLY,
            ),
            ControllerArm.PID_STOP_EARLY,
        ),
        (
            _record(
                40,
                controller=FillController.PID,
                sensor=SensorCondition.SCALE,
                gains=TUNED,
                variant=PidVariant.SMITH,
            ),
            ControllerArm.PID_SMITH,
        ),
        (
            _record(
                40,
                controller=FillController.PID_FIXED_POSE,
                sensor=SensorCondition.IDEAL,
                gains=TUNED,
                variant=PidVariant.SMITH,
            ),
            ControllerArm.PID_SMITH_FIXED_POSE,
        ),
        (
            _record(
                40,
                controller=FillController.PID_FIXED_POSE,
                sensor=SensorCondition.IDEAL,
                gains=TUNED,
                variant=PidVariant.STOP_EARLY,
            ),
            ControllerArm.PID_STOP_EARLY_FIXED_POSE,
        ),
    ],
)
def test_each_trial_is_counted_for_the_controller_it_ran(record, arm):
    assert _arm_of(record) is arm


def test_only_model_predictive_trials_at_the_fitted_parameters_with_feedback_are_compared():
    wrong = Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5)
    comparison = ControllerComparison(
        records=[
            _record(40, perturbation=wrong),
            _record(40, loop_mode=LoopMode.OPEN_LOOP),
        ],
        tuned=TunedPidGains(gains={SourceCup.CUP.value: TUNED}),
    )
    assert comparison.summary.empty


def test_a_trial_recorded_before_there_was_a_choice_counts_as_the_model_with_an_ideal_sensor():
    comparison = ControllerComparison(
        records=[_record(40, loop_mode=LoopMode.FILL_FEEDBACK)],
        tuned=TunedPidGains(gains={SourceCup.CUP.value: TUNED}),
    )
    row = comparison.summary.iloc[0]
    assert (row[ComparisonColumn.ARM], row[ComparisonColumn.SENSOR]) == (
        ControllerArm.MODEL_FILL_FEEDBACK,
        SensorCondition.IDEAL,
    )
