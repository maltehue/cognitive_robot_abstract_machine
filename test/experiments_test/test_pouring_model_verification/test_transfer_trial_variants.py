import pytest

from experiments.pouring_model_verification.parameters import (
    BASELINE,
    DEMO_PARAMETERS,
    ModelParameter,
    Perturbation,
)
from experiments.pouring_model_verification.inflow_model import FirstOrderInflowModel
from experiments.pouring_model_verification.onset_rule import OnsetResponse, OnsetRule
from experiments.pouring_model_verification.pid import PidGains, PidVariant
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    Scene,
    SourceCup,
)
from experiments.pouring_model_verification.sensor import SensorCondition, WristSensor
from experiments.pouring_model_verification.transfer_trial import (
    FillController,
    LoopMode,
    MissingInflowModelForSmithError,
    MissingPidGainsError,
    OnsetRuleWithoutModelError,
    POUR_START_TILT_MARGIN,
    PidWithoutFillFeedbackError,
    KeptReposeAngleWithoutRevisionError,
    ConceptAwareRevisionWithoutWristError,
    SequencePosition,
    TransferTrial,
    CautionWithoutBankError,
    DrainRateSpreadMissingError,
    PourFault,
)
from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    DrainRateSpread,
)
from experiments.pouring_model_verification.recovery import CautionSettings
from experiments.pouring_model_verification.sensor import SensorSet

SCENE = Scene(source=SourceCup.CUP, receiver=ReceiverCup.WIDE)
GAINS = PidGains(proportional=0.02, integral=0.01, derivative=0.0, start_tilt=0.6)


def _trial(**variant) -> TransferTrial:
    return TransferTrial(
        scene=SCENE,
        baseline=DEMO_PARAMETERS,
        perturbation=BASELINE,
        loop_mode=LoopMode.FILL_FEEDBACK,
        seed=2,
        **variant,
    )


# %% trial names


def test_a_model_predictive_trial_with_an_ideal_sensor_keeps_its_recorded_name():
    assert _trial().name == (
        f"transfer_{SCENE.label}_{BASELINE.label}_{LoopMode.FILL_FEEDBACK}_seed2"
    )


def test_a_trial_with_another_sensor_is_named_after_it():
    trial = _trial(sensor=SensorCondition.SCALE)
    assert trial.name == (
        f"transfer_{SCENE.label}_{BASELINE.label}_{LoopMode.FILL_FEEDBACK}"
        f"_{SensorCondition.SCALE}_seed2"
    )


@pytest.mark.parametrize(
    "controller", [FillController.PID, FillController.PID_FIXED_POSE]
)
def test_a_pid_trial_is_named_after_its_controller_gains_and_sensor(controller):
    trial = _trial(controller=controller, gains=GAINS, sensor=SensorCondition.NOISY)
    assert trial.name == (
        f"transfer_{SCENE.label}_{controller}_{GAINS.label}"
        f"_{SensorCondition.NOISY}_seed2"
    )


@pytest.mark.parametrize("variant", [PidVariant.STOP_EARLY, PidVariant.SMITH])
def test_a_compensated_pid_trial_is_named_after_its_variant(variant):
    trial = _trial(
        controller=FillController.PID,
        gains=GAINS,
        sensor=SensorCondition.SCALE,
        pid_variant=variant,
        inflow_model=FirstOrderInflowModel(
            gain=20.0, onset_tilt=0.8, time_constant=0.4
        ),
    )
    assert trial.name == (
        f"transfer_{SCENE.label}_{FillController.PID}_{variant}_{GAINS.label}"
        f"_{SensorCondition.SCALE}_seed2"
    )


RULE = OnsetRule(response=OnsetResponse.OVERRIDE, patience=2.0, override_velocity=0.1)


def test_a_trial_with_the_onset_rule_is_named_after_it_and_its_start_margin():
    trial = _trial(onset_rule=RULE, start_margin=0.0)
    assert trial.name == (
        f"transfer_{SCENE.label}_{BASELINE.label}_{LoopMode.FILL_FEEDBACK}"
        f"_{RULE.label}_margin0_seed2"
    )


def test_a_trial_keeping_the_usual_start_margin_does_not_name_it():
    trial = _trial(onset_rule=RULE, start_margin=POUR_START_TILT_MARGIN)
    assert trial.name.endswith(f"_{RULE.label}_seed2")


REVISION = OnsetRule(response=OnsetResponse.REVISION, patience=2.0)


def test_a_pour_of_a_sequence_is_named_after_its_place_in_it():
    trial = _trial(
        onset_rule=REVISION,
        start_margin=0.0,
        sequence=SequencePosition(pour=3, sequence_seed=1),
    )
    assert trial.name.endswith(f"_{REVISION.label}_margin0_pour3_seed2")


def test_a_pour_starting_from_a_kept_repose_angle_is_named_as_kept():
    trial = _trial(
        onset_rule=REVISION,
        start_margin=0.0,
        sequence=SequencePosition(pour=3, sequence_seed=1, kept_repose_angle=0.9),
    )
    assert trial.name.endswith("_pour3_kept_seed2")


CONCEPT_AWARE = OnsetRule(response=OnsetResponse.CONCEPT_AWARE_REVISION, patience=1.0)


def test_a_trial_weighing_the_source_at_the_wrist_is_named_after_that_sensor():
    trial = _trial(
        onset_rule=CONCEPT_AWARE,
        start_margin=0.0,
        wrist_sensor=WristSensor.REALISTIC,
    )
    assert trial.name.endswith(
        f"_{CONCEPT_AWARE.label}_margin0_wrist_{WristSensor.REALISTIC}_seed2"
    )


# %% what a trial needs


def test_a_concept_aware_revision_is_refused_without_a_wrist_sensor():
    with pytest.raises(ConceptAwareRevisionWithoutWristError):
        _trial(onset_rule=CONCEPT_AWARE, start_margin=0.0)


def test_a_concept_aware_revision_revises_the_repose_angle_too():
    assert _trial(
        onset_rule=CONCEPT_AWARE, wrist_sensor=WristSensor.IDEAL
    ).revises_repose_angle


def test_a_kept_repose_angle_is_refused_without_a_revision_to_keep_it_in():
    with pytest.raises(KeptReposeAngleWithoutRevisionError):
        _trial(
            onset_rule=RULE,
            sequence=SequencePosition(pour=2, sequence_seed=1, kept_repose_angle=0.9),
        )


def test_the_onset_rule_is_refused_for_a_controller_without_the_model():
    with pytest.raises(OnsetRuleWithoutModelError):
        _trial(controller=FillController.PID, gains=GAINS, onset_rule=RULE)


def test_a_smith_predictor_without_an_inflow_model_is_refused():
    with pytest.raises(MissingInflowModelForSmithError):
        _trial(controller=FillController.PID, gains=GAINS, pid_variant=PidVariant.SMITH)


def test_a_pid_trial_without_gains_is_refused():
    with pytest.raises(MissingPidGainsError):
        _trial(controller=FillController.PID)


def test_a_pid_trial_that_is_told_nothing_is_refused():
    with pytest.raises(PidWithoutFillFeedbackError):
        TransferTrial(
            scene=SCENE,
            baseline=DEMO_PARAMETERS,
            perturbation=BASELINE,
            loop_mode=LoopMode.OPEN_LOOP,
            seed=0,
            controller=FillController.PID,
            gains=GAINS,
        )


def test_only_the_pid_controllers_steer_without_the_model():
    assert [controller.uses_pouring_model for controller in FillController] == [
        controller is FillController.MODEL_PREDICTIVE for controller in FillController
    ]


def test_only_the_aiming_controllers_move_the_cup_to_aim():
    assert [controller.aims for controller in FillController] == [
        controller is not FillController.PID_FIXED_POSE for controller in FillController
    ]


# %% a cautious start


CAUTION = CautionSettings(cap_fraction=0.5, release_time=0.5)


def _bank_trial(**variant) -> TransferTrial:
    sensors = SensorSet.IDEAL
    return _trial(
        bank=BankSettings(),
        sensor_set=sensors,
        sensor=sensors.receiver,
        wrist_sensor=sensors.wrist,
        **variant,
    )


def test_a_cautious_pour_is_named_after_its_caution():
    assert _bank_trial(caution=CAUTION).name == _bank_trial().name.replace(
        "_seed", f"_{CAUTION.label}_seed"
    )


def test_a_cautious_start_is_refused_without_a_bank_to_confirm_the_aim():
    with pytest.raises(CautionWithoutBankError):
        _trial(caution=CAUTION)


# %% a fault outside the closed world


def test_a_baffle_at_the_lip_leaves_the_grains_sound():
    assert PourFault.LIP_BAFFLE.contact is None


def test_rough_grains_put_nothing_in_the_way():
    assert PourFault.ROUGH_GRAINS.obstacle(SourceCup.CUP) is None


def test_the_baffle_stands_inside_the_lip_the_cup_pours_over():
    geometry = SourceCup.CUP.geometry
    baffle = PourFault.LIP_BAFFLE.obstacle(SourceCup.CUP)
    origin = baffle.origin.to_np()[:3, 3]
    outer_face = origin[0] * SourceCup.CUP.pour_direction + baffle.scale.x / 2
    assert outer_face == pytest.approx(geometry.inner_radius)
    assert (
        origin[2] - baffle.scale.z / 2
        < geometry.height
        < (origin[2] + baffle.scale.z / 2)
    )


def test_the_baffle_blocks_only_part_of_the_lip():
    geometry = SourceCup.CUP.geometry
    baffle = PourFault.LIP_BAFFLE.obstacle(SourceCup.CUP)
    assert 0.0 < baffle.scale.y < 2 * geometry.inner_radius


# %% the twin's drain rate while turning

DRAINING = DEMO_PARAMETERS.with_dynamic_outflow_rate_constant(
    0.1
).with_dynamic_outflow_rate_spread(0.3)
WRONG_DRAIN = Perturbation(parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=4.0)


def _draining_trial(baseline, bank: BankSettings) -> TransferTrial:
    return TransferTrial(
        scene=SCENE,
        baseline=baseline,
        perturbation=WRONG_DRAIN,
        loop_mode=LoopMode.FILL_FEEDBACK,
        seed=2,
        bank=bank,
        sensor_set=SensorSet.IDEAL,
        sensor=SensorSet.IDEAL.receiver,
        wrist_sensor=SensorSet.IDEAL.wrist,
    )


def test_a_bank_adapting_the_drain_rate_carries_the_twins_spread_with_the_error_set_in():
    trial = _draining_trial(DRAINING, BankSettings(adapts_drain_rate=True))
    assert trial.drain_rate == DrainRateSpread(
        centre=DRAINING.dynamic_outflow_rate_constant * WRONG_DRAIN.factor,
        spread=DRAINING.dynamic_outflow_rate_spread,
    )


def test_a_bank_that_does_not_adapt_the_drain_rate_carries_no_spread_of_it():
    assert _draining_trial(DRAINING, BankSettings()).drain_rate is None


def test_a_bank_adapting_the_drain_rate_needs_the_twins_spread_of_it():
    trial = _draining_trial(DEMO_PARAMETERS, BankSettings(adapts_drain_rate=True))
    with pytest.raises(DrainRateSpreadMissingError):
        trial.drain_rate
