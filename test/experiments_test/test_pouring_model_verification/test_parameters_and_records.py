import math

import pandas
import pytest

from experiments.pouring_model_verification.analysis import (
    Column,
    ParameterSet,
    SourceAnalysis,
)
from experiments.pouring_model_verification.parameters import (
    NOMINAL,
    NOMINAL_PARAMETERS,
    BASELINE,
    Baselines,
    BaselinesNotFittedError,
    MissingBaselineError,
    DEMO_PARAMETERS,
    ModelParameter,
    ModelParameters,
    Perturbation,
    closed_loop_perturbations,
)
from experiments.pouring_model_verification.records import (
    Outcome,
    Phase,
    Sample,
    Study,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.tilt_rig import (
    MAXIMUM_TILT_RATE,
    Ramp,
    StepAndHold,
    TiltProfile,
)
from experiments.pouring_model_verification.scene import (
    ReceiverCup,
    SourceCup,
    all_scenes,
)
from experiments.pouring_model_verification.transfer_trial import LoopMode
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)

# %% parameter sets


@pytest.mark.parametrize("parameter", list(ModelParameter))
def test_scaling_changes_only_that_parameter(parameter):
    scaled = DEMO_PARAMETERS.scaled(parameter, 2.0)
    for other in ModelParameter:
        expected = DEMO_PARAMETERS.value_of(other) * (
            2.0 if other is parameter else 1.0
        )
        assert scaled.value_of(other) == expected


def test_parameters_reach_the_drain():
    equation = ArticulatedPouringEquation(container_height=0.06, container_width=0.078)
    parameters = ModelParameters(
        repose_angle=0.3,
        outflow_rate_constant=0.5,
        discharge_coefficient=0.6,
        lip_offset_scale=1.5,
    )
    drain = parameters.applied_to(equation)
    assert drain.repose_angle == parameters.repose_angle
    assert drain.outflow_rate_constant == parameters.outflow_rate_constant
    assert drain.discharge_coefficient == parameters.discharge_coefficient
    assert drain.lip_offset == equation.lip_offset * parameters.lip_offset_scale


def test_parameters_keep_a_spout_lip_offset():
    equation = ArticulatedPouringEquation(
        container_height=0.06, container_width=0.078, lip_offset=0.075
    )
    assert DEMO_PARAMETERS.applied_to(equation).lip_offset == equation.lip_offset


def test_parameters_round_trip_through_json():
    parameters = DEMO_PARAMETERS.scaled(ModelParameter.LIP_OFFSET, 1.3)
    assert ModelParameters.from_json(parameters.to_json()) == parameters


@pytest.mark.parametrize("perturbation", closed_loop_perturbations())
def test_perturbations_round_trip_through_json(perturbation):
    assert Perturbation.from_json(perturbation.to_json()) == perturbation


def test_the_baseline_perturbation_is_its_baseline_and_nominal_ignores_it():
    fitted = DEMO_PARAMETERS.scaled(ModelParameter.REPOSE_ANGLE, 1.3)
    assert BASELINE.around(fitted) == fitted
    assert NOMINAL.around(fitted) == NOMINAL_PARAMETERS


def test_a_perturbation_scales_its_baseline():
    fitted = DEMO_PARAMETERS.scaled(ModelParameter.REPOSE_ANGLE, 1.3)
    perturbation = Perturbation(
        parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=2.0
    )
    assert perturbation.around(fitted) == fitted.scaled(
        ModelParameter.OUTFLOW_RATE_CONSTANT, 2.0
    )


# %% the repose angle while the cup turns

TURNING = DEMO_PARAMETERS.with_dynamic_repose_angle(DEMO_PARAMETERS.repose_angle + 0.15)


def test_a_turning_cup_holds_its_dynamic_repose_angle():
    assert TURNING.while_turning().repose_angle == TURNING.dynamic_repose_angle


def test_without_a_dynamic_repose_angle_the_static_one_holds_while_turning():
    assert DEMO_PARAMETERS.while_turning() == DEMO_PARAMETERS


def test_an_error_in_the_repose_angle_scales_the_dynamic_one_too():
    perturbation = Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5)
    assert perturbation.around(TURNING).while_turning().repose_angle == (
        TURNING.dynamic_repose_angle * perturbation.factor
    )


def test_a_dynamic_repose_angle_survives_a_round_trip_through_json():
    assert ModelParameters.from_json(TURNING.to_json()) == TURNING


# %% baselines on disk


def test_baselines_round_trip_through_their_file(tmp_path):
    baselines = Baselines(
        parameters={
            SourceCup.CUP.value: DEMO_PARAMETERS,
            SourceCup.SPOUTED_CUP.value: NOMINAL_PARAMETERS,
        }
    )
    baselines.write(tmp_path / "baselines.json")
    assert Baselines.read(tmp_path / "baselines.json") == baselines


def test_a_cup_without_a_fit_has_no_baseline():
    baselines = Baselines(parameters={SourceCup.CUP.value: DEMO_PARAMETERS})
    with pytest.raises(MissingBaselineError):
        baselines.of(SourceCup.SPOUTED_CUP)


def test_reading_baselines_before_any_fit_is_refused(tmp_path):
    with pytest.raises(BaselinesNotFittedError):
        Baselines.read(tmp_path / "baselines.json")


# %% scenes


def test_every_scene_has_its_own_label():
    labels = [scene.label for scene in all_scenes()]
    assert len(labels) == len(SourceCup) * len(ReceiverCup)
    assert len(set(labels)) == len(labels)


def test_the_narrow_receiver_is_half_as_wide():
    assert ReceiverCup.NARROW.geometry.inner_radius == pytest.approx(
        ReceiverCup.WIDE.geometry.inner_radius / 2
    )


def test_closed_loop_perturbations_have_distinct_labels():
    labels = [perturbation.label for perturbation in closed_loop_perturbations()]
    assert len(set(labels)) == len(labels)


# %% tilt profiles


@pytest.mark.parametrize(
    "profile",
    [StepAndHold(tilt=math.radians(80), hold=5.0), Ramp(tilt_rate=0.2, hold=4.0)],
)
def test_profiles_round_trip_through_json(profile):
    assert TiltProfile.from_json(profile.to_json()) == profile


def test_step_turns_at_the_rig_rate_and_holds():
    step = StepAndHold(tilt=1.0, hold=5.0)
    assert step.tilt_at(0.5) == 0.5 * MAXIMUM_TILT_RATE
    assert step.tilt_at(step.duration) == step.tilt
    assert step.duration == step.tilt / MAXIMUM_TILT_RATE + step.hold


# %% records on disk


def test_record_round_trips_through_its_directory(tmp_path):
    record = TrialRecord(
        description={TrialKey.STUDY: Study.TILT_RIG, TrialKey.SEED: 2},
        samples=pandas.DataFrame({Sample.TIME: [0.0, 0.02], Sample.TILT: [0.0, 0.1]}),
    )
    record.write(tmp_path / "trial")
    read = TrialRecord.read(tmp_path / "trial")
    assert TrialRecord.is_complete(tmp_path / "trial")
    assert read.description == record.description
    pandas.testing.assert_frame_equal(read.samples, record.samples)


def test_a_directory_without_both_files_is_not_complete(tmp_path):
    (tmp_path / "trial").mkdir()
    assert not TrialRecord.is_complete(tmp_path / "trial")


# %% how much closing the loop compensates


def _transfer_record(
    loop_mode: LoopMode,
    seed: int,
    delivered: int,
    receiver: ReceiverCup = ReceiverCup.WIDE,
) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.STUDY: Study.TRANSFER,
            TrialKey.SEED: seed,
            TrialKey.SOURCE: SourceCup.CUP,
            TrialKey.RECEIVER: receiver,
            TrialKey.PERTURBATION: BASELINE.to_json(),
            TrialKey.LOOP_MODE: loop_mode,
            TrialKey.OUTCOME: {
                Outcome.GOAL_GRAINS: 40.0,
                Outcome.DELIVERED: delivered,
            },
        },
        samples=pandas.DataFrame(
            {
                Sample.PHASE: [Phase.TRANSFER],
                Sample.MODEL_RECEIVER_FILL: [0.06],
                Sample.COMMITTED_RECEIVER_FILL: [0.05],
            }
        ),
    )


def test_compensation_is_the_share_of_the_open_loop_error_removed():
    analysis = SourceAnalysis(
        records=[
            _transfer_record(LoopMode.OPEN_LOOP, 0, 50),
            _transfer_record(LoopMode.OPEN_LOOP, 1, 54),
            _transfer_record(LoopMode.FILL_FEEDBACK, 0, 44),
            _transfer_record(LoopMode.FILL_FEEDBACK, 1, 36),
        ],
        source=SourceCup.CUP,
    )
    compensation = analysis.compensation.set_index(Column.LOOP_MODE)
    open_loop_error = (abs(50 - 40.0) + abs(54 - 40.0)) / 2
    feedback_error = (abs(44 - 40.0) + abs(36 - 40.0)) / 2
    assert compensation.loc[
        LoopMode.FILL_FEEDBACK, Column.COMPENSATION
    ] == pytest.approx(1 - feedback_error / open_loop_error)


def test_compensation_is_kept_apart_per_receiver():
    analysis = SourceAnalysis(
        records=[
            _transfer_record(LoopMode.OPEN_LOOP, 0, 50, ReceiverCup.WIDE),
            _transfer_record(LoopMode.FILL_FEEDBACK, 0, 45, ReceiverCup.WIDE),
            _transfer_record(LoopMode.OPEN_LOOP, 0, 60, ReceiverCup.NARROW),
            _transfer_record(LoopMode.FILL_FEEDBACK, 0, 30, ReceiverCup.NARROW),
        ],
        source=SourceCup.CUP,
    )
    compensation = analysis.compensation.set_index(Column.RECEIVER)
    goal = 40.0
    assert compensation.loc[ReceiverCup.WIDE, Column.COMPENSATION] == pytest.approx(
        1 - abs(45 - goal) / abs(50 - goal)
    )
    assert compensation.loc[ReceiverCup.NARROW, Column.COMPENSATION] == pytest.approx(
        1 - abs(30 - goal) / abs(60 - goal)
    )


def test_only_the_analysed_cup_trials_are_kept():
    spouted = _transfer_record(LoopMode.OPEN_LOOP, 0, 50)
    spouted.description[TrialKey.SOURCE] = SourceCup.SPOUTED_CUP
    analysis = SourceAnalysis(
        records=[_transfer_record(LoopMode.OPEN_LOOP, 0, 50), spouted],
        source=SourceCup.CUP,
    )
    assert analysis.transfer == [analysis.records[0]]


# %% the repose angle the steady turns imply


def test_the_steady_turns_shift_the_fitted_repose_angle_by_their_mean_onset_error():
    fitted, early_by = 0.7, (0.1, 0.2)
    onsets = pandas.DataFrame(
        {
            Column.PARAMETER_SET: [
                ParameterSet.FITTED,
                ParameterSet.FITTED,
                ParameterSet.DEMO,
            ],
            Column.PREDICTED_ONSET: [1.0, 1.1, 5.0],
            Column.MEASURED_ONSET: [1.0 + early_by[0], 1.1 + early_by[1], 0.0],
        }
    )
    assert SourceAnalysis.repose_angle_matching(onsets, fitted) == pytest.approx(
        fitted + sum(early_by) / len(early_by)
    )


SCATTERING = TURNING.with_dynamic_repose_spread(0.04)


def test_the_spread_of_the_turning_repose_angle_survives_a_round_trip_through_json():
    assert ModelParameters.from_json(SCATTERING.to_json()) == SCATTERING


def test_an_error_in_the_repose_angle_moves_the_centre_but_keeps_the_spread():
    perturbation = Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5)
    assert perturbation.around(SCATTERING).dynamic_repose_spread == (
        SCATTERING.dynamic_repose_spread
    )


# %% the drain rate while the cup turns

DRAINING = DEMO_PARAMETERS.with_dynamic_outflow_rate_constant(
    DEMO_PARAMETERS.outflow_rate_constant * 0.5
).with_dynamic_outflow_rate_spread(0.3)


def test_a_turning_cup_drains_at_its_dynamic_rate():
    assert DRAINING.while_turning().outflow_rate_constant == (
        DRAINING.dynamic_outflow_rate_constant
    )


def test_an_error_in_the_drain_rate_scales_the_dynamic_one_and_keeps_its_spread():
    perturbation = Perturbation(
        parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=4.0
    )
    perturbed = perturbation.around(DRAINING)
    assert (
        perturbed.while_turning().outflow_rate_constant,
        perturbed.dynamic_outflow_rate_spread,
    ) == (
        DRAINING.dynamic_outflow_rate_constant * perturbation.factor,
        DRAINING.dynamic_outflow_rate_spread,
    )


def test_the_dynamic_drain_rate_and_its_spread_survive_a_round_trip_through_json():
    assert ModelParameters.from_json(DRAINING.to_json()) == DRAINING
