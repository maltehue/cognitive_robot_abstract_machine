import math

import numpy
import pandas
import pytest

from experiments.pouring_model_verification.bank_replay import LoggedPour
from experiments.pouring_model_verification.hypothesis_bank import (
    OnsetSpread,
    FreeParameter,
    Hypothesis,
    ModelPoint,
    ModelSwitch,
)
from experiments.pouring_model_verification.parameters import (
    BASELINE,
    DEMO_PARAMETERS,
    Baselines,
    ModelParameter,
    Perturbation,
)
from experiments.pouring_model_verification.persistence_study import SEQUENCE_LENGTH
from experiments.pouring_model_verification.records import (
    Phase,
    Sample,
    TrialKey,
    TrialRecord,
)
from experiments.pouring_model_verification.recovery import (
    CAUTION_DRAIN_FACTOR,
    DrainCorrectionValues,
    OnsetWriteBack,
    CautionSettings,
    CautiousStart,
    hypotheses_for,
    injected_hypothesis,
    sensor_knowledge,
)
from experiments.pouring_model_verification.recovery_evaluation import (
    TROUGH_HOLD,
    slot_edge_acts_as_lip,
)
from experiments.pouring_model_verification.recovery_study import (
    COUPLED_PERTURBATIONS,
    BankSequence,
    BankSettings,
    Level,
    coupled_conditions,
    missing_lip_conditions,
    one_parameter_conditions,
)
from experiments.pouring_model_verification.scene import ReceiverCup, SourceCup
from experiments.pouring_model_verification.sensor import (
    CameraCondition,
    SensorCondition,
    SensorSet,
    WristSensor,
)
from experiments.pouring_model_verification.transfer_trial import ModelBlend
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

BASELINES = Baselines(parameters={source: DEMO_PARAMETERS for source in SourceCup})

# %% which hypotheses, and which one is true


def test_only_a_cup_with_a_slot_edge_is_weighed_for_a_second_lip():
    assert Hypothesis.SECOND_LIP in hypotheses_for(SourceCup.SPOUTED_CUP)
    assert Hypothesis.SECOND_LIP not in hypotheses_for(SourceCup.CUP)


def test_with_the_onset_tracked_no_hypothesis_is_named_for_it_alone():
    assert hypotheses_for(SourceCup.CUP, tracks_onset=True) == (
        Hypothesis.NOMINAL,
        Hypothesis.AIM,
        Hypothesis.CATCH_ALL,
    )


def test_the_second_lip_can_be_forced_into_the_bank_of_a_plain_cup():
    assert Hypothesis.SECOND_LIP in hypotheses_for(SourceCup.CUP, True)


@pytest.mark.parametrize(
    "perturbation, hypothesis",
    [
        (BASELINE, Hypothesis.NOMINAL),
        (
            Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5),
            Hypothesis.ONSET,
        ),
        (
            Perturbation(parameter=ModelParameter.DISCHARGE_COEFFICIENT, factor=2.0),
            Hypothesis.AIM,
        ),
        (COUPLED_PERTURBATIONS[0], Hypothesis.COUPLED),
        (
            Perturbation(parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=4.0),
            None,
        ),
    ],
)
def test_the_error_set_into_the_model_names_its_hypothesis(perturbation, hypothesis):
    assert injected_hypothesis(perturbation) is hypothesis


# %% two parameters set wrong at once


def test_a_coupled_perturbation_scales_both_parameters():
    perturbation = COUPLED_PERTURBATIONS[0]
    parameters = perturbation.around(DEMO_PARAMETERS)
    assert (parameters.repose_angle, parameters.discharge_coefficient) == (
        DEMO_PARAMETERS.repose_angle * perturbation.factor,
        DEMO_PARAMETERS.discharge_coefficient * perturbation.second_factor,
    )


def test_a_coupled_perturbation_survives_a_round_trip_through_json():
    perturbation = COUPLED_PERTURBATIONS[1]
    assert Perturbation.from_json(perturbation.to_json()) == perturbation


def test_a_single_perturbation_keeps_the_label_it_was_recorded_under():
    perturbation = Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5)
    assert perturbation.label == "repose_angle_x0.5"


# %% the sensors


def test_a_realistic_set_weighs_with_the_scale_and_the_realistic_wrist():
    assert (SensorSet.NO_CAMERA.receiver, SensorSet.NO_CAMERA.wrist) == (
        SensorCondition.SCALE,
        WristSensor.REALISTIC,
    )


def test_without_a_camera_the_bank_knows_of_no_crossing():
    knowledge = sensor_knowledge(SensorSet.NO_CAMERA, fall_time=0.15)
    assert knowledge.crossing_noise is None


def test_the_bank_knows_the_noisy_camera_by_its_own_noise():
    knowledge = sensor_knowledge(SensorSet.NOISY_CAMERA, fall_time=0.15)
    assert knowledge.crossing_noise == CameraCondition.NOISY.model.noise


def test_with_nothing_under_the_receiver_the_set_reads_the_wrist_and_the_noisy_camera():
    sensors = SensorSet.WRIST_CAMERA
    assert (sensors.receiver, sensors.wrist, sensors.camera) == (
        None,
        WristSensor.REALISTIC,
        CameraCondition.NOISY,
    )


def test_with_nothing_under_the_receiver_the_bank_expects_no_receiver_reading():
    knowledge = sensor_knowledge(SensorSet.WRIST_CAMERA, fall_time=0.15)
    assert knowledge.receiver_noise is None


def test_the_pole_camera_set_reads_the_wrist_and_the_pole_camera_and_no_receiver():
    sensors = SensorSet.WRIST_POLE_CAMERA
    assert (sensors.receiver, sensors.wrist, sensors.camera) == (
        None,
        WristSensor.REALISTIC,
        CameraCondition.POLE,
    )


def test_the_bank_knows_the_pole_camera_by_its_own_noise():
    knowledge = sensor_knowledge(SensorSet.WRIST_POLE_CAMERA, fall_time=0.15)
    assert knowledge.crossing_noise == CameraCondition.POLE.model.noise


def test_the_camera_alone_set_reads_neither_cup_and_the_counting_pole_camera():
    sensors = SensorSet.POLE_CAMERA
    assert (sensors.receiver, sensors.wrist, sensors.camera) == (
        None,
        None,
        CameraCondition.POLE_COUNTING,
    )


def test_the_rough_camera_alone_set_reads_neither_cup_and_the_rough_counting_camera():
    sensors = SensorSet.POLE_CAMERA_ROUGH
    assert (sensors.receiver, sensors.wrist, sensors.camera) == (
        None,
        None,
        CameraCondition.POLE_COUNTING_ROUGH,
    )


def test_with_the_camera_alone_the_bank_weighs_what_the_stream_shows():
    knowledge = sensor_knowledge(SensorSet.POLE_CAMERA, fall_time=0.15)
    camera = CameraCondition.POLE_COUNTING.model
    assert (
        knowledge.receiver_noise,
        knowledge.source_noise,
        knowledge.stream_dropout,
        knowledge.count_noise,
    ) == (None, None, camera.dropout, camera.count_noise)


def test_other_cameras_leave_a_lost_stream_without_meaning():
    knowledge = sensor_knowledge(SensorSet.WRIST_POLE_CAMERA, fall_time=0.15)
    assert (knowledge.stream_dropout, knowledge.count_noise) == (None, None)


# %% the drain corrections a hypothesis calls for

BUILT = ModelPoint(
    repose_angle=0.5, discharge_coefficient=0.3, outflow_rate_constant=2.0
)


def _cup_with_fill_level() -> tuple[World, HasFillLevel]:
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    body = Body(
        name=PrefixedName("cup"),
        collision=ShapeCollection(
            [
                Box(
                    origin=HomogeneousTransformationMatrix(),
                    scale=Scale(0.07, 0.07, 0.16),
                )
            ]
        ),
    )
    with world.modify_world():
        world.add_body(body)
        world.add_connection(FixedConnection(parent=world.root, child=body))
    cup = HasFillLevel(name=PrefixedName("cup"), root=body)
    with world.modify_world():
        world.add_semantic_annotation(cup)
    cup.initialize_fill_level(world=world, initial_fill=1.0)
    return world, cup


def test_the_drain_as_built_needs_no_correction():
    values = DrainCorrectionValues.between(BUILT, BUILT, cautious=False)
    assert values == DrainCorrectionValues(
        repose_correction=0.0, log_drain_scale=0.0, log_discharge_scale=0.0
    )


def test_a_hypothesis_is_corrected_towards_by_its_differences_from_the_drain_as_built():
    point = ModelPoint(
        repose_angle=0.7, discharge_coefficient=0.6, outflow_rate_constant=1.0
    )
    values = DrainCorrectionValues.between(point, BUILT, cautious=False)
    assert (
        values.repose_correction,
        values.log_drain_scale,
        values.log_discharge_scale,
    ) == pytest.approx((0.2, math.log(0.5), math.log(2.0)))


def test_a_cautious_pour_takes_the_drain_as_faster_by_the_caution_factor():
    values = DrainCorrectionValues.between(BUILT, BUILT, cautious=True)
    assert values.log_drain_scale == pytest.approx(math.log(CAUTION_DRAIN_FACTOR))


def test_the_values_are_written_into_the_containers_corrections():
    world, cup = _cup_with_fill_level()
    values = DrainCorrectionValues(
        repose_correction=0.2, log_drain_scale=-0.5, log_discharge_scale=0.3
    )
    values.write_to(world, cup.fill_connection.drain_correction)
    assert [
        connection.position
        for connection in cup.fill_connection.drain_correction.connections_in(world)
    ] == pytest.approx([0.2, -0.5, 0.3])


# %% switching without flipping


def test_a_concept_once_wrong_stays_wrong_on_the_margin_of_the_threshold():
    switch = ModelSwitch(threshold=0.9)
    switch.decide({Hypothesis.NOMINAL: 0.0, Hypothesis.ONSET: 1.0})
    assert (
        switch.decide(
            {
                Hypothesis.NOMINAL: 0.0,
                Hypothesis.ONSET: 0.5,
                Hypothesis.AIM: 0.95,
                Hypothesis.COUPLED: 0.0,
            }
        )
        is Hypothesis.COUPLED
    )


def test_a_switch_waits_for_the_readings_to_call_for_it_again():
    switch = ModelSwitch(threshold=0.9, confirmations=3)
    probabilities = {Hypothesis.NOMINAL: 0.0, Hypothesis.ONSET: 1.0}
    calls = [switch.decide(probabilities) for _ in range(3)]
    assert calls == [None, None, Hypothesis.ONSET]


def test_a_switch_moves_the_model_part_of_the_way_at_once():
    start = ModelPoint(
        repose_angle=0.5, discharge_coefficient=0.3, outflow_rate_constant=1.0
    )
    target = start.with_values({FreeParameter.REPOSE_ANGLE: 1.0})
    blend = ModelBlend(point=start, time=0.0)
    moved = blend.towards(target, at=0.1)
    assert start.repose_angle < moved.repose_angle < target.repose_angle


def test_a_slot_edge_switched_on_opens_gradually():
    start = ModelPoint(
        repose_angle=0.5, discharge_coefficient=0.3, outflow_rate_constant=1.0
    )
    blend = ModelBlend(point=start, time=0.0)
    blend.towards(start.with_values({FreeParameter.SLOT_HEIGHT: 0.02}), at=0.1)
    assert 0.0 < blend.slot_open < 1.0


# %% the conditions


def test_every_level_has_its_conditions():
    levels = {
        condition.level
        for condition in one_parameter_conditions()
        + coupled_conditions()
        + missing_lip_conditions()
    }
    assert levels == set(Level)


def test_a_wrong_discharge_coefficient_is_also_poured_into_the_narrow_receiver():
    narrow = {
        condition.perturbation.parameter
        for condition in one_parameter_conditions()
        if condition.scene.receiver is ReceiverCup.NARROW
    }
    assert narrow == {ModelParameter.DISCHARGE_COEFFICIENT}


# %% sequences that write back


def _record(writes_back: bool, final: ModelPoint) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.WRITES_BACK: writes_back,
            TrialKey.FINAL_MODEL: final.to_json(),
        },
        samples=pandas.DataFrame(),
    )


LEARNED = ModelPoint(
    repose_angle=1.0, discharge_coefficient=0.3, outflow_rate_constant=0.3
)


def _sequence() -> BankSequence:
    return BankSequence(
        condition=missing_lip_conditions()[0],
        sensors=SensorSet.IDEAL,
        sequence_seed=4,
        settings=BankSettings(),
    )


def test_the_next_pour_starts_from_the_model_the_previous_one_wrote_back():
    sequence = _sequence()
    first = sequence.trial(1, BASELINES, None)
    following = sequence.ready_trial(BASELINES, {first.name: _record(True, LEARNED)})
    assert following.kept_model == LEARNED


def test_a_pour_that_writes_nothing_back_leaves_the_next_on_the_model_as_given():
    sequence = _sequence()
    first = sequence.trial(1, BASELINES, None)
    following = sequence.ready_trial(BASELINES, {first.name: _record(False, LEARNED)})
    assert following.kept_model is None


def test_a_poured_sequence_has_nothing_left_to_run():
    sequence = _sequence()
    records, kept = {}, None
    for pour in range(1, SEQUENCE_LENGTH + 1):
        trial = sequence.trial(pour, BASELINES, kept)
        records[trial.name] = _record(True, LEARNED)
        kept = LEARNED
    assert sequence.ready_trial(BASELINES, records) is None


# %% where the spouted cup truly starts to pour


def _spout_samples(spout, source):
    return pandas.DataFrame(
        {
            Sample.TIME: [0.1 * step for step in range(len(spout))],
            Sample.PHASE: Phase.TRANSFER,
            Sample.GRAINS_IN_SPOUT: spout,
            Sample.GRAINS_IN_SOURCE: source,
        }
    )


def test_grains_held_in_the_trough_before_any_leave_make_the_slot_edge_a_lip():
    steps = int(TROUGH_HOLD / 0.1) + 5
    samples = _spout_samples([3] * (steps + 5), [100] * steps + [99] * 5)
    assert slot_edge_acts_as_lip(samples) is True


def test_grains_that_leave_as_soon_as_they_reach_the_trough_leave_the_outlet_the_lip():
    samples = _spout_samples([0, 2, 2, 2], [100, 99, 98, 97])
    assert slot_edge_acts_as_lip(samples) is False


# %% replaying a pour on the repose angle held while turning


def test_a_logged_pour_replays_its_model_on_the_repose_angle_held_while_turning():
    baseline = DEMO_PARAMETERS.with_dynamic_repose_angle(0.9)
    perturbation = Perturbation(parameter=ModelParameter.REPOSE_ANGLE, factor=0.5)
    record = TrialRecord(
        description={
            TrialKey.PERTURBATION: perturbation.to_json(),
            TrialKey.PARAMETERS: perturbation.around(DEMO_PARAMETERS).to_json(),
        },
        samples=pandas.DataFrame({Sample.PHASE: [Phase.TRANSFER]}),
    )
    pour = LoggedPour(record=record, baseline=baseline)
    assert (pour.given.repose_angle, pour.true_point.repose_angle) == (
        perturbation.around(baseline).while_turning().repose_angle,
        baseline.dynamic_repose_angle,
    )


def test_a_logged_pour_replays_its_model_on_the_drain_rate_held_while_turning():
    baseline = DEMO_PARAMETERS.with_dynamic_outflow_rate_constant(0.1)
    perturbation = Perturbation(
        parameter=ModelParameter.OUTFLOW_RATE_CONSTANT, factor=4.0
    )
    record = TrialRecord(
        description={
            TrialKey.PERTURBATION: perturbation.to_json(),
            TrialKey.PARAMETERS: perturbation.around(DEMO_PARAMETERS).to_json(),
        },
        samples=pandas.DataFrame({Sample.PHASE: [Phase.TRANSFER]}),
    )
    pour = LoggedPour(record=record, baseline=baseline)
    assert (
        pour.given.outflow_rate_constant,
        pour.true_point.outflow_rate_constant,
    ) == (
        perturbation.around(baseline).while_turning().outflow_rate_constant,
        baseline.dynamic_outflow_rate_constant,
    )


# %% a cautious start while the aim is unconfirmed

CAUTION = CautionSettings(cap_fraction=0.5, release_time=0.5)
OPENING = 0.03


def _inside_readings(start: float, count: int, step: float = 0.1):
    return [(start + index * step, OPENING / 2) for index in range(count)]


def test_a_pour_starts_with_its_tilt_speed_capped():
    caution = CautiousStart(settings=CAUTION, opening_radius=OPENING)
    assert caution.observe(0.0, None, aim_confirmed=False) is True


def test_a_confirmed_aim_releases_the_cap():
    caution = CautiousStart(settings=CAUTION, opening_radius=OPENING)
    assert caution.observe(0.0, None, aim_confirmed=True) is False


def test_the_stream_seen_inside_for_the_release_time_releases_the_cap():
    caution = CautiousStart(settings=CAUTION, opening_radius=OPENING)
    steps = round(CAUTION.release_time / 0.1) + 1
    capped = [
        caution.observe(at, crossing, aim_confirmed=False)
        for at, crossing in _inside_readings(0.0, steps)
    ]
    assert capped == [True] * (steps - 1) + [False]


def test_a_stream_seen_outside_starts_the_count_again():
    caution = CautiousStart(settings=CAUTION, opening_radius=OPENING)
    for at, crossing in _inside_readings(0.0, 3):
        caution.observe(at, crossing, aim_confirmed=False)
    caution.observe(0.3, 2 * OPENING, aim_confirmed=False)
    assert caution.observe(0.4, OPENING / 2, aim_confirmed=False) is True


def test_a_released_cap_stays_released():
    caution = CautiousStart(settings=CAUTION, opening_radius=OPENING)
    caution.observe(0.0, None, aim_confirmed=True)
    assert caution.observe(0.1, 2 * OPENING, aim_confirmed=False) is False


def test_caution_settings_name_the_pour_they_run_in():
    assert CAUTION.label == "caution0.5_0.5s"


# %% writing a new onset back only after repeated evidence

TWIN_ONSET = OnsetSpread(centre=0.9, spread=0.04)
HIGH = TWIN_ONSET.centre + 4 * TWIN_ONSET.spread
LOW = TWIN_ONSET.centre - 4 * TWIN_ONSET.spread


def _after(estimates, explained=True):
    state = OnsetWriteBack(onset=TWIN_ONSET)
    for estimate in estimates:
        state = state.after(estimate, explained=explained)
    return state


def test_a_pour_inside_the_interval_writes_nothing_back():
    state = _after([TWIN_ONSET.centre])
    assert (state.onset, state.run, state.written) == (TWIN_ONSET, (), False)


def test_two_pours_outside_on_the_same_side_are_not_enough():
    state = _after([HIGH, HIGH])
    assert (state.onset, state.written) == (TWIN_ONSET, False)


def test_three_pours_outside_on_the_same_side_move_the_centre_to_theirs():
    estimates = [HIGH, HIGH + 0.01, HIGH + 0.02]
    state = _after(estimates)
    assert state.written is True
    assert state.onset.centre == pytest.approx(numpy.mean(estimates))
    assert state.run == ()


def test_the_spread_written_back_is_never_narrower_than_the_twins():
    state = _after([HIGH, HIGH, HIGH])
    assert state.onset.spread == TWIN_ONSET.spread


def test_a_pour_on_the_other_side_starts_the_count_again():
    state = _after([HIGH, HIGH, LOW])
    assert (state.run, state.written) == ((LOW,), False)


def test_an_explanation_that_does_not_fit_the_pour_is_never_written_back():
    state = _after([HIGH, HIGH, HIGH], explained=False)
    assert (state.onset, state.written) == (TWIN_ONSET, False)


# %% a sequence that writes the onset back

ADAPTING = BankSettings(label="adapting", adapts_onset=True)
SCATTERING_BASELINES = Baselines(
    parameters={
        source: DEMO_PARAMETERS.with_dynamic_repose_angle(
            0.9
        ).with_dynamic_repose_spread(0.04)
        for source in SourceCup
    }
)


def _onset_record(estimate: float) -> TrialRecord:
    return TrialRecord(
        description={
            TrialKey.WRITES_BACK: False,
            TrialKey.FINAL_MODEL: LEARNED.to_json(),
            TrialKey.ONSET_ESTIMATE: estimate,
            TrialKey.CONSISTENCY: None,
        },
        samples=pandas.DataFrame(),
    )


def _onset_sequence() -> BankSequence:
    return BankSequence(
        condition=missing_lip_conditions()[0],
        sensors=SensorSet.IDEAL,
        sequence_seed=4,
        settings=ADAPTING,
    )


def test_a_sequence_moves_the_twins_onset_after_three_pours_outside_it():
    sequence = _onset_sequence()
    records = {}
    for pour in range(1, 4):
        trial = sequence.ready_trial(SCATTERING_BASELINES, records)
        records[trial.name] = _onset_record(0.9 + 0.2)
    following = sequence.ready_trial(SCATTERING_BASELINES, records)
    assert following.sequence.kept_onset.centre == pytest.approx(1.1)


def test_a_sequence_keeps_the_twins_onset_while_pours_stay_inside_it():
    sequence = _onset_sequence()
    records = {}
    for pour in range(1, 4):
        trial = sequence.ready_trial(SCATTERING_BASELINES, records)
        records[trial.name] = _onset_record(0.9)
    assert (
        sequence.ready_trial(SCATTERING_BASELINES, records).sequence.kept_onset is None
    )
