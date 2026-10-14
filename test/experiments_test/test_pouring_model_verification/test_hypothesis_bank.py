import math
from dataclasses import replace

from typing_extensions import Optional

import numpy
import pytest

from experiments.pouring_model_verification.recovery import (
    BankCoupling,
    hypotheses_for,
)
from experiments.pouring_model_verification.scene import SourceCup
from experiments.pouring_model_verification.transfer_trial import BankSteering

from experiments.pouring_model_verification.hypothesis_bank import (
    INTERVAL_SPREADS,
    PARAMETER_ORDER,
    DrainRateSpread,
    OnsetSpread,
    BankSettings,
    DrainGeometry,
    FreeParameter,
    Hypothesis,
    HypothesisBank,
    ModelPoint,
    ModelSwitch,
    Reading,
    SensorKnowledge,
    StreamGeometry,
)

GEOMETRY = DrainGeometry(
    container_height=0.06, lip_offset=0.039, grains_per_fill=150.0, slot_offset=0.039
)
GIVEN = ModelPoint(
    repose_angle=0.7, discharge_coefficient=0.3, outflow_rate_constant=2.0
)
EXACT_SENSORS = SensorKnowledge(
    receiver_noise=0.0,
    receiver_delay=0.0,
    counts_falling_grains=True,
    fall_time=0.15,
    source_noise=0.0,
    source_delay=0.0,
    crossing_noise=None,
)


def _parameters(point: ModelPoint) -> numpy.ndarray:
    return numpy.array([[point.value_of(parameter) for parameter in PARAMETER_ORDER]])


# %% the hypotheses


def test_each_hypothesis_frees_the_concepts_it_is_named_for():
    assert {hypothesis: hypothesis.concepts for hypothesis in Hypothesis} == {
        Hypothesis.NOMINAL: frozenset(),
        Hypothesis.ONSET: {FreeParameter.REPOSE_ANGLE},
        Hypothesis.AIM: {FreeParameter.DISCHARGE_COEFFICIENT},
        Hypothesis.COUPLED: {
            FreeParameter.REPOSE_ANGLE,
            FreeParameter.DISCHARGE_COEFFICIENT,
        },
        Hypothesis.SECOND_LIP: {FreeParameter.SLOT_HEIGHT, FreeParameter.REPOSE_ANGLE},
        Hypothesis.CATCH_ALL: frozenset(),
        Hypothesis.OUTFLOW_CAPACITY: {FreeParameter.OUTFLOW_RATE_CONSTANT},
    }


def test_the_drain_rate_is_freed_only_where_asked_and_never_counts_as_a_concept():
    freed = Hypothesis.ONSET.free_parameters(frees_rate=True)
    assert FreeParameter.OUTFLOW_RATE_CONSTANT in freed
    assert FreeParameter.OUTFLOW_RATE_CONSTANT not in Hypothesis.ONSET.concepts
    assert Hypothesis.NOMINAL.free_parameters(frees_rate=True) == ()


def test_hypotheses_are_labelled_in_the_order_the_report_numbers_them():
    assert [hypothesis.label for hypothesis in Hypothesis] == [
        f"H{index}" for index in range(len(Hypothesis))
    ]


# %% the drain


def test_no_head_before_the_onset_the_model_gives():
    grains = 100.0
    onset = GEOMETRY.onset_tilt(grains, GIVEN)
    below = GEOMETRY.head(onset - 0.01, numpy.array([grains]), _parameters(GIVEN))
    above = GEOMETRY.head(onset + 0.01, numpy.array([grains]), _parameters(GIVEN))
    assert below[0] == 0.0 and above[0] > 0.0


def test_a_lower_slot_edge_raises_the_head_and_lowers_the_onset():
    grains = 60.0
    slotted = ModelPoint(
        repose_angle=GIVEN.repose_angle,
        discharge_coefficient=GIVEN.discharge_coefficient,
        outflow_rate_constant=GIVEN.outflow_rate_constant,
        slot_height=0.02,
    )
    tilt = GEOMETRY.onset_tilt(grains, GIVEN) + 0.05
    plain = GEOMETRY.head(tilt, numpy.array([grains]), _parameters(GIVEN))[0]
    with_slot = GEOMETRY.head(tilt, numpy.array([grains]), _parameters(slotted))[0]
    assert with_slot > plain
    assert GEOMETRY.onset_tilt(grains, slotted) < GEOMETRY.onset_tilt(grains, GIVEN)


# %% the stream


STREAM = StreamGeometry(
    horizontal_share=1.0,
    vertical_share=0.0,
    drop=0.11,
    centre_distance=0.05,
    opening_radius=0.03,
)


def test_a_stream_thrown_onto_the_centre_lands_inside():
    speed = STREAM.centre_distance / math.sqrt(2 * STREAM.drop / 9.81)
    assert STREAM.inside_share(numpy.array([speed]), 0.005)[0] == pytest.approx(1.0)


def test_a_stream_thrown_far_past_the_opening_lands_outside():
    assert STREAM.inside_share(numpy.array([2.0]), 0.005)[0] == pytest.approx(0.0)


# %% the switch


def test_two_hypotheses_sharing_a_concept_switch_to_the_simpler_one():
    switch = ModelSwitch(threshold=0.9)
    chosen = switch.decide(
        {
            Hypothesis.NOMINAL: 0.0,
            Hypothesis.ONSET: 0.5,
            Hypothesis.AIM: 0.0,
            Hypothesis.COUPLED: 0.5,
            Hypothesis.CATCH_ALL: 0.0,
        }
    )
    assert chosen is Hypothesis.ONSET


def test_both_concepts_probably_wrong_switch_to_the_coupled_hypothesis():
    switch = ModelSwitch(threshold=0.9)
    chosen = switch.decide(
        {
            Hypothesis.NOMINAL: 0.0,
            Hypothesis.ONSET: 0.05,
            Hypothesis.AIM: 0.0,
            Hypothesis.COUPLED: 0.95,
            Hypothesis.CATCH_ALL: 0.0,
        }
    )
    assert chosen is Hypothesis.COUPLED


def test_nothing_probable_enough_keeps_the_controller_where_it_is():
    switch = ModelSwitch(threshold=0.9)
    assert (
        switch.decide(
            {
                Hypothesis.NOMINAL: 0.4,
                Hypothesis.ONSET: 0.4,
                Hypothesis.CATCH_ALL: 0.2,
            }
        )
        is None
    )
    assert switch.active is Hypothesis.NOMINAL


def test_the_catch_all_wins_over_every_concept_when_probable_enough():
    switch = ModelSwitch(threshold=0.9)
    assert (
        switch.decide({Hypothesis.ONSET: 0.05, Hypothesis.CATCH_ALL: 0.95})
        is Hypothesis.CATCH_ALL
    )


def test_a_hypothesis_the_readings_rule_out_hands_over_to_the_most_probable_one():
    """
    The shape of the 2026-10-14 overpour: the burst rules the late onset out, the
    nominal model and the aim share what is left, and neither reaches the threshold.
    """
    switch = ModelSwitch(threshold=0.9, active=Hypothesis.ONSET)
    chosen = switch.decide(
        {
            Hypothesis.NOMINAL: 0.47,
            Hypothesis.ONSET: 0.0,
            Hypothesis.AIM: 0.52,
            Hypothesis.COUPLED: 0.0,
            Hypothesis.CATCH_ALL: 0.01,
        }
    )
    assert chosen is Hypothesis.AIM


def test_a_hypothesis_still_in_play_keeps_the_controller_on_it():
    switch = ModelSwitch(threshold=0.9, active=Hypothesis.ONSET)
    assert (
        switch.decide(
            {
                Hypothesis.NOMINAL: 0.4,
                Hypothesis.ONSET: 0.2,
                Hypothesis.AIM: 0.4,
                Hypothesis.CATCH_ALL: 0.0,
            }
        )
        is None
    )
    assert switch.active is Hypothesis.ONSET


# %% the onset tracked by every hypothesis


def _bank_tracking_the_onset() -> HypothesisBank:
    return HypothesisBank(
        hypotheses=hypotheses_for(SourceCup.CUP, tracks_onset=True),
        geometry=GEOMETRY,
        base=GIVEN,
        sensors=EXACT_SENSORS,
        settings=BankSettings(tracks_onset=True),
        seed=0,
    )


def test_with_the_onset_tracked_every_hypothesis_estimates_it():
    bank = _bank_tracking_the_onset()
    assert all(
        FreeParameter.REPOSE_ANGLE in bank.filters[hypothesis].free
        for hypothesis in bank.hypotheses
    )


def test_a_tracked_onset_drifts_with_the_readings():
    bank = _bank_tracking_the_onset()
    assert not any(
        bank.filters[hypothesis].keeps_onset_static for hypothesis in bank.hypotheses
    )


# %% the bank


def _poured_readings(
    truth: ModelPoint, tilt_rate: float = 0.05, count: int = 80
) -> list:
    """
    Readings of a pour the drain with the true parameters makes, read exactly.
    """
    grains, received, tilt, step = 100.0, 0.0, 0.9, 0.1
    readings = []
    for index in range(count):
        readings.append(
            Reading(
                time=index * step,
                tilt=tilt,
                receiver_grains=received,
                source_grains=grains,
            )
        )
        head = GEOMETRY.head(tilt, numpy.array([grains]), _parameters(truth))
        flow = min(grains, GEOMETRY.outflow(head, _parameters(truth))[0] * step)
        grains, received = grains - flow, received + flow
        tilt += tilt_rate * step
    return readings


def _bank(
    onset: Optional[OnsetSpread] = None,
    drain_rate: Optional[DrainRateSpread] = None,
    hypotheses: tuple = (
        Hypothesis.NOMINAL,
        Hypothesis.ONSET,
        Hypothesis.AIM,
        Hypothesis.COUPLED,
        Hypothesis.CATCH_ALL,
    ),
    sensors: SensorKnowledge = EXACT_SENSORS,
) -> HypothesisBank:
    return HypothesisBank(
        hypotheses=hypotheses,
        geometry=GEOMETRY,
        base=GIVEN,
        sensors=sensors,
        settings=BankSettings(),
        seed=0,
        onset=onset,
        drain_rate=drain_rate,
    )


def test_a_pour_that_starts_later_than_the_model_says_is_blamed_on_the_onset():
    bank = _bank()
    for reading in _poured_readings(
        GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.25})
    ):
        bank.observe(reading)
    concepts = ModelSwitch.concept_probabilities(bank.probabilities)
    assert concepts[FreeParameter.REPOSE_ANGLE] > 0.9


def test_the_onset_hypothesis_estimates_the_repose_angle_the_pour_had():
    bank = _bank()
    true_angle = GIVEN.repose_angle + 0.25
    for reading in _poured_readings(
        GIVEN.with_values({FreeParameter.REPOSE_ANGLE: true_angle})
    ):
        bank.observe(reading)
    estimate = bank.filters[Hypothesis.ONSET].mean().repose_angle
    assert estimate == pytest.approx(true_angle, abs=0.05)


def test_a_pour_as_the_model_says_leaves_the_nominal_hypothesis_alive():
    bank = _bank()
    for reading in _poured_readings(GIVEN):
        bank.observe(reading)
    assert bank.probabilities[Hypothesis.NOMINAL] > 0.1


def test_every_hypothesis_keeps_a_share_however_badly_it_fits():
    bank = _bank()
    for reading in _poured_readings(
        GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.25})
    ):
        bank.observe(reading)
    assert min(bank.probabilities.values()) > 0.0


def test_the_settings_survive_a_round_trip_through_json():
    settings = BankSettings(label="tuned", particles=300, commit_threshold=0.8)
    assert BankSettings.from_json(settings.to_json()) == settings


def test_a_model_point_survives_a_round_trip_through_json():
    point = GIVEN.with_values({FreeParameter.SLOT_HEIGHT: 0.02})
    assert ModelPoint.from_json(point.to_json()) == point


# %% a pour with nothing under the receiver

UNWEIGHED_RECEIVER_SENSORS = replace(EXACT_SENSORS, receiver_noise=None)

OFF_TARGET_STREAM = replace(STREAM, centre_distance=1.0)
"""
A receiver so far from the lip that no stream the model throws reaches it.
"""


def _unweighed(readings: list, stream: Optional[StreamGeometry] = None) -> list:
    """
    The same readings as a robot with nothing under the receiver takes them.
    """
    return [
        replace(reading, receiver_grains=None, stream=stream) for reading in readings
    ]


def test_without_a_receiver_reading_a_late_onset_is_blamed_on_the_onset():
    bank = _bank(sensors=UNWEIGHED_RECEIVER_SENSORS)
    late = GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.25})
    for reading in _unweighed(_poured_readings(late)):
        bank.observe(reading)
    concepts = ModelSwitch.concept_probabilities(bank.probabilities)
    assert concepts[FreeParameter.REPOSE_ANGLE] > 0.9


def test_without_a_receiver_reading_the_bank_estimates_what_the_receiver_holds():
    readings = _poured_readings(GIVEN)
    coupling = BankCoupling(bank=_bank(sensors=UNWEIGHED_RECEIVER_SENSORS))
    for reading in _unweighed(readings):
        coupling.observe(reading, poured_share=0.0)
    assert coupling.received_grains == pytest.approx(
        readings[-1].receiver_grains, abs=1.0
    )


def test_grains_the_stream_throws_past_the_receiver_are_not_counted_as_received():
    readings = _poured_readings(GIVEN)
    coupling = BankCoupling(bank=_bank(sensors=UNWEIGHED_RECEIVER_SENSORS))
    for reading in _unweighed(readings, stream=OFF_TARGET_STREAM):
        coupling.observe(reading, poured_share=0.0)
    assert readings[-1].receiver_grains > 10.0
    assert coupling.received_grains == pytest.approx(0.0, abs=1.0)


# %% a pour only a camera watches

CAMERA_ONLY_SENSORS = replace(
    EXACT_SENSORS,
    receiver_noise=None,
    source_noise=None,
    stream_dropout=0.05,
    count_noise=0.2,
)


def _as_the_camera_sees(
    readings: list, counts_grains: bool = True, clump: int = 1
) -> list:
    """
    The same pour as a camera alone sees it: the stream there or not, and how many
    grains it has counted crossing the opening, a clump at a time; the cups weighed only
    at the start.
    """
    seen, counted = [], 0.0
    for index, reading in enumerate(readings):
        following = readings[min(index + 1, len(readings) - 1)]
        flowing = following.receiver_grains > reading.receiver_grains
        if index % clump == 0:
            counted = reading.receiver_grains
        seen.append(
            replace(
                reading,
                receiver_grains=None,
                source_grains=reading.source_grains if index == 0 else None,
                stream_seen=flowing,
                counted_grains=counted if counts_grains else None,
            )
        )
    return seen


LATE_ONSET = GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.25})
FLOWING = dict(tilt_rate=0.1, count=80)
"""
A pour that turns fast enough for the stream to carry 5-15 grains per second.
"""


def test_with_the_camera_alone_a_late_onset_is_blamed_on_the_onset():
    bank = _bank(sensors=CAMERA_ONLY_SENSORS)
    for reading in _as_the_camera_sees(_poured_readings(LATE_ONSET)):
        bank.observe(reading)
    concepts = ModelSwitch.concept_probabilities(bank.probabilities)
    assert concepts[FreeParameter.REPOSE_ANGLE] > 0.9


def test_a_stream_not_seen_where_the_model_pours_is_blamed_on_the_onset():
    bank = _bank(sensors=replace(CAMERA_ONLY_SENSORS, count_noise=None))
    readings = _as_the_camera_sees(_poured_readings(LATE_ONSET), counts_grains=False)
    for reading in readings:
        bank.observe(reading)
    concepts = ModelSwitch.concept_probabilities(bank.probabilities)
    assert concepts[FreeParameter.REPOSE_ANGLE] > 0.9


def test_with_the_camera_alone_the_bank_estimates_what_both_cups_hold():
    readings = _poured_readings(GIVEN, **FLOWING)
    coupling = BankCoupling(bank=_bank(sensors=CAMERA_ONLY_SENSORS))
    for reading in _as_the_camera_sees(readings):
        coupling.observe(reading, poured_share=0.0)
    assert (coupling.received_grains, coupling.source_grains) == (
        pytest.approx(readings[-1].receiver_grains, abs=3.0),
        pytest.approx(readings[-1].source_grains, abs=3.0),
    )


def test_a_pour_as_the_model_says_counted_in_clumps_is_not_blamed_on_anything():
    bank = _bank(sensors=CAMERA_ONLY_SENSORS)
    for reading in _as_the_camera_sees(_poured_readings(GIVEN, **FLOWING), clump=5):
        bank.observe(reading)
    assert bank.probabilities[Hypothesis.CATCH_ALL] < 0.5


# %% a pour that ends before the first reading


def test_a_bank_that_saw_no_reading_reports_the_model_as_given():
    coupling = BankCoupling(bank=_bank())
    assert (coupling.point, coupling.summary()[Hypothesis.ONSET]["mean"]) == (
        GIVEN,
        GIVEN.to_json(),
    )


# %% a repose angle that varies from pour to pour

SPREAD = OnsetSpread(centre=GIVEN.repose_angle, spread=0.04)


def test_the_twins_interval_holds_the_centre_and_not_what_lies_three_spreads_off():
    low, high = SPREAD.interval
    assert low < SPREAD.centre < high
    assert not low < SPREAD.centre + 3 * SPREAD.spread < high


def test_the_share_of_a_posterior_inside_the_interval_counts_its_weights():
    inside, outside = SPREAD.centre, SPREAD.centre + 5 * SPREAD.spread
    share = SPREAD.share_inside(
        numpy.array([inside, outside]), numpy.array([0.25, 0.75])
    )
    assert share == pytest.approx(0.25)


def test_with_an_onset_spread_the_nominal_hypothesis_adapts_its_repose_angle():
    bank = _bank(onset=SPREAD)
    assert bank.filters[Hypothesis.NOMINAL].free == (FreeParameter.REPOSE_ANGLE,)


def test_adapting_the_repose_angle_holds_no_concept_wrong():
    assert Hypothesis.NOMINAL.concepts == frozenset()


def test_without_an_onset_spread_the_nominal_hypothesis_leaves_nothing_free():
    assert _bank().filters[Hypothesis.NOMINAL].free == ()


def _coupled_pour(truth: ModelPoint, onset: Optional[OnsetSpread]):
    coupling = BankCoupling(bank=_bank(onset=onset))
    for reading in _poured_readings(truth):
        coupling.observe(reading, poured_share=0.0)
    return coupling


def test_a_pour_inside_the_twins_spread_is_adapted_to_rather_than_blamed():
    truth = GIVEN.with_values(
        {FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + SPREAD.spread}
    )
    assert _coupled_pour(truth, SPREAD).active is Hypothesis.NOMINAL


def test_a_pour_far_outside_the_twins_spread_is_blamed_on_the_onset():
    truth = GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.25})
    assert _coupled_pour(truth, SPREAD).active is Hypothesis.ONSET


def test_a_vetoed_concept_is_not_declared_wrong():
    switch = ModelSwitch(threshold=0.9)
    probabilities = {Hypothesis.NOMINAL: 0.0, Hypothesis.ONSET: 1.0}
    assert (
        switch.decide(probabilities, vetoed=frozenset({FreeParameter.REPOSE_ANGLE}))
        is None
    )


def test_the_outflow_capacity_hypothesis_is_weighed_only_where_asked():
    assert Hypothesis.OUTFLOW_CAPACITY not in hypotheses_for(SourceCup.CUP)
    assert Hypothesis.OUTFLOW_CAPACITY in hypotheses_for(
        SourceCup.CUP, outflow_capacity=True
    )


def test_the_repose_angle_adapted_to_stays_within_the_twins_spread_through_a_pour():
    truth = GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.25})
    bank = _bank(onset=SPREAD)
    for reading in _poured_readings(truth, tilt_rate=0.01, count=500):
        bank.observe(reading)
    angles = bank.filters[Hypothesis.NOMINAL].parameters[:, 0]
    assert angles.max() < SPREAD.centre + 5 * SPREAD.spread


@pytest.mark.parametrize(
    "hypothesis, adapts",
    [
        (Hypothesis.NOMINAL, True),
        (Hypothesis.AIM, True),
        (Hypothesis.OUTFLOW_CAPACITY, True),
        (Hypothesis.CATCH_ALL, True),
        (Hypothesis.ONSET, False),
        (Hypothesis.COUPLED, False),
        (Hypothesis.SECOND_LIP, False),
    ],
)
def test_every_hypothesis_that_keeps_the_onset_draws_it_from_the_twins_spread(
    hypothesis, adapts
):
    assert hypothesis.adapts_onset is adapts


# %% a drain rate that varies from pour to pour

RATE_SPREAD = DrainRateSpread(centre=GIVEN.outflow_rate_constant, spread=0.15)


def test_the_twins_drain_rate_interval_is_a_band_of_factors_around_the_centre():
    factor = math.exp(INTERVAL_SPREADS * RATE_SPREAD.spread)
    assert RATE_SPREAD.interval == pytest.approx(
        (RATE_SPREAD.centre / factor, RATE_SPREAD.centre * factor)
    )


def test_the_share_of_a_drain_rate_posterior_inside_the_interval_counts_its_weights():
    inside, outside = RATE_SPREAD.centre, RATE_SPREAD.centre * 5.0
    share = RATE_SPREAD.share_inside(
        numpy.array([inside, outside]), numpy.array([0.25, 0.75])
    )
    assert share == pytest.approx(0.25)


def test_with_a_drain_rate_spread_the_nominal_hypothesis_adapts_its_drain_rate():
    bank = _bank(drain_rate=RATE_SPREAD)
    assert bank.filters[Hypothesis.NOMINAL].free == (
        FreeParameter.OUTFLOW_RATE_CONSTANT,
    )


@pytest.mark.parametrize(
    "hypothesis, adapts",
    [
        (Hypothesis.NOMINAL, True),
        (Hypothesis.ONSET, True),
        (Hypothesis.AIM, True),
        (Hypothesis.COUPLED, True),
        (Hypothesis.SECOND_LIP, True),
        (Hypothesis.CATCH_ALL, True),
        (Hypothesis.OUTFLOW_CAPACITY, False),
    ],
)
def test_every_hypothesis_that_keeps_the_drain_rate_draws_it_from_the_twins_spread(
    hypothesis, adapts
):
    assert hypothesis.adapts_drain_rate is adapts


def test_a_drain_rate_drawn_from_the_twins_spread_stays_put_within_the_pour():
    bank = _bank(drain_rate=RATE_SPREAD)
    bank.observe(_poured_readings(GIVEN)[0])
    hypothesis_filter = bank.filters[Hypothesis.NOMINAL]
    column = PARAMETER_ORDER.index(FreeParameter.OUTFLOW_RATE_CONSTANT)
    before = hypothesis_filter.parameters[:, column].copy()
    hypothesis_filter._walk()
    assert numpy.array_equal(hypothesis_filter.parameters[:, column], before)


def test_the_drain_rates_drawn_scatter_within_the_twins_spread():
    bank = _bank(drain_rate=RATE_SPREAD)
    bank.observe(_poured_readings(GIVEN)[0])
    column = PARAMETER_ORDER.index(FreeParameter.OUTFLOW_RATE_CONSTANT)
    rates = bank.filters[Hypothesis.NOMINAL].parameters[:, column]
    factor = math.exp(5 * RATE_SPREAD.spread)
    assert (
        rates.min() > RATE_SPREAD.centre / factor
        and rates.max() < RATE_SPREAD.centre * factor
        and rates.std() > 0.0
    )


_WITH_CAPACITY = (
    Hypothesis.NOMINAL,
    Hypothesis.ONSET,
    Hypothesis.OUTFLOW_CAPACITY,
    Hypothesis.CATCH_ALL,
)


def _coupled_drain(truth: ModelPoint) -> BankCoupling:
    coupling = BankCoupling(
        bank=_bank(drain_rate=RATE_SPREAD, hypotheses=_WITH_CAPACITY)
    )
    for reading in _poured_readings(truth):
        coupling.observe(reading, poured_share=0.0)
    return coupling


def test_a_drain_inside_the_twins_spread_is_adapted_to_rather_than_blamed():
    truth = GIVEN.with_values(
        {
            FreeParameter.OUTFLOW_RATE_CONSTANT: GIVEN.outflow_rate_constant
            * math.exp(RATE_SPREAD.spread)
        }
    )
    assert _coupled_drain(truth).active is Hypothesis.NOMINAL


def test_a_drain_far_outside_the_twins_spread_is_blamed_on_the_outflow_capacity():
    truth = GIVEN.with_values(
        {FreeParameter.OUTFLOW_RATE_CONSTANT: GIVEN.outflow_rate_constant * 5.0}
    )
    assert _coupled_drain(truth).active is Hypothesis.OUTFLOW_CAPACITY


def test_the_drain_rate_is_vetoed_while_its_posterior_lies_within_the_twins_spread():
    coupling = _coupled_drain(GIVEN)
    assert FreeParameter.OUTFLOW_RATE_CONSTANT in coupling.vetoed


# %% an onset constant within the pour


@pytest.mark.parametrize("hypothesis", [Hypothesis.ONSET, Hypothesis.COUPLED])
def test_a_static_onset_does_not_drift_between_readings(hypothesis):
    bank = HypothesisBank(
        hypotheses=(hypothesis,),
        geometry=GEOMETRY,
        base=GIVEN,
        sensors=EXACT_SENSORS,
        settings=BankSettings(static_onset=True),
        seed=0,
    )
    readings = _poured_readings(GIVEN)
    bank.observe(readings[0])
    hypothesis_filter = bank.filters[hypothesis]
    before = hypothesis_filter.parameters[:, 0].copy()
    hypothesis_filter._walk()
    assert numpy.array_equal(hypothesis_filter.parameters[:, 0], before)


def test_by_default_the_onset_hypothesis_lets_its_repose_angle_drift():
    bank = _bank()
    bank.observe(_poured_readings(GIVEN)[0])
    hypothesis_filter = bank.filters[Hypothesis.ONSET]
    before = hypothesis_filter.parameters[:, 0].copy()
    hypothesis_filter._walk()
    assert not numpy.array_equal(hypothesis_filter.parameters[:, 0], before)


# %% the controller's model at the first reading


def test_the_first_reading_starts_the_switch_from_the_model_the_controller_ran_on():
    coupling = BankCoupling(bank=_bank(onset=SPREAD))
    readings = _poured_readings(GIVEN)
    for reading in readings[:2]:
        coupling.observe(reading, poured_share=0.0)
    steering = BankSteering(
        motion=None,
        coupling=coupling,
        perception=None,
        scene=None,
        camera=None,
        camera_condition=None,
        built=GIVEN,
        opening_radius=0.03,
        random=None,
    )
    assert steering.model_at(readings[1].time) == coupling.bank.base


# %% the share of what left that the receiver got


def test_the_inside_share_is_what_the_particles_committed_of_what_left():
    bank = _bank(sensors=UNWEIGHED_RECEIVER_SENSORS)
    for reading in _unweighed(_poured_readings(GIVEN), OFF_TARGET_STREAM):
        bank.observe(reading)
    for hypothesis_filter in bank.filters.values():
        left = hypothesis_filter.committed + hypothesis_filter.spilled
        share = numpy.where(left > 0.0, hypothesis_filter.committed / left, 1.0)
        assert hypothesis_filter.inside_share == pytest.approx(
            float(numpy.average(share, weights=hypothesis_filter.weights))
        )
        assert hypothesis_filter.inside_share < 1.0


def test_a_filter_that_saw_nothing_leave_takes_everything_to_land_inside():
    bank = _bank(sensors=UNWEIGHED_RECEIVER_SENSORS)
    bank.observe(_unweighed(_poured_readings(GIVEN, count=1))[0])
    assert bank.filters[Hypothesis.NOMINAL].inside_share == 1.0
