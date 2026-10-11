import math

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
    PARAMETER_ORDER,
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


def _bank(onset: Optional[OnsetSpread] = None) -> HypothesisBank:
    return HypothesisBank(
        hypotheses=(
            Hypothesis.NOMINAL,
            Hypothesis.ONSET,
            Hypothesis.AIM,
            Hypothesis.COUPLED,
            Hypothesis.CATCH_ALL,
        ),
        geometry=GEOMETRY,
        base=GIVEN,
        sensors=EXACT_SENSORS,
        settings=BankSettings(),
        seed=0,
        onset=onset,
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
