import math

import pytest

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from experiments.pouring_model_verification.onset_rule import (
    ARRIVAL,
    ARRIVAL_CONFIRMATIONS,
    ARRIVAL_NOISE_MULTIPLE,
    ArrivalWatch,
    FILL_ROW_WEIGHT,
    OnsetRuleKey,
    RevisionCause,
    TwoSidedOverrideError,
    REVISION_MARGIN,
    OnsetOverrideSwitch,
    OnsetResponse,
    OnsetRevision,
    OnsetRule,
    StallWatch,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)

RULE = OnsetRule(response=OnsetResponse.REVISION, patience=2.0, override_velocity=0.1)
GOAL = 40.0
TOLERANCE = 7.0


def _watch() -> StallWatch:
    return StallWatch(rule=RULE, goal=GOAL, tolerance=TOLERANCE)


# %% noticing that nothing comes out


def test_nothing_arriving_for_the_patience_is_a_stall():
    watch = _watch()
    readings = [watch.observe(0.0, at=0.1 * step) for step in range(25)]
    assert readings.index(True) == round(RULE.patience / 0.1)


def test_grains_arriving_restart_the_patience():
    watch = _watch()
    for step in range(15):
        watch.observe(0.0, at=0.1 * step)
    watch.observe(3.0, at=1.5)
    assert not watch.observe(3.0, at=1.5 + RULE.patience - 0.1)
    assert watch.observe(3.0, at=1.5 + RULE.patience)


def test_a_receiver_at_its_goal_is_not_stalled():
    watch = _watch()
    watch.observe(GOAL - TOLERANCE, at=0.0)
    assert not watch.observe(GOAL - TOLERANCE, at=10.0)


def test_noise_below_one_grain_does_not_count_as_arriving():
    watch = _watch()
    stalled = [watch.observe(0.4 * (step % 2), at=0.1 * step) for step in range(25)]
    assert stalled[-1]


# %% the two responses


def test_the_override_runs_exactly_while_the_pour_is_stalled():
    variables = FloatVariableData()
    active = FloatVariable("override_active")
    variables.register_expression(active)
    switch = OnsetOverrideSwitch(active=active, variables=variables)
    switch.respond(stalled=True, at=1.0, tilt=1.0, source_fill=0.7)
    on = variables.get_value(active)
    switch.respond(stalled=False, at=1.1, tilt=1.0, source_fill=0.7)
    assert (on, variables.get_value(active)) == (1.0, 0.0)


def _revision() -> tuple[OnsetRevision, ArticulatedPouringEquation, FloatVariableData]:
    variables = FloatVariableData()
    correction = FloatVariable("repose_correction")
    variables.register_expression(correction)
    variables.set_value(correction, 0.0)
    equation = ArticulatedPouringEquation(
        container_height=0.06, container_width=0.07, repose_angle=0.6
    )
    return (
        OnsetRevision(equation=equation, correction=correction, variables=variables),
        equation,
        variables,
    )


def test_a_revision_puts_the_onset_just_below_the_tilt_nothing_came_out_at():
    revision, equation, variables = _revision()
    tilt, fill = 1.3, 0.7
    revision.respond(stalled=True, at=4.0, tilt=tilt, source_fill=fill)
    onset = float(equation.onset_tilt(fill).evaluate()[0]) + variables.get_value(
        revision.correction
    )
    assert onset == pytest.approx(tilt - REVISION_MARGIN)


def test_every_revision_is_logged_with_the_repose_angle_before_and_after():
    revision, equation, _ = _revision()
    revision.respond(stalled=True, at=4.0, tilt=1.3, source_fill=0.7)
    revision.respond(stalled=True, at=6.0, tilt=1.4, source_fill=0.7)
    first, second = revision.log
    assert (first.time, first.tilt, first.old_repose_angle) == (
        4.0,
        1.3,
        equation.repose_angle,
    )
    assert second.old_repose_angle == pytest.approx(first.new_repose_angle)
    assert second.new_repose_angle == pytest.approx(first.new_repose_angle + 0.1)


def test_a_pour_that_is_not_stalled_is_not_revised():
    revision, _, variables = _revision()
    revision.respond(stalled=False, at=4.0, tilt=1.3, source_fill=0.7)
    assert (revision.log, variables.get_value(revision.correction)) == ([], 0.0)


# %% the rule


def test_a_rule_is_named_after_its_response_and_settings():
    override = OnsetRule(
        response=OnsetResponse.OVERRIDE, patience=2.0, override_velocity=0.1
    )
    assert (RULE.label, override.label) == ("revision_t2", "override_t2_v0.1")


def test_an_override_weighted_above_the_default_names_its_weight():
    override = OnsetRule(
        response=OnsetResponse.OVERRIDE,
        patience=2.0,
        override_velocity=0.1,
        override_weight=FILL_ROW_WEIGHT,
    )
    assert override.label == f"override_t2_v0.1_w{FILL_ROW_WEIGHT:g}"


def test_a_rule_survives_a_json_round_trip():
    override = OnsetRule(
        response=OnsetResponse.OVERRIDE,
        patience=2.0,
        override_velocity=0.1,
        override_weight=FILL_ROW_WEIGHT,
    )
    assert OnsetRule.from_json(override.to_json()) == override


# %% noticing that grains arrived


def _arrival_watch(noise: float = 0.0) -> ArrivalWatch:
    return ArrivalWatch(noise=noise, reference=0.0)


def test_grains_arriving_are_confirmed_after_enough_readings_above_the_evidence():
    watch = _arrival_watch()
    watch.observe(0.0, at=0.0)
    confirmations = [
        watch.observe(ARRIVAL, at=0.1 * step)
        for step in range(1, ARRIVAL_CONFIRMATIONS + 1)
    ]
    assert confirmations == [None] * (ARRIVAL_CONFIRMATIONS - 1) + [0.1]


def test_a_reading_back_below_the_evidence_restarts_the_confirmation():
    watch = _arrival_watch()
    watch.observe(0.0, at=0.0)
    watch.observe(ARRIVAL, at=0.1)
    watch.observe(0.0, at=0.2)
    confirmed = [
        watch.observe(ARRIVAL, at=0.3 + 0.1 * step)
        for step in range(ARRIVAL_CONFIRMATIONS)
    ]
    assert confirmed[-1] == pytest.approx(0.3)


def test_the_evidence_grows_with_the_sensor_noise():
    noise = 2.0
    watch = _arrival_watch(noise)
    assert watch.evidence == ARRIVAL + ARRIVAL_NOISE_MULTIPLE * noise


def test_grains_arriving_are_reported_only_once():
    watch = _arrival_watch()
    watch.observe(0.0, at=0.0)
    readings = [watch.observe(5.0 + step, at=0.1 * step) for step in range(1, 10)]
    assert sum(reading is not None for reading in readings) == 1


# %% lowering the repose angle


def _onset(revision: OnsetRevision, equation, fill: float) -> float:
    return float(
        equation.onset_tilt(fill).evaluate()[0]
    ) + revision.variables.get_value(revision.correction)


def test_grains_arriving_below_the_predicted_onset_lower_it_to_the_tilt_they_left_at():
    revision, equation, _ = _revision()
    fill = 0.7
    tilt = _onset(revision, equation, fill) - 0.1
    assert revision.lower(at=3.0, tilt=tilt, source_fill=fill)
    assert _onset(revision, equation, fill) == pytest.approx(tilt)


def test_grains_arriving_past_the_predicted_onset_leave_it_alone():
    revision, equation, variables = _revision()
    fill = 0.7
    tilt = _onset(revision, equation, fill) + 0.1
    assert not revision.lower(at=3.0, tilt=tilt, source_fill=fill)
    assert (revision.log, variables.get_value(revision.correction)) == ([], 0.0)


def test_a_lowering_is_logged_as_lowering_the_repose_angle():
    revision, equation, _ = _revision()
    fill = 0.7
    revision.lower(
        at=3.0, tilt=_onset(revision, equation, fill) - 0.1, source_fill=fill
    )
    (logged,) = revision.log
    assert logged.new_repose_angle < logged.old_repose_angle


def test_a_revision_starts_from_a_kept_correction():
    revision, equation, variables = _revision()
    kept = 0.15
    variables.set_value(revision.correction, kept)
    assert revision.repose_angle == pytest.approx(equation.repose_angle + kept)


# %% the two-sided rule


def test_a_two_sided_rule_names_that_it_also_lowers():
    two_sided = OnsetRule(response=OnsetResponse.REVISION, patience=2.0, lowers=True)
    assert two_sided.label == f"{RULE.label}_two_sided"


def test_a_two_sided_rule_survives_a_json_round_trip():
    two_sided = OnsetRule(response=OnsetResponse.REVISION, patience=2.0, lowers=True)
    assert OnsetRule.from_json(two_sided.to_json()) == two_sided


def test_a_rule_recorded_before_lowering_existed_reads_as_one_sided():
    data = RULE.to_json()
    del data[OnsetRuleKey.LOWERS]
    assert not OnsetRule.from_json(data).lowers


def test_an_override_cannot_lower():
    with pytest.raises(TwoSidedOverrideError):
        OnsetRule(
            response=OnsetResponse.OVERRIDE,
            patience=2.0,
            override_velocity=0.1,
            lowers=True,
        )


def test_a_concept_aware_revision_may_lower_too():
    rule = OnsetRule(
        response=OnsetResponse.CONCEPT_AWARE_REVISION, patience=1.0, lowers=True
    )
    assert rule.label == "concept_aware_revision_t1_two_sided"


def test_a_revision_is_logged_with_what_caused_it():
    revision, equation, _ = _revision()
    fill = 0.7
    revision.respond(stalled=True, at=2.0, tilt=1.6, source_fill=fill)
    revision.lower(
        at=3.0, tilt=_onset(revision, equation, fill) - 0.1, source_fill=fill
    )
    assert [logged.cause for logged in revision.log] == [
        RevisionCause.STALL,
        RevisionCause.ARRIVAL,
    ]
