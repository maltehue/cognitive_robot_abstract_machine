import math
from dataclasses import dataclass, field

import numpy
import pytest

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from experiments.pouring_model_verification.concept_revision import (
    ConceptAwareRevision,
    DischargeRevision,
    Launch,
    RevisedConcept,
    SourceWatch,
    StallEvidence,
)
from experiments.pouring_model_verification.onset_rule import (
    ARRIVAL,
    OnsetRevision,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)

PERIOD = 0.1
PATIENCE = 1.0
GRAVITY = 9.81


# %% telling whether the source is losing grains


def _source_watch(noise: float = 0.0) -> SourceWatch:
    return SourceWatch(noise=noise, patience=PATIENCE)


def _feed(watch: SourceWatch, rate: float, steps: int = 20) -> float:
    at = 0.0
    for step in range(steps):
        at = PERIOD * step
        watch.observe(100.0 - rate * at, at=at)
    return at


def test_a_source_draining_steadily_has_lost_its_rate_over_the_patience():
    watch = _source_watch()
    at = _feed(watch, rate=6.0)
    assert watch.lost(at) == pytest.approx(6.0 * PATIENCE)


def test_a_source_holding_its_grains_is_not_losing_them():
    watch = _source_watch()
    at = _feed(watch, rate=0.0)
    assert not watch.is_losing(at)


def test_a_source_losing_more_than_a_grain_over_the_patience_is_losing_them():
    watch = _source_watch()
    at = _feed(watch, rate=2.0 * ARRIVAL / PATIENCE)
    assert watch.is_losing(at)


def test_a_noisier_reading_needs_a_larger_loss_to_count():
    quiet, noisy = _source_watch(0.0), _source_watch(3.0)
    at = _feed(quiet, rate=0.0)
    _feed(noisy, rate=0.0)
    assert noisy.evidence(at) > quiet.evidence(at)


# %% the stream as the model throws it


def _launch(horizontal: float = 1.0, vertical: float = 0.0, drop: float = 0.2):
    return Launch(
        speed=1.0,
        horizontal_share=horizontal,
        vertical_share=vertical,
        drop=drop,
        gravity=GRAVITY,
    )


def test_a_horizontal_stream_lands_its_speed_times_its_fall_time_away():
    launch = _launch()
    speed = 0.4
    assert launch.distance(speed) == pytest.approx(
        speed * math.sqrt(2 * launch.drop / GRAVITY)
    )


def test_the_speed_found_for_a_distance_throws_the_stream_that_far():
    launch = _launch(horizontal=math.sqrt(0.5), vertical=math.sqrt(0.5))
    distance = 0.15
    assert launch.distance(launch.speed_for(distance)) == pytest.approx(distance)


# %% revising the discharge coefficient


def _discharge_revision(coefficient: float = 0.3):
    variables = FloatVariableData()
    scale = FloatVariable("discharge_scale")
    variables.register_expression(scale)
    variables.set_value(scale, 1.0)
    return DischargeRevision(coefficient=coefficient, scale=scale, variables=variables)


def test_a_stream_seen_landing_further_out_raises_the_coefficient_to_throw_it_there():
    revision = _discharge_revision()
    launch = _launch()
    seen_at = 2.0 * launch.distance(launch.speed)
    old, new = revision.revise(launch, seen_at)
    assert new / old == pytest.approx(launch.speed_for(seen_at) / launch.speed)


def test_a_revised_coefficient_is_where_the_model_pours_from_next():
    revision = _discharge_revision(coefficient=0.3)
    launch = _launch()
    _, new = revision.revise(launch, 0.5 * launch.distance(launch.speed))
    assert revision.value == pytest.approx(new)


# %% choosing the concept


@dataclass
class SeenStream:
    """
    Stands in for what a camera sees of the stream and what the model predicts of it.
    """

    seen_distance: float | None
    launch_now: Launch = field(default_factory=_launch)

    def record(self, at: float) -> None:
        pass

    def observed_distance(self, since: float) -> float | None:
        return self.seen_distance

    def launch(self) -> Launch:
        return self.launch_now


def _concept_aware(stream: SeenStream, source_rate: float):
    variables = FloatVariableData()
    correction = FloatVariable("repose_correction")
    variables.register_expression(correction)
    variables.set_value(correction, 0.0)
    onset = OnsetRevision(
        equation=ArticulatedPouringEquation(
            container_height=0.06, container_width=0.07, repose_angle=0.6
        ),
        correction=correction,
        variables=variables,
    )
    responder = ConceptAwareRevision(
        onset=onset,
        discharge=_discharge_revision(),
        source=_source_watch(),
        stream=stream,
        patience=PATIENCE,
    )
    at = _feed(responder.source, rate=source_rate)
    return responder, at


def test_a_stall_with_the_source_holding_its_grains_revises_the_onset():
    responder, at = _concept_aware(SeenStream(seen_distance=None), source_rate=0.0)
    responder.respond(stalled=True, at=at, tilt=1.3, source_fill=0.7)
    (diagnosis,) = responder.diagnoses
    assert (diagnosis.concept, diagnosis.evidence) == (
        RevisedConcept.REPOSE_ANGLE,
        StallEvidence.SOURCE_HOLDING,
    )


def test_a_stall_with_the_source_losing_grains_revises_the_discharge_coefficient():
    launch = _launch()
    responder, at = _concept_aware(
        SeenStream(seen_distance=2.0 * launch.distance(launch.speed)),
        source_rate=6.0,
    )
    responder.respond(stalled=True, at=at, tilt=1.3, source_fill=0.7)
    (diagnosis,) = responder.diagnoses
    assert (diagnosis.concept, diagnosis.evidence) == (
        RevisedConcept.DISCHARGE_COEFFICIENT,
        StallEvidence.SOURCE_LOSING,
    )
    assert diagnosis.new_value > diagnosis.old_value


def test_a_miss_nobody_saw_land_is_logged_without_a_revision():
    responder, at = _concept_aware(SeenStream(seen_distance=None), source_rate=6.0)
    changed = responder.respond(stalled=True, at=at, tilt=1.3, source_fill=0.7)
    (diagnosis,) = responder.diagnoses
    assert (changed, diagnosis.new_value) == (False, diagnosis.old_value)


def test_a_pour_that_is_not_stalled_is_not_diagnosed():
    responder, at = _concept_aware(SeenStream(seen_distance=None), source_rate=6.0)
    responder.respond(stalled=False, at=at, tilt=1.3, source_fill=0.7)
    assert responder.diagnoses == []


def test_each_reading_feeds_the_source_watch():
    responder, at = _concept_aware(SeenStream(seen_distance=None), source_rate=0.0)
    for step in range(1, 11):
        responder.observe(at + PERIOD * step, source_grains=100.0 - 3.0 * step)
    assert responder.source.lost(at + PERIOD * 10) == pytest.approx(
        3.0 / PERIOD * PATIENCE
    )
