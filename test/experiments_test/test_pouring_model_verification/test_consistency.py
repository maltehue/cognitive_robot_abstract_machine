import numpy
import pytest

from experiments.pouring_model_verification.consistency import (
    ConsistencyCheck,
    Feature,
    PourFeatures,
    RateResiduals,
)
from experiments.pouring_model_verification.hypothesis_bank import (
    PARAMETER_ORDER,
    BankSettings,
    DrainGeometry,
    FreeParameter,
    ModelPoint,
    Reading,
    SensorKnowledge,
)

GEOMETRY = DrainGeometry(
    container_height=0.06, lip_offset=0.039, grains_per_fill=150.0, slot_offset=0.039
)
GIVEN = ModelPoint(
    repose_angle=0.7, discharge_coefficient=0.3, outflow_rate_constant=2.0
)
EXACT = SensorKnowledge(
    receiver_noise=0.0,
    receiver_delay=0.0,
    counts_falling_grains=True,
    fall_time=0.15,
    source_noise=0.0,
    source_delay=0.0,
    crossing_noise=None,
)


def _readings(truth: ModelPoint, rate_scale: float = 1.0) -> list:
    """
    Readings of a steady turn the drain with the true parameters empties, read exactly,
    its outflow scaled by a factor the model does not know.
    """
    parameters = numpy.array([[truth.value_of(p) for p in PARAMETER_ORDER]])
    grains, tilt, step, readings = 100.0, 0.9, 0.1, []
    for index in range(120):
        readings.append(
            Reading(
                time=index * step,
                tilt=tilt,
                receiver_grains=100.0 - grains,
                source_grains=grains,
            )
        )
        head = GEOMETRY.head(tilt, numpy.array([grains]), parameters)
        grains -= min(grains, rate_scale * GEOMETRY.outflow(head, parameters)[0] * step)
        tilt += 0.03 * step
    return readings


def _particles(point: ModelPoint, count: int = 100) -> numpy.ndarray:
    return numpy.tile([point.value_of(p) for p in PARAMETER_ORDER], (count, 1))


def _check() -> ConsistencyCheck:
    settings = BankSettings(rate_noise=0.1, stall_probability=0.0)
    return ConsistencyCheck(geometry=GEOMETRY, sensors=EXACT, settings=settings, seed=0)


def test_the_first_grain_leaves_at_the_tilt_of_the_first_reading_that_lost_one():
    readings = _readings(GIVEN)
    features = PourFeatures.of(readings, threshold=1.0)
    first = next(
        r for r in readings if r.source_grains <= readings[0].source_grains - 1.0
    )
    assert features.first_tilt == first.tilt


def test_a_pour_its_own_model_made_is_consistent():
    readings = _readings(GIVEN)
    assert _check().verdict(_particles(GIVEN), numpy.zeros(100), readings).consistent


def test_a_slow_drain_explained_by_a_later_onset_is_inconsistent():
    readings = _readings(GIVEN, rate_scale=0.3)
    later = GIVEN.with_values({FreeParameter.REPOSE_ANGLE: GIVEN.repose_angle + 0.15})
    verdict = _check().verdict(_particles(later), numpy.zeros(100), readings)
    assert not verdict.consistent


def test_residuals_seen_on_sound_pours_widen_the_prediction():
    readings = _readings(GIVEN, rate_scale=0.6)
    plain = _check().verdict(_particles(GIVEN), numpy.zeros(100), readings)
    calibrated = ConsistencyCheck(
        geometry=GEOMETRY,
        sensors=EXACT,
        settings=BankSettings(rate_noise=0.1, stall_probability=0.0),
        seed=0,
        residuals={Feature.EARLY_RATE: RateResiduals(edges=(), samples=((-3.0, 0.0),))},
    ).verdict(_particles(GIVEN), numpy.zeros(100), readings)
    assert (
        calibrated.intervals[Feature.EARLY_RATE].low
        < plain.intervals[Feature.EARLY_RATE].low
    )


def test_residuals_are_taken_from_the_tilt_rate_bin_the_pour_falls_in():
    residuals = RateResiduals(edges=(0.05,), samples=((1.0, 2.0), (5.0, 6.0)))
    assert (residuals.for_rate(0.01).tolist(), residuals.for_rate(0.2).tolist()) == (
        [1.0, 2.0],
        [5.0, 6.0],
    )


def test_a_pour_records_how_fast_it_turned_towards_its_first_grain():
    readings = _readings(GIVEN)
    features = PourFeatures.of(readings, threshold=1.0)
    first = next(
        r for r in readings if r.source_grains <= readings[0].source_grains - 1.0
    )
    assert features.approach_rate == pytest.approx(
        (first.tilt - readings[0].tilt) / (first.time - readings[0].time)
    )


def test_an_observation_its_own_model_made_sits_inside_its_prediction():
    readings = _readings(GIVEN)
    verdict = _check().verdict(_particles(GIVEN), numpy.zeros(100), readings)
    assert 0.0 < verdict.positions[Feature.FIRST_TILT] < 1.0
