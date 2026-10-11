import numpy
import pytest

from experiments.pouring_model_verification.sensor import (
    SensorChannel,
    SensorCondition,
    SensorModel,
    WRIST_DELAY,
    WRIST_NOISE,
    WristSensor,
)

GRAIN_SHARE = 0.01
"""
The fill share one grain amounts to in these tests.
"""


def _channel(model: SensorModel, seed: int = 0) -> SensorChannel:
    return SensorChannel(
        model=model, grain_share=GRAIN_SHARE, random=numpy.random.default_rng(seed)
    )


# %% an ideal sensor


def test_an_ideal_sensor_reports_the_true_value():
    channel = _channel(SensorCondition.IDEAL.model)
    assert [channel.read(value, at=0.1 * i) for i, value in enumerate([0.1, 0.3])] == [
        0.1,
        0.3,
    ]


# %% a delayed sensor


def test_a_delayed_sensor_reports_what_was_true_one_delay_ago():
    model = SensorModel(noise=0.0, delay=0.2)
    channel = _channel(model)
    readings = {round(0.1 * i, 1): 0.01 * i for i in range(6)}
    reported = [channel.read(value, at=at) for at, value in readings.items()]
    assert reported[-1] == readings[0.3]


def test_a_delayed_sensor_reports_its_first_reading_until_one_delay_has_passed():
    channel = _channel(SensorModel(noise=0.0, delay=0.5))
    first = channel.read(0.2, at=0.0)
    assert channel.read(0.4, at=0.1) == first


# %% a noisy sensor


def test_a_noisy_sensor_scatters_by_its_noise_in_grains():
    model = SensorModel(noise=2.0, delay=0.0)
    channel = _channel(model)
    reported = numpy.array([channel.read(0.5, at=0.1 * i) for i in range(20000)])
    assert reported.std() == pytest.approx(model.noise * GRAIN_SHARE, rel=0.05)
    assert reported.mean() == pytest.approx(0.5, abs=model.noise * GRAIN_SHARE / 20)


def test_a_noisy_sensor_never_reports_a_level_outside_the_container():
    channel = _channel(SensorModel(noise=50.0, delay=0.0))
    reported = [channel.read(0.0, at=0.1 * i) for i in range(200)]
    assert min(reported) == 0.0


def test_the_same_seed_scatters_the_same_way():
    model = SensorCondition.NOISY.model
    first, second = _channel(model, seed=3), _channel(model, seed=3)
    assert [first.read(0.5, at=i) for i in range(5)] == [
        second.read(0.5, at=i) for i in range(5)
    ]


# %% which grains a sensor sees


def test_a_scale_under_the_receiver_does_not_see_grains_still_falling():
    assert not SensorCondition.SCALE.model.counts_falling_grains
    assert not SensorCondition.LANDED_ONLY.model.counts_falling_grains


def test_only_the_ideal_condition_is_an_ideal_sensor():
    assert [condition.model.is_ideal for condition in SensorCondition] == [
        condition is SensorCondition.IDEAL for condition in SensorCondition
    ]


def test_an_ideal_wrist_weighs_the_source_at_once_without_noise():
    assert WristSensor.IDEAL.model.is_ideal


def test_a_realistic_wrist_carries_the_wrist_noise_and_delay():
    assert WristSensor.REALISTIC.model == SensorModel(
        noise=WRIST_NOISE, delay=WRIST_DELAY
    )
