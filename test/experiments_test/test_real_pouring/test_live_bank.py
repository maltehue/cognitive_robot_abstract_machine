import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    FreeParameter,
    ModelPoint,
)
from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.stream_detection import StreamReading
from experiments.pouring_model_verification.sensor import CameraCondition

from ..pouring_worlds import OPENING_RADIUS, tilted_source_above_receiver

LOADED = 100.0
"""
How many grains the source was loaded with: the count of a full cup.
"""


def live_bank_tilted(tilt: float) -> LiveBank:
    stream = tilted_source_above_receiver(tilt=tilt)
    equation = stream.source.fill_equation
    return LiveBank.for_cups(
        source=stream.source,
        receiver=stream.receiver,
        world=stream.world,
        built=ModelPoint(
            repose_angle=equation.repose_angle,
            discharge_coefficient=equation.discharge_coefficient,
            outflow_rate_constant=equation.outflow_rate_constant,
        ),
        settings=BankSettings(),
        camera=CameraCondition.POLE_COUNTING.model,
        grains_per_fill=LOADED,
        loaded_grains=LOADED,
        goal_grains=0.7 * LOADED,
        opening_radius=OPENING_RADIUS,
        prediction_window=1.5,
    )


def _no_stream(at: float) -> StreamReading:
    return StreamReading(stamp=at, stream_seen=False, counted_grains=0.0, crossing=None)


def test_the_first_reading_leaves_the_drain_as_built():
    bank = live_bank_tilted(tilt=1.0)
    bank.observe(_no_stream(0.0))
    correction = bank.source.fill_connection.drain_correction
    assert [
        connection.position for connection in correction.connections_in(bank.world)
    ] == pytest.approx([0.0, 0.0, 0.0])


def test_the_source_is_reported_as_loaded_less_what_the_camera_counted():
    bank = live_bank_tilted(tilt=1.0)
    bank.observe(_no_stream(0.0))
    bank.observe(
        StreamReading(stamp=0.1, stream_seen=True, counted_grains=19.0, crossing=None)
    )
    crossed = bank.camera.grains_crossed(19.0)
    assert bank.source.fill_level == pytest.approx((LOADED - crossed) / LOADED)


def test_a_stream_long_missing_where_the_model_pours_moves_the_onset_later():
    bank = live_bank_tilted(tilt=1.0)
    for step in range(60):
        bank.observe(_no_stream(0.1 * step))
    correction = bank.source.fill_connection.drain_correction
    assert FreeParameter.REPOSE_ANGLE in bank.coupling.active.concepts
    assert correction.repose_connection_in(bank.world).position > 0.0


def test_every_reading_is_kept_in_the_trace():
    bank = live_bank_tilted(tilt=1.0)
    for step in range(3):
        bank.observe(_no_stream(0.1 * step))
    assert [row.time for row in bank.trace] == pytest.approx([0.0, 0.1, 0.2])


def test_a_camera_that_counted_too_much_calls_for_more_area_per_grain():
    bank = live_bank_tilted(tilt=1.0)
    bank.observe(_no_stream(0.0))
    bank.observe(
        StreamReading(stamp=0.1, stream_seen=True, counted_grains=38.0, crossing=None)
    )
    weighed = 0.5 * bank.camera.grains_crossed(38.0)
    assert bank.count_calibration_factor(weighed_grains=weighed) == pytest.approx(2.0)
