import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    BankSettings,
    FreeParameter,
    ModelPoint,
)
from experiments.real_pouring.live_bank import LiveBank, LiveBankStep
from experiments.real_pouring.receiving_cup import ReceiverFillReport
from experiments.real_pouring.stream_detection import StreamReading
from experiments.pouring_model_verification.sensor import CameraCondition

from ..pouring_worlds import OPENING_RADIUS, tilted_source_above_receiver

LOADED = 100.0
"""
How many grains the source was loaded with: the count of a full cup.
"""


def live_bank_tilted(tilt: float, settings: BankSettings = BankSettings()) -> LiveBank:
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
        settings=settings,
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


def test_the_step_records_what_the_receiver_was_told_to_hold():
    bank = live_bank_tilted(tilt=1.0)
    bank.observe(_no_stream(0.0))
    bank.observe(
        StreamReading(stamp=0.1, stream_seen=True, counted_grains=19.0, crossing=None)
    )
    assert bank.trace[-1].told_grains == pytest.approx(
        bank.coupling.received_grains + bank.coupling.margin
    )
    assert bank.receiver.fill_level == pytest.approx(
        bank.trace[-1].told_grains / LOADED
    )


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


# %% the onset tracked by every hypothesis


TRACKING_THE_ONSET = BankSettings(tracks_onset=True)
"""
A bank in which every hypothesis estimates the onset.
"""


def _stream(at: float, counted: float, seen: bool = True) -> StreamReading:
    return StreamReading(
        stamp=at, stream_seen=seen, counted_grains=counted, crossing=None
    )


def test_with_the_onset_tracked_every_hypothesis_learns_a_late_onset():
    bank = live_bank_tilted(tilt=1.0, settings=TRACKING_THE_ONSET)
    for step in range(60):
        bank.observe(_no_stream(0.1 * step))
    learned = {}
    for hypothesis in bank.coupling.bank.hypotheses:
        bank.coupling.switch.active = hypothesis
        learned[hypothesis] = bank.coupling.point.repose_angle
    assert all(angle > bank.built.repose_angle for angle in learned.values())


SMALL_STREAM = 0.25
"""
Grains a reading counts while the stream runs: few enough that the cup stays nearly
full, so only the material, not the emptying cup, can stop the stream.
"""


def test_a_stream_that_stops_where_the_model_pours_raises_the_onset_again():
    bank = live_bank_tilted(tilt=1.0, settings=TRACKING_THE_ONSET)
    for step in range(20):
        bank.observe(_stream(0.1 * step, counted=SMALL_STREAM * step))
    while_flowing = bank.coupling.point.repose_angle
    for step in range(20, 60):
        bank.observe(_stream(0.1 * step, counted=SMALL_STREAM * 19, seen=False))
    assert bank.coupling.point.repose_angle > while_flowing


# %% the receiver reported from the count

STOP_LEAD_TIME = 1.0
"""
How long grains keep arriving after the controller stops, in seconds, as the tests hand
it to the bank.
"""


def live_bank_from_count(tilt: float) -> LiveBank:
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
        fill_report=ReceiverFillReport.COUNTED,
        stop_lead_time=STOP_LEAD_TIME,
    )


STEADY_RATE = 1.0
"""
Grains per reading of a count rising steadily, ten readings a second.
"""


def _steady_count(bank: LiveBank, readings: int) -> float:
    """
    Feed a count rising by :data:`STEADY_RATE` per reading.

    :return: The last count.
    """
    counted = 0.0
    for step in range(readings):
        counted = STEADY_RATE * step
        bank.observe(_stream(0.1 * step, counted=counted))
    return counted


def test_the_receiver_is_reported_as_what_crossed_plus_what_is_still_in_flight():
    bank = live_bank_from_count(tilt=1.0)
    counted = _steady_count(bank, readings=60)
    crossed = bank.camera.grains_crossed(counted)
    rate = bank.camera.grains_crossed(STEADY_RATE) / 0.1
    in_flight = bank.trace[-1].in_flight_grains
    assert in_flight == pytest.approx(rate * STOP_LEAD_TIME, rel=0.01)
    expected = crossed * bank.coupling.inside_share + in_flight
    assert bank.trace[-1].told_grains == pytest.approx(expected)
    assert bank.receiver.fill_level == pytest.approx(expected / LOADED)


def test_a_burst_in_the_count_raises_what_is_in_flight_only_gradually():
    bank = live_bank_from_count(tilt=1.0)
    for step in range(10):
        bank.observe(_stream(0.1 * step, counted=0.0))
    burst = 20.0
    at_burst = bank.observe(_stream(1.0, counted=burst)).in_flight_grains
    unfiltered = bank.camera.grains_crossed(burst) / 0.1 * STOP_LEAD_TIME
    assert 0.0 < at_burst < unfiltered
    for step in range(11, 21):
        bank.observe(_stream(0.1 * step, counted=burst))
    later = bank.trace[-1].in_flight_grains
    assert 0.0 < later < at_burst


IN_FLIGHT_LIMIT = 5.0
"""
The most grains the tests let the bank report in flight.
"""


def test_what_is_in_flight_never_exceeds_the_limit():
    stream = tilted_source_above_receiver(tilt=1.0)
    equation = stream.source.fill_equation
    bank = LiveBank.for_cups(
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
        fill_report=ReceiverFillReport.COUNTED,
        stop_lead_time=STOP_LEAD_TIME,
        in_flight_limit=IN_FLIGHT_LIMIT,
    )
    _steady_count(bank, readings=60)
    assert bank.trace[-1].in_flight_grains == IN_FLIGHT_LIMIT


def test_nothing_is_in_flight_while_the_count_stands_still():
    bank = live_bank_from_count(tilt=1.0)
    for step in range(8):
        bank.observe(_stream(0.1 * step, counted=5.0, seen=False))
    assert bank.trace[-1].in_flight_grains == 0.0
    assert bank.receiver.fill_level == pytest.approx(
        bank.camera.grains_crossed(5.0) * bank.coupling.inside_share / LOADED
    )


def test_the_step_keeps_what_was_in_flight_through_json():
    bank = live_bank_from_count(tilt=1.0)
    for step, counted in enumerate([0.0, 3.0, 6.0, 9.0, 12.0, 15.0]):
        bank.observe(_stream(0.1 * step, counted=counted))
    step = bank.trace[-1]
    assert step.in_flight_grains > 0.0
    assert LiveBankStep.from_json(step.to_json()) == step


# %% the controller's onset without a trickle

NO_TRICKLE = BankSettings(trickle=0.0)
"""
A bank that does not hold the controller's onset below the tilt.
"""


def test_without_a_trickle_the_controller_runs_on_the_onset_quantile():
    bank = live_bank_tilted(tilt=1.0, settings=NO_TRICKLE)
    for step in range(30):
        bank.observe(_no_stream(0.1 * step))
    active = bank.coupling.bank.filters[bank.coupling.active]
    assert FreeParameter.REPOSE_ANGLE in active.free
    assert bank.coupling.point.repose_angle == active.quantile(
        FreeParameter.REPOSE_ANGLE, NO_TRICKLE.onset_quantile
    )


# %% the heap in the receiver as the stop signal


def _counting_bank() -> LiveBank:
    stream = tilted_source_above_receiver(tilt=1.0)
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
        fill_report=ReceiverFillReport.COUNTED,
    )


def test_a_heap_showing_more_than_was_counted_is_what_the_receiver_holds():
    bank = _counting_bank()
    bank.observe(_no_stream(0.0))
    heap_fill = 0.68
    bank.observe(
        StreamReading(
            stamp=0.1,
            stream_seen=False,
            counted_grains=20.0,
            crossing=None,
            heap_fill=heap_fill,
        )
    )
    assert bank.receiver.fill_level == pytest.approx(heap_fill)


def test_without_a_heap_in_view_the_receiver_holds_what_was_counted():
    bank = _counting_bank()
    bank.observe(_no_stream(0.0))
    step = bank.observe(
        StreamReading(stamp=0.1, stream_seen=False, counted_grains=20.0, crossing=None)
    )
    assert bank.receiver.fill_level == pytest.approx(step.told_grains / LOADED)


# %% the receiver read through its wall


def live_bank_through_wall(tilt: float) -> LiveBank:
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
        fill_report=ReceiverFillReport.SEEN_THROUGH_WALL,
        stop_lead_time=STOP_LEAD_TIME,
    )


def test_a_translucent_receiver_holds_what_its_wall_shows_plus_what_is_in_flight():
    bank = live_bank_through_wall(tilt=1.0)
    wall_fill = 0.12
    for step, counted in enumerate([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
        bank.observe(
            StreamReading(
                stamp=0.1 * step,
                stream_seen=True,
                counted_grains=counted,
                crossing=None,
                heap_fill=wall_fill,
            )
        )
    capacity_ratio = bank.source.capacity / bank.receiver.capacity
    wall_grains = wall_fill * LOADED / capacity_ratio
    in_flight = bank.trace[-1].in_flight_grains
    assert in_flight > 0.0
    assert bank.trace[-1].told_grains == pytest.approx(wall_grains + in_flight)
    assert bank.receiver.fill_level == pytest.approx(
        wall_fill + in_flight / LOADED * capacity_ratio
    )


def test_a_wall_showing_nothing_yet_leaves_the_receiver_at_what_crossed():
    bank = live_bank_through_wall(tilt=1.0)
    bank.observe(_no_stream(0.0))
    step = bank.observe(
        StreamReading(stamp=0.1, stream_seen=True, counted_grains=4.0, crossing=None)
    )
    crossed = bank.camera.grains_crossed(4.0)
    assert step.told_grains == pytest.approx(
        crossed * bank.coupling.inside_share + step.in_flight_grains
    )
