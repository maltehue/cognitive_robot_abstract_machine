"""
Receiver measurements take precedence over a biased accumulated count.
"""

import pytest

from experiments.real_pouring.receiving_cup import ReceiverFillReport
from experiments.real_pouring.stream_detection import StreamReading

from .test_live_bank import live_bank_tilted

# %% measured contents


@pytest.mark.parametrize("fill", [0.0, 0.2])
def test_measured_contents_override_a_larger_count(fill: float) -> None:
    """
    A valid measurement, including empty, determines the reported contents.
    """
    bank = live_bank_tilted(1.0)
    bank.fill_report = ReceiverFillReport.MEASURED
    step = bank.observe(StreamReading(0.0, False, 80.0, None, heap_fill=fill))
    assert bank.receiver.fill_level == pytest.approx(fill)
    assert step.told_grains == pytest.approx(
        fill * bank.grains_per_fill * bank.receiver.capacity / bank.source.capacity
    )


def test_missing_measurement_uses_only_subsequent_count_increments() -> None:
    """
    A dropout does not restore the accumulated count's earlier bias.
    """
    bank = live_bank_tilted(1.0)
    bank.fill_report = ReceiverFillReport.MEASURED
    initial = bank.observe(StreamReading(0.0, False, 80.0, None, heap_fill=0.2))
    following = bank.observe(StreamReading(0.1, False, 80.0, None))
    assert following.told_grains == pytest.approx(initial.told_grains)


def test_measurement_keeps_the_separate_stop_allowance() -> None:
    """
    Already received material and predicted late arrivals are added once.
    """
    bank = live_bank_tilted(1.0)
    bank.fill_report = ReceiverFillReport.MEASURED
    bank.stop_lead_time = 1.0
    bank.in_flight_limit = 2.0
    bank.observe(StreamReading(0.0, False, 0.0, None, heap_fill=0.0))
    step = bank.observe(StreamReading(0.1, True, 80.0, None, heap_fill=0.2))
    measured = (
        0.2 * bank.grains_per_fill * bank.receiver.capacity / bank.source.capacity
    )
    assert step.in_flight_grains == bank.in_flight_limit
    assert step.told_grains == pytest.approx(measured + step.in_flight_grains)


def test_count_is_available_before_the_first_measurement() -> None:
    """
    Warmup retains an estimate until the wall supplies a measurement.
    """
    bank = live_bank_tilted(1.0)
    bank.fill_report = ReceiverFillReport.MEASURED
    step = bank.observe(StreamReading(0.0, False, 20.0, None))
    assert step.told_grains == pytest.approx(
        bank.camera.grains_crossed(step.counted_grains) * bank.coupling.inside_share
    )


def test_count_increments_continue_from_the_measured_contents() -> None:
    """
    Missing frames add only newly observed arrivals to the last measurement.
    """
    bank = live_bank_tilted(1.0)
    bank.fill_report = ReceiverFillReport.MEASURED
    initial = bank.observe(StreamReading(0.0, False, 80.0, None, heap_fill=0.2))
    following = bank.observe(StreamReading(0.1, False, 81.0, None))
    assert following.told_grains == pytest.approx(
        initial.told_grains
        + bank.camera.grains_crossed(1.0) * bank.coupling.inside_share
    )


def test_a_returning_measurement_replaces_accumulated_dropout_estimates() -> None:
    """
    Recovered perception replaces the count-based extrapolation.
    """
    bank = live_bank_tilted(1.0)
    bank.fill_report = ReceiverFillReport.MEASURED
    bank.observe(StreamReading(0.0, False, 80.0, None, heap_fill=0.2))
    bank.observe(StreamReading(0.1, False, 85.0, None))
    bank.observe(StreamReading(0.2, False, 85.0, None, heap_fill=0.21))
    assert bank.receiver.fill_level == pytest.approx(0.21)
