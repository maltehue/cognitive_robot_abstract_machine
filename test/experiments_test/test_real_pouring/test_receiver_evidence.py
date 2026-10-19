"""
Wall observations constrain the internal bank without a biased count likelihood.
"""

from dataclasses import replace

import pytest

from experiments.real_pouring.live_bank import LiveBank
from experiments.real_pouring.receiver_evidence import ReceiverEvidence
from experiments.real_pouring.receiving_cup import ReceiverFillReport
from experiments.real_pouring.stream_detection import StreamReading
from krrood.adapters.json_serializer import from_json, to_json

from .test_live_bank import live_bank_tilted

# %% observation selection and units


@pytest.fixture
def evidence() -> ReceiverEvidence:
    """
    Distinct calibrated and fallback uncertainties with nonoverlapping windows.
    """
    return ReceiverEvidence(0.02, 0.08, 0.33, 0.33)


@pytest.fixture
def measured_bank(evidence: ReceiverEvidence) -> LiveBank:
    """
    A camera bank configured to learn from receiver contents.
    """
    return receiver_bank(evidence)


def receiver_bank(evidence: ReceiverEvidence, tilt: float = 1.0) -> LiveBank:
    """
    Construct an independent deterministic bank for receiver-evidence comparisons.
    """
    original = live_bank_tilted(tilt)
    return LiveBank.for_cups(
        source=original.source,
        receiver=original.receiver,
        world=original.world,
        built=original.built,
        settings=original.coupling.bank.settings,
        camera=original.camera,
        grains_per_fill=original.grains_per_fill,
        loaded_grains=original.loaded_grains,
        goal_grains=original.goal_grains,
        opening_radius=original.opening_radius,
        prediction_window=original.prediction_window,
        fill_report=ReceiverFillReport.MEASURED,
        receiver_evidence=evidence,
    )


def test_receiver_measurements_replace_the_count_likelihood(
    measured_bank: LiveBank,
) -> None:
    """
    The bank cannot prefer an inflated count to the receiver observation.
    """
    assert measured_bank.coupling.bank.sensors.count_noise is None
    assert (
        measured_bank.coupling.bank.sensors.receiver_delay
        == measured_bank.receiver_evidence.delay
    )


def test_observation_is_converted_to_receiver_grains(measured_bank: LiveBank) -> None:
    """
    Fill and its uncertainty use the same receiver-capacity conversion.
    """
    reading = StreamReading(0.0, False, 80.0, None, heap_fill=0.2, wall_calibrated=True)
    step = measured_bank.observe(reading)
    conversion = (
        measured_bank.grains_per_fill
        * measured_bank.receiver.capacity
        / measured_bank.source.capacity
    )
    assert step.receiver_observation == reading.heap_fill * conversion
    assert (
        step.receiver_noise
        == measured_bank.receiver_evidence.calibrated_noise * conversion
    )


def test_geometric_fallback_has_its_own_uncertainty(measured_bank: LiveBank) -> None:
    """
    An out-of-range geometric estimate must not receive calibrated confidence.
    """
    step = measured_bank.observe(StreamReading(0.0, False, 0.0, None, heap_fill=0.2))
    conversion = (
        measured_bank.grains_per_fill
        * measured_bank.receiver.capacity
        / measured_bank.source.capacity
    )
    assert (
        step.receiver_noise
        == measured_bank.receiver_evidence.geometric_noise * conversion
    )


def test_overlapping_wall_windows_are_not_repeated_evidence(
    measured_bank: LiveBank,
) -> None:
    """
    Camera frames inside one evidence window still update motion, but not fill evidence.
    """
    reading = StreamReading(0.0, False, 0.0, None, heap_fill=0.0, wall_calibrated=True)
    measured_bank.observe(reading)
    skipped = measured_bank.observe(
        replace(reading, stamp=measured_bank.receiver_evidence.interval / 2)
    )
    accepted = measured_bank.observe(
        replace(reading, stamp=measured_bank.receiver_evidence.interval)
    )
    assert skipped.receiver_observation is None
    assert accepted.receiver_observation == reading.heap_fill


def test_missing_wall_reading_is_not_a_zero_measurement(
    measured_bank: LiveBank,
) -> None:
    """
    Unavailable contents do not empty the modeled receiver.
    """
    step = measured_bank.observe(StreamReading(0.0, False, 80.0, None))
    assert step.receiver_observation is None


def test_source_report_uses_the_posterior_not_the_count(
    measured_bank: LiveBank,
) -> None:
    """
    A count spike cannot directly empty the source reported to the controller.
    """
    measured_bank.observe(
        StreamReading(0.0, False, 80.0, None, heap_fill=0.0, wall_calibrated=True)
    )
    assert measured_bank.source.fill_level == pytest.approx(
        min(measured_bank.loaded_grains, measured_bank.coupling.source_grains)
        / measured_bank.grains_per_fill
    )


def test_wall_provenance_survives_serialization() -> None:
    """
    The listener receives the same quality designation the detector published.
    """
    reading = StreamReading(0.0, False, 0.0, None, heap_fill=0.0, wall_calibrated=True)
    assert from_json(to_json(reading)) == reading


def test_count_bias_cannot_change_the_receiver_posterior(
    evidence: ReceiverEvidence,
) -> None:
    """
    Identical wall and stream evidence yield the same posterior despite count bias.
    """
    first = receiver_bank(evidence, tilt=1.8)
    second = receiver_bank(evidence, tilt=1.8)
    for index in range(20):
        reading = StreamReading(
            index * 0.1,
            True,
            float(index),
            None,
            heap_fill=index * 0.005,
            wall_calibrated=True,
        )
        first.observe(reading)
        second.observe(replace(reading, counted_grains=reading.counted_grains * 5))
    assert first.coupling.received_grains == second.coupling.received_grains


def test_wall_contents_change_the_internal_posterior(
    evidence: ReceiverEvidence,
) -> None:
    """
    Receiver measurements affect inferred contents rather than only trace fields.
    """
    empty = receiver_bank(evidence, tilt=1.8)
    filling = receiver_bank(evidence, tilt=1.8)
    for index in range(20):
        reading = StreamReading(
            index * 0.1, True, float(index), None, heap_fill=0.0, wall_calibrated=True
        )
        empty.observe(reading)
        filling.observe(replace(reading, heap_fill=index * 0.01))
    assert filling.coupling.received_grains > empty.coupling.received_grains


def test_evidence_and_source_posterior_survive_trace_serialization(
    measured_bank: LiveBank,
) -> None:
    """
    Saved bank traces retain exactly the evidence admitted and source posterior.
    """
    step = measured_bank.observe(
        StreamReading(0.0, False, 0.0, None, heap_fill=0.0, wall_calibrated=True)
    )
    assert from_json(to_json(step)) == step
