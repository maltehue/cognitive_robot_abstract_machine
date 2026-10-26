"""
Continuous wall evidence, calibration, and rejection of transient streams.
"""

from __future__ import annotations

from dataclasses import replace

import numpy
import pytest

from experiments.real_pouring.wall_fill import (
    InvalidWallCalibrationError,
    InvalidWallObservationError,
    WallColourMeasurement,
    WallFillCalibration,
    WallFillEstimator,
    WallFillSample,
)
from krrood.adapters.json_serializer import from_json, to_json

# %% colour evidence and spatial weighting


@pytest.fixture
def measurement() -> WallColourMeasurement:
    """
    A continuous colour measurement with an explicit noise floor and saturation.
    """
    return WallColourMeasurement(noise_floor=2.0, colour_span=16.0, height_power=2.0)


@pytest.fixture
def region() -> numpy.ndarray:
    """
    A receiver crop with no excluded pixels.
    """
    return numpy.ones((5, 4), dtype=bool)


def test_unchanged_wall_is_empty(
    measurement: WallColourMeasurement, region: numpy.ndarray
) -> None:
    """
    The empty image must remain zero without a learned intercept.
    """
    assert measurement.measure(numpy.zeros(region.shape), region) == 0.0


def test_subthreshold_colour_is_continuous(
    measurement: WallColourMeasurement, region: numpy.ndarray
) -> None:
    """
    Faint colour above the noise floor contributes before a binary mask would.
    """
    faint = numpy.full(
        region.shape, measurement.noise_floor + measurement.colour_span / 4
    )
    stronger = numpy.full(
        region.shape, measurement.noise_floor + measurement.colour_span / 2
    )
    assert measurement.measure(stronger, region) == pytest.approx(
        2 * measurement.measure(faint, region)
    )
    assert measurement.measure(faint, region) > 0.0


def test_higher_contents_contribute_more(
    measurement: WallColourMeasurement, region: numpy.ndarray
) -> None:
    """
    Identical coloured areas at different heights have different volume evidence.
    """
    upper = numpy.zeros(region.shape)
    lower = numpy.zeros(region.shape)
    upper[1] = measurement.noise_floor + measurement.colour_span
    lower[3] = measurement.noise_floor + measurement.colour_span
    assert measurement.measure(upper, region) == pytest.approx(
        9 * measurement.measure(lower, region)
    )


def test_pixels_outside_receiver_do_not_contribute(
    measurement: WallColourMeasurement, region: numpy.ndarray
) -> None:
    """
    Background colour must not affect the receiver's estimate.
    """
    region[:, 0] = False
    image = numpy.zeros(region.shape)
    image[:, 0] = measurement.noise_floor + measurement.colour_span
    assert measurement.measure(image, region) == 0.0


def test_narrow_persistent_stream_does_not_become_contents(
    measurement: WallColourMeasurement, region: numpy.ndarray
) -> None:
    """
    A stream covering less than half a row is rejected even when persistent.
    """
    image = numpy.zeros(region.shape)
    image[:, region.shape[1] // 2] = measurement.noise_floor + measurement.colour_span
    assert measurement.measure(image, region) == 0.0


@pytest.mark.parametrize(
    "noise_floor,colour_span,height_power",
    [(-1, 1, 1), (0, 0, 1), (0, 1, -1), (float("nan"), 1, 1)],
)
def test_invalid_measurement_parameters_are_rejected(
    noise_floor: float, colour_span: float, height_power: float
) -> None:
    """
    Invalid colour models must fail at construction.
    """
    with pytest.raises(InvalidWallCalibrationError):
        WallColourMeasurement(noise_floor, colour_span, height_power)


# %% calibration from independent weighed recordings


def test_scale_is_fitted_from_weighed_samples(
    measurement: WallColourMeasurement,
) -> None:
    """
    A proportional set of scale readings determines the conversion exactly.
    """
    samples = [
        WallFillSample(signal=0.1, fill=0.2),
        WallFillSample(signal=0.2, fill=0.4),
    ]
    calibration = WallFillCalibration.fit(measurement, samples)
    assert calibration.scale == pytest.approx(samples[0].fill / samples[0].signal)
    assert calibration.fill_for(samples[1].signal) == pytest.approx(samples[1].fill)


def test_calibration_round_trips(measurement: WallColourMeasurement) -> None:
    """
    Saved calibration retains its concrete measurement and fitted scale.
    """
    calibration = WallFillCalibration(measurement=measurement, scale=2.0)
    assert from_json(to_json(calibration)) == calibration


def test_calibration_without_colour_evidence_is_rejected(
    measurement: WallColourMeasurement,
) -> None:
    """
    Weighing alone cannot identify the colour-to-fill conversion.
    """
    with pytest.raises(InvalidWallCalibrationError):
        WallFillCalibration.fit(measurement, [WallFillSample(signal=0.0, fill=0.2)])


def test_fitted_calibration_declines_extrapolation(
    measurement: WallColourMeasurement,
) -> None:
    """
    Colour evidence outside the weighed calibration range is not a measured fill.
    """
    samples = [
        WallFillSample(signal=0.1, fill=0.2),
        WallFillSample(signal=0.2, fill=0.4),
    ]
    calibration = WallFillCalibration.fit(measurement, samples)
    assert calibration.fill_for(samples[0].signal / 2) is None
    assert calibration.fill_for(samples[-1].signal * 2) is None


def test_no_colour_still_means_empty_after_calibration(
    measurement: WallColourMeasurement,
) -> None:
    """
    The empty-view origin remains known when only nonempty pours were weighed.
    """
    calibration = WallFillCalibration.fit(
        measurement, [WallFillSample(signal=0.1, fill=0.2)]
    )
    assert calibration.fill_for(0.0) == 0.0


# %% offset of the colour-to-fill conversion


def test_offset_is_fitted_from_weighed_samples(
    measurement: WallColourMeasurement,
) -> None:
    """
    Readings that lie on a line not through the origin determine its scale and offset.
    """
    samples = [
        WallFillSample(signal=0.1, fill=0.15),
        WallFillSample(signal=0.2, fill=0.35),
        WallFillSample(signal=0.3, fill=0.55),
    ]
    calibration = WallFillCalibration.fit(measurement, samples)
    scale = (samples[1].fill - samples[0].fill) / (
        samples[1].signal - samples[0].signal
    )
    assert calibration.scale == pytest.approx(scale)
    assert calibration.offset == pytest.approx(
        samples[0].fill - scale * samples[0].signal
    )
    assert calibration.fill_for(samples[2].signal) == pytest.approx(samples[2].fill)


def test_one_recording_fits_no_offset(measurement: WallColourMeasurement) -> None:
    """
    A single colour level cannot tell an offset from a scale, so the line keeps the
    origin.
    """
    sample = WallFillSample(signal=0.1, fill=0.2)
    calibration = WallFillCalibration.fit(measurement, [sample])
    assert calibration.offset == 0.0
    assert calibration.scale == pytest.approx(sample.fill / sample.signal)


def test_offset_never_reports_a_negative_fill(
    measurement: WallColourMeasurement,
) -> None:
    """
    Faint colour below the offset reads as an empty receiver, not less than empty.
    """
    calibration = WallFillCalibration(measurement=measurement, scale=2.0, offset=-0.3)
    assert calibration.fill_for(0.1) == 0.0


def test_calibration_with_offset_round_trips(
    measurement: WallColourMeasurement,
) -> None:
    """
    Saved calibration retains its fitted offset.
    """
    calibration = WallFillCalibration(measurement=measurement, scale=2.0, offset=-0.02)
    assert from_json(to_json(calibration)) == calibration


def test_non_finite_offset_is_rejected(measurement: WallColourMeasurement) -> None:
    """
    An undefined offset cannot convert colour into fill.
    """
    with pytest.raises(InvalidWallCalibrationError):
        WallFillCalibration(measurement=measurement, scale=2.0, offset=float("nan"))


# %% causal filtering


@pytest.fixture
def estimator(
    measurement: WallColourMeasurement, region: numpy.ndarray
) -> WallFillEstimator:
    """
    A short filter for testing stream rejection without a camera.
    """
    return WallFillEstimator(
        calibration=WallFillCalibration(measurement=measurement, scale=2.0),
        region=region,
        window_frames=3,
    )


def test_warmup_has_no_measurement(
    estimator: WallFillEstimator, region: numpy.ndarray
) -> None:
    """
    A full observation window is required before publishing a fill.
    """
    for _ in range(estimator.window_frames - 1):
        assert estimator.observe(numpy.zeros(region.shape)) is None


def test_transient_stream_does_not_become_contents(
    estimator: WallFillEstimator, region: numpy.ndarray
) -> None:
    """
    A brief yellow stream over an empty wall must not increase fill.
    """
    empty = numpy.zeros(region.shape)
    stream = numpy.full(region.shape, estimator.calibration.measurement.colour_span)
    for _ in range(estimator.window_frames):
        estimator.observe(empty)
    assert estimator.observe(stream) == 0.0
    assert estimator.observe(empty) == 0.0


def test_persistent_contents_are_measured(
    estimator: WallFillEstimator, region: numpy.ndarray
) -> None:
    """
    Stable contents become measurable after the causal window.
    """
    contents = numpy.full(region.shape, estimator.calibration.measurement.colour_span)
    for _ in range(estimator.window_frames):
        result = estimator.observe(contents)
    expected = estimator.calibration.fill_for(
        estimator.calibration.measurement.measure(contents, region)
    )
    assert result == expected


def test_observations_are_copied(
    estimator: WallFillEstimator, region: numpy.ndarray
) -> None:
    """
    A camera reusing its input array must not rewrite observation history.
    """
    image = numpy.zeros(region.shape)
    estimator.observe(image)
    image.fill(estimator.calibration.measurement.colour_span)
    for _ in range(estimator.window_frames - 1):
        result = estimator.observe(image)
    assert result == 0.0


def test_empty_region_is_rejected(
    estimator: WallFillEstimator, region: numpy.ndarray
) -> None:
    """
    A missing receiver cannot produce a fill measurement.
    """
    with pytest.raises(InvalidWallObservationError):
        replace(estimator, region=numpy.zeros_like(region))


def test_wrong_image_shape_is_rejected(estimator: WallFillEstimator) -> None:
    """
    A frame from another crop must not be mixed into the filter.
    """
    with pytest.raises(InvalidWallObservationError):
        estimator.observe(numpy.zeros((1, 1)))
