"""
Cup-specific fill estimates from persistent colour through a translucent wall.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy
from typing_extensions import Sequence

# %% measurement contracts


@dataclass
class InvalidWallCalibrationError(ValueError):
    """
    Raised when colour measurements cannot define a finite fill calibration.
    """


@dataclass
class InvalidWallObservationError(ValueError):
    """
    Raised when an observation does not contain a finite receiver image.
    """


@dataclass(frozen=True)
class WallColourMeasurement:
    """
    Median colour shift across each receiver row, weighted by its image height.
    """

    noise_floor: float
    """
    Colour shift discarded before accumulating colour evidence, in the shift's units.
    """

    colour_span: float
    """
    Colour shift above the floor that saturates a pixel's contribution.
    """

    height_power: float
    """
    Exponent weighting image height from zero at the crop bottom to one at its top.
    """

    def __post_init__(self) -> None:
        """
        Require finite, nonnegative weights and a positive colour span.
        """
        parameters = (self.noise_floor, self.colour_span, self.height_power)
        if not all(math.isfinite(value) for value in parameters):
            raise InvalidWallCalibrationError()
        if self.noise_floor < 0 or self.colour_span <= 0 or self.height_power < 0:
            raise InvalidWallCalibrationError()

    def measure(self, colour_shift: numpy.ndarray, region: numpy.ndarray) -> float:
        """
        Return colour evidence per receiver pixel in a tightly cropped wall image.
        """
        if (
            colour_shift.ndim != 2
            or region.dtype != numpy.bool_
            or colour_shift.shape != region.shape
            or not region.any()
            or not numpy.isfinite(colour_shift).all()
        ):
            raise InvalidWallObservationError()
        height = numpy.linspace(1.0, 0.0, region.shape[0])
        populated = region.any(axis=1)
        profile = numpy.zeros(region.shape[0])
        profile[populated] = numpy.nanmedian(
            numpy.where(region[populated], colour_shift[populated], numpy.nan), axis=1
        )
        colour = numpy.clip((profile - self.noise_floor) / self.colour_span, 0.0, 1.0)
        return float(
            (colour * height**self.height_power * region.sum(axis=1)).sum()
            / region.sum()
        )


# %% calibration against weighed pours


@dataclass(frozen=True)
class WallFillSample:
    """
    One recording's settled colour evidence and independently weighed fill.
    """

    signal: float
    """
    Settled colour evidence measured without the weighing.
    """

    fill: float
    """
    Received mass divided by the receiver's full mass.
    """

    def __post_init__(self) -> None:
        """
        Reject measurements outside the calibration domain.
        """
        if (
            not math.isfinite(self.signal)
            or not math.isfinite(self.fill)
            or not 0 <= self.signal <= 1
            or not 0 <= self.fill <= 1
        ):
            raise InvalidWallCalibrationError()


@dataclass(frozen=True)
class WallFillCalibration:
    """
    A linear colour-to-fill calibration for one cup, material and camera view.
    """

    measurement: WallColourMeasurement
    """
    The colour and spatial weighting used to measure each image.
    """

    scale: float
    """
    Fill per unit colour evidence, fitted from independent weighed recordings.
    """

    offset: float = 0.0
    """
    Fill at zero colour evidence on the fitted line, applied only to nonempty evidence:

    negative where faint colour shows before any contents do.
    """

    minimum_signal: float = 0.0
    """
    Least nonempty colour evidence supported by the calibration recordings.
    """

    maximum_signal: float = 1.0
    """
    Greatest colour evidence supported by the calibration recordings.
    """

    def __post_init__(self) -> None:
        """
        Require a finite positive conversion from colour to fill.
        """
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise InvalidWallCalibrationError()
        if not math.isfinite(self.offset):
            raise InvalidWallCalibrationError()
        if not 0 <= self.minimum_signal <= self.maximum_signal <= 1:
            raise InvalidWallCalibrationError()

    @classmethod
    def fit(
        cls, measurement: WallColourMeasurement, samples: Sequence[WallFillSample]
    ) -> WallFillCalibration:
        """
        Fit scale and offset by least squares, giving each recording equal weight.

        Samples of a single colour level cannot separate an offset from the scale; their
        line passes through the origin.
        """
        signals = numpy.array([sample.signal for sample in samples])
        fills = numpy.array([sample.fill for sample in samples])
        if not (signals > 0).any():
            raise InvalidWallCalibrationError()
        if len(numpy.unique(signals)) < 2:
            scale, offset = float(signals @ fills / (signals @ signals)), 0.0
        else:
            scale, offset = numpy.polyfit(signals, fills, 1)
        return cls(
            measurement=measurement,
            scale=float(scale),
            offset=float(offset),
            minimum_signal=float(signals.min()),
            maximum_signal=float(signals.max()),
        )

    def fill_for(self, signal: float) -> float | None:
        """
        Return fill within calibration support, or ``None`` for extrapolation.
        """
        if not math.isfinite(signal) or signal < 0:
            raise InvalidWallObservationError()
        if signal == 0:
            return 0.0
        if not self.minimum_signal <= signal <= self.maximum_signal:
            return None
        return min(1.0, max(0.0, self.scale * signal + self.offset))


# %% causal rejection of transient yellow pixels


@dataclass
class WallFillEstimator:
    """
    Read persistent colour using only the current and preceding observations.

    The pixelwise minimum rejects colour excursions shorter than the observation window.
    Persistent occlusion and changing illumination still require validation.
    """

    calibration: WallFillCalibration
    """
    The conversion for this receiver, material and view.
    """

    region: numpy.ndarray
    """
    Receiver pixels inside a tightly cropped wall image.
    """

    window_frames: int
    """
    Consecutive observations required before colour is treated as contents.
    """

    _observations: deque[numpy.ndarray] = field(init=False, repr=False)
    """
    Recent colour shift images, copied so camera buffers cannot alter history.
    """

    def __post_init__(self) -> None:
        """
        Create a fresh observation window for a nonempty receiver mask.
        """
        if (
            self.region.ndim != 2
            or self.region.dtype != numpy.bool_
            or not self.region.any()
            or not isinstance(self.window_frames, int)
            or self.window_frames < 1
        ):
            raise InvalidWallObservationError()
        self.region = self.region.copy()
        self._observations = deque(maxlen=self.window_frames)

    def observe(self, colour_shift: numpy.ndarray) -> float | None:
        """
        Return fill after a full causal window, or ``None`` during warmup.
        """
        if (
            colour_shift.shape != self.region.shape
            or not numpy.isfinite(colour_shift).all()
        ):
            raise InvalidWallObservationError()
        self._observations.append(colour_shift.astype(float, copy=True))
        if len(self._observations) < self.window_frames:
            return None
        persistent = numpy.minimum.reduce(self._observations)
        return self.calibration.fill_for(
            self.calibration.measurement.measure(persistent, self.region)
        )
