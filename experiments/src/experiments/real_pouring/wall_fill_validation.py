"""Reproducible recording-level validation of calibrated translucent-cup perception."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import numpy
from typing_extensions import Sequence

from experiments.real_pouring.receiver_heap import MINIMUM_HEAP_PIXELS
from experiments.real_pouring.stream_detection import (
    MINIMUM_YELLOWING,
    WALL_STILL_SECONDS,
)
from experiments.real_pouring.wall_fill import (
    WallColourMeasurement,
    WallFillCalibration,
    WallFillEstimator,
    WallFillSample,
)
from giskardpy.middleware.ros2.particle_weighing import WeighedParticles
from krrood.adapters.json_serializer import from_json, to_json

# %% recording and artifact formats


SETTLED_FRAMES = 30
"""Final one-second camera window used for each recording's weighed comparison."""


class WallFillPartition(StrEnum):
    """Whether a whole recording may contribute to calibration."""

    DEVELOPMENT = "development"
    """Available for fitting and model development."""
    VALIDATION = "validation"
    """Excluded from fitting."""


class WallCacheKey(StrEnum):
    """Arrays written by the wall-yellowing cache extractor."""

    DIFFERENCES = "diffs"
    """Signed Lab yellowing images."""
    REGION = "region"
    """Receiver pixels in the cropped images."""
    TIMES = "times"
    """Frame times relative to the pour start."""
    FILLS = "map_fills"
    """Fill fractions in the geometric lookup."""
    AREAS = "map_areas"
    """Projected contents areas in the geometric lookup."""
    DETECTED_SHARE = "share"
    """Legacy detected fraction of projected contents."""


class WallFillArtifact(StrEnum):
    """Files produced by an offline wall-fill evaluation."""

    CALIBRATION = "calibration.json"
    """Fitted colour model and its supported signal interval."""
    RESULTS = "results.json"
    """Recording-level predictions and dataset provenance."""
    DATASET = "dataset.json"
    """Declared recording partitions and colour features."""
    JSON_SUFFIX = ".json"
    """Extension of an adjacent weighing file."""
    CACHE_SUFFIX = ".npz"
    """Extension of a compressed wall-image cache."""


class WallFillArgument(StrEnum):
    """Command-line inputs to the offline evaluation."""

    CACHE = "cache"
    """Directory containing cached images and weighings."""
    DATASET = "dataset"
    """Path to the declared recording split."""
    OUTPUT = "output"
    """Directory receiving calibration and evaluation results."""


@dataclass
class InvalidWallDatasetError(ValueError):
    """Raised when recording identities, partitions, or settled observations are invalid."""


# %% weighed endpoint observations


@dataclass(frozen=True)
class WallFillResult:
    """One recording's estimated fill against its independent weighing."""

    recording: str
    """Identity of the complete recording."""

    partition: WallFillPartition
    """Whether the recording was available for model development."""

    weighed: float
    """Received mass divided by full receiver mass."""

    estimate: float
    """Median estimated fill over the settled observation window."""

    baseline: float
    """Median legacy wall-area estimate over the same window."""

    calibrated: bool
    """Whether every settled frame was inside the colour calibration range."""

    @property
    def error(self) -> float:
        """Return signed error in receiver fill fraction."""
        return self.estimate - self.weighed


@dataclass(frozen=True)
class WallFillObservation:
    """One equally weighted calibration sample and its settled frame measurements."""

    recording: str
    """Identity used to keep all frames from a pour in one partition."""

    partition: WallFillPartition
    """Whether this recording may influence calibration."""

    sample: WallFillSample
    """Median settled colour signal paired with the received fill."""

    baseline: float
    """Median geometric wall reading over the settled frames."""

    signals: tuple[float, ...] = ()
    """Individual settled signals, retaining range and fallback behaviour."""

    baselines: tuple[float, ...] = ()
    """Geometric readings corresponding to the settled colour signals."""

    def __post_init__(self) -> None:
        """Require corresponding colour and geometric observations."""
        if len(self.signals) != len(self.baselines):
            raise InvalidWallDatasetError()

    def evaluate(self, calibration: WallFillCalibration) -> WallFillResult:
        """Compare the calibrated-or-fallback estimate with independent weighing."""
        signals = self.signals or (self.sample.signal,)
        baselines = self.baselines or (self.baseline,)
        calibrated = [calibration.fill_for(signal) for signal in signals]
        estimates = [
            baseline if estimate is None else estimate
            for estimate, baseline in zip(calibrated, baselines)
        ]
        return WallFillResult(
            recording=self.recording,
            partition=self.partition,
            weighed=self.sample.fill,
            estimate=float(numpy.median(estimates)),
            baseline=self.baseline,
            calibrated=all(estimate is not None for estimate in calibrated),
        )


# %% reading causal image evidence


@dataclass
class WallFillRecording:
    """Cached wall images and their final weighing, independent of the pour goal."""

    recording: str
    """Bag directory name identifying the recording."""

    partition: WallFillPartition
    """The explicitly assigned calibration partition."""

    differences: numpy.ndarray
    """Signed per-frame Lab yellowing relative to the learned empty view."""

    region: numpy.ndarray
    """Receiver mask in the wall crop."""

    times: numpy.ndarray
    """Camera times relative to the pour start, in seconds."""

    fills: numpy.ndarray
    """Fill fractions in the legacy geometric lookup."""

    areas: numpy.ndarray
    """Projected areas corresponding to the lookup's fill fractions."""

    detected_share: float
    """Legacy fraction of projected contents detected through the wall."""

    weighed: float
    """Final received fill measured on the scale."""

    @classmethod
    def read(cls, path: Path, partition: WallFillPartition) -> WallFillRecording:
        """Read a colour cache and its adjacent weighed-particles JSON file."""
        weighing = WeighedParticles.from_json(
            json.loads(path.with_suffix(WallFillArtifact.JSON_SUFFIX).read_text())
        )
        with numpy.load(path, allow_pickle=False) as cache:
            return cls(
                recording=path.stem,
                partition=partition,
                differences=cache[WallCacheKey.DIFFERENCES],
                region=cache[WallCacheKey.REGION],
                times=cache[WallCacheKey.TIMES],
                fills=cache[WallCacheKey.FILLS],
                areas=cache[WallCacheKey.AREAS],
                detected_share=float(cache[WallCacheKey.DETECTED_SHARE]),
                weighed=weighing.received_fill_level,
            )

    @property
    def window_frames(self) -> int:
        """Return the causal stillness window at the recorded camera frequency."""
        intervals = numpy.diff(self.times)
        if (
            not len(intervals)
            or not numpy.isfinite(intervals).all()
            or (intervals <= 0).any()
        ):
            raise InvalidWallDatasetError()
        return max(1, round(WALL_STILL_SECONDS / float(numpy.median(intervals))))

    def baseline_for(self, persistent: numpy.ndarray) -> float | None:
        """Apply the legacy threshold and geometric conversion to persistent colour."""
        pixels = int(((persistent >= MINIMUM_YELLOWING) & self.region).sum())
        if pixels < MINIMUM_HEAP_PIXELS:
            return None
        return float(numpy.interp(pixels / self.detected_share, self.areas, self.fills))

    def observe(self, measurement: WallColourMeasurement) -> WallFillObservation:
        """Summarize final frames without using their weighing to derive image features."""
        required = SETTLED_FRAMES + self.window_frames - 1
        if len(self.differences) < required:
            raise InvalidWallDatasetError()
        recent = self.differences[-required:].astype(float)
        signals = []
        baselines = []
        for index in range(self.window_frames - 1, len(recent)):
            persistent = recent[index + 1 - self.window_frames : index + 1].min(axis=0)
            baseline = self.baseline_for(persistent)
            if baseline is None:
                raise InvalidWallDatasetError()
            signals.append(measurement.measure(persistent, self.region))
            baselines.append(baseline)
        return WallFillObservation(
            recording=self.recording,
            partition=self.partition,
            sample=WallFillSample(float(numpy.median(signals)), self.weighed),
            baseline=float(numpy.median(baselines)),
            signals=tuple(signals),
            baselines=tuple(baselines),
        )

    def replay(self, calibration: WallFillCalibration) -> list[float | None]:
        """Replay calibrated readings with the same geometric fallback as the detector."""
        estimator = WallFillEstimator(calibration, self.region, self.window_frames)
        history: deque[numpy.ndarray] = deque(maxlen=self.window_frames)
        readings = []
        for difference in self.differences:
            history.append(difference)
            estimate = estimator.observe(difference)
            if estimate is None and len(history) == self.window_frames:
                estimate = self.baseline_for(numpy.minimum.reduce(history))
            readings.append(estimate)
        return readings


# %% partition-safe calibration and validation


@dataclass(frozen=True)
class WallFillDataset:
    """An explicit split of whole recordings for one cup, material and camera view."""

    measurement: WallColourMeasurement
    """Colour features chosen during model development."""

    development: tuple[str, ...]
    """Recording identities allowed to determine calibration."""

    validation: tuple[str, ...]
    """Recording identities excluded from calibration."""

    excluded: tuple[str, ...]
    """Known recordings intentionally omitted from the evaluation."""

    def __post_init__(self) -> None:
        """Reject overlapping partitions and insufficient calibration data."""
        recordings = (*self.development, *self.validation, *self.excluded)
        if len(self.development) < 2 or len(set(recordings)) != len(recordings):
            raise InvalidWallDatasetError()

    def read(self, directory: Path) -> list[WallFillRecording]:
        """Load only the declared development and validation caches."""
        return [
            WallFillRecording.read(
                directory / (recording + WallFillArtifact.CACHE_SUFFIX), partition
            )
            for partition, recordings in (
                (WallFillPartition.DEVELOPMENT, self.development),
                (WallFillPartition.VALIDATION, self.validation),
            )
            for recording in recordings
        ]

    def _fit(self, observations: Sequence[WallFillObservation]) -> WallFillCalibration:
        """Fit equal recording weights with support bounded by their settled frames."""
        calibration = WallFillCalibration.fit(
            self.measurement, [item.sample for item in observations]
        )
        signals = [
            signal
            for item in observations
            for signal in (item.signals or (item.sample.signal,))
        ]
        return replace(
            calibration, minimum_signal=min(signals), maximum_signal=max(signals)
        )

    def _development(
        self, observations: Sequence[WallFillObservation]
    ) -> list[WallFillObservation]:
        """Verify recording identities before selecting calibration observations."""
        identities = [item.recording for item in observations]
        if len(set(identities)) != len(identities) or set(identities) != set(
            (*self.development, *self.validation)
        ):
            raise InvalidWallDatasetError()
        for item in observations:
            expected = (
                WallFillPartition.DEVELOPMENT
                if item.recording in self.development
                else WallFillPartition.VALIDATION
            )
            if item.partition != expected:
                raise InvalidWallDatasetError()
        return [
            item
            for item in observations
            if item.partition is WallFillPartition.DEVELOPMENT
        ]

    def calibrate(
        self, observations: Sequence[WallFillObservation]
    ) -> WallFillCalibration:
        """Fit exclusively on the declared development recordings."""
        return self._fit(self._development(observations))

    def cross_validate(
        self, observations: Sequence[WallFillObservation]
    ) -> list[WallFillResult]:
        """Predict each development recording with that entire recording left out."""
        development = self._development(observations)
        return [
            item.evaluate(
                self._fit(
                    [
                        other
                        for other in development
                        if other.recording != item.recording
                    ]
                )
            )
            for item in development
        ]


@dataclass(frozen=True)
class WallFillEvaluation:
    """Calibration and recording-level results with a reproducible dataset split."""

    dataset: WallFillDataset
    """The feature configuration and whole-recording split."""

    calibration: WallFillCalibration
    """Model fitted exclusively to development recordings."""

    cross_validation: list[WallFillResult]
    """Development predictions fitted without the recording being predicted."""

    validation: list[WallFillResult]
    """Predictions of recordings excluded from calibration."""


def main() -> None:
    """Fit and evaluate a declared dataset of wall caches without ROS or live calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in WallFillArgument:
        parser.add_argument(argument, type=Path)
    arguments = parser.parse_args()
    dataset = from_json(json.loads(arguments.dataset.read_text()))
    if not isinstance(dataset, WallFillDataset):
        raise InvalidWallDatasetError()
    recordings = dataset.read(arguments.cache)
    observations = [recording.observe(dataset.measurement) for recording in recordings]
    calibration = dataset.calibrate(observations)
    evaluation = WallFillEvaluation(
        dataset,
        calibration,
        dataset.cross_validate(observations),
        [
            item.evaluate(calibration)
            for item in observations
            if item.partition is WallFillPartition.VALIDATION
        ],
    )
    arguments.output.mkdir(parents=True, exist_ok=True)
    (arguments.output / WallFillArtifact.CALIBRATION).write_text(
        json.dumps(to_json(calibration), indent=2)
    )
    (arguments.output / WallFillArtifact.RESULTS).write_text(
        json.dumps(to_json(evaluation), indent=2)
    )
    for result in (*evaluation.cross_validation, *evaluation.validation):
        print(
            f"{result.recording} {result.partition}: weighed {result.weighed:.4f}, estimate {result.estimate:.4f}, baseline {result.baseline:.4f}, calibrated {result.calibrated}"
        )
