"""
Recording-level calibration splits and comparisons against independent weighing.
"""

from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy
import pytest

from experiments.real_pouring.wall_fill import WallColourMeasurement, WallFillSample
from experiments.real_pouring.wall_fill_validation import (
    InvalidWallDatasetError,
    WallFillDataset,
    WallFillObservation,
    WallFillPartition,
    WallFillRecording,
    WallFillArtifact,
    SETTLED_FRAMES,
    RETURN_MARGIN_FRAMES,
    HeldTiltFrames,
    SourceReturn,
    WallCacheKey,
    causal_window_frames,
    main,
)
from krrood.adapters.json_serializer import from_json

# %% complete recordings stay in one partition


@pytest.fixture
def dataset() -> WallFillDataset:
    """
    Two calibration pours and an independent validation pour.
    """
    return WallFillDataset(
        measurement=WallColourMeasurement(2.0, 32.0, 2.0),
        development=("first", "second"),
        validation=("third",),
        excluded=(),
    )


@pytest.fixture
def observations() -> list[WallFillObservation]:
    """
    Calibrated signals with a validation label that can be varied independently.
    """
    return [
        WallFillObservation(
            "first", WallFillPartition.DEVELOPMENT, WallFillSample(0.1, 0.2), 0.1
        ),
        WallFillObservation(
            "second", WallFillPartition.DEVELOPMENT, WallFillSample(0.2, 0.4), 0.2
        ),
        WallFillObservation(
            "third", WallFillPartition.VALIDATION, WallFillSample(0.15, 0.3), 0.15
        ),
    ]


def test_validation_weighing_cannot_change_calibration(
    dataset: WallFillDataset, observations: list[WallFillObservation]
) -> None:
    """
    Calibration must be independent of every held-out target.
    """
    changed = [
        *observations[:-1],
        replace(observations[-1], sample=WallFillSample(0.15, 0.9)),
    ]
    assert dataset.calibrate(observations) == dataset.calibrate(changed)


def test_overlapping_partitions_are_rejected(dataset: WallFillDataset) -> None:
    """
    The same recording cannot contribute both training and validation evidence.
    """
    with pytest.raises(InvalidWallDatasetError):
        replace(dataset, validation=dataset.development)


def test_missing_development_recording_is_rejected(
    dataset: WallFillDataset, observations: list[WallFillObservation]
) -> None:
    """
    A partial cache must not silently change the declared calibration cohort.
    """
    with pytest.raises(InvalidWallDatasetError):
        dataset.calibrate(observations[1:])


def test_each_development_prediction_excludes_its_own_weighing(
    dataset: WallFillDataset, observations: list[WallFillObservation]
) -> None:
    """
    A left-out recording's label cannot influence its predicted fill.
    """
    original = dataset.cross_validate(observations)
    changed = [
        replace(
            observations[0], sample=WallFillSample(observations[0].sample.signal, 0.8)
        ),
        *observations[1:],
    ]
    repeated = dataset.cross_validate(changed)
    assert original[0].estimate == repeated[0].estimate


def test_out_of_range_observation_uses_baseline(
    dataset: WallFillDataset, observations: list[WallFillObservation]
) -> None:
    """
    A recording below the calibration range retains the measured baseline.
    """
    observation = replace(observations[-1], sample=WallFillSample(0.01, 0.02))
    result = observation.evaluate(dataset.calibrate(observations))
    assert result.estimate == observation.baseline
    assert result.calibrated is False


# %% recorded camera crops, available without ROS or Downloads in CI


MAXIMUM_VALIDATION_ERROR = 0.03
"""
Acceptance bound of three receiver percentage points for the reserved pours.
"""

MAXIMUM_RELATIVE_ERROR = 0.7
"""
Require at least a thirty-percent error reduction against the legacy wall gauge.
"""


@pytest.fixture
def recording_directory() -> Path:
    """
    Compact camera windows at the held tilt and the original weighing for twenty-four
    pours.
    """
    return Path(__file__).parent / "dataset" / "wall_fill"


@pytest.fixture
def recorded_dataset(recording_directory: Path) -> WallFillDataset:
    """
    The declared development and reserved-pour split.
    """
    return from_json(
        json.loads((recording_directory / WallFillArtifact.DATASET).read_text())
    )


@pytest.fixture
def recorded_observations(
    recorded_dataset: WallFillDataset, recording_directory: Path
) -> list[WallFillObservation]:
    """
    Production features measured from the retained camera windows.
    """
    return [
        recording.observe(recorded_dataset.measurement)
        for recording in recorded_dataset.read(recording_directory)
    ]


def test_reserved_pours_improve_on_geometric_wall_readings(
    recorded_dataset: WallFillDataset, recorded_observations: list[WallFillObservation]
) -> None:
    """
    Whole pours excluded from calibration satisfy the offline accuracy bound.
    """
    calibration = recorded_dataset.calibrate(recorded_observations)
    results = [
        item.evaluate(calibration)
        for item in recorded_observations
        if item.partition is WallFillPartition.VALIDATION
    ]
    errors = numpy.array([abs(result.error) for result in results])
    baseline = numpy.array(
        [abs(result.baseline - result.weighed) for result in results]
    )
    assert errors.max() <= MAXIMUM_VALIDATION_ERROR
    assert errors.mean() < MAXIMUM_RELATIVE_ERROR * baseline.mean()


def test_left_out_development_pours_improve_on_geometric_wall_readings(
    recorded_dataset: WallFillDataset, recorded_observations: list[WallFillObservation]
) -> None:
    """
    Each development prediction is evaluated without fitting to its own weighing.
    """
    results = recorded_dataset.cross_validate(recorded_observations)
    error = numpy.mean([abs(result.error) for result in results])
    baseline = numpy.mean([abs(result.baseline - result.weighed) for result in results])
    assert error < MAXIMUM_RELATIVE_ERROR * baseline


def test_endpoint_evaluation_matches_causal_replay(
    recorded_dataset: WallFillDataset,
    recorded_observations: list[WallFillObservation],
    recording_directory: Path,
) -> None:
    """
    Reported endpoint accuracy includes the production range fallback per frame.
    """
    calibration = recorded_dataset.calibrate(recorded_observations)
    for name in recorded_dataset.validation:
        recording = WallFillRecording.read(
            recording_directory / (name + WallFillArtifact.CACHE_SUFFIX),
            WallFillPartition.VALIDATION,
        )
        replay = recording.replay(calibration)
        result = recording.observe(recorded_dataset.measurement).evaluate(calibration)
        assert numpy.median(replay[-SETTLED_FRAMES:]) == pytest.approx(result.estimate)


def test_command_writes_reloadable_calibration(
    recording_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    The offline command emits a model that the production detector can load.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_wall_fill",
            str(recording_directory),
            str(recording_directory / WallFillArtifact.DATASET),
            str(tmp_path),
        ],
    )
    main()
    calibration = from_json(
        json.loads((tmp_path / WallFillArtifact.CALIBRATION).read_text())
    )
    result = from_json(json.loads((tmp_path / WallFillArtifact.RESULTS).read_text()))
    assert calibration == result.calibration
    assert (
        tuple(row.recording for row in result.validation) == result.dataset.validation
    )


# %% frames read while the source is still held over the receiver


@pytest.fixture
def held_tilts() -> numpy.ndarray:
    """
    A source turned to its pouring tilt, held there with jitter, returned upright and
    left standing.
    """
    rising = numpy.linspace(40.0, 85.0, 50)
    holding = 85.0 + 0.01 * numpy.sin(numpy.arange(40))
    returning = numpy.linspace(84.0, 0.0, 30)
    standing = numpy.zeros(20)
    return numpy.concatenate([rising, holding, returning, standing])


def test_hold_ends_before_the_source_turns_back(held_tilts: numpy.ndarray) -> None:
    """
    The last held frame is the one before the return, less the margin.
    """
    last_held = 50 + 40 - 1
    assert SourceReturn(held_tilts).held_tilt_end() == last_held - RETURN_MARGIN_FRAMES


def test_a_source_that_never_returns_is_rejected(held_tilts: numpy.ndarray) -> None:
    """
    A recording cut before the return has no known end of the hold.
    """
    with pytest.raises(InvalidWallDatasetError):
        SourceReturn(held_tilts[:90]).held_tilt_end()


def test_a_source_that_never_tilts_is_rejected() -> None:
    """
    A recording without a pour has no hold.
    """
    with pytest.raises(InvalidWallDatasetError):
        SourceReturn(numpy.zeros(30)).held_tilt_end()


def test_retained_frames_end_at_the_held_tilt(
    held_tilts: numpy.ndarray, tmp_path: Path
) -> None:
    """
    A compact cache keeps the scored frames and their causal history up to the end of
    the hold, labelled here by each frame's index.
    """
    count = len(held_tilts)
    times = numpy.arange(count) / 30.0
    full = tmp_path / "full.npz"
    numpy.savez_compressed(
        full,
        **{
            WallCacheKey.DIFFERENCES: numpy.arange(count, dtype=numpy.int16)[
                :, None, None
            ]
            * numpy.ones((1, 2, 2), dtype=numpy.int16),
            WallCacheKey.REGION: numpy.ones((2, 2), dtype=bool),
            WallCacheKey.TIMES: times,
            WallCacheKey.TILTS: held_tilts,
            WallCacheKey.FILLS: numpy.linspace(0.0, 1.0, 3),
            WallCacheKey.AREAS: numpy.linspace(0.0, 10.0, 3),
            WallCacheKey.DETECTED_SHARE: numpy.array(0.5),
        },
    )
    compact = tmp_path / "compact.npz"
    HeldTiltFrames(full).write(compact)
    end = SourceReturn(held_tilts).held_tilt_end()
    with numpy.load(compact) as cache:
        retained = cache[WallCacheKey.DIFFERENCES][:, 0, 0]
        assert WallCacheKey.TILTS not in cache.files
        assert cache[WallCacheKey.TIMES] == pytest.approx(
            times[end + 1 - len(retained) : end + 1]
        )
    assert retained[-1] == end
    assert len(retained) == SETTLED_FRAMES + causal_window_frames(times) - 1
