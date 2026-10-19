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
    Compact final camera windows and the original weighing for eighteen pours.
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
