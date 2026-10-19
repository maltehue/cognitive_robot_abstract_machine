"""
Calibrated wall readings passed through the production stream detector.
"""

from __future__ import annotations

import numpy
import pytest

from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import (
    NO_OFFSET,
    WALL_STILL_SECONDS,
    Yellowing,
)
from experiments.real_pouring.wall_fill import (
    InvalidWallCalibrationError,
    WallColourMeasurement,
    WallFillCalibration,
    WallFillEstimator,
)

from .test_receiver_wall import CORN_BEHIND_THE_WALL, HEIGHT, VIEW, _cup_on_mat

# %% opt-in calibrated perception


def test_receiver_factory_passes_calibration_to_detector() -> None:
    """
    Explicit calibration reaches the detector that supplies the stream reading.
    """
    calibration = WallFillCalibration(WallColourMeasurement(0.0, 32.0, 3.0), 2.0)
    detector = ReceivingCup.TRANSLUCENT_CUP.detector_for(
        material=PouredMaterial.CORN,
        view=VIEW,
        opening_centre=numpy.array([0.0, 0.0, HEIGHT]),
        frame_period=WALL_STILL_SECONDS / 3,
        wall_calibration=calibration,
    )
    empty = _cup_on_mat(NO_OFFSET)
    detector.learn_background([empty])
    contents = empty.copy()
    contents[detector.wall_mask(empty.shape[:2])] = CORN_BEHIND_THE_WALL
    region = detector._wall_region
    yellowing = Yellowing().difference(
        contents[detector._wall_window], detector._wall_background
    )
    reference = WallFillEstimator(calibration, region, 3)
    for index in range(reference.window_frames):
        expected = reference.observe(yellowing)
        reading = detector.observe(contents, stamp=index * detector.frame_period)
    assert reading.heap_fill == expected
    assert reading.heap_fill > 0.0
    detector.learn_background([empty])
    assert detector.observe(empty, stamp=0.0).heap_fill is None


def test_opaque_receiver_rejects_wall_calibration() -> None:
    """
    A wall model must not silently apply to an opaque cup.
    """
    with pytest.raises(InvalidWallCalibrationError):
        ReceivingCup.JEROEN_CUP.detector_for(
            material=PouredMaterial.CORN,
            view=VIEW,
            opening_centre=numpy.array([0.0, 0.0, HEIGHT]),
            frame_period=WALL_STILL_SECONDS,
            wall_calibration=WallFillCalibration(
                WallColourMeasurement(0.0, 32.0, 3.0), 2.0
            ),
        )
