"""
The camera identifies calibrated readings and geometric fallback.
"""

from dataclasses import replace

import numpy
import pytest

from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import (
    NO_OFFSET,
    WALL_STILL_SECONDS,
    StreamDetector,
)
from experiments.real_pouring.wall_fill import (
    WallColourMeasurement,
    WallFillCalibration,
)

from .test_receiver_wall import CORN_BEHIND_THE_WALL, HEIGHT, VIEW, _cup_on_mat

# %% measurement provenance


@pytest.fixture
def wall_detector() -> StreamDetector:
    """
    A supported receiver whose uniform colour fits the calibration domain.
    """
    return ReceivingCup.TRANSLUCENT_CUP.detector_for(
        material=PouredMaterial.CORN,
        view=VIEW,
        opening_centre=numpy.array([0.0, 0.0, HEIGHT]),
        frame_period=WALL_STILL_SECONDS,
        wall_calibration=WallFillCalibration(
            WallColourMeasurement(0.0, 32.0, 3.0), 2.0
        ),
    )


@pytest.mark.parametrize("supported", [True, False])
def test_detector_publishes_the_selected_wall_measurement_source(
    wall_detector: StreamDetector, supported: bool
) -> None:
    """
    The bank can distinguish calibrated colour from out-of-range fallback.
    """
    if not supported:
        wall_detector.wall_calibration = replace(
            wall_detector.wall_calibration, maximum_signal=0.0
        )
    empty = _cup_on_mat(NO_OFFSET)
    wall_detector.learn_background([empty])
    contents = empty.copy()
    contents[wall_detector.wall_mask(empty.shape[:2])] = CORN_BEHIND_THE_WALL
    reading = wall_detector.observe(contents, stamp=0.0)
    assert reading.wall_calibrated is supported
