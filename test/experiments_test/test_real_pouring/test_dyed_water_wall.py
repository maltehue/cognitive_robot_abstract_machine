from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy
import pytest

from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import (
    TRANSLUCENT_CUP_WATER_MASS_GRAMS,
    ReceivingCup,
)
from experiments.real_pouring.stream_detection import WALL_STILL_SECONDS, CameraView

# %% a hand pour of dyed water into the translucent cup, held still three times

RECORDING = (
    Path(__file__).parent / "dataset" / "dyed_water" / "2026-10-29_hand_pour.npz"
)
"""
The pole camera's view of water dyed near black poured by hand into the translucent cup
(2026-10-29), cut around the cup: the empty view, and the median of each stretch the
pour was held still, with what the scale read then.
"""

FRAME_PERIOD = 1 / 22
"""
How long apart the camera took the video's images, in seconds.
"""

READING_TOLERANCE = 0.02
"""
How far the wall's reading may lie from the weighed fill: two image rows of the water
line.
"""


class HandPourKey(StrEnum):
    """
    The arrays the recording holds.
    """

    EMPTY = "empty"
    """
    The cut of the empty view.
    """

    STOPS = "stops"
    """
    The cuts of the median view of each stop.
    """

    STOP_MASSES_GRAMS = "stop_masses_grams"
    """
    What the scale read at each stop.
    """

    CORNER = "corner"
    """
    The image row and column of the cuts' first pixel.
    """

    IMAGE_SIZE = "image_size"
    """
    Height and width of the camera's images.
    """

    INTRINSICS = "intrinsics"
    """
    The camera's intrinsic matrix.
    """

    WORLD_T_CAMERA = "world_T_camera"
    """
    The camera's pose in the world.
    """

    OPENING_CENTRE = "opening_centre"
    """
    The centre of the cup's opening, in the world.
    """


@dataclass
class HandPour:
    """
    The recording, its cut-outs put back where they lie in the camera's image.
    """

    view: CameraView
    """
    The pole camera's calibration.
    """

    opening_centre: numpy.ndarray
    """
    The centre of the cup's opening, in the world.
    """

    empty: numpy.ndarray
    """
    The view before the pour, BGR.
    """

    stops: list[numpy.ndarray]
    """
    The view while the pour was held still, each time, BGR.
    """

    stop_masses_grams: list[float]
    """
    What the scale read at each stop, in grams.
    """

    @classmethod
    def read(cls, path: Path) -> HandPour:
        """
        :param path: The recording.
        :return: Its views put back into whole camera images, black around the cut.
        """
        with numpy.load(path) as recording:
            image_size = tuple(recording[HandPourKey.IMAGE_SIZE])
            row, column = recording[HandPourKey.CORNER]

            def placed(cut: numpy.ndarray) -> numpy.ndarray:
                image = numpy.zeros((*image_size, 3), dtype=numpy.uint8)
                image[row : row + cut.shape[0], column : column + cut.shape[1]] = cut
                return image

            return cls(
                view=CameraView(
                    intrinsics=recording[HandPourKey.INTRINSICS],
                    world_T_camera=recording[HandPourKey.WORLD_T_CAMERA],
                ),
                opening_centre=recording[HandPourKey.OPENING_CENTRE],
                empty=placed(recording[HandPourKey.EMPTY]),
                stops=[placed(stop) for stop in recording[HandPourKey.STOPS]],
                stop_masses_grams=list(recording[HandPourKey.STOP_MASSES_GRAMS]),
            )


@pytest.fixture(scope="module")
def hand_pour() -> HandPour:
    """
    The recorded hand pour.
    """
    return HandPour.read(RECORDING)


@pytest.mark.parametrize("stop", range(3))
def test_the_wall_reads_the_weighed_fill_of_dyed_water_held_still(hand_pour, stop):
    detector = ReceivingCup.TRANSLUCENT_CUP.detector_for(
        material=PouredMaterial.DYED_WATER,
        view=hand_pour.view,
        opening_centre=hand_pour.opening_centre,
        frame_period=FRAME_PERIOD,
    )
    detector.learn_background([hand_pour.empty])
    for frame in range(round(WALL_STILL_SECONDS / FRAME_PERIOD) + 1):
        reading = detector.observe(hand_pour.stops[stop], stamp=frame * FRAME_PERIOD)
    assert reading.heap_fill == pytest.approx(
        hand_pour.stop_masses_grams[stop] / TRANSLUCENT_CUP_WATER_MASS_GRAMS,
        abs=READING_TOLERANCE,
    )
