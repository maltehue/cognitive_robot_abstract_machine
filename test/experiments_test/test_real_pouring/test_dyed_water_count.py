from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy
import pytest

from experiments.real_pouring.pour_replay import FRAME_PERIOD, GRAINS_PER_FILL
from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup
from experiments.real_pouring.stream_detection import CameraView

# %% a robot pour of dyed water into the translucent cup, weighed after

RECORDING = (
    Path(__file__).parent
    / "dataset"
    / "dyed_water"
    / "2026-10-29_10-22-11_robot_pour.npz"
)
"""
The pole camera's view of the robot pouring water dyed near black into the translucent
cup (2026-10-29 10:22): the empty view cut around the cup, the pixels of its opening in
every frame the stream fell, and what the scale weighed in the cup after.
"""

COUNT_TOLERANCE = 0.02
"""
How far, as a share, the counted mass may lie from the weighed one.
"""


class RobotPourKey(StrEnum):
    """
    The arrays the recording holds.
    """

    EMPTY = "empty"
    """
    The cut of the empty view.
    """

    OPENING = "opening"
    """
    Which pixels of the cut are the cup's opening.
    """

    OPENING_PIXELS = "opening_pixels"
    """
    The opening's pixels in each frame of the stream.
    """

    STAMPS = "stamps"
    """
    When each frame was taken, in seconds after the pour started.
    """

    RECEIVED_MASS_GRAMS = "received_mass_grams"
    """
    What the scale weighed in the cup after the pour.
    """

    CORNER = "corner"
    """
    The image row and column of the cut's first pixel.
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
class RobotPour:
    """
    The recording, its frames put back together in whole camera images.
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

    frames: list[numpy.ndarray]
    """
    The view while the stream fell, the empty view outside the opening, BGR.
    """

    stamps: list[float]
    """
    When each frame was taken, in seconds after the pour started.
    """

    received_mass_grams: float
    """
    What the scale weighed in the cup after the pour.
    """

    @classmethod
    def read(cls, path: Path) -> RobotPour:
        """
        :param path: The recording.
        :return: Its frames put back into whole camera images, black around the cut.
        """
        with numpy.load(path) as recording:
            image_size = tuple(recording[RobotPourKey.IMAGE_SIZE])
            row, column = recording[RobotPourKey.CORNER]
            empty_cut = recording[RobotPourKey.EMPTY]
            opening = recording[RobotPourKey.OPENING]

            def placed(cut: numpy.ndarray) -> numpy.ndarray:
                image = numpy.zeros((*image_size, 3), dtype=numpy.uint8)
                image[row : row + cut.shape[0], column : column + cut.shape[1]] = cut
                return image

            def with_opening(pixels: numpy.ndarray) -> numpy.ndarray:
                cut = empty_cut.copy()
                cut[opening] = pixels
                return placed(cut)

            return cls(
                view=CameraView(
                    intrinsics=recording[RobotPourKey.INTRINSICS],
                    world_T_camera=recording[RobotPourKey.WORLD_T_CAMERA],
                ),
                opening_centre=recording[RobotPourKey.OPENING_CENTRE],
                empty=placed(empty_cut),
                frames=[
                    with_opening(pixels)
                    for pixels in recording[RobotPourKey.OPENING_PIXELS]
                ],
                stamps=list(recording[RobotPourKey.STAMPS]),
                received_mass_grams=float(recording[RobotPourKey.RECEIVED_MASS_GRAMS]),
            )


def test_the_count_reads_the_weighed_mass_of_dyed_water_poured_by_the_robot():
    robot_pour = RobotPour.read(RECORDING)
    detector = ReceivingCup.TRANSLUCENT_CUP.detector_for(
        material=PouredMaterial.DYED_WATER,
        view=robot_pour.view,
        opening_centre=robot_pour.opening_centre,
        frame_period=FRAME_PERIOD,
    )
    detector.learn_background([robot_pour.empty])
    for frame, stamp in zip(robot_pour.frames, robot_pour.stamps):
        reading = detector.observe(frame, stamp=stamp)
    grain_mass_grams = PouredMaterial.DYED_WATER.full_cup_mass_grams / GRAINS_PER_FILL
    assert reading.counted_grains * grain_mass_grams == pytest.approx(
        robot_pour.received_mass_grams, rel=COUNT_TOLERANCE
    )
