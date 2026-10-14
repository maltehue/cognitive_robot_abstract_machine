"""
What is poured, and what the camera and the scale need to know about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Optional

from giskardpy.middleware.ros2.particle_weighing import FULL_CUP_MASS_GRAMS

from experiments.real_pouring.stream_detection import (
    ChangeSegmentation,
    GrainSegmentation,
    WhiteSegmentation,
)

STYROFOAM_PIXEL_SECONDS_PER_GRAIN = 29.15
"""
How much stream area, times the time it shows, one of the live bank's grains (a percent
of a full Jeroen cup) adds for the styrofoam balls: 1100 pixel-seconds per gram in the
five demo pours of 2026-10-12, times 2.65 g per full cup over 100 grains.
"""


CORN_FULL_CUP_MASS_GRAMS: Optional[float] = 361.83
"""
What the corn kernels weigh when they fill the cup, in grams; ``None`` until a full cup
of them has been weighed. Weighed on 2026-10-14 as the received, spilled and left-over
masses of one pour: 50.24 + 4.77 + 306.82 g.
"""

CORN_PIXEL_SECONDS_PER_GRAIN = 173.0
"""
How much changed area, times the time it shows, one of the live bank's grains of corn
adds: the styrofoam value times 5.93, the factor between the camera's count and the
scale in the first corn pour on 2026-10-14.
"""


@dataclass
class FullCupMassUnknownError(Exception):
    """
    Raised when a pour of a material is to be weighed before anyone weighed a full cup
    of it.
    """

    material: PouredMaterial
    """
    The material no full cup was weighed of.
    """

    def __str__(self) -> str:
        return (
            f"Nobody has weighed a full cup of {self.material} yet: fill the cup as for "
            f"a pour, weigh what is in it, and set the mass in "
            f"experiments/real_pouring/poured_material.py."
        )


class PouredMaterial(StrEnum):
    """
    A material the robot pours.
    """

    STYROFOAM = "styrofoam"
    """
    White styrofoam balls.
    """

    CORN = "corn"
    """
    Corn kernels, yellow to brownish.
    """

    @property
    def segmentation(self) -> GrainSegmentation:
        """
        :return: How the camera tells this material's grains from the empty view.
        """
        match self:
            case PouredMaterial.STYROFOAM:
                return WhiteSegmentation()
            case PouredMaterial.CORN:
                return ChangeSegmentation()

    @property
    def pixel_seconds_per_grain(self) -> float:
        """
        :return: How much stream area, times the time it shows, one of the live bank's
            grains of this material adds; the demo prints the factor to correct it by
            after a weighed pour.
        """
        match self:
            case PouredMaterial.STYROFOAM:
                return STYROFOAM_PIXEL_SECONDS_PER_GRAIN
            case PouredMaterial.CORN:
                return CORN_PIXEL_SECONDS_PER_GRAIN

    @property
    def full_cup_mass_grams(self) -> float:
        """
        :return: The mass of this material when it fills the cup, in grams.
        :raises FullCupMassUnknownError: If no full cup of it has been weighed.
        """
        match self:
            case PouredMaterial.STYROFOAM:
                return FULL_CUP_MASS_GRAMS
            case PouredMaterial.CORN if CORN_FULL_CUP_MASS_GRAMS is not None:
                return CORN_FULL_CUP_MASS_GRAMS
            case PouredMaterial.CORN:
                raise FullCupMassUnknownError(material=self)
