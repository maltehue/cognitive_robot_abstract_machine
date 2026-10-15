"""
What is poured, and what the camera and the scale need to know about it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum

from typing_extensions import Optional

from giskardpy.middleware.ros2.particle_weighing import FULL_CUP_MASS_GRAMS

from experiments.pouring_model_verification.hypothesis_bank import ModelPoint
from experiments.real_pouring.stream_detection import (
    GrainSegmentation,
    MovingGrainSegmentation,
)

STYROFOAM_PIXEL_SECONDS_PER_GRAIN = 24.8
"""
How much moving grain area in the opening, times the time it shows, one of the live
bank's grains (a percent of a full Jeroen cup) received adds for the styrofoam balls:
the mean of five weighed pours of 2026-10-14, 22.1 to 28.2 (within 12 %), counted on
the opening moved onto the rim
(experiments/results/real_pouring/analysis_2026-10-15/count_calibration.py).
"""


CORN_FULL_CUP_MASS_GRAMS: Optional[float] = 361.83
"""
What the corn kernels weigh when they fill the cup, in grams; ``None`` until a full cup
of them has been weighed. Weighed on 2026-10-14 as the received, spilled and left-over
masses of one pour: 50.24 + 4.77 + 306.82 g.
"""

CORN_PIXEL_SECONDS_PER_GRAIN = 48.8
"""
How much moving grain area in the opening, times the time it shows, one of the live
bank's grains of corn received adds: the mean of the three weighed corn pours of
2026-10-14 and 2026-10-15, 43.1 to 54.0 (within 11 %), a thin slow stream and a dense
fast one alike, counted on the opening moved onto the rim
(experiments/results/real_pouring/analysis_2026-10-15/count_calibration.py).
"""


STYROFOAM_HEAP_PIXELS_PER_SHARE = 576.0
"""
How many heap pixels the pole camera would see of the styrofoam surface filling the
whole receiver, for the heap gauge: the mean of the three weighed pours of 2026-10-14
that ended near the goal (fills 0.63 to 0.70, 513 to 675). The two fuller pours (0.79,
0.91) show four times as much, the light balls not lying flat as the gauge assumes, so
above the goal the gauge reads too full and stops a pour early rather than late
(experiments/results/real_pouring/analysis_2026-10-15/heap_calibration.py).
"""

CORN_HEAP_PIXELS_PER_SHARE = 2186.0
"""
How many heap pixels the pole camera would see of the corn surface filling the whole
receiver, for the heap gauge: from the corn pour of 2026-10-14 17:50, which ended at a
fill of 0.69, near the goal. The brim-full pour of 2026-10-15 gives 2806, so above the
goal the gauge reads up to 0.07 too full and stops a pour early rather than late
(experiments/results/real_pouring/analysis_2026-10-15/heap_calibration.py).
"""

STYROFOAM_REPOSE_HYSTERESIS = math.radians(60.0)
"""
How far below the repose angle the styrofoam balls keep flowing once they flow: the
bulk starts at about 70 degrees with the cup full and stops on the way back at 74, 82
and 90 degrees with 32, 17 and 6 % left (2026-10-14 13:44, 15:56, 14:49), which the
model's lip geometry puts 67, 59 and 53 degrees below the starting angle
(experiments/results/real_pouring/analysis_2026-10-15/stop_angles.py).
"""

CORN_REPOSE_HYSTERESIS = math.radians(45.0)
"""
How far below the repose angle the corn keeps flowing once it flows: the bulk starts at
about 66 degrees with the cup full and stops on the way back at 92 degrees with 20 % left
(2026-10-14 17:50), 45 degrees below the starting angle in the model's lip geometry; with
85 % left (17:14) only 27 below, but a pour to the goal ends near a fifth left
(experiments/results/real_pouring/analysis_2026-10-15/stop_angles.py).
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
        :return: How the camera tells this material's grains from the empty view:
            for every material, by what moves through the opening in another colour
            than the empty view's, so only the calibration differs.
        """
        match self:
            case PouredMaterial.STYROFOAM | PouredMaterial.CORN:
                return MovingGrainSegmentation()

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
    def heap_pixels_per_share(self) -> float:
        """
        :return: How many heap pixels the camera would see of this material's surface
            filling the whole receiver: the scale of the heap gauge.
        """
        match self:
            case PouredMaterial.STYROFOAM:
                return STYROFOAM_HEAP_PIXELS_PER_SHARE
            case PouredMaterial.CORN:
                return CORN_HEAP_PIXELS_PER_SHARE

    @property
    def repose_hysteresis(self) -> float:
        """
        :return: How far below the repose angle this material keeps flowing once it
            flows, in radians.
        """
        match self:
            case PouredMaterial.STYROFOAM:
                return STYROFOAM_REPOSE_HYSTERESIS
            case PouredMaterial.CORN:
                return CORN_REPOSE_HYSTERESIS

    def drain_of(self, drain: ModelPoint) -> ModelPoint:
        """
        :param drain: A drain's parameters.
        :return: The drain as it pours this material: with its hysteresis.
        """
        return replace(drain, repose_hysteresis=self.repose_hysteresis)

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
