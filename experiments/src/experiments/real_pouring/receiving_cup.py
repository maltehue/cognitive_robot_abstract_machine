"""
The cups the robot pours into, and what the camera and the model need to know of each.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy

from experiments.real_pouring.poured_material import (
    MaterialNotSeenThroughWallError,
    PouredMaterial,
)
from experiments.real_pouring.receiver_heap import HeapGauge
from experiments.real_pouring.receiver_wall import BLUE_MAT, WallGauge
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    CameraView,
    OpeningRegion,
    GrainSegmentation,
    MovingYellowSegmentation,
    StreamDetector,
)

JEROEN_CUP_HEIGHT = 0.16
"""
The Jeroen cup's height, from its bottom to its rim, in metres.
"""

JEROEN_CUP_RADIUS = 0.035
"""
The Jeroen cup's opening radius, in metres.
"""

JEROEN_CUP_GOAL_FILL = 0.7
"""
The fill the demo pours the Jeroen cup to.
"""

TRANSLUCENT_CUP_GOAL_FILL = 0.3
"""
The fill the demo pours the translucent cup to.
"""

TRANSLUCENT_CUP_HEIGHT = 0.155
"""
The translucent cup's height, from its bottom to its rim, in metres.
"""

TRANSLUCENT_CUP_RADIUS = 0.045
"""
The translucent cup's inner radius, in metres: 9 cm across.
"""

TRANSLUCENT_CUP_CORN_MASS_GRAMS = 722.0
"""
What the corn kernels weigh when they fill the translucent cup to the brim, in grams.
"""

TRANSLUCENT_CUP_CORN_PIXEL_SECONDS_PER_GRAIN = 9.7
"""
How much moving yellow area in the translucent cup's opening, times the time it shows,
one of the live bank's grains of corn received adds: from the one weighed pour into the
cup of 2026-10-15 18:33 (326 g received), less the count's creep while no stream fell
(experiments/results/real_pouring/analysis_2026-10-15/translucent_calibration.py).
"""

TRANSLUCENT_CUP_CORN_WALL_DETECTED_SHARE = 0.68
"""
The share of the corn's projected area the pole camera picks out through the
translucent cup's frosted wall: from the same pour, the corn lying still at its end
against the area the weighed fill of 0.45 projects to
(experiments/results/real_pouring/analysis_2026-10-15/translucent_calibration.py).
"""


RECEIVING_CUP_FILE_NAME = "receiving_cup.txt"
"""
Name of the file in a demo bag that names the cup poured into.
"""


@dataclass
class ReceiverFullMassUnknownError(Exception):
    """
    Raised when a receiving cup's capacity is needed for a material nobody has filled it
    with on the scale.
    """

    cup: ReceivingCup
    """
    The cup not weighed full.
    """

    material: PouredMaterial
    """
    The material it was not weighed full of.
    """

    def __str__(self) -> str:
        return (
            f"Nobody has weighed the {self.cup} full of {self.material} yet: fill it to "
            f"the brim, weigh what is in it, and set the mass in "
            f"experiments/real_pouring/receiving_cup.py."
        )


@dataclass
class ReceivingCupChangedError(Exception):
    """
    Raised when the controller still holds the receiving cup of an earlier pour into
    another cup: its size and capacity would be the old cup's.
    """

    cup: ReceivingCup
    """
    The cup poured into now.
    """

    def __str__(self) -> str:
        return (
            f"The controller still holds the receiving cup of an earlier pour into "
            f"another cup than the {self.cup}: restart Giskard so the demo creates the "
            f"cups anew."
        )


CAPACITY_TOLERANCE = 1e-6
"""
How far, relatively, two capacities may differ and still be the same cup's.
"""


class ReceiverFillReport(StrEnum):
    """
    What the controller is told the receiver holds.
    """

    ESTIMATED = "estimated"
    """
    What the bank estimates arrived, with the margin it stops short by; a heap seen
    lying in the receiver raises it.
    """

    COUNTED = "counted"
    """
    What the camera counted crossing the opening, with the grains still in flight
    added; a heap seen lying in the receiver raises it.
    """

    SEEN_THROUGH_WALL = "seen_through_wall"
    """
    What the camera sees lying in the receiver through its wall, with the grains still
    in flight added; what crossed until the wall shows anything.
    """


class ReceivingCup(StrEnum):
    """
    A cup the robot pours into.
    """

    JEROEN_CUP = "jeroen_cup"
    """
    The blue Jeroen cup the robot also pours from: opaque, so the camera sees its
    contents only over the rim.
    """

    TRANSLUCENT_CUP = "translucent_cup"
    """
    A frosted plastic cup, 9 cm across inside and 15.5 cm high: the camera sees its
    contents through the wall.
    """

    @property
    def height(self) -> float:
        """
        :return: The cup's height, from its bottom to its rim, in metres.
        """
        match self:
            case ReceivingCup.JEROEN_CUP:
                return JEROEN_CUP_HEIGHT
            case ReceivingCup.TRANSLUCENT_CUP:
                return TRANSLUCENT_CUP_HEIGHT

    @property
    def inner_radius(self) -> float:
        """
        :return: The radius of the cup's opening, in metres.
        """
        match self:
            case ReceivingCup.JEROEN_CUP:
                return JEROEN_CUP_RADIUS
            case ReceivingCup.TRANSLUCENT_CUP:
                return TRANSLUCENT_CUP_RADIUS

    @property
    def goal_fill(self) -> float:
        """
        :return: The fill the demo pours this cup to: for the translucent cup, which
            holds two source cups, 0.3 holds about as much as 0.6 of the Jeroen cup and
            stays short of overflowing even if the source empties.
        """
        match self:
            case ReceivingCup.JEROEN_CUP:
                return JEROEN_CUP_GOAL_FILL
            case ReceivingCup.TRANSLUCENT_CUP:
                return TRANSLUCENT_CUP_GOAL_FILL

    @property
    def fill_report(self) -> ReceiverFillReport:
        """
        :return: How the controller is told what this cup holds: the translucent cup
            shows its contents through the wall, the opaque Jeroen cup only what crosses
            its rim.
        """
        match self:
            case ReceivingCup.JEROEN_CUP:
                return ReceiverFillReport.COUNTED
            case ReceivingCup.TRANSLUCENT_CUP:
                return ReceiverFillReport.SEEN_THROUGH_WALL

    def full_mass_grams(self, material: PouredMaterial) -> float:
        """
        :param material: What fills the cup.
        :return: What that much of the material weighs, in grams.
        :raises ReceiverFullMassUnknownError: If nobody weighed the cup full of it.
        """
        match self, material:
            case ReceivingCup.JEROEN_CUP, _:
                return material.full_cup_mass_grams
            case ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN:
                return TRANSLUCENT_CUP_CORN_MASS_GRAMS
        raise ReceiverFullMassUnknownError(cup=self, material=material)

    def source_cups_held(self, material: PouredMaterial) -> float:
        """
        :param material: What is poured.
        :return: How many full Jeroen cups, the cup poured from, this cup holds.
        """
        return self.full_mass_grams(material) / material.full_cup_mass_grams

    def check_reused(
        self,
        receiver_capacity: float,
        source_capacity: float,
        material: PouredMaterial,
    ) -> None:
        """
        :param receiver_capacity: The capacity of the receiver an earlier pour left.
        :param source_capacity: The capacity of the cup poured from.
        :param material: What is poured.
        :raises ReceivingCupChangedError: If that receiver was another cup.
        """
        expected = source_capacity * self.source_cups_held(material)
        if not math.isclose(receiver_capacity, expected, rel_tol=CAPACITY_TOLERANCE):
            raise ReceivingCupChangedError(cup=self)

    def segmentation(self, material: PouredMaterial) -> GrainSegmentation:
        """
        :param material: What is poured.
        :return: How the camera tells the material falling into this cup from the cup.
        :raises MaterialNotSeenThroughWallError: For a translucent cup and a material
            that does not turn its light inside yellow.
        """
        match self, material:
            case ReceivingCup.JEROEN_CUP, _:
                return material.segmentation
            case ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN:
                return MovingYellowSegmentation()
        raise MaterialNotSeenThroughWallError(material=material)

    def pixel_seconds_per_grain(self, material: PouredMaterial) -> float:
        """
        :param material: What is poured.
        :return: How much stream area in this cup's opening, times the time it shows,
            one of the live bank's grains of the material adds.
        :raises MaterialNotSeenThroughWallError: For a translucent cup and a material
            that does not turn its light inside yellow.
        """
        match self, material:
            case ReceivingCup.JEROEN_CUP, _:
                return material.pixel_seconds_per_grain
            case ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.CORN:
                return TRANSLUCENT_CUP_CORN_PIXEL_SECONDS_PER_GRAIN
        raise MaterialNotSeenThroughWallError(material=material)

    def wall_detected_share(self, material: PouredMaterial) -> float:
        """
        :param material: What is poured.
        :return: The share of the material's projected area the camera picks out
            through this cup's frosted wall.
        :raises MaterialNotSeenThroughWallError: For a material the wall hides, or a
            cup without a wall to see through.
        """
        if self is ReceivingCup.TRANSLUCENT_CUP and material is PouredMaterial.CORN:
            return TRANSLUCENT_CUP_CORN_WALL_DETECTED_SHARE
        raise MaterialNotSeenThroughWallError(material=material)

    def detector_for(
        self,
        material: PouredMaterial,
        view: CameraView,
        opening_centre: numpy.ndarray,
        frame_period: float,
    ) -> StreamDetector:
        """
        :param material: What is poured.
        :param view: The camera's calibration.
        :param opening_centre: The centre of the cup's opening, in the world.
        :param frame_period: How long apart the camera takes its images, in seconds.
        :return: A detector watching this cup's opening for the material, reading the
            cup's fill the way the cup lets the camera see it.
        :raises MaterialNotSeenThroughWallError: For a translucent cup and a material
            its wall hides.
        """
        opening = OpeningRegion(centre=opening_centre, radius=self.inner_radius)
        common = dict(
            view=view,
            opening=opening,
            segmentation=self.segmentation(material),
            pixel_seconds_per_grain=self.pixel_seconds_per_grain(material),
            frame_period=frame_period,
        )
        match self:
            case ReceivingCup.JEROEN_CUP:
                return StreamDetector(
                    **common,
                    outline=BLUE_CUP_OUTLINE,
                    heap_gauge=HeapGauge.for_receiver(
                        camera_position=view.world_T_camera[:3, 3],
                        opening_centre=opening_centre,
                        height=self.height,
                        radius=self.inner_radius,
                        pixels_per_share=material.heap_pixels_per_share,
                    ),
                )
            case ReceivingCup.TRANSLUCENT_CUP:
                return StreamDetector(
                    **common,
                    cup_on_mat=BLUE_MAT,
                    wall_gauge=WallGauge.for_receiver(
                        view=view,
                        opening_centre=opening_centre,
                        height=self.height,
                        radius=self.inner_radius,
                        detected_share=self.wall_detected_share(material),
                    ),
                )
