"""
How full the receiver is, read off how much of its contents' surface the camera sees
over the rim.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy
from typing_extensions import Optional

SURFACE_SAMPLES = 141
"""
How many points across the surface's diameter its visibility is sampled at.
"""

FILL_SAMPLES = 201
"""
How many fills from empty to full the visible share is tabulated at.
"""

MINIMUM_HEAP_PIXELS = 30
"""
How many heap pixels the camera has to see for the heap to tell a fill: one grain of a
few millimetres.
"""


@dataclass(frozen=True)
class HeapGauge:
    """
    Reads the receiver's fill off the area of its contents the camera sees: a camera
    looking in from the side sees the surface only once it has risen far enough for the
    rim not to hide it, and then more of it the higher it stands.
    """

    camera_position: numpy.ndarray
    """
    Where the camera is, in the world, in metres.
    """

    receiver_base: numpy.ndarray
    """
    The centre of the receiver's bottom, in the world, in metres.
    """

    height: float
    """
    The receiver's height, from its bottom to its rim, in metres.
    """

    radius: float
    """
    The receiver's inner radius, in metres.
    """

    pixels_per_share: float
    """
    How many heap pixels the camera would see of the whole surface: the scale one
    weighed pour of a material sets.
    """

    @classmethod
    def for_receiver(
        cls,
        camera_position: numpy.ndarray,
        opening_centre: numpy.ndarray,
        height: float,
        radius: float,
        pixels_per_share: float,
    ) -> HeapGauge:
        """
        :param camera_position: Where the camera is, in the world, in metres.
        :param opening_centre: The centre of the receiver's opening, in the world.
        :param height: The receiver's height, in metres.
        :param radius: The receiver's opening radius, in metres.
        :param pixels_per_share: Heap pixels the whole surface would show.
        :return: The gauge of a receiver standing below its opening.
        """
        return cls(
            camera_position=camera_position,
            receiver_base=opening_centre - numpy.array([0.0, 0.0, height]),
            height=height,
            radius=radius,
            pixels_per_share=pixels_per_share,
        )

    def visible_share(self, fill: float) -> float:
        """
        :param fill: How full the receiver is, from 0 to 1.
        :return: The share of the contents' surface the camera sees over the rim: the
            points whose line of sight leaves the receiver through its opening.
        """
        across = numpy.linspace(-self.radius, self.radius, SURFACE_SAMPLES)
        x, y = numpy.meshgrid(across, across)
        on_surface = x**2 + y**2 <= self.radius**2
        surface_height = self.receiver_base[2] + fill * self.height
        towards_camera = self.camera_position - numpy.stack(
            [
                self.receiver_base[0] + x,
                self.receiver_base[1] + y,
                numpy.full_like(x, surface_height),
            ],
            axis=-1,
        )
        to_rim = (
            self.receiver_base[2] + self.height - surface_height
        ) / towards_camera[..., 2]
        at_rim_x = x + to_rim * towards_camera[..., 0]
        at_rim_y = y + to_rim * towards_camera[..., 1]
        through_opening = at_rim_x**2 + at_rim_y**2 <= self.radius**2
        return float((on_surface & through_opening).sum() / on_surface.sum())

    @cached_property
    def _shares(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        :return: Fills from empty to full and the visible share at each.
        """
        fills = numpy.linspace(0.0, 1.0, FILL_SAMPLES)
        return fills, numpy.array([self.visible_share(fill) for fill in fills])

    def fill_for(self, pixels: float) -> Optional[float]:
        """
        :param pixels: How many heap pixels the camera sees.
        :return: The fill at which it would see that many, or ``None`` where too few
            show to tell: below the fill the rim hides the surface up to, the heap says
            only that the receiver holds less.
        """
        if pixels < MINIMUM_HEAP_PIXELS:
            return None
        fills, shares = self._shares
        rising = numpy.concatenate([[True], numpy.diff(shares) > 0])
        return float(
            numpy.interp(pixels / self.pixels_per_share, shares[rising], fills[rising])
        )
