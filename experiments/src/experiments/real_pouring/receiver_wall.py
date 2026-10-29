"""
How full a translucent receiver is, read off the contents it shows through the wall and
the opening.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import cv2
import numpy
from typing_extensions import Optional, Tuple

from experiments.real_pouring.receiver_heap import FILL_SAMPLES, MINIMUM_HEAP_PIXELS
from experiments.real_pouring.stream_detection import (
    OUTLINE_SEARCH_MARGIN,
    TOP_EDGE_ROWS,
    CameraView,
    PixelOffset,
    ColourShift,
    ReceiverNotFoundError,
    WallContents,
)

OUTLINE_SAMPLES = 90
"""
How many points around the receiver's circumference are projected to outline it.
"""


WATER_LINE_ROW_SHARE = 0.5
"""
The share of a row of the receiver that has to show the contents for the row to lie
below the water line: the stream falling inside and the frosting's blur leave a few
pixels of a row out.
"""


@dataclass(frozen=True)
class WallGauge(ABC):
    """
    Reads the fill of a translucent receiver off the contents the camera picks out
    through its wall and its opening.
    """

    view: CameraView
    """
    The camera's calibration.
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

    colour_shift: ColourShift
    """
    How the contents are told from the empty view.
    """

    @staticmethod
    def base_below(opening_centre: numpy.ndarray, height: float) -> numpy.ndarray:
        """
        :param opening_centre: The centre of the receiver's opening, in the world.
        :param height: The receiver's height, in metres.
        :return: The centre of the bottom of a receiver standing below its opening.
        """
        return opening_centre - numpy.array([0.0, 0.0, height])

    def _circle_at(self, fill: float) -> numpy.ndarray:
        """
        :param fill: How full the receiver is, from 0 to 1.
        :return: Points around the receiver's inside at the height of that fill, in the
            world.
        """
        angles = numpy.linspace(0.0, 2 * numpy.pi, OUTLINE_SAMPLES, endpoint=False)
        return self.receiver_base + numpy.stack(
            [
                self.radius * numpy.cos(angles),
                self.radius * numpy.sin(angles),
                numpy.full(OUTLINE_SAMPLES, fill * self.height),
            ],
            axis=1,
        )

    def outline(self, fill: float) -> numpy.ndarray:
        """
        :param fill: How full the receiver is, from 0 to 1.
        :return: The outline of its contents in the image, a convex polygon of
            ``(column, row)`` points.
        """
        points = self.view.to_image(
            numpy.concatenate([self._circle_at(0.0), self._circle_at(fill)])
        )
        return cv2.convexHull(points.astype(numpy.float32))

    def projected_area(self, fill: float) -> float:
        """
        :param fill: How full the receiver is, from 0 to 1.
        :return: How many pixels its contents cover in the image.
        """
        return float(cv2.contourArea(self.outline(fill)))

    def region_in(
        self, image_size: Tuple[int, int], offset: PixelOffset
    ) -> numpy.ndarray:
        """
        :param image_size: Height and width of the camera's images.
        :param offset: How far the camera's pose error moves the receiver in the image.
        :return: Which pixels show the receiver.
        """
        mask = numpy.zeros(image_size, dtype=numpy.uint8)
        cv2.fillPoly(mask, [numpy.round(self.outline(1.0)).astype(numpy.int32)], 1)
        return offset.applied_to(mask.astype(bool))

    def shows_contents(
        self, image: numpy.ndarray, background: numpy.ndarray
    ) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param background: What the same view showed before the pour, BGR.
        :return: Which pixels turned far enough to show the contents.
        """
        return self.colour_shift.shows(image, background)

    @abstractmethod
    def fill_shown_by(self, contents: WallContents) -> Optional[float]:
        """
        :param contents: Where the receiver shows its contents lying still.
        :return: How full they show it, from 0 to 1, or ``None`` where too little shows
            to tell.
        """


@dataclass(frozen=True)
class WallAreaGauge(WallGauge):
    """
    Reads the fill off the area of the contents: contents up to a height show as the
    receiver itself would up to that height, through the wall below the surface and
    through the opening above it.
    """

    detected_share: float
    """
    The share of the contents' projected area the camera picks out through the frosted
    wall: the scale one weighed pour of a material sets.
    """

    @classmethod
    def for_receiver(
        cls,
        view: CameraView,
        opening_centre: numpy.ndarray,
        height: float,
        radius: float,
        detected_share: float,
        colour_shift: ColourShift,
    ) -> WallAreaGauge:
        """
        :param view: The camera's calibration.
        :param opening_centre: The centre of the receiver's opening, in the world.
        :param height: The receiver's height, in metres.
        :param radius: The receiver's inner radius, in metres.
        :param detected_share: The share of the contents' area the camera picks out.
        :param colour_shift: How the contents are told from the empty view.
        :return: The gauge of a receiver standing below its opening.
        """
        return cls(
            view=view,
            receiver_base=cls.base_below(opening_centre, height),
            height=height,
            radius=radius,
            detected_share=detected_share,
            colour_shift=colour_shift,
        )

    @cached_property
    def _areas(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        :return: Fills from empty to full and the area the contents cover at each.
        """
        fills = numpy.linspace(0.0, 1.0, FILL_SAMPLES)
        return fills, numpy.array([self.projected_area(fill) for fill in fills])

    def fill_for(self, pixels: float) -> Optional[float]:
        """
        :param pixels: How many pixels of the receiver show its contents.
        :return: The fill at which the camera would pick out that many, or ``None``
            where too few show to tell.
        """
        if pixels < MINIMUM_HEAP_PIXELS:
            return None
        fills, areas = self._areas
        return float(numpy.interp(pixels / self.detected_share, areas, fills))

    def fill_shown_by(self, contents: WallContents) -> Optional[float]:
        return self.fill_for(int(contents.shown.sum()))


@dataclass(frozen=True)
class WaterLineGauge(WallGauge):
    """
    Reads the fill off the line dark contents reach up to on the wall: the surface's
    edge nearest the camera, below which the wall shows the contents in every row.
    """

    @classmethod
    def for_receiver(
        cls,
        view: CameraView,
        opening_centre: numpy.ndarray,
        height: float,
        radius: float,
        colour_shift: ColourShift,
    ) -> WaterLineGauge:
        """
        :param view: The camera's calibration.
        :param opening_centre: The centre of the receiver's opening, in the world.
        :param height: The receiver's height, in metres.
        :param radius: The receiver's inner radius, in metres.
        :param colour_shift: How the contents are told from the empty view.
        :return: The gauge of a receiver standing below its opening.
        """
        return cls(
            view=view,
            receiver_base=cls.base_below(opening_centre, height),
            height=height,
            radius=radius,
            colour_shift=colour_shift,
        )

    @cached_property
    def _surface_rows(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        """
        :return: Fills from empty to full and the image row of the surface's edge
            nearest the camera at each, rising as the fill does.
        """
        fills = numpy.linspace(0.0, 1.0, FILL_SAMPLES)
        rows = numpy.array(
            [self.view.to_image(self._circle_at(fill))[:, 1].max() for fill in fills]
        )
        return fills, rows

    def fill_shown_by(self, contents: WallContents) -> Optional[float]:
        """
        :param contents: Where the receiver shows its contents lying still.
        :return: The fill whose surface lies on the topmost of the rows that, from the
            receiver's bottom up without a gap, show the contents, or ``None`` where its
            bottom row does not.
        """
        receiver_pixels = contents.receiver.sum(axis=1)
        rows = numpy.nonzero(receiver_pixels)[0]
        under_water = (contents.shown[rows] & contents.receiver[rows]).sum(
            axis=1
        ) >= WATER_LINE_ROW_SHARE * receiver_pixels[rows]
        if not under_water[-1]:
            return None
        dry = numpy.nonzero(~under_water)[0]
        top = rows[dry[-1] + 1] if dry.size else rows[0]
        line = contents.first_row + top - 0.5
        fills, surface_rows = self._surface_rows
        return float(numpy.interp(-line, -surface_rows, fills))


# %% finding the cup as the gap it makes in the mat it stands on

MAT_BESIDE_PIXELS = 10
"""
How many pixels of the mat have to show on either side of an interruption, in its row,
for the interruption to be something standing on the mat rather than the mat's edge,
the gripper reaching in over it, or its shadow.
"""

MINIMUM_GAP_PIXELS = 100
"""
How many pixels an interruption of the mat has to cover to be a thing standing on it
rather than the mat's own texture: the cup's gap spans its width, dozens of pixels.
"""

MAXIMUM_CUP_OFFSET = 50.0
"""
How far the cup's gap may lie from where the cup projects and still be the cup, in
pixels: the camera's pose is off by a few centimetres at most, a millimetre per pixel
under the pole camera, while a joint cap of the arm interrupting the mat lies about a
hundred pixels away.
"""


@dataclass(frozen=True)
class CupOnMat:
    """
    A cup found in the empty view by the gap it makes in the coloured mat it stands on,
    whatever its own colour: a translucent cup has no outline of its own to find, but the
    mat seen through its frosted wall is far less saturated than the bare mat.
    """

    minimum_hue: int
    """
    The lowest hue of the bare mat, on OpenCV's scale from 0 to 179.
    """

    maximum_hue: int
    """
    The highest hue of the bare mat, on OpenCV's scale from 0 to 179.
    """

    minimum_saturation: int
    """
    The least saturation of the bare mat, from 0 to 255: above what the mat seen through
    the cup's wall keeps.
    """

    def shows_mat(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :return: Which of its pixels show the bare mat.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return (
            (hsv[..., 0] >= self.minimum_hue)
            & (hsv[..., 0] <= self.maximum_hue)
            & (hsv[..., 1] >= self.minimum_saturation)
        )

    def offset(self, cup: numpy.ndarray, empty_view: numpy.ndarray) -> PixelOffset:
        """
        :param cup: Which pixels the cup projects to.
        :param empty_view: The empty view, BGR.
        :return: How far the cup has to move for the bottom of its projection to meet
            the bottom of the gap it makes in the mat, and the centres of the two
            bottoms to meet: of the gaps in the mat, the one nearest the projection,
            since the arm can interrupt the mat too, further away.
        :raises ReceiverNotFoundError: If the mat shows no gap within
            :data:`MAXIMUM_CUP_OFFSET` of the projection.
        """
        rows, columns = numpy.nonzero(cup)
        height, width = cup.shape
        top, bottom = max(0, rows.min() - OUTLINE_SEARCH_MARGIN), min(
            height, rows.max() + OUTLINE_SEARCH_MARGIN
        )
        left, right = max(0, columns.min() - OUTLINE_SEARCH_MARGIN), min(
            width, columns.max() + OUTLINE_SEARCH_MARGIN
        )
        mat = self.shows_mat(empty_view[top:bottom, left:right])
        mat_to_the_left = numpy.cumsum(mat, axis=1)
        mat_to_the_right = mat.sum(axis=1, keepdims=True) - mat_to_the_left
        gap = (
            ~mat
            & (mat_to_the_left >= MAT_BESIDE_PIXELS)
            & (mat_to_the_right >= MAT_BESIDE_PIXELS)
        )
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            gap.astype(numpy.uint8)
        )
        cup_bottom = rows.max()
        cup_centre = columns[rows > cup_bottom - TOP_EDGE_ROWS].mean()
        nearest: Optional[PixelOffset] = None
        nearest_distance = MAXIMUM_CUP_OFFSET
        for label in range(1, count):
            if statistics[label, cv2.CC_STAT_AREA] < MINIMUM_GAP_PIXELS:
                continue
            gap_rows, gap_columns = numpy.nonzero(labels == label)
            gap_bottom = gap_rows.max()
            gap_centre = (
                gap_columns[gap_rows > gap_bottom - TOP_EDGE_ROWS].mean() + left
            )
            offset = PixelOffset(
                rows=int(gap_bottom + top - cup_bottom),
                columns=int(round(gap_centre - cup_centre)),
            )
            distance = math.hypot(offset.rows, offset.columns)
            if distance <= nearest_distance:
                nearest, nearest_distance = offset, distance
        if nearest is None:
            raise ReceiverNotFoundError()
        return nearest


BLUE_MAT = CupOnMat(minimum_hue=95, maximum_hue=125, minimum_saturation=150)
"""
The blue mat the receiving cup stands on under the Tracy pole camera: bare, its
saturation is about 215 to 245; seen through the translucent cup's wall, 60 to 100.
"""
