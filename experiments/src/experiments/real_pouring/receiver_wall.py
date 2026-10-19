"""
How full a translucent receiver is, read off the area its contents show through the wall
and the opening.
"""

from __future__ import annotations

import math
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
    ReceiverNotFoundError,
    Yellowing,
)

OUTLINE_SAMPLES = 90
"""
How many points around the receiver's circumference are projected to outline it.
"""


@dataclass(frozen=True)
class WallGauge:
    """
    Reads the fill of a translucent receiver off the area of its contents the camera
    picks out: contents up to a height show as the receiver itself would up to that
    height, through the wall below the surface and through the opening above it.
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

    detected_share: float
    """
    The share of the contents' projected area the camera picks out through the frosted
    wall: the scale one weighed pour of a material sets.
    """

    yellowing: Yellowing = Yellowing()
    """
    How the contents are told from the empty view.
    """

    @classmethod
    def for_receiver(
        cls,
        view: CameraView,
        opening_centre: numpy.ndarray,
        height: float,
        radius: float,
        detected_share: float,
    ) -> WallGauge:
        """
        :param view: The camera's calibration.
        :param opening_centre: The centre of the receiver's opening, in the world.
        :param height: The receiver's height, in metres.
        :param radius: The receiver's inner radius, in metres.
        :param detected_share: The share of the contents' area the camera picks out.
        :return: The gauge of a receiver standing below its opening.
        """
        return cls(
            view=view,
            receiver_base=opening_centre - numpy.array([0.0, 0.0, height]),
            height=height,
            radius=radius,
            detected_share=detected_share,
        )

    def outline(self, fill: float) -> numpy.ndarray:
        """
        :param fill: How full the receiver is, from 0 to 1.
        :return: The outline of its contents in the image, a convex polygon of
            ``(column, row)`` points.
        """
        angles = numpy.linspace(0.0, 2 * numpy.pi, OUTLINE_SAMPLES, endpoint=False)
        circle = numpy.stack(
            [
                self.radius * numpy.cos(angles),
                self.radius * numpy.sin(angles),
                numpy.zeros(OUTLINE_SAMPLES),
            ],
            axis=1,
        )
        surface = circle + numpy.array([0.0, 0.0, fill * self.height])
        points = self.view.to_image(
            numpy.concatenate([circle, surface]) + self.receiver_base
        )
        return cv2.convexHull(points.astype(numpy.float32))

    def projected_area(self, fill: float) -> float:
        """
        :param fill: How full the receiver is, from 0 to 1.
        :return: How many pixels its contents cover in the image.
        """
        return float(cv2.contourArea(self.outline(fill)))

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
        :return: Which pixels turned yellow enough to show the contents.
        """
        return self.yellowing.shows(image, background)


# %% finding the cup as the gap it makes in the mat it stands on

MAT_EDGE_COLUMNS = 10
"""
How many columns at either end of a row of the search window have to show the mat for
whatever interrupts it in that row to count as the cup standing on it, in pixels.
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
        on_mat = (mat[:, :MAT_EDGE_COLUMNS].mean(axis=1) > 0.5) & (
            mat[:, -MAT_EDGE_COLUMNS:].mean(axis=1) > 0.5
        )
        gap = ~mat & on_mat[:, numpy.newaxis]
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
