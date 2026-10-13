"""
What a fixed camera sees of a pour: whether grains cross the receiver's opening, how
many have crossed so far, and where.

The grains are told apart from what stood there before the pour by being bright,
unsaturated and new, which suits white styrofoam, rice or corn against a dark receiver.
The opening is found in the image from the camera's pose and intrinsics, so the camera
has to be calibrated; nothing here depends on ROS.
"""

# %% imports

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import Any, Dict, List, Optional, Tuple

import cv2
import numpy

from krrood.adapters.json_serializer import SubclassJSONSerializer

# %% the camera


@dataclass(frozen=True)
class PlanePoint(SubclassJSONSerializer):
    """
    A point on a horizontal plane, in the world frame.
    """

    x: float
    """
    Its x coordinate, in metres.
    """

    y: float
    """
    Its y coordinate, in metres.
    """

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "x": self.x, "y": self.y}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> PlanePoint:
        return cls(x=data["x"], y=data["y"])


@dataclass
class CameraView:
    """
    Where a camera stands and how it projects, as its calibration says.
    """

    intrinsics: numpy.ndarray
    """
    The camera matrix: focal lengths and principal point, in pixels.
    """

    world_T_camera: numpy.ndarray
    """
    The pose of the camera's optical frame in the world: z along the optical axis, x to
    the right of the image and y down it.
    """

    def to_image(self, points: numpy.ndarray) -> numpy.ndarray:
        """
        :param points: World points, one per row, in metres.
        :return: Where they show in the image, one ``(column, row)`` per row, in pixels.
        """
        camera_T_world = numpy.linalg.inv(self.world_T_camera)
        in_camera = (camera_T_world[:3, :3] @ points.T).T + camera_T_world[:3, 3]
        projected = (self.intrinsics @ in_camera.T).T
        return projected[:, :2] / projected[:, 2:3]

    def on_plane(self, pixel: numpy.ndarray, plane_height: float) -> PlanePoint:
        """
        :param pixel: A ``(column, row)`` in the image.
        :param plane_height: The height of a horizontal plane in the world, in metres.
        :return: Where the ray through that pixel meets the plane.
        """
        ray = numpy.linalg.inv(self.intrinsics) @ numpy.array([pixel[0], pixel[1], 1.0])
        direction = self.world_T_camera[:3, :3] @ ray
        origin = self.world_T_camera[:3, 3]
        distance = (plane_height - origin[2]) / direction[2]
        point = origin + distance * direction
        return PlanePoint(x=float(point[0]), y=float(point[1]))


# %% the receiver's opening

OPENING_OUTLINE_POINTS = 64
"""
How many points of the opening's rim are projected to outline it in the image.
"""


@dataclass(frozen=True)
class OpeningRegion:
    """
    The receiver's opening: a horizontal disk the stream has to cross.
    """

    centre: numpy.ndarray
    """
    The centre of the opening in the world, in metres.
    """

    radius: float
    """
    The opening's radius, in metres.
    """

    def mask_in(self, view: CameraView, image_size: Tuple[int, int]) -> numpy.ndarray:
        """
        :param view: The camera looking at the opening.
        :param image_size: Height and width of its images, in pixels.
        :return: Which pixels show the opening.
        """
        angles = numpy.linspace(
            0.0, 2 * numpy.pi, OPENING_OUTLINE_POINTS, endpoint=False
        )
        rim = self.centre + self.radius * numpy.stack(
            [numpy.cos(angles), numpy.sin(angles), numpy.zeros_like(angles)], axis=1
        )
        outline = numpy.round(view.to_image(rim)).astype(numpy.int32)
        mask = numpy.zeros(image_size, dtype=numpy.uint8)
        cv2.fillPoly(mask, [outline], 1)
        return mask.astype(bool)


# %% telling grains from what stood there


@dataclass(frozen=True)
class WhiteSegmentation:
    """
    Picks out pixels showing grains: bright, unsaturated, and different from what the
    same pixel showed before the pour.
    """

    minimum_value: int = 150
    """
    The least brightness a grain pixel has, from 0 to 255.
    """

    maximum_saturation: int = 70
    """
    The most saturation a grain pixel has, from 0 to 255; white grains have little.
    """

    minimum_difference: int = 80
    """
    How far a grain pixel's colour has to differ from the background's, summed over the
    three channels.
    """

    def grains(self, image: numpy.ndarray, background: numpy.ndarray) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param background: What the same view showed before the pour, BGR.
        :return: Which pixels show grains.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        changed = (
            numpy.abs(image.astype(numpy.int16) - background.astype(numpy.int16)).sum(
                axis=2
            )
            > self.minimum_difference
        )
        return (
            (hsv[..., 2] > self.minimum_value)
            & (hsv[..., 1] < self.maximum_saturation)
            & changed
        )


# %% what the camera reports


@dataclass(frozen=True)
class StreamReading(SubclassJSONSerializer):
    """
    What the camera saw of the stream in one image.
    """

    stamp: float
    """
    When the image was taken, in seconds.
    """

    stream_seen: bool
    """
    Whether grains showed in the receiver's opening.
    """

    counted_grains: float
    """
    How many grains the camera has counted crossing the opening since it started, from
    the white area it saw there.
    """

    crossing: Optional[PlanePoint]
    """
    Where the grains it saw cross the opening plane, on average, or ``None`` where it
    saw none.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "stamp": self.stamp,
            "stream_seen": self.stream_seen,
            "counted_grains": self.counted_grains,
            "crossing": None if self.crossing is None else self.crossing.to_json(),
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> StreamReading:
        crossing = data["crossing"]
        return cls(
            stamp=data["stamp"],
            stream_seen=data["stream_seen"],
            counted_grains=data["counted_grains"],
            crossing=None if crossing is None else PlanePoint._from_json(crossing),
        )


MINIMUM_STREAM_PIXELS = 30
"""
How many grain pixels the opening has to show for a stream to count as seen: one grain
of a few millimetres, at the pole camera's millimetre per pixel.
"""


@dataclass
class StreamDetector:
    """
    Turns the images of a fixed camera into readings of the stream crossing the
    receiver's opening.
    """

    view: CameraView
    """
    The camera's calibration.
    """

    opening: OpeningRegion
    """
    The receiver's opening.
    """

    segmentation: WhiteSegmentation
    """
    How grain pixels are picked out.
    """

    pixel_seconds_per_grain: float
    """
    How much grain area, times the time it shows, one grain adds while it crosses, in
    pixel-seconds; calibrated by weighing what a pour delivered.
    """

    frame_period: float
    """
    How long apart the camera takes its images, in seconds: the time the first image
    after the background stands for.
    """

    _background: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    What the camera showed before the pour.
    """

    _mask: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    Which pixels show the opening.
    """

    _counted: float = field(init=False, default=0.0)
    """
    The grains counted so far.
    """

    _window: Tuple[slice, slice] = field(init=False, default=(slice(0), slice(0)))
    """
    The rows and columns of the smallest image window holding the opening.
    """

    _last_stamp: Optional[float] = field(init=False, default=None)
    """
    When the image observed last was taken, or ``None`` before the first.
    """

    def learn_background(self, images: List[numpy.ndarray]) -> None:
        """
        Remember what the view shows before the pour, and start counting from nothing.

        :param images: Images taken before any grain leaves the source, BGR.
        """
        mask = self.opening.mask_in(self.view, images[0].shape[:2])
        rows, columns = numpy.nonzero(mask)
        self._window = (
            slice(rows.min(), rows.max() + 1),
            slice(columns.min(), columns.max() + 1),
        )
        self._mask = mask[self._window]
        self._background = numpy.median(
            numpy.stack([image[self._window] for image in images]), axis=0
        ).astype(numpy.uint8)
        self._counted = 0.0
        self._last_stamp = None

    def observe(self, image: numpy.ndarray, stamp: float) -> StreamReading:
        """
        :param image: The latest image, BGR.
        :param stamp: When it was taken, in seconds.
        :return: What it shows of the stream.
        """
        grains = (
            self.segmentation.grains(image[self._window], self._background) & self._mask
        )
        pixels = int(grains.sum())
        shown_for = (
            self.frame_period if self._last_stamp is None else stamp - self._last_stamp
        )
        self._last_stamp = stamp
        self._counted += pixels * shown_for / self.pixel_seconds_per_grain
        if pixels < MINIMUM_STREAM_PIXELS:
            return StreamReading(
                stamp=stamp,
                stream_seen=False,
                counted_grains=self._counted,
                crossing=None,
            )
        rows, columns = numpy.nonzero(grains)
        crossing = self.view.on_plane(
            numpy.array(
                [
                    columns.mean() + self._window[1].start,
                    rows.mean() + self._window[0].start,
                ]
            ),
            plane_height=float(self.opening.centre[2]),
        )
        return StreamReading(
            stamp=stamp,
            stream_seen=True,
            counted_grains=self._counted,
            crossing=crossing,
        )
