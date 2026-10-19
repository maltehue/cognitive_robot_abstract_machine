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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy

from experiments.real_pouring.receiver_heap import HeapGauge
from krrood.adapters.json_serializer import SubclassJSONSerializer

if TYPE_CHECKING:
    from experiments.real_pouring.receiver_wall import CupOnMat, WallGauge

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


MINIMUM_BACKGROUND_DIFFERENCE = 80
"""
How far a grain pixel's colour has to differ from what the pixel showed before the pour,
summed over the three channels, from 0 to 765.
"""

SAME_HUE_DISTANCE = 20
"""
How close two hues are, on OpenCV's scale from 0 to 179, to count as the same colour.
"""

COLOURED_SATURATION = 50
"""
The least saturation, from 0 to 255, at which a pixel's hue means something.
"""

SHADOW_SATURATION_CHANGE = 40
"""
The most a pixel's saturation changes, from 0 to 255, when only a shadow falls on it.
"""


@dataclass(frozen=True)
class GrainSegmentation(ABC):
    """
    Picks out the pixels of an image that show grains.
    """

    minimum_difference: int = MINIMUM_BACKGROUND_DIFFERENCE
    """
    How far a grain pixel's colour has to differ from the background's, summed over the
    three channels.
    """

    def changed(self, image: numpy.ndarray, background: numpy.ndarray) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param background: What the same view showed before the pour, BGR.
        :return: Which pixels differ from the background by at least
            :attr:`minimum_difference`.
        """
        return (
            numpy.abs(image.astype(numpy.int16) - background.astype(numpy.int16)).sum(
                axis=2
            )
            > self.minimum_difference
        )

    def differs_from_empty_view(
        self, image: numpy.ndarray, background: numpy.ndarray
    ) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param background: What the same view showed before the pour, BGR.
        :return: Which pixels changed from the empty view to another colour: neither a
            shadow falling on it nor its own colour brightening or darkening.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(numpy.int16)
        background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV).astype(numpy.int16)
        hue_distance = numpy.abs(hsv[..., 0] - background_hsv[..., 0])
        hue_distance = numpy.minimum(hue_distance, 180 - hue_distance)
        both_coloured = (hsv[..., 1] >= COLOURED_SATURATION) & (
            background_hsv[..., 1] >= COLOURED_SATURATION
        )
        both_grey = (hsv[..., 1] < COLOURED_SATURATION) & (
            background_hsv[..., 1] < COLOURED_SATURATION
        )
        same_colour = (hue_distance <= SAME_HUE_DISTANCE) | both_grey
        shadow = (
            (hsv[..., 2] < background_hsv[..., 2])
            & (
                numpy.abs(hsv[..., 1] - background_hsv[..., 1])
                < SHADOW_SATURATION_CHANGE
            )
            & same_colour
        )
        background_colour = (hue_distance <= SAME_HUE_DISTANCE) & both_coloured
        return self.changed(image, background) & ~shadow & ~background_colour

    def moved(self, image: numpy.ndarray, previous: numpy.ndarray) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param previous: The image before it, BGR.
        :return: Which pixels changed since the image before by at least
            :attr:`minimum_difference`.
        """
        return self.changed(image, previous)

    @abstractmethod
    def grains(
        self,
        image: numpy.ndarray,
        background: numpy.ndarray,
        previous: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param background: What the same view showed before the pour, BGR.
        :param previous: The image before it, BGR, or ``None`` for the first.
        :return: Which pixels show grains.
        """


@dataclass(frozen=True)
class WhiteSegmentation(GrainSegmentation):
    """
    Grains are bright, unsaturated pixels that differ from what the same pixel showed
    before the pour: white grains such as styrofoam balls or rice, which a shadow or a
    coloured object passing by does not pass for.
    """

    minimum_value: int = 150
    """
    The least brightness a grain pixel has, from 0 to 255.
    """

    maximum_saturation: int = 70
    """
    The most saturation a grain pixel has, from 0 to 255; white grains have little.
    """

    def grains(
        self,
        image: numpy.ndarray,
        background: numpy.ndarray,
        previous: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return (
            (hsv[..., 2] > self.minimum_value)
            & (hsv[..., 1] < self.maximum_saturation)
            & self.changed(image, background)
        )


@dataclass(frozen=True)
class ChangeSegmentation(GrainSegmentation):
    """
    Grains are pixels that differ from what the same pixel showed before the pour,
    whatever their colour: for grains such as corn kernels, yellow to brown, that no
    single colour picks out. Anything else moving through the opening counts as well.
    """

    def grains(
        self,
        image: numpy.ndarray,
        background: numpy.ndarray,
        previous: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        return self.changed(image, background)


@dataclass(frozen=True)
class HueBandSegmentation(GrainSegmentation):
    """
    Grains are pixels that differ from what the same pixel showed before the pour and
    whose colour lies in a band of hues, saturated and bright enough for the hue to mean
    something: coloured grains such as corn kernels, which grey, black or white things
    moving behind the receiver do not pass for.
    """

    minimum_hue: int = field(kw_only=True)
    """
    The lowest hue of a grain pixel, on OpenCV's scale from 0 to 179.
    """

    maximum_hue: int = field(kw_only=True)
    """
    The highest hue of a grain pixel, on OpenCV's scale from 0 to 179.
    """

    minimum_saturation: int = field(kw_only=True)
    """
    The least saturation a grain pixel has, from 0 to 255.
    """

    minimum_value: int = field(kw_only=True)
    """
    The least brightness a grain pixel has, from 0 to 255.
    """

    def grains(
        self,
        image: numpy.ndarray,
        background: numpy.ndarray,
        previous: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return (
            (hsv[..., 0] >= self.minimum_hue)
            & (hsv[..., 0] <= self.maximum_hue)
            & (hsv[..., 1] >= self.minimum_saturation)
            & (hsv[..., 2] >= self.minimum_value)
            & self.changed(image, background)
        )


@dataclass(frozen=True)
class MovingGrainSegmentation(GrainSegmentation):
    """
    Grains are pixels that changed since the previous frame and whose colour differs
    from what the empty view showed there: falling grains of any colour, however thin
    and blurred the stream, but neither grains lying still in a filled receiver, nor
    shadows, nor the background's own colour brightening or darkening as something
    passes in front of it.
    """

    def grains(
        self,
        image: numpy.ndarray,
        background: numpy.ndarray,
        previous: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        if previous is None:
            return numpy.zeros(image.shape[:2], dtype=bool)
        return self.moved(image, previous) & self.differs_from_empty_view(
            image, background
        )


MINIMUM_YELLOWING = 8
"""
How much yellower than the empty view, on the blue-yellow axis of the Lab colour space
from 0 to 255, a pixel has to turn to show corn against a light background: corn behind
the translucent cup's frosted wall turned it 10 to 30 in a video of a pour into the cup,
the bare wall above the corn no more than 7.
"""

LAB_BLUE_YELLOW_CHANNEL = 2
"""
The channel of an OpenCV Lab image running from blue to yellow.
"""


@dataclass(frozen=True)
class Yellowing:
    """
    Corn told from a light background by turning it yellower, however pale and blurred
    it shows: a blue cup, a grey arm or a white wall passing by do not turn it so.
    """

    minimum: int = MINIMUM_YELLOWING
    """
    How much yellower than the background a pixel has to turn.
    """

    def shows(self, image: numpy.ndarray, background: numpy.ndarray) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :param background: What the same view showed before the pour, BGR.
        :return: Which pixels turned yellow enough to show corn.
        """
        channel = LAB_BLUE_YELLOW_CHANNEL
        yellowing = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., channel].astype(
            numpy.int16
        ) - cv2.cvtColor(background, cv2.COLOR_BGR2LAB)[..., channel].astype(
            numpy.int16
        )
        return yellowing >= self.minimum


MINIMUM_YELLOW_MOVEMENT = 45
"""
How much a pixel's colour has to change from the previous frame, summed over the three
channels, for corn moving over a light background to count as moving: on the corn pour
into the translucent cup of 2026-10-15 18:33, the stream's count stood 45 times above
the creep while no stream fell (the source cup moving over the opening), against 20
times at 20 and 70 times at 60, which leaves too few pixels of a thin trickle.
"""


@dataclass(frozen=True)
class MovingYellowSegmentation(GrainSegmentation):
    """
    Grains are pixels that changed since the previous frame and turned yellower than the
    empty view: corn falling through the light inside of a translucent cup, against which
    a thin, blurred stream barely changes the brightness.
    """

    minimum_difference: int = MINIMUM_YELLOW_MOVEMENT
    """
    How much a pixel's colour has to change from the previous frame to count as moving.
    """

    yellowing: Yellowing = Yellowing()
    """
    How corn is told from the light background.
    """

    def grains(
        self,
        image: numpy.ndarray,
        background: numpy.ndarray,
        previous: Optional[numpy.ndarray] = None,
    ) -> numpy.ndarray:
        if previous is None:
            return numpy.zeros(image.shape[:2], dtype=bool)
        return self.moved(image, previous) & self.yellowing.shows(image, background)


# %% the receiver's rim seen in the image


OUTLINE_SEARCH_MARGIN = 100
"""
How far around the projected opening the receiver's outline is looked for, in pixels.
"""

TOP_EDGE_ROWS = 15
"""
How many rows below its top edge the outline's centre is taken from, in pixels.
"""


@dataclass(frozen=True)
class PixelOffset:
    """
    How far the opening was moved in the image, in pixels.
    """

    rows: int
    """
    How many rows down it was moved.
    """

    columns: int
    """
    How many columns right it was moved.
    """

    def applied_to(self, mask: numpy.ndarray) -> numpy.ndarray:
        """
        :param mask: Which pixels of an image are set.
        :return: The same pixels moved by this offset, those moved off the image held at
            its edge.
        """
        rows, columns = numpy.nonzero(mask)
        height, width = mask.shape
        moved = numpy.zeros_like(mask)
        moved[
            numpy.clip(rows + self.rows, 0, height - 1),
            numpy.clip(columns + self.columns, 0, width - 1),
        ] = True
        return moved

    def undone(self, pixel: numpy.ndarray) -> numpy.ndarray:
        """
        :param pixel: A ``(column, row)`` in the image.
        :return: Where that pixel lay before the move: where the camera's pose projects
            the point of the opening it shows.
        """
        return pixel - numpy.array([self.columns, self.rows])


NO_OFFSET = PixelOffset(rows=0, columns=0)
"""
An opening left where it projects.
"""


@dataclass
class ReceiverNotFoundError(Exception):
    """
    Raised when the receiver's outline does not show near where its opening projects.
    """

    def __str__(self) -> str:
        return (
            "The receiving cup does not show near where its opening projects into the "
            "image: check that it stands in view and that the camera's pose is right."
        )


@dataclass(frozen=True)
class ReceiverOutline:
    """
    The receiving cup's colour, by which its outline is found in the empty view, so the
    opening projected from a camera pose that is a few centimetres off can be moved onto
    the rim the image shows.
    """

    minimum_hue: int
    """
    The lowest hue of the cup, on OpenCV's scale from 0 to 179.
    """

    maximum_hue: int
    """
    The highest hue of the cup, on OpenCV's scale from 0 to 179.
    """

    minimum_saturation: int
    """
    The least saturation of the cup, from 0 to 255.
    """

    def shows_cup(self, image: numpy.ndarray) -> numpy.ndarray:
        """
        :param image: A colour image, BGR.
        :return: Which of its pixels show the cup's colour.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return (
            (hsv[..., 0] >= self.minimum_hue)
            & (hsv[..., 0] <= self.maximum_hue)
            & (hsv[..., 1] >= self.minimum_saturation)
        )

    def align(self, opening: numpy.ndarray, background: numpy.ndarray) -> numpy.ndarray:
        """
        :param opening: Which pixels the opening projects to.
        :param background: The empty view, BGR.
        :return: The same opening, moved so its top meets the top edge of the cup's
            outline and its centre the centre of that edge.
        :raises ReceiverNotFoundError: If the cup does not show near the projection.
        """
        return self.offset(opening, background).applied_to(opening)

    def offset(self, opening: numpy.ndarray, background: numpy.ndarray) -> PixelOffset:
        """
        :param opening: Which pixels the opening projects to.
        :param background: The empty view, BGR.
        :return: How far the opening has to move for its top to meet the top edge of the
            cup's outline and its centre the centre of that edge.
        :raises ReceiverNotFoundError: If the cup does not show near the projection.
        """
        rows, columns = numpy.nonzero(opening)
        height, width = opening.shape
        top, bottom = max(0, rows.min() - OUTLINE_SEARCH_MARGIN), min(
            height, rows.max() + OUTLINE_SEARCH_MARGIN
        )
        left, right = max(0, columns.min() - OUTLINE_SEARCH_MARGIN), min(
            width, columns.max() + OUTLINE_SEARCH_MARGIN
        )
        cup = self.shows_cup(background[top:bottom, left:right])
        count, labels, statistics, _ = cv2.connectedComponentsWithStats(
            cup.astype(numpy.uint8)
        )
        if count < 2:
            raise ReceiverNotFoundError()
        largest = 1 + int(numpy.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        cup_rows, cup_columns = numpy.nonzero(labels == largest)
        cup_top = cup_rows.min()
        centre = cup_columns[cup_rows < cup_top + TOP_EDGE_ROWS].mean() + left
        return PixelOffset(
            rows=int(cup_top + top - rows.min()),
            columns=int(round(centre - columns.mean())),
        )


BLUE_CUP_OUTLINE = ReceiverOutline(
    minimum_hue=95, maximum_hue=125, minimum_saturation=80
)
"""
The blue Jeroen cups' outline: the hue band the receiving cup shows in the pole camera.
"""


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

    heap_fill: Optional[float] = None
    """
    How full the receiver is by the heap the camera sees lying in it, or ``None`` where
    too little of it shows to tell.
    """

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "stamp": self.stamp,
            "stream_seen": self.stream_seen,
            "counted_grains": self.counted_grains,
            "crossing": None if self.crossing is None else self.crossing.to_json(),
            "heap_fill": self.heap_fill,
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> StreamReading:
        crossing = data["crossing"]
        return cls(
            stamp=data["stamp"],
            stream_seen=data["stream_seen"],
            counted_grains=data["counted_grains"],
            crossing=None if crossing is None else PlanePoint._from_json(crossing),
            heap_fill=data.get("heap_fill"),
        )


HEAP_STILL_SECONDS = 1.0
"""
How long a pixel inside the receiver has to show the same grains to count as the heap
lying there, in seconds: longer than any grain of the stream takes to fall through.
"""

WALL_STILL_SECONDS = 0.33
"""
How long a pixel of a translucent receiver has to show its contents to count as contents
lying in it, in seconds: long enough to drop the stream falling past the wall and the
flicker at the threshold, short enough that the reading follows the heap under a dense
stream, which resets the clock of every pixel it crosses.

Held for a second, the reading stood still for three seconds of a pour on 2026-10-19
while the receiver filled from 0.28 to 0.38.
"""

MINIMUM_STREAM_PIXELS = 30
"""
How many grain pixels the opening has to show for a stream to count as seen: one grain
of a few millimetres, at the pole camera's millimetre per pixel.
"""


@dataclass
class TwoReceiverGaugesError(Exception):
    """
    Raised when a detector is given both the heap gauge of an opaque receiver and the
    wall gauge of a translucent one: a receiver is one or the other.
    """

    def __str__(self) -> str:
        return (
            "A stream detector reads the receiver with the heap gauge of an opaque cup "
            "or the wall gauge of a translucent one, not both."
        )


@dataclass
class CupOnMatWithoutWallGaugeError(Exception):
    """
    Raised when a detector is to find a translucent receiver on its mat without the wall
    gauge that knows where the receiver projects.
    """

    def __str__(self) -> str:
        return (
            "A stream detector finds a translucent receiver on its mat only with the "
            "wall gauge that reads it."
        )


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

    segmentation: GrainSegmentation
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

    outline: Optional[ReceiverOutline] = None
    """
    The receiving cup's outline the opening is moved onto in the empty view, or ``None``
    to keep the opening where it projects.
    """

    cup_on_mat: Optional[CupOnMat] = None
    """
    The mat a translucent receiver is found standing on in the empty view, the opening
    moved with it, or ``None``.

    Needs :attr:`wall_gauge`, which knows where the receiver projects.
    """

    heap_gauge: Optional[HeapGauge] = None
    """
    What the heap seen lying in the receiver says of its fill, or ``None`` not to read
    the heap.

    Needs :attr:`outline`, which tells the receiver's inside from the rest.
    """

    wall_gauge: Optional[WallGauge] = None
    """
    What the contents seen lying in a translucent receiver say of its fill, or ``None``
    not to read them.
    """

    _background: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    What the camera showed before the pour.
    """

    _mask: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    Which pixels show the opening.
    """

    _opening_offset: PixelOffset = field(init=False, default=NO_OFFSET)
    """
    How far the opening was moved from where the camera's pose projects it.
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

    _heap_region: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    Which pixels of the window show the receiver's inside, or ``None`` without an
    outline.
    """

    _still_frames: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    For every pixel of the window, how many frames in a row it has shown grains in the
    receiver.
    """

    _previous: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    The window around the opening of the image observed last, or ``None`` before the
    first.
    """

    _wall_window: Tuple[slice, slice] = field(init=False, default=(slice(0), slice(0)))
    """
    The rows and columns of the smallest image window holding the translucent receiver.
    """

    _wall_region: Optional[numpy.ndarray] = field(init=False, default=None, repr=False)
    """
    Which pixels of the wall window show the translucent receiver, or ``None`` without a
    wall gauge.
    """

    _wall_background: Optional[numpy.ndarray] = field(
        init=False, default=None, repr=False
    )
    """
    What the wall window showed before the pour.
    """

    _wall_still_frames: Optional[numpy.ndarray] = field(
        init=False, default=None, repr=False
    )
    """
    For every pixel of the wall window, how many frames in a row it has shown the
    contents.
    """

    def __post_init__(self) -> None:
        if self.heap_gauge is not None and self.wall_gauge is not None:
            raise TwoReceiverGaugesError()
        if self.cup_on_mat is not None and self.wall_gauge is None:
            raise CupOnMatWithoutWallGaugeError()

    def learn_background(self, images: List[numpy.ndarray]) -> None:
        """
        Remember what the view shows before the pour, and start counting from nothing.

        :param images: Images taken before any grain leaves the source, BGR.
        """
        image_size = images[0].shape[:2]
        mask = self.opening.mask_in(self.view, image_size)
        empty_view = numpy.median(numpy.stack(images), axis=0).astype(numpy.uint8)
        self._opening_offset = self._offset_of(mask, empty_view)
        mask = self._opening_offset.applied_to(mask)
        self._learn_wall(empty_view)
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
        self._previous = None
        self._heap_region = (
            None
            if self.outline is None
            else self._mask & self.outline.shows_cup(self._background)
        )
        self._still_frames = numpy.zeros(self._mask.shape, dtype=int)

    def _offset_of(
        self, opening: numpy.ndarray, empty_view: numpy.ndarray
    ) -> PixelOffset:
        """
        :param opening: Which pixels the opening projects to.
        :param empty_view: What the camera showed before the pour, BGR.
        :return: How far the camera's pose error moves the opening in the image: found
            from the receiver's outline, from the gap it makes in its mat, or none.
        """
        if self.outline is not None:
            return self.outline.offset(opening, empty_view)
        if self.cup_on_mat is not None:
            return self.cup_on_mat.offset(
                self.wall_gauge.region_in(empty_view.shape[:2], NO_OFFSET), empty_view
            )
        return NO_OFFSET

    def _learn_wall(self, empty_view: numpy.ndarray) -> None:
        """
        Remember what the translucent receiver looks like empty, where there is a wall
        gauge to read it with.

        :param empty_view: What the camera showed before the pour, BGR.
        """
        if self.wall_gauge is None:
            return
        region = self.wall_gauge.region_in(empty_view.shape[:2], self._opening_offset)
        rows, columns = numpy.nonzero(region)
        self._wall_window = (
            slice(rows.min(), rows.max() + 1),
            slice(columns.min(), columns.max() + 1),
        )
        self._wall_region = region[self._wall_window]
        self._wall_background = empty_view[self._wall_window]
        self._wall_still_frames = numpy.zeros(self._wall_region.shape, dtype=int)

    def wall_mask(self, image_size: Tuple[int, int]) -> numpy.ndarray:
        """
        :param image_size: Height and width of the camera's images.
        :return: Which pixels the detector reads the translucent receiver's fill in,
            where it found the receiver.
        """
        mask = numpy.zeros(image_size, dtype=bool)
        mask[self._wall_window] = self._wall_region
        return mask

    @property
    def wall_pixels(self) -> int:
        """
        :return: How many pixels of the translucent receiver have shown its contents
            lying still for :data:`WALL_STILL_SECONDS`.
        """
        still_for = round(WALL_STILL_SECONDS / self.frame_period)
        return int((self._wall_still_frames >= still_for).sum())

    def _watch_wall(self, image: numpy.ndarray) -> Optional[float]:
        """
        Count, for every pixel of the translucent receiver, how long it has shown the
        contents.

        :param image: The latest image, BGR.
        :return: The fill the contents lying still show, or ``None`` without a wall
            gauge or with too few of them to tell.
        """
        if self.wall_gauge is None:
            return None
        contents = (
            self.wall_gauge.shows_contents(
                image[self._wall_window], self._wall_background
            )
            & self._wall_region
        )
        self._wall_still_frames = numpy.where(contents, self._wall_still_frames + 1, 0)
        return self.wall_gauge.fill_for(self.wall_pixels)

    @property
    def heap_pixels(self) -> int:
        """
        :return: How many pixels of the receiver's inside have shown grains lying still
            for :data:`HEAP_STILL_SECONDS`.
        """
        still_for = round(HEAP_STILL_SECONDS / self.frame_period)
        return int((self._still_frames >= still_for).sum())

    def _watch_heap(self, window: numpy.ndarray) -> None:
        """
        Count, for every pixel of the receiver's inside, how long it has shown grains.

        :param window: The window around the opening of the latest image, BGR.
        """
        if self._heap_region is None:
            return
        grains = (
            self.segmentation.differs_from_empty_view(window, self._background)
            & self._heap_region
        )
        self._still_frames = numpy.where(grains, self._still_frames + 1, 0)

    def opening_mask(self, image_size: Tuple[int, int]) -> numpy.ndarray:
        """
        :param image_size: Height and width of the camera's images.
        :return: Which pixels the detector watches as the opening, where it moved it.
        """
        mask = numpy.zeros(image_size, dtype=bool)
        mask[self._window] = self._mask
        return mask

    def grain_mask(
        self, image: numpy.ndarray, previous: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        :param image: An image of the view, BGR.
        :param previous: The image before it, BGR, or ``None``.
        :return: Which of its pixels show grains in the opening, over the whole image.
        """
        mask = numpy.zeros(image.shape[:2], dtype=bool)
        mask[self._window] = self._grains_in_window(
            image[self._window], None if previous is None else previous[self._window]
        )
        return mask

    def _grains_in_window(
        self, window: numpy.ndarray, previous: Optional[numpy.ndarray]
    ) -> numpy.ndarray:
        """
        :param window: The window around the opening of an image, BGR.
        :param previous: The same window of the image before, BGR, or ``None``.
        :return: Which of its pixels show grains in the opening.
        """
        return (
            self.segmentation.grains(window, self._background, previous=previous)
            & self._mask
        )

    def observe(self, image: numpy.ndarray, stamp: float) -> StreamReading:
        """
        :param image: The latest image, BGR.
        :param stamp: When it was taken, in seconds.
        :return: What it shows of the stream.
        """
        window = image[self._window]
        grains = self._grains_in_window(window, self._previous)
        self._previous = window
        self._watch_heap(window)
        heap_fill = (
            self._watch_wall(image)
            if self.heap_gauge is None
            else self.heap_gauge.fill_for(self.heap_pixels)
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
                heap_fill=heap_fill,
            )
        rows, columns = numpy.nonzero(grains)
        crossing = self.view.on_plane(
            self._opening_offset.undone(
                numpy.array(
                    [
                        columns.mean() + self._window[1].start,
                        rows.mean() + self._window[0].start,
                    ]
                )
            ),
            plane_height=float(self.opening.centre[2]),
        )
        return StreamReading(
            stamp=stamp,
            stream_seen=True,
            counted_grains=self._counted,
            crossing=crossing,
            heap_fill=heap_fill,
        )
