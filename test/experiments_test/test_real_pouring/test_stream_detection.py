from dataclasses import replace

import cv2
import numpy
import pytest

from krrood.adapters.json_serializer import from_json, to_json

from experiments.real_pouring.receiver_heap import HeapGauge
from experiments.real_pouring.receiver_wall import BLUE_MAT

from .test_receiver_wall import (
    CORN_BEHIND_THE_WALL,
    _cup_on_mat,
    GAUGE,
    HEIGHT,
    IMAGE_SIZE as WALL_IMAGE_SIZE,
    RADIUS,
)
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    HEAP_STILL_SECONDS,
    WALL_STILL_SECONDS,
    COLOURED_EDGE_DISTANCE,
    CupOnMatWithoutWallGaugeError,
    FLICKER_MARGIN,
    NO_OFFSET,
    CameraView,
    ChangeSegmentation,
    GrainSegmentation,
    HueBandSegmentation,
    MINIMUM_WATER_MOVEMENT,
    MovingGrainSegmentation,
    MovingWaterSegmentation,
    MINIMUM_YELLOW_MOVEMENT,
    MovingColourShiftSegmentation,
    Yellowing,
    OpeningRegion,
    PixelOffset,
    ReceiverNotFoundError,
    StreamDetector,
    TwoReceiverGaugesError,
    WhiteSegmentation,
    StreamReading,
)

# %% a camera looking straight down at a receiver's opening

CAMERA_HEIGHT = 1.0
"""
How far above the opening plane the camera hangs, in metres.
"""

FOCAL_LENGTH = 1000.0
"""
The camera's focal length, in pixels.
"""

IMAGE_SIZE = (480, 640)
"""
Height and width of the camera's images, in pixels.
"""


def _looking_down() -> CameraView:
    """
    A camera centred above the world origin, its optical axis pointing straight down.
    """
    intrinsics = numpy.array(
        [
            [FOCAL_LENGTH, 0.0, IMAGE_SIZE[1] / 2],
            [0.0, FOCAL_LENGTH, IMAGE_SIZE[0] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    world_T_camera = numpy.eye(4)
    world_T_camera[:3, :3] = numpy.diag([1.0, -1.0, -1.0])
    world_T_camera[2, 3] = CAMERA_HEIGHT
    return CameraView(intrinsics=intrinsics, world_T_camera=world_T_camera)


OPENING = OpeningRegion(centre=numpy.array([0.0, 0.0, 0.0]), radius=0.05)


def _image_with_white_square(row: int, column: int, size: int = 6) -> numpy.ndarray:
    """
    A dark image with one white square, as a grain against the receiver's inside.
    """
    image = numpy.full((*IMAGE_SIZE, 3), 30, dtype=numpy.uint8)
    image[row : row + size, column : column + size] = 250
    return image


def _dark_image() -> numpy.ndarray:
    return numpy.full((*IMAGE_SIZE, 3), 30, dtype=numpy.uint8)


FRAME_PERIOD = 1 / 30
"""
How long apart the camera takes its images, in seconds.
"""

GRAIN_SIZE = 6
"""
The side of a grain's square in the test images, in pixels.
"""

PIXEL_SECONDS_PER_GRAIN = GRAIN_SIZE**2 * FRAME_PERIOD
"""
What one grain adds while it shows for one frame.
"""


def _detector(
    segmentation: GrainSegmentation = WhiteSegmentation(),
) -> StreamDetector:
    detector = StreamDetector(
        view=_looking_down(),
        opening=OPENING,
        segmentation=segmentation,
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
    )
    detector.learn_background([_dark_image()])
    return detector


# %% the opening in the image


def test_the_opening_projects_to_a_disk_around_the_image_centre():
    mask = OPENING.mask_in(_looking_down(), IMAGE_SIZE)
    radius_in_pixels = FOCAL_LENGTH * OPENING.radius / CAMERA_HEIGHT
    assert mask[IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2]
    assert not mask[0, 0]
    assert mask.sum() == pytest.approx(numpy.pi * radius_in_pixels**2, rel=0.05)


def test_a_pixel_is_traced_back_to_where_it_lies_on_the_opening_plane():
    view = _looking_down()
    point = view.on_plane(
        pixel=numpy.array([IMAGE_SIZE[1] / 2 + 20.0, IMAGE_SIZE[0] / 2]),
        plane_height=0.0,
    )
    assert (point.x, point.y) == pytest.approx(
        (20.0 * CAMERA_HEIGHT / FOCAL_LENGTH, 0.0)
    )


# %% readings


def test_nothing_white_in_the_opening_is_no_stream():
    reading = _detector().observe(_dark_image(), stamp=0.0)
    assert (reading.stream_seen, reading.counted_grains, reading.crossing) == (
        False,
        0.0,
        None,
    )


def test_a_grain_in_the_opening_is_a_stream_crossing_where_it_shows():
    centre = (IMAGE_SIZE[0] // 2 - 3, IMAGE_SIZE[1] // 2 + 17)
    reading = _detector().observe(_image_with_white_square(*centre), stamp=0.0)
    assert reading.stream_seen
    assert (reading.crossing.x, reading.crossing.y) == pytest.approx(
        (20.0 * CAMERA_HEIGHT / FOCAL_LENGTH, 0.0), abs=1e-3
    )


def test_a_grain_outside_the_opening_is_not_a_stream():
    reading = _detector().observe(_image_with_white_square(10, 10), stamp=0.0)
    assert not reading.stream_seen


def test_the_count_adds_up_the_white_area_seen_over_time():
    detector = _detector()
    centre = (IMAGE_SIZE[0] // 2 - 3, IMAGE_SIZE[1] // 2 - 3)
    for frame in range(3):
        reading = detector.observe(
            _image_with_white_square(*centre), stamp=frame * FRAME_PERIOD
        )
    assert reading.counted_grains == pytest.approx(3.0)


def test_a_frame_the_detector_did_not_get_to_does_not_lose_grains():
    detector = _detector()
    centre = (IMAGE_SIZE[0] // 2 - 3, IMAGE_SIZE[1] // 2 - 3)
    for frame in (0, 2):
        reading = detector.observe(
            _image_with_white_square(*centre), stamp=frame * FRAME_PERIOD
        )
    assert reading.counted_grains == pytest.approx(3.0)


def test_what_stood_in_the_opening_before_the_pour_is_not_counted():
    detector = StreamDetector(
        view=_looking_down(),
        opening=OPENING,
        segmentation=WhiteSegmentation(),
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
    )
    centre = (IMAGE_SIZE[0] // 2 - 3, IMAGE_SIZE[1] // 2 - 3)
    detector.learn_background([_image_with_white_square(*centre)])
    reading = detector.observe(_image_with_white_square(*centre), stamp=0.0)
    assert (reading.stream_seen, reading.counted_grains) == (False, 0.0)


def test_a_reading_survives_a_round_trip_through_json():
    reading = _detector().observe(
        _image_with_white_square(IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), stamp=1.5
    )
    restored = from_json(to_json(reading))
    assert restored == reading


# %% grains that are not white


YELLOW_KERNEL = (60, 170, 220)
"""
A yellow corn kernel's colour, BGR: bright but strongly saturated.
"""

BROWN_KERNEL = (40, 90, 140)
"""
A brownish corn kernel's colour, BGR: saturated and darker than a white grain.
"""


def _image_with_square(row: int, column: int, colour: tuple[int, int, int]):
    """
    A dark image with one coloured square of a grain's size.
    """
    image = _dark_image()
    image[row : row + GRAIN_SIZE, column : column + GRAIN_SIZE] = colour
    return image


CENTRE = (IMAGE_SIZE[0] // 2 - 3, IMAGE_SIZE[1] // 2 - 3)
"""
Where a grain's square sits so that it lies inside the opening.
"""


@pytest.mark.parametrize("colour", [YELLOW_KERNEL, BROWN_KERNEL])
def test_whiteness_does_not_see_a_coloured_grain(colour):
    reading = _detector().observe(_image_with_square(*CENTRE, colour), stamp=0.0)
    assert not reading.stream_seen


@pytest.mark.parametrize("colour", [YELLOW_KERNEL, BROWN_KERNEL])
def test_a_change_from_the_empty_view_counts_a_grain_of_any_colour(colour):
    reading = _detector(ChangeSegmentation()).observe(
        _image_with_square(*CENTRE, colour), stamp=0.0
    )
    assert (reading.stream_seen, reading.counted_grains) == (True, pytest.approx(1.0))


def test_a_change_smaller_than_the_threshold_is_not_a_grain():
    segmentation = ChangeSegmentation()
    barely_changed = tuple(30 + segmentation.minimum_difference // 3 for _ in range(3))
    reading = _detector(segmentation).observe(
        _image_with_square(*CENTRE, barely_changed), stamp=0.0
    )
    assert not reading.stream_seen


# %% grains told apart by their hue


CORN_HUES = HueBandSegmentation(
    minimum_hue=8, maximum_hue=35, minimum_saturation=80, minimum_value=60
)
"""
A hue band from orange-brown to yellow, as for corn kernels.
"""

DARK_TROUSERS = (35, 30, 30)
"""
Dark clothing passing behind the receiver, BGR.
"""

WHITE_SHOE = (235, 235, 235)
"""
A white shoe passing behind the receiver, BGR.
"""

GREY_SHADOW = (90, 90, 90)
"""
A shadow falling on the grey table behind the receiver, BGR.
"""


@pytest.mark.parametrize("colour", [YELLOW_KERNEL, BROWN_KERNEL])
def test_a_grain_within_the_hue_band_is_counted(colour):
    reading = _detector(CORN_HUES).observe(
        _image_with_square(*CENTRE, colour), stamp=0.0
    )
    assert (reading.stream_seen, reading.counted_grains) == (True, pytest.approx(1.0))


@pytest.mark.parametrize("colour", [DARK_TROUSERS, WHITE_SHOE, GREY_SHADOW])
def test_a_change_outside_the_hue_band_is_not_a_grain(colour):
    reading = _detector(CORN_HUES).observe(
        _image_with_square(*CENTRE, colour), stamp=0.0
    )
    assert not reading.stream_seen


# %% the window moved onto the receiver's rim seen in the image


BLUE = (200, 110, 30)
"""
The receiving cup's blue, BGR.
"""

CUP_TOP_ROW = IMAGE_SIZE[0] // 2 + 30
"""
The image row the receiving cup's top edge shows at: below where the opening projects.
"""

CUP_COLUMNS = (IMAGE_SIZE[1] // 2 - 15, IMAGE_SIZE[1] // 2 + 45)
"""
The image columns the receiving cup spans: right of where the opening projects.
"""


def _view_with_cup() -> numpy.ndarray:
    image = _dark_image()
    image[CUP_TOP_ROW : CUP_TOP_ROW + 120, CUP_COLUMNS[0] : CUP_COLUMNS[1]] = BLUE
    return image


def test_the_window_moves_onto_the_cups_top_edge():
    projected = OPENING.mask_in(_looking_down(), IMAGE_SIZE)
    aligned = BLUE_CUP_OUTLINE.align(projected, _view_with_cup())
    rows, columns = numpy.nonzero(aligned)
    assert rows.min() == CUP_TOP_ROW
    assert columns.mean() == pytest.approx(sum(CUP_COLUMNS) / 2 - 0.5, abs=1.0)
    assert aligned.sum() == projected.sum()


def test_a_view_without_the_cup_says_so():
    projected = OPENING.mask_in(_looking_down(), IMAGE_SIZE)
    with pytest.raises(ReceiverNotFoundError):
        BLUE_CUP_OUTLINE.align(projected, _dark_image())


# %% grains that move


MOVING = MovingGrainSegmentation()
"""
Grains are what moved since the last frame and differs from the empty view's colour.
"""


def _grain_at(column_offset: int, colour=YELLOW_KERNEL, background=None):
    image = _dark_image() if background is None else background.copy()
    row, column = CENTRE
    image[
        row : row + GRAIN_SIZE,
        column + column_offset : column + column_offset + GRAIN_SIZE,
    ] = colour
    return image


def test_a_grain_that_moved_is_counted_where_it_is_now():
    grains = MOVING.grains(_grain_at(10), _dark_image(), previous=_grain_at(0))
    assert grains.sum() == GRAIN_SIZE**2


def test_a_grain_lying_still_is_not_counted():
    grains = MOVING.grains(_grain_at(0), _dark_image(), previous=_grain_at(0))
    assert grains.sum() == 0


def test_nothing_is_counted_without_a_previous_frame():
    assert MOVING.grains(_grain_at(10), _dark_image(), previous=None).sum() == 0


def test_a_change_in_the_backgrounds_own_colour_is_not_a_grain():
    blue = numpy.full((*IMAGE_SIZE, 3), BLUE, dtype=numpy.uint8)
    lighter_blue = tuple(min(255, value + 60) for value in BLUE)
    grains = MOVING.grains(
        _grain_at(10, lighter_blue, blue),
        blue,
        previous=_grain_at(0, lighter_blue, blue),
    )
    assert grains.sum() == 0


def test_a_shadow_is_not_a_grain():
    grey = numpy.full((*IMAGE_SIZE, 3), 160, dtype=numpy.uint8)
    shadow = (70, 70, 70)
    grains = MOVING.grains(
        _grain_at(10, shadow, grey), grey, previous=_grain_at(0, shadow, grey)
    )
    assert grains.sum() == 0


# %% the detector on the moved window, counting moving grains


def _detector_on_the_cups_rim() -> StreamDetector:
    detector = StreamDetector(
        view=_looking_down(),
        opening=OPENING,
        segmentation=MOVING,
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
        outline=BLUE_CUP_OUTLINE,
    )
    detector.learn_background([_view_with_cup()])
    return detector


def _falling_grain(row: int, column: int) -> numpy.ndarray:
    image = _view_with_cup()
    image[row : row + GRAIN_SIZE, column : column + GRAIN_SIZE] = YELLOW_KERNEL
    return image


RIM_CENTRE = (CUP_TOP_ROW + 10, sum(CUP_COLUMNS) // 2)
"""
A point just inside the cup's rim, where the moved window lies.
"""


def test_a_grain_falling_through_the_rim_is_counted():
    detector = _detector_on_the_cups_rim()
    detector.observe(_falling_grain(RIM_CENTRE[0] - 12, RIM_CENTRE[1]), stamp=0.0)
    reading = detector.observe(_falling_grain(*RIM_CENTRE), stamp=FRAME_PERIOD)
    assert reading.stream_seen


def test_a_grain_falling_where_the_opening_projects_but_the_cup_is_not_is_not_counted():
    detector = _detector_on_the_cups_rim()
    projected_top = (IMAGE_SIZE[0] // 2 - 45, IMAGE_SIZE[1] // 2 - 40)
    detector.observe(_falling_grain(projected_top[0] - 12, projected_top[1]), stamp=0.0)
    reading = detector.observe(_falling_grain(*projected_top), stamp=FRAME_PERIOD)
    assert not reading.stream_seen


def test_a_grain_falling_through_the_moved_opening_crosses_where_the_opening_lies():
    detector = _detector_on_the_cups_rim()
    rows, columns = numpy.nonzero(detector.opening_mask(IMAGE_SIZE))
    centre = (
        int(round(rows.mean())) - GRAIN_SIZE // 2,
        int(round(columns.mean())) - GRAIN_SIZE // 2,
    )
    detector.observe(_falling_grain(centre[0] - 12, centre[1]), stamp=0.0)
    reading = detector.observe(_falling_grain(*centre), stamp=FRAME_PERIOD)
    pixel = CAMERA_HEIGHT / FOCAL_LENGTH
    assert (reading.crossing.x, reading.crossing.y) == pytest.approx(
        (OPENING.centre[0], OPENING.centre[1]), abs=2 * pixel
    )


def test_the_opening_drawn_is_the_moved_one():
    detector = _detector_on_the_cups_rim()
    rows, _ = numpy.nonzero(detector.opening_mask(IMAGE_SIZE))
    assert rows.min() == CUP_TOP_ROW


# %% the heap lying still in the receiver


STILL_FRAMES = round(HEAP_STILL_SECONDS / FRAME_PERIOD)
"""
How many frames a heap pixel has to stay still for.
"""


def _heap_in_cup(column_offset: int = 0) -> numpy.ndarray:
    image = _view_with_cup()
    row, column = RIM_CENTRE
    image[
        row : row + GRAIN_SIZE,
        column + column_offset : column + column_offset + GRAIN_SIZE,
    ] = YELLOW_KERNEL
    return image


def test_grains_lying_still_in_the_cup_for_a_second_are_the_heap():
    detector = _detector_on_the_cups_rim()
    for frame in range(STILL_FRAMES):
        detector.observe(_heap_in_cup(), stamp=frame * FRAME_PERIOD)
    assert detector.heap_pixels == GRAIN_SIZE**2


def test_grains_not_yet_still_for_a_second_are_not_the_heap():
    detector = _detector_on_the_cups_rim()
    for frame in range(STILL_FRAMES - 1):
        detector.observe(_heap_in_cup(), stamp=frame * FRAME_PERIOD)
    assert detector.heap_pixels == 0


def test_grains_that_keep_moving_are_not_the_heap():
    detector = _detector_on_the_cups_rim()
    for frame in range(2 * STILL_FRAMES):
        detector.observe(
            _heap_in_cup(column_offset=frame % 2 * GRAIN_SIZE),
            stamp=frame * FRAME_PERIOD,
        )
    assert detector.heap_pixels == 0


def test_a_reading_carries_the_fill_the_heap_shows():
    gauge = HeapGauge(
        camera_position=numpy.array([0.0, 0.0, CAMERA_HEIGHT]),
        receiver_base=numpy.array([0.0, 0.0, -0.16]),
        height=0.16,
        radius=OPENING.radius,
        pixels_per_share=GRAIN_SIZE**2 / 0.5,
    )
    detector = StreamDetector(
        view=_looking_down(),
        opening=OPENING,
        segmentation=MOVING,
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
        outline=BLUE_CUP_OUTLINE,
        heap_gauge=gauge,
    )
    detector.learn_background([_view_with_cup()])
    for frame in range(STILL_FRAMES):
        reading = detector.observe(_heap_in_cup(), stamp=frame * FRAME_PERIOD)
    assert reading.heap_fill == gauge.fill_for(GRAIN_SIZE**2)


def test_a_reading_from_before_the_heap_was_read_loads_without_it():
    reading = StreamReading(
        stamp=0.0, stream_seen=False, counted_grains=0.0, crossing=None
    )
    data = to_json(reading)
    del data["heap_fill"]
    assert from_json(data).heap_fill is None


# %% a translucent receiver whose opening is placed by a measured offset


WALL_HEIGHT = HEIGHT
"""
The translucent cup's height, in metres.
"""

WALL_RADIUS = RADIUS
"""
The translucent cup's inner radius, in metres.
"""

WALL_OPENING = OpeningRegion(
    centre=numpy.array([0.0, 0.0, WALL_HEIGHT]), radius=WALL_RADIUS
)
"""
The translucent cup's opening, the cup standing on the world origin.
"""

WALL_GAUGE = replace(GAUGE, detected_share=1.0)
"""
The translucent cup seen at an angle, like the pole camera sees it, every pixel of its
contents picked out.
"""

FROSTED_WALL = (215, 205, 190)
"""
The empty cup's frosted wall, BGR.
"""


def _frosted_view() -> numpy.ndarray:
    return numpy.full((*WALL_IMAGE_SIZE, 3), FROSTED_WALL, dtype=numpy.uint8)


def _detector_on_a_translucent_cup() -> StreamDetector:
    detector = StreamDetector(
        view=WALL_GAUGE.view,
        opening=WALL_OPENING,
        segmentation=MOVING,
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
        wall_gauge=WALL_GAUGE,
    )
    detector.learn_background([_frosted_view()])
    return detector


def _contents_up_to(fill: float) -> numpy.ndarray:
    """
    The frosted cup with corn up to ``fill``, where the gauge says it shows.
    """
    image = _frosted_view()
    cv2.fillPoly(
        image,
        [numpy.round(WALL_GAUGE.outline(fill)).astype(numpy.int32)],
        CORN_BEHIND_THE_WALL,
    )
    return image


def test_a_translucent_cup_found_on_its_mat_moves_the_opening_with_it():
    shifted = PixelOffset(rows=12, columns=-18)
    detector = StreamDetector(
        view=WALL_GAUGE.view,
        opening=WALL_OPENING,
        segmentation=MOVING,
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
        cup_on_mat=BLUE_MAT,
        wall_gauge=WALL_GAUGE,
    )
    detector.learn_background([_cup_on_mat(shifted)])
    projected = WALL_OPENING.mask_in(WALL_GAUGE.view, WALL_IMAGE_SIZE)
    assert (
        detector.opening_mask(WALL_IMAGE_SIZE) == shifted.applied_to(projected)
    ).all()


def test_a_translucent_cup_is_found_on_its_mat_only_with_its_wall_gauge():
    with pytest.raises(CupOnMatWithoutWallGaugeError):
        StreamDetector(
            view=WALL_GAUGE.view,
            opening=WALL_OPENING,
            segmentation=MOVING,
            pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
            frame_period=FRAME_PERIOD,
            cup_on_mat=BLUE_MAT,
        )


def test_contents_lying_still_behind_the_wall_tell_the_fill():
    detector = _detector_on_a_translucent_cup()
    for frame in range(STILL_FRAMES):
        reading = detector.observe(_contents_up_to(0.5), stamp=frame * FRAME_PERIOD)
    assert reading.heap_fill == pytest.approx(0.5, abs=0.02)


def test_contents_not_yet_shown_through_the_wall_long_enough_tell_no_fill():
    detector = _detector_on_a_translucent_cup()
    for frame in range(round(WALL_STILL_SECONDS / FRAME_PERIOD) - 1):
        reading = detector.observe(_contents_up_to(0.5), stamp=frame * FRAME_PERIOD)
    assert reading.heap_fill is None


def test_a_detector_reads_the_receiver_with_one_gauge_only():
    with pytest.raises(TwoReceiverGaugesError):
        StreamDetector(
            view=_looking_down(),
            opening=OPENING,
            segmentation=MOVING,
            pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
            frame_period=FRAME_PERIOD,
            outline=BLUE_CUP_OUTLINE,
            heap_gauge=HeapGauge.for_receiver(
                camera_position=numpy.array([0.0, 0.0, CAMERA_HEIGHT]),
                opening_centre=OPENING.centre,
                height=WALL_HEIGHT,
                radius=WALL_RADIUS,
                pixels_per_share=1.0,
            ),
            wall_gauge=WALL_GAUGE,
        )


# %% grains that move and turn the light inside of a translucent cup yellow


MOVING_YELLOW = MovingColourShiftSegmentation(
    colour_shift=Yellowing(), minimum_difference=MINIMUM_YELLOW_MOVEMENT
)
"""
Grains are what moved since the last frame and turned yellower than the empty view.
"""

PALE_KERNEL = (150, 195, 215)
"""
A motion-blurred corn kernel against the cup's frosted inside, BGR: pale, but yellow.
"""

GREY_THING = (150, 150, 150)
"""
Something grey passing over the opening, BGR.
"""


def test_a_pale_yellow_grain_moving_over_a_light_inside_is_a_grain():
    frosted = _frosted_view()
    grains = MOVING_YELLOW.grains(
        _grain_at(10, PALE_KERNEL, frosted),
        frosted,
        previous=_grain_at(0, PALE_KERNEL, frosted),
    )
    assert grains.sum() == GRAIN_SIZE**2


def test_a_grey_thing_moving_over_a_light_inside_is_no_grain():
    frosted = _frosted_view()
    grains = MOVING_YELLOW.grains(
        _grain_at(10, GREY_THING, frosted),
        frosted,
        previous=_grain_at(0, GREY_THING, frosted),
    )
    assert grains.sum() == 0


def test_a_pale_yellow_grain_lying_still_is_no_grain():
    frosted = _frosted_view()
    lying = _grain_at(0, PALE_KERNEL, frosted)
    assert MOVING_YELLOW.grains(lying, frosted, previous=lying).sum() == 0


def test_the_contents_lying_still_are_counted_pixel_by_pixel():
    detector = _detector_on_a_translucent_cup()
    image = _contents_up_to(0.5)
    for frame in range(STILL_FRAMES):
        detector.observe(image, stamp=frame * FRAME_PERIOD)
    painted = (image == CORN_BEHIND_THE_WALL).all(axis=2)
    assert detector.wall_pixels == int(painted.sum())


WALL_STILL_FRAMES = round(WALL_STILL_SECONDS / FRAME_PERIOD)
"""
How many frames the contents have to show through the wall to count.
"""


def test_contents_shown_through_the_wall_for_a_third_of_a_second_count():
    detector = _detector_on_a_translucent_cup()
    image = _contents_up_to(0.5)
    for frame in range(WALL_STILL_FRAMES):
        detector.observe(image, stamp=frame * FRAME_PERIOD)
    painted = (image == CORN_BEHIND_THE_WALL).all(axis=2)
    assert WALL_STILL_FRAMES < STILL_FRAMES
    assert detector.wall_pixels == int(painted.sum())


def test_contents_not_yet_shown_that_long_do_not_count():
    detector = _detector_on_a_translucent_cup()
    image = _contents_up_to(0.5)
    for frame in range(WALL_STILL_FRAMES - 1):
        detector.observe(image, stamp=frame * FRAME_PERIOD)
    assert detector.wall_pixels == 0


def test_the_translucent_cup_is_drawn_where_it_was_found():
    detector = _detector_on_a_translucent_cup()
    assert (
        detector.wall_mask(WALL_IMAGE_SIZE)
        == WALL_GAUGE.region_in(WALL_IMAGE_SIZE, NO_OFFSET)
    ).all()


# %% water moving without colour of its own over the light inside of a translucent cup


MOVING_WATER = MovingWaterSegmentation()
"""
Water is what moved since the last frame, differs from the empty view and has no colour.
"""

WATER_GLINT = (240, 235, 225)
"""
The frosted inside seen through a clear stream, BGR: brightened, still colourless.
"""

COLOUR_CHANNELS = 3
"""
How many channels a colour image has, over which a change is summed.
"""


def _faint_flicker() -> tuple[int, int, int]:
    """
    The frosted inside changed by just less than water has to change it.
    """
    step = (MINIMUM_WATER_MOVEMENT - 1) // COLOUR_CHANNELS
    return tuple(value + step for value in FROSTED_WALL)


def test_a_colourless_glint_moving_over_a_light_inside_is_water():
    frosted = _frosted_view()
    water = MOVING_WATER.grains(
        _grain_at(10, WATER_GLINT, frosted),
        frosted,
        previous=_grain_at(0, WATER_GLINT, frosted),
    )
    assert water.sum() == GRAIN_SIZE**2


def test_the_coloured_source_cup_moving_over_the_opening_is_no_water():
    frosted = _frosted_view()
    water = MOVING_WATER.grains(
        _grain_at(10, BLUE, frosted), frosted, previous=_grain_at(0, BLUE, frosted)
    )
    assert water.sum() == 0


def test_water_lying_still_is_no_stream():
    frosted = _frosted_view()
    lying = _grain_at(0, WATER_GLINT, frosted)
    assert MOVING_WATER.grains(lying, frosted, previous=lying).sum() == 0


def test_no_water_is_seen_without_a_previous_frame():
    frosted = _frosted_view()
    image = _grain_at(0, WATER_GLINT, frosted)
    assert MOVING_WATER.grains(image, frosted, previous=None).sum() == 0


def test_a_flicker_smaller_than_the_threshold_is_no_water():
    frosted = _frosted_view()
    flicker = _grain_at(0, _faint_flicker(), frosted)
    assert MOVING_WATER.grains(flicker, frosted, previous=frosted).sum() == 0


DARK_SHIRT = (35, 35, 40)
"""
A person's black shirt passing behind the opening, BGR: colourless, but dark.
"""


def test_a_dark_colourless_thing_moving_over_the_opening_is_no_water():
    frosted = _frosted_view()
    water = MOVING_WATER.grains(
        _grain_at(10, DARK_SHIRT, frosted),
        frosted,
        previous=_grain_at(0, DARK_SHIRT, frosted),
    )
    assert water.sum() == 0


# %% pixels that flicker in the empty view


FLICKERING_EDGE = -30
"""
Column offset from :data:`CENTRE` of an edge inside the opening that flickers between
the frosted inside and a glint while nothing pours.
"""

EMPTY_VIEW_FRAMES = 5
"""
How many images of the empty view the detector learns from.
"""


def _frosted_opening() -> numpy.ndarray:
    return numpy.full((*IMAGE_SIZE, 3), FROSTED_WALL, dtype=numpy.uint8)


def _flickering(frames: int, column_offset: int = FLICKERING_EDGE) -> list:
    """
    Images of the frosted inside whose edge at ``column_offset`` glints in every other
    frame.
    """
    frosted = _frosted_opening()
    return [
        _grain_at(column_offset, WATER_GLINT, frosted) if frame % 2 else frosted
        for frame in range(frames)
    ]


def _water_detector_learned_on(images: list) -> StreamDetector:
    detector = StreamDetector(
        view=_looking_down(),
        opening=OPENING,
        segmentation=MOVING_WATER,
        pixel_seconds_per_grain=PIXEL_SECONDS_PER_GRAIN,
        frame_period=FRAME_PERIOD,
    )
    detector.learn_background(images)
    return detector


def _counted(detector: StreamDetector, images: list) -> float:
    for frame, image in enumerate(images):
        reading = detector.observe(image, stamp=frame * FRAME_PERIOD)
    return reading.counted_grains


def test_pixels_that_flickered_in_the_empty_view_are_not_counted():
    detector = _water_detector_learned_on(_flickering(EMPTY_VIEW_FRAMES))
    assert _counted(detector, _flickering(EMPTY_VIEW_FRAMES)) == 0.0


def test_a_flicker_shifted_by_the_margin_is_not_counted():
    detector = _water_detector_learned_on(_flickering(EMPTY_VIEW_FRAMES))
    shifted = _flickering(EMPTY_VIEW_FRAMES, FLICKERING_EDGE + FLICKER_MARGIN)
    assert _counted(detector, shifted) == 0.0


def test_a_stream_beside_the_pixels_that_flickered_is_counted():
    detector = _water_detector_learned_on(_flickering(EMPTY_VIEW_FRAMES))
    frosted = _frosted_opening()
    stream = [_grain_at(offset, WATER_GLINT, frosted) for offset in (0, 10, 20)]
    assert _counted(detector, stream) == pytest.approx(len(stream) - 1)


# %% highlights on the edge of the coloured source cup


def _glint_beside_the_source_cup(gap: int) -> numpy.ndarray:
    """
    The frosted inside with the blue source cup at :data:`CENTRE` and a glint ``gap``
    pixels to its right.
    """
    image = _frosted_opening()
    row, column = CENTRE
    image[row : row + GRAIN_SIZE, column : column + GRAIN_SIZE] = BLUE
    glint = column + GRAIN_SIZE + gap
    image[row : row + GRAIN_SIZE, glint : glint + GRAIN_SIZE] = WATER_GLINT
    return image


def test_a_glint_on_the_edge_of_the_coloured_source_cup_is_no_water():
    frosted = _frosted_opening()
    water = MOVING_WATER.grains(
        _glint_beside_the_source_cup(gap=0), frosted, previous=frosted
    )
    assert water.sum() == 0


def test_a_glint_beyond_the_edge_of_the_coloured_source_cup_is_water():
    frosted = _frosted_opening()
    water = MOVING_WATER.grains(
        _glint_beside_the_source_cup(gap=COLOURED_EDGE_DISTANCE + 1),
        frosted,
        previous=frosted,
    )
    assert water.sum() == GRAIN_SIZE**2
