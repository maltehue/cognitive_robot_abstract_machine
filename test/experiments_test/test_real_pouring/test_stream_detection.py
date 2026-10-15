import numpy
import pytest

from krrood.adapters.json_serializer import from_json, to_json

from experiments.real_pouring.receiver_heap import HeapGauge
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    HEAP_STILL_SECONDS,
    CameraView,
    ChangeSegmentation,
    GrainSegmentation,
    HueBandSegmentation,
    MovingGrainSegmentation,
    OpeningRegion,
    ReceiverNotFoundError,
    StreamDetector,
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
