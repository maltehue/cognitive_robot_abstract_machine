import numpy
import pytest

from krrood.adapters.json_serializer import from_json, to_json

from experiments.real_pouring.stream_detection import (
    CameraView,
    OpeningRegion,
    StreamDetector,
    WhiteSegmentation,
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


def _detector() -> StreamDetector:
    detector = StreamDetector(
        view=_looking_down(),
        opening=OPENING,
        segmentation=WhiteSegmentation(),
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
