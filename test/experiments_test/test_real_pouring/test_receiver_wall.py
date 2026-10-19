import cv2
import numpy
import pytest

from experiments.real_pouring.receiver_heap import MINIMUM_HEAP_PIXELS
from experiments.real_pouring.receiver_wall import BLUE_MAT, WallGauge
from experiments.real_pouring.stream_detection import (
    NO_OFFSET,
    CameraView,
    PixelOffset,
    ReceiverNotFoundError,
)

# %% a translucent cup seen from the side and above, like the pole camera sees it

HEIGHT = 0.155
"""
The cup's height, in metres.
"""

RADIUS = 0.045
"""
The cup's inner radius, in metres.
"""

DETECTED_SHARE = 0.8
"""
The share of the contents' projected area the camera picks out.
"""

IMAGE_SIZE = (960, 1280)
"""
Height and width of the camera's images, in pixels.
"""


def _looking_at(eye: numpy.ndarray, target: numpy.ndarray) -> CameraView:
    """
    A camera at ``eye`` whose optical axis points at ``target``, image rows downwards.
    """
    forward = (target - eye) / numpy.linalg.norm(target - eye)
    right = numpy.cross(forward, [0.0, 0.0, 1.0])
    right /= numpy.linalg.norm(right)
    down = numpy.cross(forward, right)
    world_T_camera = numpy.eye(4)
    world_T_camera[:3, :3] = numpy.column_stack([right, down, forward])
    world_T_camera[:3, 3] = eye
    intrinsics = numpy.array(
        [
            [1000.0, 0.0, IMAGE_SIZE[1] / 2],
            [0.0, 1000.0, IMAGE_SIZE[0] / 2],
            [0.0, 0.0, 1.0],
        ]
    )
    return CameraView(intrinsics=intrinsics, world_T_camera=world_T_camera)


VIEW = _looking_at(
    eye=numpy.array([0.6, 0.0, HEIGHT + 0.8]), target=numpy.array([0, 0, HEIGHT / 2])
)

GAUGE = WallGauge(
    view=VIEW,
    receiver_base=numpy.zeros(3),
    height=HEIGHT,
    radius=RADIUS,
    detected_share=DETECTED_SHARE,
)


# %% the area the contents show


def test_the_fuller_the_cup_the_larger_the_area_its_contents_show():
    areas = [GAUGE.projected_area(fill) for fill in numpy.linspace(0.0, 1.0, 11)]
    assert areas == sorted(areas)


def test_contents_lying_flat_on_the_floor_show_the_floor():
    angles = numpy.linspace(0.0, 2 * numpy.pi, 360, endpoint=False)
    floor = numpy.stack(
        [RADIUS * numpy.cos(angles), RADIUS * numpy.sin(angles), numpy.zeros(360)],
        axis=1,
    )
    floor_area = cv2.contourArea(VIEW.to_image(floor).astype(numpy.float32))
    assert GAUGE.projected_area(0.0) == pytest.approx(floor_area, rel=0.01)


def test_the_area_the_camera_picks_out_reads_back_as_the_fill_it_shows():
    pixels = DETECTED_SHARE * GAUGE.projected_area(0.4)
    assert GAUGE.fill_for(pixels) == pytest.approx(0.4, abs=0.01)


def test_too_few_pixels_tell_no_fill():
    assert GAUGE.fill_for(MINIMUM_HEAP_PIXELS - 1) is None


# %% what the contents look like through the wall


BEHIND_THE_WALL = (215, 205, 190)
"""
The empty cup's wall with the blue mat behind it, BGR.
"""

CORN_BEHIND_THE_WALL = (120, 150, 160)
"""
Corn kernels seen through the frosted wall, BGR: duller and yellower than the wall.
"""


def _patch(colour: tuple) -> numpy.ndarray:
    return numpy.full((4, 4, 3), colour, dtype=numpy.uint8)


def test_corn_behind_the_wall_shows_as_contents():
    assert GAUGE.shows_contents(
        _patch(CORN_BEHIND_THE_WALL), _patch(BEHIND_THE_WALL)
    ).all()


def test_the_unchanged_wall_shows_no_contents():
    assert not GAUGE.shows_contents(
        _patch(BEHIND_THE_WALL), _patch(BEHIND_THE_WALL)
    ).any()


def test_a_blue_cup_in_front_of_the_wall_shows_no_contents():
    assert not GAUGE.shows_contents(
        _patch((200, 110, 30)), _patch(BEHIND_THE_WALL)
    ).any()


# %% where the cup shows in the image


def test_the_cup_shows_where_it_projects_moved_by_the_offset():
    offset = PixelOffset(rows=5, columns=7)
    assert (
        GAUGE.region_in(IMAGE_SIZE, offset)
        == offset.applied_to(GAUGE.region_in(IMAGE_SIZE, NO_OFFSET))
    ).all()


def test_the_cup_shows_as_large_as_full_contents_would():
    assert GAUGE.region_in(IMAGE_SIZE, NO_OFFSET).sum() == pytest.approx(
        GAUGE.projected_area(1.0), rel=0.02
    )


# %% finding the cup as the gap it makes in the mat it stands on


MAT_BLUE = (200, 90, 10)
"""
The blue mat's colour, BGR: deep and saturated.
"""

FROSTED_CUP = (215, 205, 190)
"""
The frosted cup's colour over the mat, BGR: pale.
"""


def _cup_on_mat(offset: PixelOffset) -> numpy.ndarray:
    image = numpy.full((*IMAGE_SIZE, 3), MAT_BLUE, dtype=numpy.uint8)
    image[GAUGE.region_in(IMAGE_SIZE, offset)] = FROSTED_CUP
    return image


def test_the_cup_is_found_where_it_interrupts_the_mat():
    shifted = PixelOffset(rows=12, columns=-18)
    found = BLUE_MAT.offset(
        GAUGE.region_in(IMAGE_SIZE, NO_OFFSET), _cup_on_mat(shifted)
    )
    assert (found.rows, found.columns) == pytest.approx(
        (shifted.rows, shifted.columns), abs=1
    )


def test_a_view_without_the_cup_on_the_mat_says_so():
    bare_mat = numpy.full((*IMAGE_SIZE, 3), MAT_BLUE, dtype=numpy.uint8)
    with pytest.raises(ReceiverNotFoundError):
        BLUE_MAT.offset(GAUGE.region_in(IMAGE_SIZE, NO_OFFSET), bare_mat)


ELBOW_CAP_GREY = (190, 180, 170)
"""
The colour of a joint cap of the arm lying over the mat, BGR: unsaturated like the cup.
"""

CAP_SIZE = (22, 104)
"""
Rows and columns a joint cap covers of the mat in the empty view.
"""


def _cap_below_the_cup(
    image: numpy.ndarray, rows_below: int, columns_left: int
) -> None:
    """
    Paint a joint cap over the mat, its bottom the given rows below the projected cup's
    bottom and its centre the given columns left of the cup's.
    """
    rows, columns = numpy.nonzero(GAUGE.region_in(IMAGE_SIZE, NO_OFFSET))
    bottom = rows.max() + rows_below
    centre = int(columns.mean()) - columns_left
    image[
        bottom - CAP_SIZE[0] : bottom,
        centre - CAP_SIZE[1] // 2 : centre + CAP_SIZE[1] // 2,
    ] = ELBOW_CAP_GREY


def test_the_gap_nearest_the_projection_is_the_cup_not_the_largest():
    shifted = PixelOffset(rows=10, columns=-12)
    image = _cup_on_mat(shifted)
    rows, columns = numpy.nonzero(GAUGE.region_in(IMAGE_SIZE, shifted))
    # the arm hides all but the cup's lowest rows, as it does at the pre-tilt pose
    image[: rows.max() - 20, columns.min() : columns.max() + 1] = MAT_BLUE
    _cap_below_the_cup(image, rows_below=90, columns_left=60)
    found = BLUE_MAT.offset(GAUGE.region_in(IMAGE_SIZE, NO_OFFSET), image)
    assert (found.rows, found.columns) == pytest.approx(
        (shifted.rows, shifted.columns), abs=1
    )


def test_a_gap_too_far_from_the_projection_is_not_the_cup():
    image = numpy.full((*IMAGE_SIZE, 3), MAT_BLUE, dtype=numpy.uint8)
    _cap_below_the_cup(image, rows_below=90, columns_left=60)
    with pytest.raises(ReceiverNotFoundError):
        BLUE_MAT.offset(GAUGE.region_in(IMAGE_SIZE, NO_OFFSET), image)
