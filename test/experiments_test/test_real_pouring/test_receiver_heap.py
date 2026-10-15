import numpy
import pytest

from experiments.real_pouring.receiver_heap import HeapGauge

HEIGHT = 0.16
"""
The receiving cup's height, in metres.
"""

RADIUS = 0.035
"""
The receiving cup's opening radius, in metres.
"""

PIXELS_PER_SHARE = 2000.0
"""
Heap pixels the whole surface would show.
"""


def _gauge(camera_offset: float) -> HeapGauge:
    """
    A cup standing at the origin, seen from a camera one metre above its rim and
    ``camera_offset`` metres to the side.
    """
    return HeapGauge(
        camera_position=numpy.array([camera_offset, 0.0, HEIGHT + 1.0]),
        receiver_base=numpy.zeros(3),
        height=HEIGHT,
        radius=RADIUS,
        pixels_per_share=PIXELS_PER_SHARE,
    )


OBLIQUE = _gauge(camera_offset=1.0)
"""
A camera looking down into the cup at 45 degrees, like the pole camera.
"""


def test_a_camera_straight_above_sees_the_whole_surface_at_any_fill():
    assert _gauge(camera_offset=0.0).visible_share(0.3) == pytest.approx(1.0)


def test_from_the_side_a_low_surface_hides_behind_the_wall():
    assert OBLIQUE.visible_share(0.4) == 0.0


def test_the_visible_share_grows_with_the_fill_to_the_whole_surface():
    shares = [OBLIQUE.visible_share(fill) for fill in numpy.linspace(0.5, 1.0, 11)]
    assert shares == sorted(shares)
    assert shares[-1] == pytest.approx(1.0)


def test_the_heap_area_reads_back_as_the_fill_it_shows():
    pixels = PIXELS_PER_SHARE * OBLIQUE.visible_share(0.7)
    assert OBLIQUE.fill_for(pixels) == pytest.approx(0.7, abs=0.01)


def test_no_heap_in_view_reads_as_no_fill_known():
    assert OBLIQUE.fill_for(0) is None
