import pytest

from ..pouring_worlds import OPENING_RADIUS, tilted_source_above_receiver


def test_the_receivers_centre_lies_where_the_geometry_says_along_the_stream():
    stream = tilted_source_above_receiver(tilt=1.0)
    centre = stream.world.compute_forward_kinematics_np(
        stream.world.root, stream.receiver.root
    )[:2, 3]
    assert stream.distance_from_lip([centre]) == pytest.approx(
        stream.geometry(OPENING_RADIUS).centre_distance
    )


def test_the_opening_stands_a_receiver_height_above_its_base():
    stream = tilted_source_above_receiver(tilt=1.0)
    assert stream.opening_height == pytest.approx(stream.receiver.root.collision.height)


def test_the_stream_leaves_the_lip_above_the_opening():
    stream = tilted_source_above_receiver(tilt=1.0)
    assert stream.launch().drop > 0.0
