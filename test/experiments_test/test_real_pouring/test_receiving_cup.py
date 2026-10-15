import numpy
import pytest

from experiments.real_pouring import poured_material, receiving_cup
from experiments.real_pouring.poured_material import (
    MaterialNotSeenThroughWallError,
    PouredMaterial,
)
from experiments.real_pouring.receiving_cup import (
    ReceiverFullMassUnknownError,
    ReceivingCupChangedError,
    ReceivingCup,
)
from experiments.real_pouring.receiver_wall import BLUE_MAT
from experiments.real_pouring.stream_detection import (
    BLUE_CUP_OUTLINE,
    CameraView,
    MovingYellowSegmentation,
)

VIEW = CameraView(
    intrinsics=numpy.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0, 0, 1]]),
    world_T_camera=numpy.eye(4),
)
"""
A camera; where it stands does not matter for building a detector.
"""

OPENING_CENTRE = numpy.array([0.0, 0.0, 1.0])
"""
Where the receiver's opening is, in front of the camera.
"""


def _detector(cup: ReceivingCup, material: PouredMaterial = PouredMaterial.CORN):
    return cup.detector_for(
        material=material,
        view=VIEW,
        opening_centre=OPENING_CENTRE,
        frame_period=1 / 30,
    )


def test_the_jeroen_cup_is_found_by_its_blue_outline_and_read_by_its_heap():
    detector = _detector(ReceivingCup.JEROEN_CUP)
    assert (
        detector.outline,
        detector.heap_gauge.pixels_per_share,
        detector.wall_gauge,
    ) == (BLUE_CUP_OUTLINE, PouredMaterial.CORN.heap_pixels_per_share, None)


def test_the_translucent_cup_is_found_on_its_mat_and_read_through_its_wall():
    detector = _detector(ReceivingCup.TRANSLUCENT_CUP)
    assert (
        detector.outline,
        detector.cup_on_mat,
        detector.wall_gauge.detected_share,
        detector.heap_gauge,
    ) == (
        None,
        BLUE_MAT,
        receiving_cup.TRANSLUCENT_CUP_CORN_WALL_DETECTED_SHARE,
        None,
    )


def test_corn_falling_into_the_translucent_cup_is_counted_by_turning_it_yellow():
    detector = _detector(ReceivingCup.TRANSLUCENT_CUP)
    assert (
        type(detector.segmentation),
        detector.pixel_seconds_per_grain,
    ) == (
        MovingYellowSegmentation,
        receiving_cup.TRANSLUCENT_CUP_CORN_PIXEL_SECONDS_PER_GRAIN,
    )


def test_corn_falling_into_the_jeroen_cup_is_counted_as_the_material_says():
    detector = _detector(ReceivingCup.JEROEN_CUP)
    assert (detector.segmentation, detector.pixel_seconds_per_grain) == (
        PouredMaterial.CORN.segmentation,
        PouredMaterial.CORN.pixel_seconds_per_grain,
    )


@pytest.mark.parametrize("cup", list(ReceivingCup))
def test_the_detector_watches_the_cups_own_opening(cup):
    assert _detector(cup).opening.radius == cup.inner_radius


def test_styrofoam_does_not_show_through_a_frosted_wall():
    with pytest.raises(MaterialNotSeenThroughWallError):
        _detector(ReceivingCup.TRANSLUCENT_CUP, PouredMaterial.STYROFOAM)


def test_the_jeroen_cup_holds_one_source_cup():
    assert ReceivingCup.JEROEN_CUP.source_cups_held(PouredMaterial.CORN) == 1.0


def test_the_translucent_cup_holds_as_many_source_cups_as_the_scale_said():
    assert ReceivingCup.TRANSLUCENT_CUP.source_cups_held(PouredMaterial.CORN) == (
        receiving_cup.TRANSLUCENT_CUP_CORN_MASS_GRAMS
        / poured_material.CORN_FULL_CUP_MASS_GRAMS
    )


def test_a_cup_nobody_filled_with_a_material_says_so():
    with pytest.raises(ReceiverFullMassUnknownError):
        ReceivingCup.TRANSLUCENT_CUP.full_mass_grams(PouredMaterial.STYROFOAM)


SOURCE_CAPACITY = 0.001
"""
The source cup's capacity in the controller's model, in cubic metres.
"""


def test_a_receiver_left_from_a_pour_into_the_same_cup_may_be_reused():
    cup = ReceivingCup.TRANSLUCENT_CUP
    cup.check_reused(
        receiver_capacity=SOURCE_CAPACITY * cup.source_cups_held(PouredMaterial.CORN),
        source_capacity=SOURCE_CAPACITY,
        material=PouredMaterial.CORN,
    )


def test_a_receiver_left_from_a_pour_into_another_cup_may_not_be_reused():
    with pytest.raises(ReceivingCupChangedError):
        ReceivingCup.TRANSLUCENT_CUP.check_reused(
            receiver_capacity=SOURCE_CAPACITY,
            source_capacity=SOURCE_CAPACITY,
            material=PouredMaterial.CORN,
        )


def test_the_translucent_cup_is_poured_to_about_the_share_of_a_source_cup_the_jeroen_cup_was():
    corn = PouredMaterial.CORN
    assert (
        ReceivingCup.TRANSLUCENT_CUP.goal_fill
        * ReceivingCup.TRANSLUCENT_CUP.source_cups_held(corn)
        == pytest.approx(
            ReceivingCup.JEROEN_CUP.goal_fill
            * ReceivingCup.JEROEN_CUP.source_cups_held(corn),
            abs=0.15,
        )
    )
