import pytest

from experiments.real_pouring import poured_material
from experiments.real_pouring.poured_material import (
    FullCupMassUnknownError,
    PouredMaterial,
)
from experiments.real_pouring.stream_detection import MovingGrainSegmentation
from giskardpy.middleware.ros2.particle_weighing import FULL_CUP_MASS_GRAMS


@pytest.mark.parametrize("material", list(PouredMaterial))
def test_every_material_is_counted_by_what_moves_through_the_opening(material):
    assert isinstance(material.segmentation, MovingGrainSegmentation)


def test_a_full_cup_of_styrofoam_weighs_what_the_scale_said():
    assert PouredMaterial.STYROFOAM.full_cup_mass_grams == FULL_CUP_MASS_GRAMS


def test_a_material_never_weighed_says_so_instead_of_guessing(monkeypatch):
    monkeypatch.setattr(poured_material, "CORN_FULL_CUP_MASS_GRAMS", None)
    with pytest.raises(FullCupMassUnknownError):
        PouredMaterial.CORN.full_cup_mass_grams


def test_a_full_cup_of_corn_weighs_what_was_set_for_it():
    assert (
        PouredMaterial.CORN.full_cup_mass_grams
        == poured_material.CORN_FULL_CUP_MASS_GRAMS
    )


def test_corn_is_counted_with_its_own_calibration():
    assert (
        PouredMaterial.CORN.pixel_seconds_per_grain
        == poured_material.CORN_PIXEL_SECONDS_PER_GRAIN
    )


def test_styrofoam_reads_its_heap_on_its_own_scale():
    assert (
        PouredMaterial.STYROFOAM.heap_pixels_per_share
        == poured_material.STYROFOAM_HEAP_PIXELS_PER_SHARE
    )


def test_corn_reads_its_heap_on_its_own_scale():
    assert (
        PouredMaterial.CORN.heap_pixels_per_share
        == poured_material.CORN_HEAP_PIXELS_PER_SHARE
    )
