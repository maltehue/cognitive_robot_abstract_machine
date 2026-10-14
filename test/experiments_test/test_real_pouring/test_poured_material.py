import pytest

from experiments.real_pouring import poured_material
from experiments.real_pouring.poured_material import (
    FullCupMassUnknownError,
    PouredMaterial,
)
from experiments.real_pouring.stream_detection import (
    HueBandSegmentation,
    WhiteSegmentation,
)
from giskardpy.middleware.ros2.particle_weighing import FULL_CUP_MASS_GRAMS


def test_styrofoam_is_seen_by_its_whiteness():
    assert isinstance(PouredMaterial.STYROFOAM.segmentation, WhiteSegmentation)


def test_corn_is_seen_by_its_hue():
    assert isinstance(PouredMaterial.CORN.segmentation, HueBandSegmentation)


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
