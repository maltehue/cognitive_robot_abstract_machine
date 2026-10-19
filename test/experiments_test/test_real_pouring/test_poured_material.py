from dataclasses import replace

import pytest

from experiments.pouring_model_verification.hypothesis_bank import ModelPoint

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


def test_styrofoam_keeps_flowing_below_its_repose_angle_by_its_own_hysteresis():
    assert (
        PouredMaterial.STYROFOAM.repose_hysteresis
        == poured_material.STYROFOAM_REPOSE_HYSTERESIS
    )


def test_corn_keeps_flowing_below_its_repose_angle_by_its_own_hysteresis():
    assert (
        PouredMaterial.CORN.repose_hysteresis == poured_material.CORN_REPOSE_HYSTERESIS
    )


@pytest.mark.parametrize("material", list(PouredMaterial))
def test_a_drain_poured_with_a_material_takes_on_its_hysteresis(material):
    drain = ModelPoint(
        repose_angle=0.1, discharge_coefficient=0.2, outflow_rate_constant=0.8
    )
    assert material.drain_of(drain) == replace(
        drain,
        repose_hysteresis=material.repose_hysteresis,
        repose_hysteresis_growth=material.repose_hysteresis_growth,
    )


def test_corn_keeps_flowing_further_back_the_emptier_the_cup_gets():
    assert (
        PouredMaterial.CORN.repose_hysteresis_growth
        == poured_material.CORN_REPOSE_HYSTERESIS_GROWTH
    )
    assert PouredMaterial.CORN.repose_hysteresis_growth > 0.0


def test_styrofoam_keeps_flowing_as_far_back_whatever_is_left():
    assert PouredMaterial.STYROFOAM.repose_hysteresis_growth == 0.0
