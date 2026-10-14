import pytest

from giskardpy.middleware.ros2.particle_weighing import WeighedParticles
from krrood.adapters.json_serializer import from_json, to_json

CORN_FULL_CUP_MASS_GRAMS = 30.0
"""
A full cup's mass for a heavier material than the styrofoam balls, in grams.
"""


def test_the_received_fill_is_the_received_mass_over_a_full_cups_mass():
    weighed = WeighedParticles(
        received_mass_grams=21.0,
        spilled_mass_grams=1.0,
        full_cup_mass_grams=CORN_FULL_CUP_MASS_GRAMS,
    )
    assert weighed.received_fill_level == pytest.approx(21.0 / CORN_FULL_CUP_MASS_GRAMS)


def test_the_full_cups_mass_survives_a_round_trip_through_json():
    weighed = WeighedParticles(
        received_mass_grams=21.0,
        spilled_mass_grams=1.0,
        full_cup_mass_grams=CORN_FULL_CUP_MASS_GRAMS,
    )
    assert from_json(to_json(weighed)) == weighed


def test_the_answers_are_weighed_against_the_full_cup_given():
    answers = iter(["21", "1"])
    weighed = WeighedParticles.ask(
        answer_prompt=lambda question: next(answers),
        full_cup_mass_grams=CORN_FULL_CUP_MASS_GRAMS,
    )
    assert weighed.full_cup_mass_grams == CORN_FULL_CUP_MASS_GRAMS
