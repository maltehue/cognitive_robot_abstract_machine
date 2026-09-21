"""
Tests for holding a drain's scale at what the observed pour says it should be.
"""

from __future__ import annotations

import krrood.symbolic_math.symbolic_math as sm
import pytest

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable

from semantic_digital_twin.physics.drain_calibration import CalibratedDrainScale

UNCORRECTED_INFLOW = 0.2
"""
What the model predicts before anything corrects it, per second.
"""


def build_scale(
    uncorrected_inflow: float = UNCORRECTED_INFLOW,
    starting_at: float = 1.0,
    **kwargs,
) -> CalibratedDrainScale:
    """
    A drain whose model predicts ``uncorrected_inflow`` at a factor of one.

    :param uncorrected_inflow: What the model predicts before a correction.
    :param starting_at: The factor the drain starts scaled by.
    :param kwargs: Overrides for the scale's own settings.
    :return: The scale.
    """
    scale = FloatVariable("drain_scale")
    variables = FloatVariableData()
    variables.register_expression(scale)
    variables.set_value(scale, starting_at)
    return CalibratedDrainScale(
        scale=scale,
        inflow=sm.Scalar(uncorrected_inflow) * scale,
        variables=variables,
        **kwargs,
    )


# %% what the model predicts


def test_the_prediction_follows_the_factor():
    """
    The point of the factor is that the compiled prediction reads it, so changing it
    changes what the controller acts on without anything being rebuilt.
    """
    calibrated = build_scale()
    assert calibrated.predicted() == pytest.approx(UNCORRECTED_INFLOW)

    calibrated.variables.set_value(calibrated.scale, 0.25)

    assert calibrated.predicted() == pytest.approx(0.25 * UNCORRECTED_INFLOW)
    assert calibrated.uncorrected() == pytest.approx(UNCORRECTED_INFLOW)


# %% correcting it


def test_a_correction_makes_the_model_predict_what_was_measured():
    """
    Taken all the way, the factor is the one that would have made the model right.
    """
    measured = 0.05
    calibrated = build_scale(smoothing=1.0)

    calibrated.calibrate(measured_inflow=measured)

    assert calibrated.predicted() == pytest.approx(measured)
    assert calibrated.value == pytest.approx(measured / UNCORRECTED_INFLOW)


def test_a_correction_moves_part_of_the_way_when_it_is_smoothed():
    """
    A factor that jumps to each observation chases the noise in it.
    """
    calibrated = build_scale(smoothing=0.5)

    corrected = calibrated.calibrate(measured_inflow=0.0)

    assert corrected == pytest.approx(0.5)


def test_nothing_arriving_drives_the_factor_towards_nothing_pouring():
    """
    Told repeatedly that nothing arrives, the model has to stop predicting that
    something does, or the controller never learns its tilt is not enough.
    """
    calibrated = build_scale(smoothing=0.5)

    for _ in range(10):
        calibrated.calibrate(measured_inflow=0.0)

    assert calibrated.value == pytest.approx(0.0, abs=1e-2)
    assert calibrated.predicted() == pytest.approx(0.0, abs=1e-3)


def test_a_model_predicting_nothing_leaves_the_factor_alone():
    """
    That a pour has not started says nothing about how fast it will run once it has, so
    there is nothing to learn from it.
    """
    calibrated = build_scale(uncorrected_inflow=0.0, starting_at=0.4)

    assert calibrated.calibrate(measured_inflow=0.1) == pytest.approx(0.4)
    assert calibrated.value == pytest.approx(0.4)


def test_the_factor_never_makes_the_model_pour_faster_than_it_was_calibrated_to():
    """
    A correction answers for a pour that is slower than predicted; a burst of arrivals
    must not talk the model into a drain its own calibration never claimed.
    """
    calibrated = build_scale(smoothing=1.0, maximum=1.0)

    calibrated.calibrate(measured_inflow=10 * UNCORRECTED_INFLOW)

    assert calibrated.value == pytest.approx(1.0)


# %% the factor on both paths


def test_a_registered_factor_is_readable_without_a_compiled_expression():
    """
    The physics integrates the same drain between control cycles, by evaluating it
    rather than through the compiled controller, so the factor has to answer there too.
    """
    calibrated = build_scale(smoothing=1.0)

    calibrated.calibrate(measured_inflow=0.5 * UNCORRECTED_INFLOW)

    assert calibrated.inflow.evaluate()[0] == pytest.approx(0.5 * UNCORRECTED_INFLOW)
