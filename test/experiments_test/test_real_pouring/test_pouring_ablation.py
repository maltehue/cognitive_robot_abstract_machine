"""
Frozen forecasts preserve mass and never consume future observations.
"""

import numpy
import pytest

from experiments.pouring_model_verification.hypothesis_bank import (
    DrainGeometry,
    ModelPoint,
)
from experiments.real_pouring.pouring_ablation import FrozenForecast, TiltTrajectory

# %% forecast contracts


@pytest.fixture
def forecast():
    """
    A flowing source with known geometry and fixed parameters.
    """
    return FrozenForecast(
        geometry=DrainGeometry(0.16, 0.035, 100.0, 0.035),
        point=ModelPoint(0.0, 0.2, 0.8),
        source_grains=50.0,
        grains_per_receiver=200.0,
    )


def test_upright_forecast_has_no_outflow(forecast):
    """
    An upright cup cannot deliver material over the rim.
    """
    trajectory = TiltTrajectory(numpy.array([0.0, 3.0]), numpy.zeros(2))
    assert forecast.predict(trajectory, 0.0, 3.0) == 0.0


def test_forecast_cannot_deliver_more_than_source(forecast):
    """
    Long forecasts respect the initially available contents.
    """
    trajectory = TiltTrajectory(numpy.array([0.0, 100.0]), numpy.full(2, numpy.pi / 2))
    predicted = forecast.predict(trajectory, 0.0, 100.0)
    assert 0 < predicted <= forecast.source_grains / forecast.grains_per_receiver


def test_forecast_does_not_mutate_initial_state(forecast):
    """
    Independent forecast horizons start from the same frozen state.
    """
    trajectory = TiltTrajectory(numpy.array([0.0, 3.0]), numpy.full(2, numpy.pi / 2))
    first = forecast.predict(trajectory, 0.0, 1.5)
    forecast.predict(trajectory, 0.0, 3.0)
    assert forecast.predict(trajectory, 0.0, 1.5) == first
    assert forecast.source_grains == 50.0


def test_forecast_ignores_motion_after_horizon(forecast):
    """
    Motion outside the forecast interval cannot affect its result.
    """
    first = TiltTrajectory(numpy.array([0.0, 1.5, 3.0]), numpy.array([1.5, 1.5, 0.0]))
    second = TiltTrajectory(first.times, numpy.array([1.5, 1.5, 2.0]))
    assert forecast.predict(first, 0.0, 1.5) == forecast.predict(second, 0.0, 1.5)


def test_forecast_refuses_extrapolated_motion(forecast):
    """
    Missing future motion must not silently become a held-pose forecast.
    """
    trajectory = TiltTrajectory(numpy.array([0.0, 1.0]), numpy.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        forecast.predict(trajectory, 0.0, 1.5)
