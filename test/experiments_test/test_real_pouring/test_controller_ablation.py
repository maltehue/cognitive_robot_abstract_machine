"""
Offline command probes preserve state and isolate drain corrections.
"""

from copy import deepcopy
import math

import numpy
import pytest

from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.controller_ablation import ControllerProbe

# %% full arm command probes


@pytest.fixture
def probe(tracy_world):
    """
    A disconnected Tracy controller with the demo's pouring tasks.
    """
    return ControllerProbe.build(deepcopy(tracy_world), goal=0.3)


def test_command_probe_preserves_world_state(probe):
    """
    A counterfactual solve must not advance the replay state.
    """
    before = probe.world.state._data.copy()
    probe.command(DrainCorrectionValues(0.0, 0.0, 0.0))
    numpy.testing.assert_array_equal(probe.world.state._data, before)


def test_command_probe_changes_predicted_flow(probe):
    """
    A changed drain rate reaches the production symbolic fill expression.
    """
    probe.source.fill_connection.position = 1.0
    repose = float(probe.source.pour_tilt_expression.evaluate()[0]) - math.pi / 2
    first = probe.command(DrainCorrectionValues(repose, 0.0, 0.0))
    second = probe.command(DrainCorrectionValues(repose, math.log(2.0), 0.0))
    assert first.inflow > 0.0
    assert second.inflow == pytest.approx(2.0 * first.inflow)


def test_repeated_probe_is_independent_of_other_models(probe):
    """
    Solver reuse must not mix alternative model states.
    """
    corrections = DrainCorrectionValues(0.0, 0.0, 0.0)
    first = probe.command(corrections)
    probe.command(DrainCorrectionValues(0.5, math.log(2.0), 0.0))
    repeated = probe.command(corrections)
    numpy.testing.assert_allclose(repeated.velocities, first.velocities, atol=1e-6)


def test_first_planned_tilt_velocity_matches_command(probe):
    """
    The extracted plan must use the solver's velocity units and joint order.
    """
    result = probe.command(DrainCorrectionValues(0.0, 0.0, 0.0))
    assert result.planned_tilt_velocities[0] == pytest.approx(
        result.tilt_velocity, abs=1e-7
    )
    assert (
        len(result.planned_tilt_velocities)
        == probe.executor.qp_controller.config.control_horizon
    )


def test_disabled_fill_constraint_removes_drain_rate_influence(probe):
    """
    Removing the predictive fill row isolates the other demo tasks.
    """
    before = probe.executor.motion_statechart.life_cycle_state.data.copy()
    first = probe.command(
        DrainCorrectionValues(0.0, 0.0, 0.0), use_fill_constraint=False
    )
    second = probe.command(
        DrainCorrectionValues(0.0, math.log(2.0), 0.0), use_fill_constraint=False
    )
    numpy.testing.assert_allclose(first.velocities, second.velocities, atol=1e-6)
    numpy.testing.assert_array_equal(
        probe.executor.motion_statechart.life_cycle_state.data, before
    )


def test_forecast_duration_preserves_motion_horizon(tracy_world):
    """
    Changing fill lookahead must leave the arm's optimization horizon unchanged.
    """
    short = ControllerProbe.build(
        deepcopy(tracy_world), goal=0.3, prediction_duration=1.5
    )
    long = ControllerProbe.build(
        deepcopy(tracy_world), goal=0.3, prediction_duration=3.0
    )
    assert long.fill_task.prediction_duration == 2 * short.fill_task.prediction_duration
    assert (
        long.executor.qp_controller.config.control_horizon
        == short.executor.qp_controller.config.control_horizon
    )
    correction = DrainCorrectionValues(0.0, 0.0, 0.0)
    short_command = short.command(correction, use_fill_constraint=False)
    long_command = long.command(correction, use_fill_constraint=False)
    numpy.testing.assert_allclose(
        short_command.planned_tilt_velocities,
        long_command.planned_tilt_velocities,
        atol=1e-6,
    )
