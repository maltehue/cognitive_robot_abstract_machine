"""
Consecutive offline solves retain the controller's commanded motion state.
"""

from __future__ import annotations

from copy import deepcopy
import numpy

from experiments.pouring_model_verification.recovery import DrainCorrectionValues
from experiments.real_pouring.controller_ablation import ControllerProbe
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world import World

from .test_controller_ablation import probe

# %% consecutive controller steps


def test_probe_uses_supplied_motion_configuration(tracy_world: World) -> None:
    """
    Offline tuning must reach the compiled controller without changing defaults.
    """
    configuration = QPControllerConfig(
        target_frequency=80, prediction_horizon=15, verbose=False
    )
    controller = ControllerProbe.build(
        deepcopy(tracy_world), goal=0.3, configuration=configuration
    )
    assert controller.executor.qp_controller.config is configuration


def test_advancing_probe_preserves_commanded_velocity_and_acceleration(
    probe: ControllerProbe,
) -> None:
    """
    A consecutive solve leaves its integrated derivatives for the next cycle.
    """
    initial = probe.world.state._data.copy()
    result = probe.advance(DrainCorrectionValues(0.0, 0.0, 0.0))
    numpy.testing.assert_array_equal(probe.world.state.velocities, result.velocities)
    assert numpy.any(probe.world.state.accelerations != 0.0)
    assert not numpy.array_equal(probe.world.state._data, initial)


def test_advancing_probe_matches_snapshot_command_from_same_state(
    probe: ControllerProbe,
) -> None:
    """
    Advancing changes state retention, without changing the controller solve.
    """
    correction = DrainCorrectionValues(0.0, 0.0, 0.0)
    snapshot = probe.command(correction)
    advanced = probe.advance(correction)
    numpy.testing.assert_allclose(advanced.velocities, snapshot.velocities, atol=1e-7)
    numpy.testing.assert_allclose(
        advanced.planned_tilt_velocities, snapshot.planned_tilt_velocities, atol=1e-7
    )
