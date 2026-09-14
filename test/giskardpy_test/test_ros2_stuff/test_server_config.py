from giskardpy.executor import NoPacing, RealTimePacer, SimulationPacer
from giskardpy.middleware.ros2.server_config import ExecutionMode, GiskardServerConfig

# %% pacing the control loop

REAL_TIME_FACTOR = 1.0
"""
A factor that holds a simulated motion at the speed of the wall clock.
"""


def test_a_standalone_server_without_a_factor_runs_unpaced():
    pacer = GiskardServerConfig(execution_mode=ExecutionMode.STANDALONE).create_pacer()

    assert isinstance(pacer, NoPacing)


def test_a_standalone_server_with_a_factor_paces_its_simulation():
    pacer = GiskardServerConfig(
        execution_mode=ExecutionMode.STANDALONE, real_time_factor=REAL_TIME_FACTOR
    ).create_pacer()

    assert isinstance(pacer, SimulationPacer)
    assert pacer.real_time_factor == REAL_TIME_FACTOR


def test_a_closed_loop_server_runs_in_wall_clock_time():
    pacer = GiskardServerConfig(
        execution_mode=ExecutionMode.CLOSED_LOOP, real_time_factor=REAL_TIME_FACTOR
    ).create_pacer()

    assert isinstance(pacer, RealTimePacer)


# %% serving the world model


def test_a_server_publishes_its_world_by_default():
    assert GiskardServerConfig().publishes_world
