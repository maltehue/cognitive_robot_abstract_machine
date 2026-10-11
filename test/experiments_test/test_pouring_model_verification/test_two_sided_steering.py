from dataclasses import dataclass

import pytest

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable
from experiments.pouring_model_verification.onset_rule import (
    ARRIVAL_CONFIRMATIONS,
    ArrivalWatch,
    OnsetResponse,
    OnsetRevision,
    OnsetRule,
    StallWatch,
)
from experiments.pouring_model_verification.transfer_trial import (
    OnsetSteering,
    TiltHistory,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)

PERIOD = 0.1
FILL = 0.7


# %% the tilt a reading looks back to


def test_the_tilt_at_a_time_is_the_last_one_read_by_then():
    history = TiltHistory()
    for step in range(5):
        history.record(at=PERIOD * step, tilt=1.0 + step)
    assert history.at(2.5 * PERIOD) == 3.0


def test_a_time_before_the_first_reading_gives_the_first_tilt():
    history = TiltHistory()
    history.record(at=1.0, tilt=0.8)
    assert history.at(0.0) == 0.8


# %% lowering the onset while the cup turns


@dataclass
class ReportedReadings:
    """
    Stands in for what the sensor reports, set by the test.
    """

    reported_receiver_grains: float = 0.0
    reported_source_fill: float = FILL
    reported_source_grains: float = 100.0
    predicted_receiver_inflow: float = 0.0


@dataclass
class TurningCup:
    """
    Stands in for the scene, whose cup tilt the test turns.
    """

    cup_tilt: float = 0.0


def _steering(lag: float):
    variables = FloatVariableData()
    correction = FloatVariable("repose_correction")
    variables.register_expression(correction)
    variables.set_value(correction, 0.0)
    equation = ArticulatedPouringEquation(
        container_height=0.06, container_width=0.07, repose_angle=0.9
    )
    rule = OnsetRule(response=OnsetResponse.REVISION, patience=2.0, lowers=True)
    readings, cup = ReportedReadings(), TurningCup()
    steering = OnsetSteering(
        motion=None,
        watch=StallWatch(rule=rule, goal=40.0, tolerance=7.0),
        responder=OnsetRevision(
            equation=equation, correction=correction, variables=variables
        ),
        perception=readings,
        scene=cup,
        arrival=ArrivalWatch(noise=0.0, reference=0.0),
        lag=lag,
    )
    return steering, readings, cup, equation, variables, correction


def test_grains_arriving_early_lower_the_onset_to_the_tilt_one_lag_before_they_were_seen():
    lag = 0.3
    steering, readings, cup, equation, variables, correction = _steering(lag)
    first_seen = 1.0
    for step in range(20):
        at = PERIOD * step
        cup.cup_tilt = 0.5 + 0.1 * step
        readings.reported_receiver_grains = 5.0 if at >= first_seen - 1e-9 else 0.0
        steering.steer(at)
    onset = float(equation.onset_tilt(FILL).evaluate()[0]) + variables.get_value(
        correction
    )
    assert onset == pytest.approx(0.5 + 0.1 * round((first_seen - lag) / PERIOD))


def test_the_onset_is_lowered_once_the_arrival_is_confirmed():
    steering, readings, cup, *_ = _steering(lag=0.0)
    cup.cup_tilt = 0.5
    readings.reported_receiver_grains = 5.0
    lowered_at = []
    for step in range(10):
        steering.steer(PERIOD * step)
        lowered_at.append(len(steering.revisions))
    assert lowered_at.index(1) == ARRIVAL_CONFIRMATIONS - 1
