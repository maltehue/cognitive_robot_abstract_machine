import math

import pytest

import krrood.symbolic_math.symbolic_math as sm
from experiments.pouring_model_verification.pid import (
    PidFillController,
    PidGains,
    signed_tilt_about_pour_axis,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)

GOAL = 40.0
TOLERANCE = 5.0
MAXIMUM_TILT = 2.0


def _controller(gains: PidGains) -> PidFillController:
    return PidFillController(
        gains=gains, goal=GOAL, tolerance=TOLERANCE, maximum_tilt=MAXIMUM_TILT
    )


# %% the command


def test_before_any_measurement_the_cup_is_held_at_the_start_tilt():
    gains = PidGains(proportional=0.01, integral=0.0, derivative=0.0, start_tilt=0.7)
    assert _controller(gains).setpoint == gains.start_tilt


def test_the_proportional_term_tilts_by_the_missing_grains():
    gains = PidGains(proportional=0.01, integral=0.0, derivative=0.0, start_tilt=0.5)
    controller = _controller(gains)
    measured = 10.0
    assert controller.update(measured, at=0.0) == pytest.approx(
        gains.start_tilt + gains.proportional * (GOAL - measured)
    )


def test_the_integral_term_adds_up_the_missing_grains_over_time():
    gains = PidGains(proportional=0.0, integral=0.01, derivative=0.0, start_tilt=0.0)
    controller = _controller(gains)
    controller.update(0.0, at=0.0)
    controller.update(0.0, at=1.0)
    assert controller.update(0.0, at=2.0) == pytest.approx(gains.integral * GOAL * 2.0)


def test_the_derivative_term_eases_off_as_grains_arrive():
    gains = PidGains(proportional=0.0, integral=0.0, derivative=0.1, start_tilt=1.0)
    controller = _controller(gains)
    controller.update(0.0, at=0.0)
    arriving = 2.0
    assert controller.update(arriving, at=1.0) == pytest.approx(
        gains.start_tilt - gains.derivative * arriving
    )


def test_the_command_stays_between_upright_and_the_largest_tilt():
    gains = PidGains(proportional=1.0, integral=0.0, derivative=0.0, start_tilt=0.0)
    controller = _controller(gains)
    assert controller.update(0.0, at=0.0) == MAXIMUM_TILT
    assert controller.update(2 * GOAL, at=1.0) == 0.0


def test_the_integral_does_not_wind_up_while_the_command_is_saturated():
    gains = PidGains(proportional=0.1, integral=0.01, derivative=0.0, start_tilt=0.0)
    controller = _controller(gains)
    for second in range(100):
        controller.update(0.0, at=float(second))
    missing = TOLERANCE + 1.0
    assert controller.update(GOAL - missing, at=100.0) == pytest.approx(
        gains.proportional * missing + gains.integral * missing * 1.0
    )


# %% when it is done


def test_the_goal_counts_as_reached_within_the_tolerance():
    gains = PidGains(proportional=0.01, integral=0.0, derivative=0.0, start_tilt=0.0)
    controller = _controller(gains)
    controller.update(GOAL - TOLERANCE - 1.0, at=0.0)
    assert not controller.reached
    controller.update(GOAL - TOLERANCE, at=0.1)
    assert controller.reached


def test_a_reached_goal_stays_reached_when_a_reading_drops_again():
    gains = PidGains(proportional=0.01, integral=0.0, derivative=0.0, start_tilt=0.0)
    controller = _controller(gains)
    controller.update(GOAL, at=0.0)
    controller.update(0.0, at=0.1)
    assert controller.reached


def test_once_the_goal_is_reached_the_cup_is_turned_upright_whatever_is_read_next():
    gains = PidGains(proportional=0.01, integral=0.002, derivative=0.03, start_tilt=0.9)
    controller = _controller(gains)
    controller.update(GOAL, at=0.0)
    commands = [
        controller.update(reading, at=0.1 * i)
        for i, reading in enumerate([0.0, GOAL, 2 * GOAL], start=1)
    ]
    assert commands == [controller.minimum_tilt] * 3


# %% the gains


def test_scaling_the_gains_leaves_the_start_tilt_alone():
    gains = PidGains(proportional=0.02, integral=0.01, derivative=0.03, start_tilt=0.6)
    scaled = gains.scaled(2.0)
    assert (
        scaled.proportional,
        scaled.integral,
        scaled.derivative,
        scaled.start_tilt,
    ) == (
        2.0 * gains.proportional,
        2.0 * gains.integral,
        2.0 * gains.derivative,
        gains.start_tilt,
    )


def test_gains_survive_a_json_round_trip():
    gains = PidGains(proportional=0.02, integral=0.01, derivative=0.03, start_tilt=0.6)
    assert PidGains.from_json(gains.to_json()) == gains


# %% the tilt the controller steers


@pytest.mark.parametrize("tilt", [0.0, 0.4, -0.3, 1.9])
def test_the_tilt_about_the_pour_axis_is_the_turn_away_from_the_carry_orientation(
    tilt,
):
    carry = HomogeneousTransformationMatrix.from_xyz_quaternion(
        quat_x=0.5, quat_y=0.5, quat_z=0.5, quat_w=0.5
    )
    tool = carry @ HomogeneousTransformationMatrix.from_xyz_rpy(yaw=tilt)
    expression = signed_tilt_about_pour_axis(tool, carry.to_rotation_matrix())
    assert float(sm.Scalar(expression).evaluate()[0]) == pytest.approx(tilt)
