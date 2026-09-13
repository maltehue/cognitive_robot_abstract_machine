from __future__ import annotations

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import (
    GiskardEqualityConstraint,
    GiskardInequalityConstraint,
    LargeNumber,
)
from giskardpy.qp.enforcement_strategy import (
    PredictedValueStrategy,
    normalize_slack_weight,
)
from giskardpy.qp.exceptions import ConstraintTypeMismatchError
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

from .test_integral_strategy import _world_with_joint

# %% operating point constants

_PREDICTION_HORIZON = 12
"""
Prediction horizon of the controller configuration under test, giving a control horizon
of ten velocity blocks.
"""

_CONTROL_FREQUENCY = 80
"""
Control frequency, in hertz, of the controller configuration under test.
"""

_REFERENCE_VELOCITY = 1.0
"""
Reference velocity of the constraint, large enough that the reachable change never caps
a scheduled bound unless a test asks for it.
"""

_WEIGHT = 2500.0
"""
Quadratic weight of the constraint.
"""

_INSIDE_LOWER_ERROR = -0.03
"""
Lower error of a constraint whose expression sits inside its band.
"""

_INSIDE_UPPER_ERROR = 0.02
"""
Upper error of a constraint whose expression sits inside its band.
"""

_BELOW_LOWER_ERROR = 0.05
"""
Lower error of a constraint whose expression sits below its band by this gap.
"""

_BELOW_UPPER_ERROR = 0.10
"""
Upper error of a constraint whose expression sits below its band.
"""

# %% fixtures


def _config() -> QPControllerConfig:
    return QPControllerConfig(
        target_frequency=_CONTROL_FREQUENCY, prediction_horizon=_PREDICTION_HORIZON
    )


def _inequality_constraint(
    degree_of_freedom: DegreeOfFreedom,
    lower_error: float,
    upper_error: float,
    reference_velocity: float = _REFERENCE_VELOCITY,
) -> GiskardInequalityConstraint:
    """
    Builds an inequality constraint on the joint position with the given errors.
    """
    return GiskardInequalityConstraint(
        name="clearance",
        expression=degree_of_freedom.variables.position,
        quadratic_weight=_WEIGHT,
        normalization_factor=reference_velocity,
        enforcement_strategy=PredictedValueStrategy,
        lower_bound=sm.Scalar(lower_error),
        upper_bound=sm.Scalar(upper_error),
    )


def _strategy_over(*constraints) -> PredictedValueStrategy:
    """
    Builds the strategy under test over the given constraints on a single joint.
    """
    _world, degree_of_freedom = _world_with_joint()
    return PredictedValueStrategy(
        degrees_of_freedom=[degree_of_freedom],
        constraints=list(constraints),
        qp_controller_config=_config(),
    )


def _inside_band_strategy() -> PredictedValueStrategy:
    _world, degree_of_freedom = _world_with_joint()
    return PredictedValueStrategy(
        degrees_of_freedom=[degree_of_freedom],
        constraints=[
            _inequality_constraint(
                degree_of_freedom, _INSIDE_LOWER_ERROR, _INSIDE_UPPER_ERROR
            )
        ],
        qp_controller_config=_config(),
    )


def _below_band_strategy(
    reference_velocity: float = _REFERENCE_VELOCITY,
) -> PredictedValueStrategy:
    _world, degree_of_freedom = _world_with_joint()
    return PredictedValueStrategy(
        degrees_of_freedom=[degree_of_freedom],
        constraints=[
            _inequality_constraint(
                degree_of_freedom,
                _BELOW_LOWER_ERROR,
                _BELOW_UPPER_ERROR,
                reference_velocity,
            )
        ],
        qp_controller_config=_config(),
    )


def _evaluated(vector: sm.Vector) -> list[float]:
    return vector.evaluate().flatten().tolist()


# %% row layout


class TestRowsSitDenselyNearTheExecutedBlock:
    """
    The predicted value is bounded after the executed block, at doubling distances
    behind it, and at the end of the horizon.
    """

    def test_constrained_steps_double_and_end_at_the_last_block(self) -> None:
        strategy = _inside_band_strategy()

        assert strategy.constrained_steps() == [0, 1, 3, 7, 9]

    def test_each_row_accumulates_the_jacobian_up_to_its_step(self) -> None:
        """
        The row of step ``t`` carries the jacobian times the time step in every velocity
        block up to ``t`` and nothing in the later blocks or the jerk columns.
        """
        strategy = _inside_band_strategy()
        time_step = strategy.qp_controller_config.model_predictive_control_time_step
        horizon = strategy.qp_controller_config.control_horizon

        matrix = strategy.create_matrix().evaluate()

        for row, step in enumerate(strategy.constrained_steps()):
            velocity_part = matrix[row, :horizon].tolist()
            assert velocity_part == pytest.approx(
                [time_step] * (step + 1) + [0.0] * (horizon - step - 1)
            )
            assert not matrix[row, horizon:].any()

    def test_rows_are_named_after_their_step(self) -> None:
        strategy = _inside_band_strategy()

        assert strategy.create_names() == [
            f"t{step:03}/clearance" for step in strategy.constrained_steps()
        ]


# %% bounds


class TestBoundsKeepTheValueInsideOnceItIsInside:
    """
    While the expression is inside its band, every row only forbids leaving it.
    """

    def test_lower_bounds_equal_the_lower_error(self) -> None:
        strategy = _inside_band_strategy()

        assert _evaluated(strategy.create_lower_bounds()) == pytest.approx(
            [_INSIDE_LOWER_ERROR] * len(strategy.constrained_steps())
        )

    def test_upper_bounds_equal_the_upper_error(self) -> None:
        strategy = _inside_band_strategy()

        assert _evaluated(strategy.create_upper_bounds()) == pytest.approx(
            [_INSIDE_UPPER_ERROR] * len(strategy.constrained_steps())
        )


class TestBoundsCloseTheGapProportionallyOverTheHorizon:
    """
    While the expression is outside its band, row ``t`` asks for the share ``(t + 1) /
    horizon`` of the gap, so the last row asks for all of it and no row demands more
    than the reference velocity reaches by its step.
    """

    def test_lower_bounds_follow_the_schedule(self) -> None:
        strategy = _below_band_strategy()
        horizon = strategy.qp_controller_config.control_horizon

        assert _evaluated(strategy.create_lower_bounds()) == pytest.approx(
            [
                _BELOW_LOWER_ERROR * (step + 1) / horizon
                for step in strategy.constrained_steps()
            ]
        )

    def test_last_row_asks_for_the_whole_gap(self) -> None:
        strategy = _below_band_strategy()

        assert _evaluated(strategy.create_lower_bounds())[-1] == pytest.approx(
            _BELOW_LOWER_ERROR
        )

    def test_permissive_side_stays_uncapped(self) -> None:
        strategy = _below_band_strategy()

        assert _evaluated(strategy.create_upper_bounds()) == pytest.approx(
            [_BELOW_UPPER_ERROR] * len(strategy.constrained_steps())
        )

    def test_reference_velocity_caps_a_demanded_rise(self) -> None:
        """
        With a reference velocity too slow to close the scheduled share of the gap, a
        row asks only for what that velocity reaches by its step.
        """
        slow_reference_velocity = 0.1
        strategy = _below_band_strategy(slow_reference_velocity)
        time_step = strategy.qp_controller_config.model_predictive_control_time_step

        assert _evaluated(strategy.create_lower_bounds())[0] == pytest.approx(
            slow_reference_velocity * time_step
        )


# %% slack


class TestSlackIsSharedAcrossTheRows:
    """
    Each row gets its own slack variable carrying the constraint's limits, and the
    constraint's horizon-normalized weight is split evenly over its rows.
    """

    def test_one_slack_variable_per_row_with_the_constraint_limits(self) -> None:
        strategy = _inside_band_strategy()
        rows = len(strategy.constrained_steps())

        slack = strategy.create_slack_variables()

        assert _evaluated(slack.lower_bounds) == [-LargeNumber] * rows
        assert _evaluated(slack.upper_bounds) == [LargeNumber] * rows
        assert slack.names == strategy.create_names()

    def test_weights_are_the_horizon_normalized_weight_over_the_rows(self) -> None:
        strategy = _inside_band_strategy()
        rows = len(strategy.constrained_steps())
        expected = normalize_slack_weight(
            _WEIGHT,
            _REFERENCE_VELOCITY,
            strategy.qp_controller_config.control_horizon,
        ).evaluate()[0]

        slack = strategy.create_slack_variables()

        assert _evaluated(slack.quadratic_weights) == pytest.approx(
            [expected / rows] * rows
        )

    def test_slack_matrix_couples_each_row_to_its_own_slack(self) -> None:
        strategy = _inside_band_strategy()
        rows = len(strategy.constrained_steps())
        time_step = strategy.qp_controller_config.model_predictive_control_time_step

        matrix = strategy.create_slack_matrix().evaluate()

        assert np.allclose(matrix, np.eye(rows) * time_step)


# %% constraint types


class TestEqualityConstraintsAreRejected:
    """
    The strategy bounds values between two limits and refuses an equality constraint.
    """

    def test_equality_bounds_raise(self) -> None:
        _world, degree_of_freedom = _world_with_joint()
        strategy = PredictedValueStrategy(
            degrees_of_freedom=[degree_of_freedom],
            constraints=[
                GiskardEqualityConstraint(
                    name="target",
                    expression=degree_of_freedom.variables.position,
                    bound=sm.Scalar(0.1),
                    normalization_factor=_REFERENCE_VELOCITY,
                    quadratic_weight=_WEIGHT,
                    enforcement_strategy=PredictedValueStrategy,
                )
            ],
            qp_controller_config=_config(),
        )

        with pytest.raises(ConstraintTypeMismatchError):
            strategy.create_equality_bounds()
