"""Symbolic-goal retargeting of the terminal-state prediction row.

Widening the terminal goal from a plain float to a :class:`~krrood.symbolic_math.symbolic_math.\
ScalarData` lets a registered ``FloatVariable`` retarget the row at runtime through the QP's
float-variable channel, with no recompile. These tests pin that the retargeted bound is exactly the
bound the strategy would compute for the same goal as a float.
"""

from __future__ import annotations

import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.float_variable_data import FloatVariableData

from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.terminal_state_prediction_strategy import (
    TerminalStatePredictionConstraint,
    TerminalStatePredictionStrategy,
)

STATE_SENSITIVITY = 0.02
"""Constant ``df/dx`` of the toy linear state rate ``f = df/dx * x``."""

STATE_VALUE = 0.3
"""Operating-point value of the passive state DOF."""


def _strategy_with_goal(
    goal_value: sm.ScalarData, name: str
) -> tuple[TerminalStatePredictionStrategy, sm.FloatVariable]:
    """Builds a single-constraint strategy whose state rate is linear in its own state."""
    state_variable = sm.FloatVariable(name=f"{name}_state")
    constraint = TerminalStatePredictionConstraint(
        name=name,
        expression=sm.Scalar(STATE_SENSITIVITY) * state_variable,
        normalization_factor=sm.Scalar(0.05),
        quadratic_weight=sm.Scalar(100.0),
        enforcement_strategy=TerminalStatePredictionStrategy,
        linear_weight=0,
        state_variable=state_variable,
        goal_value=goal_value,
    )
    strategy = TerminalStatePredictionStrategy(
        degrees_of_freedom=[],
        constraints=[constraint],
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )
    return strategy, state_variable


def _float_goal_bound(goal_value: float, name: str) -> float:
    """The equality bound the strategy computes for a plain-float goal."""
    strategy, state_variable = _strategy_with_goal(goal_value, name)
    bound = strategy.create_equality_bounds()[0]
    data = FloatVariableData()
    data.register_expression(state_variable)
    data.set_value(state_variable, STATE_VALUE)
    return float(bound.evaluate().item())


class TestSymbolicGoalRetargeting:
    """Whether a registered ``FloatVariable`` goal retargets the terminal bound without recompiling."""

    def test_goal_variable_is_a_free_variable_of_the_bound(self):
        goal = sm.FloatVariable(name="goal")
        strategy, _ = _strategy_with_goal(goal, "in_bound")
        bound = strategy.create_equality_bounds()[0]
        assert any(variable is goal for variable in bound.free_variables())

    def test_retargeted_bound_matches_float_goal_bound(self):
        goal = sm.FloatVariable(name="goal")
        strategy, state_variable = _strategy_with_goal(goal, "match")
        bound = strategy.create_equality_bounds()[0]
        data = FloatVariableData()
        data.register_expression(goal)
        data.register_expression(state_variable)
        data.set_value(state_variable, STATE_VALUE)
        for goal_value in (0.305, 0.31, 0.5, 0.9):
            data.set_value(goal, goal_value)
            retargeted = float(bound.evaluate().item())
            assert retargeted == _float_goal_bound(goal_value, f"float_{goal_value}")

    def test_rewriting_goal_variable_changes_the_unsaturated_bound(self):
        goal = sm.FloatVariable(name="goal")
        strategy, state_variable = _strategy_with_goal(goal, "track")
        bound = strategy.create_equality_bounds()[0]
        data = FloatVariableData()
        data.register_expression(goal)
        data.register_expression(state_variable)
        data.set_value(state_variable, STATE_VALUE)
        data.set_value(goal, 0.305)
        near = float(bound.evaluate().item())
        data.set_value(goal, 0.310)
        farther = float(bound.evaluate().item())
        assert farther - near == pytest.approx(0.005)
