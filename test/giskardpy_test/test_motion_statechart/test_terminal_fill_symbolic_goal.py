"""Runtime behaviour of a terminal-fill task carrying a symbolic goal.

A :class:`~giskardpy.motion_statechart.tasks.pouring.TerminalFillConstraintTask` may carry a
registered ``FloatVariable`` goal so the reasoner can retarget it mid-motion. Two things then have to
hold: the tick-time convergence check must read the goal's live value instead of comparing a Python
float against a symbol (which would raise), and such a task must refuse JSON serialization rather
than silently shipping a goal of zero.
"""

from __future__ import annotations

import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.adapters.exceptions import SymbolicValueNotSerializableError
from krrood.adapters.json_serializer import to_json
from krrood.symbolic_math.float_variable_data import FloatVariableData

from giskardpy.motion_statechart.tasks.pouring import FillByTransferTask


class _FloatVariableContext:
    """Minimal stand-in exposing only the float-variable data the goal resolution reads."""

    def __init__(self, float_variable_data: FloatVariableData) -> None:
        self.float_variable_data = float_variable_data


def _task_with_symbolic_goal(
    goal: sm.FloatVariable,
) -> tuple[FillByTransferTask, FloatVariableData]:
    """A fill-by-transfer task whose goal is a registered float variable."""
    data = FloatVariableData()
    data.register_expression(goal)
    task = FillByTransferTask(receiver=None, goal_value=goal, fill_level_tolerance=0.05)
    return task, data


class TestFillGoalReachedReadsLiveValue:
    """Whether the convergence check resolves a symbolic goal to its live value at tick time."""

    def test_current_goal_value_resolves_the_live_symbolic_value(self):
        goal = sm.FloatVariable(name="goal")
        task, data = _task_with_symbolic_goal(goal)
        data.set_value(goal, 0.8)
        assert task._current_goal_value(_FloatVariableContext(data)) == 0.8

    def test_current_goal_value_returns_a_plain_float_goal_unchanged(self):
        task = FillByTransferTask(
            receiver=None, goal_value=0.6, fill_level_tolerance=0.05
        )
        assert (
            task._current_goal_value(_FloatVariableContext(FloatVariableData())) == 0.6
        )

    def test_fill_goal_reached_uses_the_resolved_goal_without_raising(self):
        goal = sm.FloatVariable(name="goal")
        task, data = _task_with_symbolic_goal(goal)
        data.set_value(goal, 0.8)
        resolved = task._current_goal_value(_FloatVariableContext(data))
        assert task._fill_goal_reached(0.74, resolved) is False
        assert task._fill_goal_reached(0.78, resolved) is True


class TestSymbolicGoalTaskRefusesSerialization:
    """Whether a task carrying a symbolic goal fails serialization instead of losing the goal."""

    def test_serializing_a_symbolic_goal_task_raises(self):
        goal = sm.FloatVariable(name="goal")
        task, _ = _task_with_symbolic_goal(goal)
        with pytest.raises(SymbolicValueNotSerializableError):
            to_json(task)
