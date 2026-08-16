"""Tasks that command a velocity rather than bound one.

These are the first users of the QP's velocity *equality* path, so the assertions below are as much
about that path working as about the tasks: a commanded rate has to show up as an actual rate.
"""

from __future__ import annotations

import pytest

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.commanded_velocity import (
    CommandedTranslationVelocity,
)

COMMANDED_SPEED = 0.05
"""Commanded vertical speed, in metres per second."""

TICKS = 40
"""Control cycles the command is held for."""


@pytest.fixture
def commanded_translation(prismatic_bot):
    """A single-prismatic-joint robot whose tip velocity is commanded along world z."""
    world = prismatic_bot
    tip = world.get_body_by_name("robot")
    task = CommandedTranslationVelocity(
        name="commanded", root_link=world.root, tip_link=tip
    )
    statechart = MotionStatechart()
    statechart.add_node(task)
    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    return executor, task, tip


def _tip_height(world, tip) -> float:
    return float(world.compute_forward_kinematics_np(world.root, tip)[2, 3])


class TestCommandedTranslationVelocity:
    """Whether a commanded rate is realized as motion at that rate."""

    def test_a_commanded_velocity_moves_the_tip_at_that_speed(
        self, commanded_translation
    ):
        executor, task, tip = commanded_translation
        world = executor.context.world
        executor.context.float_variable_data.set_value(
            task.commanded_velocity[2], COMMANDED_SPEED
        )
        start_height = _tip_height(world, tip)

        for _ in range(TICKS):
            executor.tick()

        control_dt = executor.qp_controller.config.control_dt
        travelled = _tip_height(world, tip) - start_height
        assert travelled == pytest.approx(
            COMMANDED_SPEED * TICKS * control_dt, rel=0.25
        )

    def test_reversing_the_command_reverses_the_motion(self, commanded_translation):
        executor, task, tip = commanded_translation
        world = executor.context.world
        float_variable_data = executor.context.float_variable_data

        float_variable_data.set_value(task.commanded_velocity[2], COMMANDED_SPEED)
        for _ in range(TICKS):
            executor.tick()
        height_after_rising = _tip_height(world, tip)

        float_variable_data.set_value(task.commanded_velocity[2], -COMMANDED_SPEED)
        for _ in range(TICKS):
            executor.tick()

        assert _tip_height(world, tip) < height_after_rising

    def test_a_zero_command_holds_position(self, commanded_translation):
        executor, task, tip = commanded_translation
        world = executor.context.world
        executor.context.float_variable_data.set_value(task.commanded_velocity[2], 0.0)
        start_height = _tip_height(world, tip)

        for _ in range(TICKS):
            executor.tick()

        assert _tip_height(world, tip) == pytest.approx(start_height, abs=1e-3)
