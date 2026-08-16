"""
Bounding the head above the pouring lip.

The head is the effect model's own measure of how hard a container pours, so a bound on
it is a dynamic constraint stated without reference to the robot. These tests pin that
it restrains a motion that would otherwise exceed it, that the restraint shows up as
lost tilt angle without the constraint ever naming a tilt, and that because the head
depends on fill as well as tilt the same bound implies different tilt limits at
different fill levels.

The motion driven here is a joint goal rather than a pouring task: a terminal-state
prediction row wins against this constraint over the horizon (see
:class:`BoundedPourHead`), so pairing the two would test that interaction rather than
the bound.
"""

from __future__ import annotations

import math

import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import RootLinkNotWorldRootError
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.pouring import BoundedPourHead
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.physics.equations.pouring_equations import (
    SymbolicFillContext,
)

from .single_cup_world import build_single_cup_world

MAXIMUM_HEAD = 0.05
"""
Head bound used in these tests, in metres.

Chosen above the head the world already has at its initial tilt and full fill: a bound
below the starting value asks the optimizer to undo a state it did not create, which is
a different scenario from holding a bound during a pour.
"""


def _head_at(cup, tilt: float, fill: float) -> float:
    """
    Evaluates the analytic head above the lip at a given tilt and fill.
    """
    return float(
        cup.fill_equation.head_above_lip(
            SymbolicFillContext(
                tilt_expression=sm.Scalar(tilt), fill_position=sm.Scalar(fill)
            )
        ).evaluate()[0]
    )


class TestHeadBoundIsFillDependent:
    """
    That a single head bound implies different tilt limits at different fill levels.

    This is the property that makes the bound a statement about the task rather than about the
    robot: nobody writes the fill-dependent rule, it follows from the effect model.
    """

    def test_a_fuller_container_reaches_the_bound_at_a_smaller_tilt(self):
        tilt = math.radians(70)
        assert _head_at(build_single_cup_world()[1], tilt, 0.9) > _head_at(
            build_single_cup_world()[1], tilt, 0.4
        )

    def test_the_head_is_zero_while_the_container_is_upright(self):
        assert _head_at(build_single_cup_world()[1], 0.0, 1.0) == pytest.approx(0.0)


def _run_tilting_to(
    goal_tilt: float, maximum_head: float | None
) -> tuple[float, float]:
    """
    Tilts the cup toward a goal angle, optionally under a head bound.

    :param goal_tilt: Tilt angle the joint task drives toward, in radians.
    :param maximum_head: Head bound to hold, or ``None`` for an unconstrained run.
    :return: The final tilt angle and the head at that tilt.
    """
    world, cup = build_single_cup_world()
    tilt_connection = cup.root.parent_connection
    statechart = MotionStatechart()
    statechart.add_node(
        JointPositionList(
            goal_state=JointState.from_mapping({tilt_connection: goal_tilt})
        )
    )
    if maximum_head is not None:
        statechart.add_node(
            BoundedPourHead(source=cup, root_link=world.root, maximum_head=maximum_head)
        )
    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    for _ in range(300):
        executor.tick()
    final_tilt = float(tilt_connection.position)
    return final_tilt, _head_at(cup, final_tilt, float(cup.fill_level))


class TestBoundedPourHeadHoldsTheBound:
    """
    Whether the bound restrains a motion that would otherwise exceed it.
    """

    def test_an_unconstrained_tilt_exceeds_the_bound(self):
        _tilt, head = _run_tilting_to(1.5, maximum_head=None)
        assert head > MAXIMUM_HEAD

    def test_the_bound_stops_the_tilt_at_the_allowed_head(self):
        _tilt, head = _run_tilting_to(1.5, maximum_head=MAXIMUM_HEAD)
        assert head == pytest.approx(MAXIMUM_HEAD, abs=0.005)

    def test_the_bound_costs_tilt_angle(self):
        """
        The bound is realized as a tilt limit without naming a tilt: the optimizer gives
        up angle to keep the head within bound.
        """
        unbounded_tilt, _ = _run_tilting_to(1.5, maximum_head=None)
        bounded_tilt, _ = _run_tilting_to(1.5, maximum_head=MAXIMUM_HEAD)
        assert bounded_tilt < unbounded_tilt


class TestBoundedPourHeadValidation:
    """
    What the task refuses to build against.
    """

    def test_a_non_world_root_is_rejected(self):
        world, cup = build_single_cup_world()
        statechart = MotionStatechart()
        statechart.add_node(
            BoundedPourHead(source=cup, root_link=cup.root, maximum_head=MAXIMUM_HEAD)
        )
        executor = Executor(MotionStatechartContext(world=world))
        with pytest.raises(RootLinkNotWorldRootError):
            executor.compile(motion_statechart=statechart)
