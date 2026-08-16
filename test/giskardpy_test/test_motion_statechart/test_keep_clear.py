"""
The planar keep-clear constraint.

Pinned on a one-axis world: a goal drives the subject straight at the obstacle, and the
constraint must stop it at the clearance without any damping of the motion that stays
outside it.
"""

from __future__ import annotations

import pytest

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.tasks.keep_clear import MaintainHorizontalClearance
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    PrismaticConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

MINIMUM_CLEARANCE = 0.2
"""
Clearance the constraint must hold, in metres.
"""

OBSTACLE_POSITION = 0.6
"""
Where the obstacle stands on the motion axis, in metres.
"""


def _one_axis_world() -> tuple[World, Body, Body, PrismaticConnection]:
    """
    A world with one body sliding along x toward a fixed obstacle.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        world.add_body(map_body)
        subject = Body(name=PrefixedName("subject"))
        degree_of_freedom = DegreeOfFreedom(
            limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(
                    position=-1.0, velocity=-1.0, acceleration=None, jerk=None
                ),
                upper=DerivativeMap(
                    position=1.0, velocity=1.0, acceleration=None, jerk=None
                ),
            ),
            has_hardware_interface=True,
        )
        world.add_degree_of_freedom(degree_of_freedom)
        connection = PrismaticConnection(
            parent=map_body, child=subject, raw_dof=degree_of_freedom, axis=Vector3.X()
        )
        world.add_connection(connection)
        obstacle = Body(name=PrefixedName("obstacle"))
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=map_body,
                child=obstacle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=OBSTACLE_POSITION
                ),
            )
        )
    return world, subject, obstacle, connection


def _run(with_clearance: bool) -> float:
    """
    Drives the subject at the obstacle and returns where it ends up.
    """
    world, subject, obstacle, connection = _one_axis_world()
    statechart = MotionStatechart()
    statechart.add_node(
        JointPositionList(
            goal_state=JointState.from_mapping({connection: OBSTACLE_POSITION})
        )
    )
    if with_clearance:
        statechart.add_node(
            MaintainHorizontalClearance(
                root_link=world.root,
                subject_link=subject,
                obstacle_link=obstacle,
                minimum_clearance=MINIMUM_CLEARANCE,
            )
        )
    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    for _ in range(300):
        executor.tick()
    return float(connection.position)


class TestMaintainHorizontalClearance:
    """
    Whether the clearance stops a motion aimed at the obstacle, and only that.
    """

    def test_without_the_constraint_the_goal_is_reached(self):
        assert _run(with_clearance=False) == pytest.approx(OBSTACLE_POSITION, abs=0.02)

    def test_the_constraint_stops_the_approach_at_the_clearance(self):
        final_position = _run(with_clearance=True)
        assert final_position == pytest.approx(
            OBSTACLE_POSITION - MINIMUM_CLEARANCE, abs=0.02
        )
