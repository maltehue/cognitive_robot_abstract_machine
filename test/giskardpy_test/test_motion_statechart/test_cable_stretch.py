from __future__ import annotations

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cable_stretch import (
    MinimizeCableStretch,
    NonPositiveRestLengthError,
)
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.wrist_cable_guide import WristCableGuide
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% fixtures


WRIST_OFFSET_Z = 0.3
SLACK_LENGTH = 0.05
OVERSHOOTING_WRIST_GOAL = 2.5


def _build_wrist_cable_world() -> tuple[World, WristCableGuide]:
    """
    Build a wrist arm with a soft cable routed around the wrist.

    The cable anchor and the wrist guide start close together, so rotating the wrist
    swings the guide away and forces the cable to stretch.

    :return: The world and the cable guide routed around its wrist.
    """
    world = World()
    with world.modify_world():
        forearm = Body(name=PrefixedName("forearm"))
        world.add_body(forearm)
        hand = Body(name=PrefixedName("hand"))
        world.add_body(hand)
        wrist = RevoluteConnection.create_with_dofs(
            world=world,
            parent=forearm,
            child=hand,
            name=PrefixedName("wrist"),
            axis=Vector3.Z(),
            dof_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-3.0, velocity=-1.0),
                upper=DerivativeMap(position=3.0, velocity=1.0),
            ),
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=WRIST_OFFSET_Z
            ),
        )
        world.add_connection(wrist)
    cable = WristCableGuide.route_around_wrist(
        world=world,
        proximal_body=world.get_body_by_name("forearm"),
        distal_body=world.get_body_by_name("hand"),
        proximal_body_T_anchor=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.1, z=0.28
        ),
        distal_body_T_guide=HomogeneousTransformationMatrix.from_xyz_rpy(x=0.1, z=0.02),
        slack_length=SLACK_LENGTH,
    )
    return world, cable


def _rotate_wrist(
    world: World, cable: WristCableGuide, with_stretch_task: bool
) -> tuple[float, float]:
    """
    Command an overshooting wrist rotation, optionally under the stretch task.

    :param world: World holding the wrist and cable.
    :param cable: Cable routed around the wrist.
    :param with_stretch_task: Whether the cable stretch task is added.
    :return: The final wrist angle and the final cable stretch.
    """
    motion_statechart = MotionStatechart()
    motion = JointPositionList(
        goal_state=JointState.from_str_dict(
            {"wrist": OVERSHOOTING_WRIST_GOAL}, world=world
        ),
        weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
    )
    motion_statechart.add_node(motion)
    if with_stretch_task:
        motion_statechart.add_node(
            MinimizeCableStretch(
                cable_anchor=cable.cable_anchor,
                wrist_guide=cable.wrist_guide,
                rest_length=cable.rest_length,
            )
        )
    # End on velocity convergence: the stretch task caps the wrist short of its goal, so
    # the motion settles at the cap rather than reaching the commanded angle.
    motion_statechart.add_node(EndMotion())

    executor = Executor(MotionStatechartContext(world=world))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick_until_end()

    wrist_angle = world.state[world.get_connection_by_name("wrist").dof.id].position
    return wrist_angle, cable.current_stretch()


# %% stretch constraint


class TestMinimizeCableStretch:
    def test_unconstrained_wrist_overstretches_cable(self):
        world, cable = _build_wrist_cable_world()
        wrist_angle, stretch = _rotate_wrist(world, cable, with_stretch_task=False)
        assert wrist_angle == pytest.approx(OVERSHOOTING_WRIST_GOAL, abs=1e-2)
        assert stretch > 0.05

    def test_stretch_task_caps_wrist_before_overstretch(self):
        world, cable = _build_wrist_cable_world()
        wrist_angle, stretch = _rotate_wrist(world, cable, with_stretch_task=True)
        assert stretch == pytest.approx(0.0, abs=1e-3)
        assert wrist_angle < OVERSHOOTING_WRIST_GOAL - 1.0

    def test_stretch_task_holds_span_at_rest_length(self):
        world, cable = _build_wrist_cable_world()
        _rotate_wrist(world, cable, with_stretch_task=True)
        assert cable.current_span() == pytest.approx(cable.rest_length, abs=1e-3)

    def test_non_positive_rest_length_is_rejected(self):
        world, cable = _build_wrist_cable_world()
        task = MinimizeCableStretch(
            cable_anchor=cable.cable_anchor,
            wrist_guide=cable.wrist_guide,
            rest_length=0.0,
        )
        motion_statechart = MotionStatechart()
        motion_statechart.add_node(task)
        executor = Executor(MotionStatechartContext(world=world))
        with pytest.raises(NonPositiveRestLengthError):
            executor.compile(motion_statechart=motion_statechart)
