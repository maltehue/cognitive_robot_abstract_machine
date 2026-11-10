"""
A controller that may move only the degrees of freedom a hardware interface commands, as
on a real robot, whose other joints follow their own state topics.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection

COMMANDED_GOAL = 0.3
"""
Where the commanded joint is moved to.
"""

UNCOMMANDED_GOAL = 0.5
"""
Where a task asks the uncommanded joint to go, which it never does.
"""

TICK_LIMIT = 1000
"""
Ticks after which a motion that has not ended counts as stuck.
"""


@dataclass
class TwoJointMotion:
    """
    A motion that ends once the commanded joint reaches its goal, while another task keeps
    asking a joint no hardware interface commands to move.
    """

    world: World
    """
    The world holding both joints.
    """

    commanded: PrismaticConnection
    """
    The joint a hardware interface commands.
    """

    uncommanded: PrismaticConnection
    """
    The joint nothing commands; its position is what its sensor reports, never changing.
    """

    motion_statechart: MotionStatechart
    """
    The motion, ended by reaching :data:`COMMANDED_GOAL`.
    """

    def run(self, only_commanded: bool) -> Executor:
        """
        Run the motion, putting the uncommanded joint back where its sensor reports it
        after every tick.

        :param only_commanded: Whether the controller may move only commanded joints.
        :return: The executor after the motion ended or :data:`TICK_LIMIT` ticks passed.
        """
        executor = Executor(
            MotionStatechartContext(
                world=self.world,
                moves_only_commanded_degrees_of_freedom=only_commanded,
            )
        )
        executor.compile(motion_statechart=self.motion_statechart)
        sensed_position = JointState.from_mapping({self.uncommanded: 0.0})
        for _ in range(TICK_LIMIT):
            executor.tick()
            sensed_position.apply_to(self.world)
            if self.motion_statechart.is_end_motion():
                break
        return executor


@pytest.fixture()
def two_joint_motion(prismatic_bot2: World) -> TwoJointMotion:
    commanded, uncommanded = prismatic_bot2.get_connections_by_type(PrismaticConnection)
    uncommanded.has_hardware_interface = False
    reach = JointPositionList(
        goal_state=JointState.from_mapping({commanded: COMMANDED_GOAL})
    )
    hold_out = JointPositionList(
        goal_state=JointState.from_mapping({uncommanded: UNCOMMANDED_GOAL})
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(reach)
    motion_statechart.add_node(hold_out)
    motion_statechart.add_node(EndMotion.when_true(reach))
    return TwoJointMotion(
        world=prismatic_bot2,
        commanded=commanded,
        uncommanded=uncommanded,
        motion_statechart=motion_statechart,
    )


# %% which joints the controller moves


def test_the_controller_moves_only_commanded_joints_when_told_to(two_joint_motion):
    executor = two_joint_motion.run(only_commanded=True)

    assert executor.qp_controller.active_dofs == [two_joint_motion.commanded.dof]


def test_the_controller_moves_every_active_joint_by_default(two_joint_motion):
    executor = two_joint_motion.run(only_commanded=False)

    assert executor.qp_controller.active_dofs == [
        two_joint_motion.commanded.dof,
        two_joint_motion.uncommanded.dof,
    ]


# %% ending the motion


def test_an_uncommanded_joint_does_not_keep_the_motion_from_ending(two_joint_motion):
    """
    The uncommanded joint never reaches the goal another task gives it, so a controller
    that commands it never sees every joint at rest; one that leaves it alone ends once
    the commanded joint settles.
    """
    two_joint_motion.run(only_commanded=True)

    assert two_joint_motion.motion_statechart.is_end_motion()
    assert two_joint_motion.commanded.position == pytest.approx(
        COMMANDED_GOAL, abs=0.01
    )


def test_a_commanded_uncommanded_joint_keeps_the_motion_from_ending(two_joint_motion):
    """
    The failure the switch exists for: a velocity the robot never executes keeps the end
    of the motion waiting for every joint to come to rest.
    """
    two_joint_motion.run(only_commanded=False)

    assert not two_joint_motion.motion_statechart.is_end_motion()
