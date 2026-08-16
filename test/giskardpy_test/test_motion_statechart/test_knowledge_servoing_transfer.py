"""A liquid transfer driven end to end by the substance-transfer theory.

The statechart holds no pouring logic of its own: which constraints are active and what the fill
goal is are conclusions the theory reaches from the twin each control cycle, and the motion ends
because the theory concludes the transfer is finished, not because a task was told a target up
front.
"""

from __future__ import annotations

import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.graph_node import CancelMotion, EndMotion
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.knowledge_servoing.concluded_monitor import (
    ConcludedMonitor,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_binding_policy import (
    DecisionBindingPolicy,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot
from giskardpy.motion_statechart.knowledge_servoing.decision_transcript import (
    DecisionTranscript,
)
from giskardpy.motion_statechart.knowledge_servoing.symbolic_theory_node import (
    SymbolicTheoryNode,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.reasoning.substance_transfer import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
    RetargetFillLevel,
    TransferSituationGrounding,
    build_substance_transfer_theory,
)

from .test_pouring import (  # noqa: F401 - fixtures are used by name
    tracy_pouring_world,
    tracy_transfer_world,
)

REQUESTED_FILL_LEVEL = 0.4
"""Fill level the theory is asked to reach in the receiving cup."""

FILL_LEVEL_TOLERANCE = 0.05
"""Band around the requested level within which the transfer counts as done."""


@pytest.fixture
def transfer_statechart(tracy_transfer_world):  # noqa: F811 - pytest fixture injection
    """Assembles the reasoner-driven transfer statechart over the coupled cups.

    :returns: ``(executor, statechart_parts, receiving_cup)``
    """
    world, source_cup, receiving_cup, left_tool_frame = tracy_transfer_world
    # The shared fixture couples a torrent: the source empties in under two seconds, far faster than
    # the arm can tilt back, so any controller would overshoot. Re-couple at a pouring rate the
    # actuation can actually track.
    source_equation = source_cup.fill_equation
    receiving_cup.recouple_outflow_from(
        source=source_cup,
        world=world,
        fill_equation=ArticulatedPouringEquation(
            container_height=source_equation.container_height,
            container_width=source_equation.container_width,
            outflow_rate_constant=0.12,
            discharge_coefficient=source_equation.discharge_coefficient,
        ),
    )

    grounding = TransferSituationGrounding(
        source=source_cup,
        receiver=receiving_cup,
        requested_fill_level=REQUESTED_FILL_LEVEL,
        fill_level_tolerance=FILL_LEVEL_TOLERANCE,
    )
    theory = build_substance_transfer_theory()
    decision_slot = DecisionSlot()
    goal_fill_variable = sm.FloatVariable(name="transfer_goal_fill_level")

    aim = KeepProjectileInReceiver(
        receiver=receiving_cup,
        source=source_cup,
        weight=DefaultWeights.WEIGHT_MAXIMUM,
    )
    clearance = KeepSourceRimAboveReceiverRim(
        receiver=receiving_cup, source=source_cup, minimum_clearance=0.08
    )
    transfer = FillByTransferTask(
        receiver=receiving_cup,
        goal_value=goal_fill_variable,
        fill_level_tolerance=FILL_LEVEL_TOLERANCE,
    )
    # Concluding the transfer has to actively close the pour: the outflow gate is geometric, so a
    # statechart that merely stopped driving the fill would leave the source tilted and aimed, and
    # the receiver would keep filling while the arm settles.
    return_upright = AlignPlanes(
        root_link=world.root,
        tip_link=source_cup.root,
        goal_normal=Vector3.Z(reference_frame=world.root),
        tip_normal=Vector3.Z(reference_frame=source_cup.root),
    )

    align_monitor = ConcludedMonitor(
        decision_type=AlignSourceOverReceiver, decision_slot=decision_slot
    )
    pour_monitor = ConcludedMonitor(
        decision_type=PourIntoReceiver, decision_slot=decision_slot
    )
    concluded_monitor = ConcludedMonitor(
        decision_type=ConcludeTransfer, decision_slot=decision_slot
    )
    abandoned_monitor = ConcludedMonitor(
        decision_type=AbandonTransfer, decision_slot=decision_slot
    )
    abandon_motion = CancelMotion.when_true(abandoned_monitor)

    binding_policy = DecisionBindingPolicy()
    binding_policy.activate(AlignSourceOverReceiver, aim)
    binding_policy.activate(PourIntoReceiver, transfer)
    binding_policy.activate(ConcludeTransfer, return_upright)
    binding_policy.activate(AbandonTransfer, abandon_motion)
    binding_policy.parameterize(
        RetargetFillLevel,
        lambda decision: decision.goal_fill_level,
        goal_fill_variable,
    )

    theory_node = SymbolicTheoryNode(
        grounding=grounding,
        theory=theory,
        binding_policy=binding_policy,
        decision_slot=decision_slot,
    )

    aim.start_condition = align_monitor.observation_variable
    clearance.start_condition = align_monitor.observation_variable
    transfer.start_condition = pour_monitor.observation_variable
    return_upright.start_condition = concluded_monitor.observation_variable

    statechart = MotionStatechart()
    for node in (
        theory_node,
        align_monitor,
        pour_monitor,
        concluded_monitor,
        abandoned_monitor,
        aim,
        clearance,
        transfer,
        return_upright,
        abandon_motion,
    ):
        statechart.add_node(node)
    statechart.add_node(EndMotion.when_true(concluded_monitor))

    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    return executor, transfer, pour_monitor, goal_fill_variable, receiving_cup


class TestTransferGrounding:
    """Whether the qualitative facts read the live world rather than a stale or default state."""

    def test_facts_describe_the_initial_pose(self, tracy_transfer_world):  # noqa: F811
        world, source_cup, receiving_cup, _tool = tracy_transfer_world
        grounding = TransferSituationGrounding(
            source=source_cup,
            receiver=receiving_cup,
            requested_fill_level=REQUESTED_FILL_LEVEL,
        )

        [situation] = grounding.ground(world)

        assert situation.near
        assert situation.source_above_receiver
        assert situation.receiver_fill_level == 0.0
        assert not situation.goal_reached
        assert not situation.pours_to

    def test_facts_follow_the_world_when_the_receiver_fills(
        self, tracy_transfer_world
    ):  # noqa: F811
        """
        Grounding compiles its expressions once, so this pins that the compiled functions read the
        live state array rather than the values present when they were compiled.
        """
        world, source_cup, receiving_cup, _tool = tracy_transfer_world
        grounding = TransferSituationGrounding(
            source=source_cup,
            receiver=receiving_cup,
            requested_fill_level=REQUESTED_FILL_LEVEL,
        )
        grounding.ground(world)

        receiving_cup.fill_connection.position = REQUESTED_FILL_LEVEL

        [situation] = grounding.ground(world)
        assert situation.receiver_fill_level == pytest.approx(REQUESTED_FILL_LEVEL)
        assert situation.goal_reached


class TestReasonerDrivenTransfer:
    """Whether a transfer runs to completion driven only by the theory's conclusions."""

    def test_the_receiver_reaches_the_requested_fill_level(self, transfer_statechart):
        executor, _transfer, _pour_monitor, _goal, receiving_cup = transfer_statechart
        executor.tick_until_end(timeout=2000)
        assert receiving_cup.fill_level == pytest.approx(
            REQUESTED_FILL_LEVEL, abs=FILL_LEVEL_TOLERANCE
        )

    def test_the_theory_supplies_the_fill_goal_through_the_parameter_channel(
        self, transfer_statechart
    ):
        """
        The goal arrives with the pour regime rather than at build time: the theory concludes it
        only once pouring is warranted, so the variable stays at its registered default until then.
        """
        executor, _transfer, pour_monitor, goal_fill_variable, _cup = (
            transfer_statechart
        )
        float_variable_data = executor.context.float_variable_data
        assert float_variable_data.get_value(goal_fill_variable) == 0.0

        while pour_monitor.observation_state != ObservationStateValues.TRUE:
            executor.tick()

        assert float_variable_data.get_value(goal_fill_variable) == pytest.approx(
            REQUESTED_FILL_LEVEL
        )

    def test_the_transfer_task_stays_inactive_until_the_theory_concludes_pouring(
        self, transfer_statechart
    ):
        executor, transfer, pour_monitor, _goal, _cup = transfer_statechart
        assert pour_monitor.observation_state == ObservationStateValues.UNKNOWN
        assert transfer.life_cycle_state == LifeCycleValues.NOT_STARTED


class TestTransferDecisionTranscript:
    """Whether a completed run can be explained afterwards in the theory's own vocabulary."""

    def test_the_transcript_records_the_regime_sequence_of_a_full_transfer(
        self, transfer_statechart
    ):
        executor, _transfer, _pour_monitor, _goal, receiving_cup = transfer_statechart
        transcript = DecisionTranscript()
        decision_slot = next(
            node.decision_slot
            for node in executor.motion_statechart.nodes
            if isinstance(node, SymbolicTheoryNode)
        )

        for control_cycle in range(2000):
            executor.tick()
            transcript.record(decision_slot.latest, control_cycle)
            if executor.motion_statechart.is_end_motion():
                break

        assert transcript.cycle_of_first(AlignSourceOverReceiver) is not None
        assert transcript.cycle_of_first(PourIntoReceiver) > transcript.cycle_of_first(
            AlignSourceOverReceiver
        )
        assert transcript.cycle_of_first(ConcludeTransfer) > transcript.cycle_of_first(
            PourIntoReceiver
        )
        assert transcript.cycle_of_first(AbandonTransfer) is None

    def test_the_transcript_records_the_goal_the_theory_supplied(
        self, transfer_statechart
    ):
        executor, _transfer, pour_monitor, _goal, _cup = transfer_statechart
        transcript = DecisionTranscript()
        decision_slot = next(
            node.decision_slot
            for node in executor.motion_statechart.nodes
            if isinstance(node, SymbolicTheoryNode)
        )

        while pour_monitor.observation_state != ObservationStateValues.TRUE:
            executor.tick()
            transcript.record(decision_slot.latest, executor.control_cycles)

        [retarget] = [
            decision
            for decision in transcript.changes[-1].decisions
            if isinstance(decision, RetargetFillLevel)
        ]
        assert retarget.goal_fill_level == pytest.approx(REQUESTED_FILL_LEVEL)
