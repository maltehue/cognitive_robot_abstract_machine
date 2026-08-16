"""Assembles and runs the two-theory knowledge-servoing demonstration.

The statechart built here contains no pouring logic and no safety logic. It holds tasks and the
monitors that gate them; which tasks are active, and what the fill goal is, are conclusions two
independent theories reach from the twin each reasoning cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import EndMotion
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
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianVelocityLimit
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from semantic_digital_twin.reasoning.contextual_safety import (
    EnforceCaution,
    SafetySituationGrounding,
    build_contextual_safety_theory,
)
from semantic_digital_twin.reasoning.substance_transfer import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
    RetargetFillLevel,
    TransferSituationGrounding,
    build_substance_transfer_theory,
)
from semantic_digital_twin.spatial_types import Vector3

from experiments.knowledge_servoing.scenario import TransferScenario

REQUESTED_FILL_LEVEL = 0.4
"""Fill level the transfer theory is asked to reach."""

FILL_LEVEL_TOLERANCE = 0.05
"""Band around the requested level within which the transfer counts as done."""

CAUTIOUS_LINEAR_VELOCITY = 0.03
"""Linear speed cap the caution regime imposes, in metres per second."""


@dataclass
class TransferDemonstration:
    """A compiled, runnable two-theory transfer and the handles needed to observe it."""

    executor: Executor
    """The in-process controller ticking the statechart."""

    scenario: TransferScenario
    """The world being manipulated."""

    transfer_decisions: DecisionSlot
    """What the substance-transfer theory last concluded."""

    safety_decisions: DecisionSlot
    """What the contextual-safety theory last concluded."""

    transfer_transcript: DecisionTranscript = field(default_factory=DecisionTranscript)
    """Regime turnovers of the transfer theory over the run."""

    safety_transcript: DecisionTranscript = field(default_factory=DecisionTranscript)
    """Regime turnovers of the safety theory over the run."""

    def run(self, maximum_control_cycles: int = 4000) -> None:
        """Ticks the controller to completion, transcribing both theories as it goes.

        :param maximum_control_cycles: Cycle budget after which the run stops regardless.
        """
        for _ in range(maximum_control_cycles):
            self.executor.tick()
            control_cycle = self.executor.control_cycles
            self.transfer_transcript.record(
                self.transfer_decisions.latest, control_cycle
            )
            self.safety_transcript.record(self.safety_decisions.latest, control_cycle)
            if self.executor.motion_statechart.is_end_motion():
                return

    def plot_gantt_chart(self, path: str) -> None:
        """Renders the statechart's life-cycle history, which is the regime timeline.

        :param path: Where to write the PDF.
        """
        self.executor.motion_statechart.plot_gantt_chart(
            path=path, context=self.executor.context
        )


def build_transfer_demonstration(
    scenario: TransferScenario,
    requested_fill_level: float = REQUESTED_FILL_LEVEL,
) -> TransferDemonstration:
    """Wires both theories to the controller and compiles the statechart.

    :param scenario: The world to manipulate.
    :param requested_fill_level: Fill level the transfer theory is asked to reach.
    :return: The compiled demonstration, ready to run.
    """
    world = scenario.world
    goal_fill_variable = sm.FloatVariable(name="transfer_goal_fill_level")

    aim = KeepProjectileInReceiver(
        receiver=scenario.receiving_cup,
        source=scenario.source_cup,
        weight=DefaultWeights.WEIGHT_MAXIMUM,
    )
    clearance = KeepSourceRimAboveReceiverRim(
        receiver=scenario.receiving_cup,
        source=scenario.source_cup,
        minimum_clearance=0.08,
    )
    transfer = FillByTransferTask(
        receiver=scenario.receiving_cup,
        goal_value=goal_fill_variable,
        fill_level_tolerance=FILL_LEVEL_TOLERANCE,
    )
    return_upright = AlignPlanes(
        root_link=world.root,
        tip_link=scenario.source_cup.root,
        goal_normal=Vector3.Z(reference_frame=world.root),
        tip_normal=Vector3.Z(reference_frame=scenario.source_cup.root),
    )
    speed_cap = CartesianVelocityLimit(
        root_link=world.root,
        tip_link=scenario.source_cup.root,
        max_linear_velocity=CAUTIOUS_LINEAR_VELOCITY,
    )

    transfer_decisions = DecisionSlot()
    safety_decisions = DecisionSlot()
    align_monitor = ConcludedMonitor(
        name="aligned",
        decision_type=AlignSourceOverReceiver,
        decision_slot=transfer_decisions,
    )
    pour_monitor = ConcludedMonitor(
        name="pouring", decision_type=PourIntoReceiver, decision_slot=transfer_decisions
    )
    concluded_monitor = ConcludedMonitor(
        name="concluded",
        decision_type=ConcludeTransfer,
        decision_slot=transfer_decisions,
    )
    caution_monitor = ConcludedMonitor(
        name="cautious", decision_type=EnforceCaution, decision_slot=safety_decisions
    )

    transfer_policy = DecisionBindingPolicy()
    transfer_policy.activate(AlignSourceOverReceiver, aim)
    transfer_policy.activate(PourIntoReceiver, transfer)
    transfer_policy.activate(ConcludeTransfer, return_upright)
    transfer_policy.activate(AbandonTransfer, concluded_monitor)
    transfer_policy.parameterize(
        RetargetFillLevel, lambda decision: decision.goal_fill_level, goal_fill_variable
    )
    safety_policy = DecisionBindingPolicy()
    safety_policy.activate(EnforceCaution, speed_cap)

    transfer_node = SymbolicTheoryNode(
        name="transfer_theory",
        grounding=TransferSituationGrounding(
            source=scenario.source_cup,
            receiver=scenario.receiving_cup,
            requested_fill_level=requested_fill_level,
            fill_level_tolerance=FILL_LEVEL_TOLERANCE,
        ),
        theory=build_substance_transfer_theory(),
        binding_policy=transfer_policy,
        decision_slot=transfer_decisions,
    )
    safety_node = SymbolicTheoryNode(
        name="safety_theory",
        grounding=SafetySituationGrounding(
            carried_container=scenario.source_cup,
            sensitive_bodies=[scenario.sensitive_body],
        ),
        theory=build_contextual_safety_theory(),
        binding_policy=safety_policy,
        decision_slot=safety_decisions,
    )

    aim.start_condition = align_monitor.observation_variable
    clearance.start_condition = align_monitor.observation_variable
    transfer.start_condition = pour_monitor.observation_variable
    return_upright.start_condition = concluded_monitor.observation_variable
    speed_cap.start_condition = caution_monitor.observation_variable

    statechart = MotionStatechart()
    for node in (
        transfer_node,
        safety_node,
        align_monitor,
        pour_monitor,
        concluded_monitor,
        caution_monitor,
        aim,
        clearance,
        transfer,
        return_upright,
        speed_cap,
    ):
        statechart.add_node(node)
    statechart.add_node(EndMotion.when_true(concluded_monitor))

    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    return TransferDemonstration(
        executor=executor,
        scenario=scenario,
        transfer_decisions=transfer_decisions,
        safety_decisions=safety_decisions,
    )
