"""
Assembles and runs the two-theory knowledge-servoing demonstration.

The statechart built here contains no pouring logic and no safety logic. It holds tasks
and the monitors that gate them; which tasks are active, and what the fill goal is, are
conclusions two independent theories reach from the twin each reasoning cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.knowledge_servoing.chart_assembler import (
    PluggedTheory,
    TheoryChartAssembler,
)
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
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.motion_statechart.tasks.commanded_velocity import (
    CommandedTiltVelocity,
    CommandedTranslationVelocity,
)
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ToolSpeedLimitDeclaration,
)
from semantic_digital_twin.reasoning.contextual_safety import (
    EnforceCaution,
    SafetySituationGrounding,
    build_contextual_safety_theory,
)
from semantic_digital_twin.reasoning.substance_transfer import (
    TransferSituationGrounding,
    build_substance_transfer_theory,
)
from semantic_digital_twin.reasoning.substance_transfer.declarations import (
    transfer_constraint_declarations,
)
from semantic_digital_twin.reasoning.substance_transfer.motion_primitives import (
    DecreaseTilt,
    IncreaseTilt,
    MoveBack,
    MoveForward,
    MoveLeft,
    MoveRight,
)
from semantic_digital_twin.reasoning.substance_transfer.primitive_theory import (
    build_motion_primitive_theory,
)

from experiments.knowledge_servoing.constraint_factories import build_transfer_catalog
from experiments.knowledge_servoing.scenario import (
    TransferScenario,
    pouring_plane_stabilization,
)
from experiments.knowledge_servoing.twist_bridge import TwistBridgeNode

POURING_TARGET_FREQUENCY = 80
"""
Control frequency the pouring effect-model goals run at, in hertz.
"""

POURING_PREDICTION_HORIZON = 60
"""
Prediction horizon for charts with pouring effect-model goals.

The canonical pouring configuration is horizon 120 (`test_pouring._pouring_context`),
and ungated charts run there. Gated charts do not: measured on the transfer
demonstration, 40 and 120 are infeasible mid-run while 20 and 60 complete, and the non-
monotonic pattern points at the per-solve row filtering that life-cycle gating causes
interacting badly with the solver at long horizons. Until that is fixed in the
controller, 60 is the longest horizon a gated pouring chart reliably runs at.
"""


def pouring_controller_configuration() -> QPControllerConfig:
    """
    The controller configuration any chart with pouring effect-model goals must run
    under.

    The terminal-state prediction rows plan the pour over the horizon; at the simulation
    default of seven steps they see almost none of it and the transfer overshoots, so
    every executor ticking such a chart uses this configuration.
    """
    return QPControllerConfig(
        target_frequency=POURING_TARGET_FREQUENCY,
        prediction_horizon=POURING_PREDICTION_HORIZON,
    )


REQUESTED_FILL_LEVEL = 0.4
"""
Fill level the transfer theory is asked to reach.
"""

FILL_LEVEL_TOLERANCE = 0.05
"""
Band around the requested level within which the transfer counts as done.
"""

CAUTIOUS_LINEAR_VELOCITY = 0.03
"""
Linear speed cap the caution regime imposes, in metres per second.
"""


@dataclass
class TransferDemonstration:
    """
    A compiled, runnable two-theory transfer and the handles needed to observe it.
    """

    executor: Executor
    """
    The in-process controller ticking the statechart.
    """

    scenario: TransferScenario
    """
    The world being manipulated.
    """

    transfer_decisions: DecisionSlot
    """
    What the substance-transfer theory last concluded.
    """

    safety_decisions: DecisionSlot
    """
    What the contextual-safety theory last concluded.
    """

    transfer_transcript: DecisionTranscript = field(default_factory=DecisionTranscript)
    """
    Regime turnovers of the transfer theory over the run.
    """

    safety_transcript: DecisionTranscript = field(default_factory=DecisionTranscript)
    """
    Regime turnovers of the safety theory over the run.
    """

    def run(self, maximum_control_cycles: int = 4000) -> None:
        """
        Ticks the controller to completion, transcribing both theories as it goes.

        :param maximum_control_cycles: Cycle budget after which the run stops
            regardless.
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
        """
        Renders the statechart's life-cycle history, which is the regime timeline.

        :param path: Where to write the PDF.
        """
        self.executor.motion_statechart.plot_gantt_chart(
            path=path, context=self.executor.context
        )


def build_transfer_demonstration(
    scenario: TransferScenario,
    requested_fill_level: float = REQUESTED_FILL_LEVEL,
) -> TransferDemonstration:
    """
    Assembles the statechart from what both theories declare, and compiles it.

    Nothing about the chart is wired by hand: every task, gate and parameter binding follows from
    the theories' constraint declarations, so plugging in a further theory would be one more entry
    in the assembler's input. Only termination is stated here, because what ends the motion is a
    property of the demonstration, not of any one theory.

    :param scenario: The world to manipulate.
    :param requested_fill_level: Fill level the transfer theory is asked to reach.
    :return: The compiled demonstration, ready to run.
    """
    world = scenario.world
    transfer_theory = build_substance_transfer_theory(
        transfer_constraint_declarations(
            source_name="source_cup",
            receiver_name="receiving_cup",
            fill_level_tolerance=FILL_LEVEL_TOLERANCE,
        )
    )
    safety_theory = build_contextual_safety_theory(
        (
            ToolSpeedLimitDeclaration(
                identifier="caution_speed_cap",
                subject_name="source_cup",
                maximum_speed=CAUTIOUS_LINEAR_VELOCITY,
                gating_decision_type=EnforceCaution,
            ),
        )
    )

    statechart = MotionStatechart()
    assembler = TheoryChartAssembler(catalog=build_transfer_catalog(), world=world)
    transfer_assembled, safety_assembled = assembler.assemble(
        [
            PluggedTheory(
                name="transfer_theory",
                theory=transfer_theory,
                grounding=TransferSituationGrounding(
                    source=scenario.source_cup,
                    receiver=scenario.receiving_cup,
                    requested_fill_level=requested_fill_level,
                    fill_level_tolerance=FILL_LEVEL_TOLERANCE,
                ),
            ),
            PluggedTheory(
                name="safety_theory",
                theory=safety_theory,
                grounding=SafetySituationGrounding(
                    carried_container=scenario.source_cup,
                    sensitive_bodies=[scenario.sensitive_body],
                ),
            ),
        ],
        statechart,
    )
    statechart.add_node(pouring_plane_stabilization(scenario))
    statechart.add_node(
        EndMotion.when_true(transfer_assembled.monitors["return_upright"])
    )

    executor = Executor(
        MotionStatechartContext(
            world=world, qp_controller_config=pouring_controller_configuration()
        ),
        pacer=SimulationPacer(real_time_factor=1),
    )
    executor.compile(motion_statechart=statechart)
    return TransferDemonstration(
        executor=executor,
        scenario=scenario,
        transfer_decisions=transfer_assembled.decision_slot,
        safety_decisions=safety_assembled.decision_slot,
    )


def build_primitive_arm_demonstration(
    scenario: TransferScenario,
    requested_fill_level: float = REQUESTED_FILL_LEVEL,
) -> TransferDemonstration:
    """
    Wires the replication arm: the same facts, driven through the fixed-gain twist
    bridge.

    The reasoner, the grounding, the scene and the robot are the ones the regime arm
    uses. What differs is the vocabulary the theory concludes and the bridge that turns
    it into motion, so a difference in the outcome is attributable to the bridge.

    :param scenario: The world to manipulate.
    :param requested_fill_level: Fill level the theory is asked to reach.
    :return: The compiled demonstration, ready to run.
    """
    world = scenario.world
    translation = CommandedTranslationVelocity(
        name="commanded_translation",
        root_link=world.root,
        tip_link=scenario.source_cup.root,
    )
    tilt = CommandedTiltVelocity(
        name="commanded_tilt",
        root_link=world.root,
        tip_link=scenario.source_cup.root,
    )

    primitive_decisions = DecisionSlot()
    bridge = TwistBridgeNode(
        name="twist_bridge",
        decision_slot=primitive_decisions,
        translation=translation,
        tilt=tilt,
    )
    goal_reached_monitor = ConcludedMonitor(
        name="tilting_back",
        decision_type=DecreaseTilt,
        decision_slot=primitive_decisions,
    )

    policy = DecisionBindingPolicy()
    for primitive in (
        MoveForward,
        MoveBack,
        MoveLeft,
        MoveRight,
        IncreaseTilt,
        DecreaseTilt,
    ):
        policy.activate(primitive, bridge)

    theory_node = SymbolicTheoryNode(
        name="primitive_theory",
        grounding=TransferSituationGrounding(
            source=scenario.source_cup,
            receiver=scenario.receiving_cup,
            requested_fill_level=requested_fill_level,
            fill_level_tolerance=FILL_LEVEL_TOLERANCE,
        ),
        theory=build_motion_primitive_theory(),
        binding_policy=policy,
        decision_slot=primitive_decisions,
    )

    statechart = MotionStatechart()
    for node in (theory_node, bridge, goal_reached_monitor, translation, tilt):
        statechart.add_node(node)
    statechart.add_node(EndMotion.when_true(goal_reached_monitor))

    executor = Executor(
        MotionStatechartContext(world=world), pacer=SimulationPacer(real_time_factor=1)
    )
    executor.compile(motion_statechart=statechart)
    return TransferDemonstration(
        executor=executor,
        scenario=scenario,
        transfer_decisions=primitive_decisions,
        safety_decisions=primitive_decisions,
    )
