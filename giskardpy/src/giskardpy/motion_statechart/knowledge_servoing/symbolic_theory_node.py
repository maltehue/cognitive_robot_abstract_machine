"""
The statechart node that runs a symbolic theory and applies its decisions to the
controller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    SituationGrounding,
    SymbolicTheory,
)

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.knowledge_servoing.decision_binding_policy import (
    DecisionBindingPolicy,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot


@dataclass(eq=False, repr=False)
class SymbolicTheoryNode(MotionStatechartNode):
    """
    Runs a symbolic theory at its own rate and applies its decisions to both write
    channels.

    On an inference cycle it grounds the world into situations, infers a decision set, publishes it
    to the decision slot the :class:`ConcludedMonitor`s read (channel 1), and writes the parameter
    decisions into their registered float variables through the binding policy (channel 2). Control
    cycles in between leave the last decision set standing, so the controller keeps running on the
    reasoner's most recent conclusions.

    .. note:: Inference runs synchronously on the control thread. The spike measured it at tens of
        microseconds over a handful of situations — far inside the control budget — so no reasoner
        thread or cross-thread state hand-off is needed; grounding, inference and application all
        happen on the one thread that already owns the world.
    """

    grounding: SituationGrounding = field(kw_only=True)
    """
    Produces the theory's situations from the world.
    """

    theory: SymbolicTheory = field(kw_only=True)
    """
    The theory whose decisions drive the controller.
    """

    binding_policy: DecisionBindingPolicy = field(kw_only=True)
    """
    Maps the theory's decisions onto the two write channels.
    """

    decision_slot: DecisionSlot = field(kw_only=True)
    """
    The slot this node publishes to and the monitors read from.
    """

    control_cycles_per_inference: int = field(default=5, kw_only=True)
    """
    How many control cycles pass between inferences.

    Reasoning is a decision layer, not a servo loop: rerunning it every control cycle
    multiplies its cost without changing its conclusions, since the qualitative facts it
    reads move far more slowly than the joint state. Between inferences the monitors
    keep reading the last published decision set, which is what a slower reasoner
    driving a faster controller means.
    """

    _cycles_until_inference: int = field(default=0, init=False, repr=False)
    """
    Control cycles still to skip before the next inference; zero means infer on this
    tick.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Registers the policy's float-variable targets and validates it against the
        theory.

        Registering the targets before validation lets the build-time check confirm
        every parameter decision writes a float variable the solver can actually read.
        """
        self.binding_policy.register_targets(context.float_variable_data)
        self.binding_policy.validate(self.theory, context.float_variable_data)
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        """
        Infers on the first tick after starting, so the monitors are never left unknown.
        """
        self._cycles_until_inference = 0

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self._cycles_until_inference > 0:
            self._cycles_until_inference -= 1
            return ObservationStateValues.TRUE
        self._cycles_until_inference = self.control_cycles_per_inference - 1
        situations = self.grounding.ground(context.world)
        decisions = self.theory.infer(situations)
        self.decision_slot.publish(decisions)
        self.binding_policy.apply_parameters(decisions, context.float_variable_data)
        return ObservationStateValues.TRUE
