"""The statechart node that runs a symbolic theory and applies its decisions to the controller."""

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
    """Runs a symbolic theory each control tick and applies its decisions to both write channels.

    On every tick it grounds the world into situations, infers a decision set, publishes it to the
    decision slot the :class:`ConcludedMonitor`s read (channel 1), and writes the parameter decisions
    into their registered float variables through the binding policy (channel 2).

    .. note:: Inference runs synchronously on the control thread. The spike measured it at tens of
        microseconds over a handful of situations — far inside the control budget — so no reasoner
        thread or cross-thread state hand-off is needed; grounding, inference and application all
        happen on the one thread that already owns the world.
    """

    grounding: SituationGrounding = field(kw_only=True)
    """Produces the theory's situations from the world."""

    theory: SymbolicTheory = field(kw_only=True)
    """The theory whose decisions drive the controller."""

    binding_policy: DecisionBindingPolicy = field(kw_only=True)
    """Maps the theory's decisions onto the two write channels."""

    decision_slot: DecisionSlot = field(kw_only=True)
    """The slot this node publishes to and the monitors read from."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """Registers the policy's float-variable targets and validates it against the theory.

        Registering the targets before validation lets the build-time check confirm every parameter
        decision writes a float variable the solver can actually read.
        """
        self.binding_policy.register_targets(context.float_variable_data)
        self.binding_policy.validate(self.theory, context.float_variable_data)
        return NodeArtifacts()

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        situations = self.grounding.ground(context.world)
        decisions = self.theory.infer(situations)
        self.decision_slot.publish(decisions)
        self.binding_policy.apply_parameters(decisions, context.float_variable_data)
        return ObservationStateValues.TRUE
