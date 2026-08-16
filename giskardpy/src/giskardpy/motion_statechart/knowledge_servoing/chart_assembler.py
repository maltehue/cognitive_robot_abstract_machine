"""
Assembles a motion statechart from the constraints its theories declare.

This inverts the dependency the hand-wired demonstrations had: instead of a theory
having to fit a chart someone else built, the chart is built from what each theory
declares it needs. Plugging a theory in is adding one entry to the assembler's input;
every gate, monitor and parameter binding follows from its declarations, and the binding
policy's build-time validation passes by construction for whatever it assembled.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from operator import attrgetter

from typing_extensions import Dict, List, Sequence

from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
    ParameterChannel,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    SituationGrounding,
    SymbolicTheory,
)
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.knowledge_servoing.concluded_monitor import (
    ConcludedMonitor,
)
from giskardpy.motion_statechart.knowledge_servoing.constraint_catalog import (
    ConstraintCatalog,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_binding_policy import (
    DecisionBindingPolicy,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot
from giskardpy.motion_statechart.knowledge_servoing.exceptions import (
    MissingParameterTargetError,
    UnknownParameterAttributeError,
)
from giskardpy.motion_statechart.knowledge_servoing.symbolic_theory_node import (
    SymbolicTheoryNode,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart


@dataclass
class PluggedTheory:
    """
    One theory and the grounding that feeds it, as handed to the assembler.
    """

    name: str
    """Names the theory's node and prefixes the names of everything assembled for it."""

    theory: SymbolicTheory
    """
    The theory whose declarations the chart is assembled from.
    """

    grounding: SituationGrounding
    """Produces the theory's situations from the world."""


@dataclass
class AssembledTheory:
    """
    Everything the assembler built for one plugged theory.
    """

    theory_node: SymbolicTheoryNode
    """The node running the theory each reasoning cycle."""

    decision_slot: DecisionSlot
    """
    Where the theory's decisions are published.
    """

    constraint_nodes: Dict[str, MotionStatechartNode]
    """The enforcing node of each declared constraint, by declaration identifier."""

    monitors: Dict[str, ConcludedMonitor]
    """
    The gating monitor of each gated constraint, by declaration identifier.
    """


@dataclass
class TheoryChartAssembler:
    """
    Builds the statechart a set of declared theories asks for.
    """

    catalog: ConstraintCatalog
    """The constraint vocabulary declarations are enforced from."""

    world: World
    """
    The world subject names are resolved in.
    """

    def assemble(
        self, plugged_theories: Sequence[PluggedTheory], statechart: MotionStatechart
    ) -> List[AssembledTheory]:
        """
        Adds every plugged theory and everything its declarations require to the chart.

        Termination is deliberately not assembled: what ends a motion is the caller's statement
        about the task, not a property of any one theory.

        :param plugged_theories: The theories to assemble, each with its grounding.
        :param statechart: The chart the assembled nodes are added to.
        :return: What was assembled, one entry per plugged theory.
        """
        return [
            self._assemble_theory(plugged, statechart) for plugged in plugged_theories
        ]

    def _assemble_theory(
        self, plugged: PluggedTheory, statechart: MotionStatechart
    ) -> AssembledTheory:
        """
        Assembles one theory's constraints, gates and bindings into the chart.

        :param plugged: The theory and its grounding.
        :param statechart: The chart the nodes are added to.
        :return: What was assembled for this theory.
        """
        decision_slot = DecisionSlot()
        binding_policy = DecisionBindingPolicy()
        constraint_nodes: Dict[str, MotionStatechartNode] = {}
        monitors: Dict[str, ConcludedMonitor] = {}
        monitors_by_decision_type: Dict[type, ConcludedMonitor] = {}

        for declaration in plugged.theory.required_constraints:
            instantiation = self.catalog.instantiate(declaration, self.world)
            statechart.add_node(instantiation.node)
            constraint_nodes[declaration.identifier] = instantiation.node

            if declaration.gating_decision_type is not None:
                # One monitor per gating decision type: several constraints gated by the same
                # decision share it, since they all read the same conclusion.
                monitor = monitors_by_decision_type.get(
                    declaration.gating_decision_type
                )
                if monitor is None:
                    monitor = ConcludedMonitor(
                        name=f"{plugged.name}_{declaration.gating_decision_type.__name__}_concluded",
                        decision_type=declaration.gating_decision_type,
                        decision_slot=decision_slot,
                    )
                    statechart.add_node(monitor)
                    binding_policy.activate(
                        declaration.gating_decision_type, instantiation.node
                    )
                    monitors_by_decision_type[declaration.gating_decision_type] = (
                        monitor
                    )
                instantiation.node.start_condition = monitor.observation_variable
                monitors[declaration.identifier] = monitor

            if declaration.parameter_channel is not None:
                self._bind_parameter_channel(
                    declaration, instantiation.parameter_target, binding_policy
                )

        theory_node = SymbolicTheoryNode(
            name=plugged.name,
            grounding=plugged.grounding,
            theory=plugged.theory,
            binding_policy=binding_policy,
            decision_slot=decision_slot,
        )
        statechart.add_node(theory_node)
        return AssembledTheory(
            theory_node=theory_node,
            decision_slot=decision_slot,
            constraint_nodes=constraint_nodes,
            monitors=monitors,
        )

    @staticmethod
    def _bind_parameter_channel(
        declaration: ConstraintDeclaration,
        parameter_target,
        binding_policy: DecisionBindingPolicy,
    ) -> None:
        """
        Binds a declaration's parameter channel to its instantiation's target.

        :param declaration: The declaration carrying the channel.
        :param parameter_target: The float variable the factory returned, if any.
        :param binding_policy: The policy the binding is added to.
        :raises MissingParameterTargetError: if the factory returned nothing to write
            into.
        :raises UnknownParameterAttributeError: if the channel names a field the
            decision type does not have.
        """
        channel: ParameterChannel = declaration.parameter_channel
        if parameter_target is None:
            raise MissingParameterTargetError(identifier=declaration.identifier)
        field_names = {
            declared_field.name
            for declared_field in dataclass_fields(channel.decision_type)
        }
        if channel.attribute_name not in field_names:
            raise UnknownParameterAttributeError(
                decision_type=channel.decision_type,
                attribute_name=channel.attribute_name,
            )
        binding_policy.parameterize(
            channel.decision_type, attrgetter(channel.attribute_name), parameter_target
        )
