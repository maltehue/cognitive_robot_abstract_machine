"""The declarative map from a theory's decision types onto the controller's two write channels.

This is the pluggable part of the controller: which regime decision gates which task (channel 1) and
which parameter decision writes which float variable (channel 2). It is built once and validated
against the theory's declared decision types, so a misconfiguration raises at build rather than
misbehaving at run time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Callable, Dict, Type

from krrood.symbolic_math.float_variable_data import (
    FloatVariableData,
    hidden_index_name,
)
from krrood.symbolic_math.symbolic_math import SymbolicMathType

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    DecisionSet,
    ParameterDecision,
    RegimeDecision,
    SymbolicTheory,
)

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.knowledge_servoing.exceptions import (
    DecisionChannelMismatchError,
    DecisionTypeAlreadyBoundError,
    UnboundDecisionTypeError,
    UnregisteredFloatVariableTargetError,
)


@dataclass
class ParameterBinding:
    """How a parameter decision's value reaches the solver: read it, then write the target variable."""

    read_value: Callable[[ParameterDecision], float]
    """Extracts the numeric value from a concluded parameter decision."""

    target: SymbolicMathType
    """The registered float variable the value is written into."""


@dataclass
class DecisionBindingPolicy:
    """Maps a theory's decision types onto statechart activations and float-variable writes."""

    _activations: Dict[Type[RegimeDecision], MotionStatechartNode] = field(
        default_factory=dict, init=False
    )
    """Regime decision type to the node its presence gates (channel 1)."""

    _parameterizations: Dict[Type[ParameterDecision], ParameterBinding] = field(
        default_factory=dict, init=False
    )
    """Parameter decision type to how its value is written (channel 2)."""

    def activate(
        self, decision_type: Type[RegimeDecision], node: MotionStatechartNode
    ) -> None:
        """Binds a regime decision type to a node it gates (channel 1)."""
        self._require_channel(decision_type, RegimeDecision)
        self._require_not_already_bound(decision_type)
        self._activations[decision_type] = node

    def parameterize(
        self,
        decision_type: Type[ParameterDecision],
        read_value: Callable[[ParameterDecision], float],
        target: SymbolicMathType,
    ) -> None:
        """Binds a parameter decision type to the float variable its value is written into (channel 2)."""
        self._require_channel(decision_type, ParameterDecision)
        self._require_not_already_bound(decision_type)
        self._parameterizations[decision_type] = ParameterBinding(
            read_value=read_value, target=target
        )

    def validate(
        self, theory: SymbolicTheory, float_variable_data: FloatVariableData
    ) -> None:
        """Checks every declared decision type is bound and every parameter target is registered.

        :raises UnboundDecisionTypeError: if the theory declares a decision type bound to neither
            channel.
        :raises UnregisteredFloatVariableTargetError: if a parameter target was never registered with
            the float-variable data.
        """
        bound = set(self._activations) | set(self._parameterizations)
        for decision_type in theory.decision_types:
            if decision_type not in bound:
                raise UnboundDecisionTypeError(decision_type=decision_type)
        for decision_type, binding in self._parameterizations.items():
            if not hasattr(binding.target, hidden_index_name):
                raise UnregisteredFloatVariableTargetError(decision_type=decision_type)

    def register_targets(self, float_variable_data: FloatVariableData) -> None:
        """Registers every parameter target with the float-variable data so its value can be written."""
        for binding in self._parameterizations.values():
            float_variable_data.register_expression(binding.target)

    def apply_parameters(
        self, decisions: DecisionSet, float_variable_data: FloatVariableData
    ) -> None:
        """Writes each concluded parameter decision's value into its registered float variable."""
        for decision in decisions:
            binding = self._parameterizations.get(type(decision))
            if binding is None:
                continue
            float_variable_data.set_value(binding.target, binding.read_value(decision))

    def gated_node(self, decision_type: Type[RegimeDecision]) -> MotionStatechartNode:
        """The node a regime decision type gates."""
        return self._activations[decision_type]

    def _require_channel(
        self, decision_type: Type[ControlDecision], expected_base: Type[ControlDecision]
    ) -> None:
        if not issubclass(decision_type, expected_base):
            raise DecisionChannelMismatchError(
                decision_type=decision_type, expected_base=expected_base
            )

    def _require_not_already_bound(self, decision_type: Type[ControlDecision]) -> None:
        if (
            decision_type in self._activations
            or decision_type in self._parameterizations
        ):
            raise DecisionTypeAlreadyBoundError(decision_type=decision_type)
