"""
Binding a symbolic theory's decisions to the controller's two write channels.

Exercises the statechart binding end to end without a robot: a theory node runs a (fake)
theory each tick, publishes its decisions to a slot a :class:`ConcludedMonitor` reads
(channel 1), and writes a parameter decision into a registered float variable through
the :class:`DecisionBindingPolicy` (channel 2). The build-time checks that make the
policy hard to misuse are pinned too.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.float_variable_data import FloatVariableData

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    DecisionSet,
    ParameterDecision,
    RegimeDecision,
    SituationGrounding,
    SymbolicTheory,
)
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.knowledge_servoing.concluded_monitor import (
    ConcludedMonitor,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_binding_policy import (
    DecisionBindingPolicy,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot
from giskardpy.motion_statechart.knowledge_servoing.exceptions import (
    DecisionChannelMismatchError,
    DecisionTypeAlreadyBoundError,
    UnboundDecisionTypeError,
    UnregisteredFloatVariableTargetError,
)
from giskardpy.motion_statechart.knowledge_servoing.symbolic_theory_node import (
    SymbolicTheoryNode,
)


@dataclass(frozen=True)
class Advance(RegimeDecision):
    """
    A regime decision that gates a task.
    """


@dataclass(frozen=True)
class SetSpeed(ParameterDecision):
    """
    A parameter decision carrying a speed to write into a float variable.
    """

    speed: float


@dataclass
class FixedGrounding(SituationGrounding):
    """
    A grounding that returns preset situations regardless of the world.
    """

    situations: list

    def ground(self, world: World) -> list:
        return self.situations


@dataclass
class FixedTheory(SymbolicTheory):
    """
    A theory that returns a preset decision set, standing in for a real reasoner.
    """

    decisions: DecisionSet
    declared_decision_types: frozenset

    @property
    def decision_types(self) -> frozenset:
        return self.declared_decision_types

    def infer(self, situations) -> DecisionSet:
        return self.decisions


def _context() -> MotionStatechartContext:
    return MotionStatechartContext(
        world=World(), float_variable_data=FloatVariableData()
    )


class TestConcludedMonitor:
    """
    Whether the monitor reflects the presence of a decision type in the latest decision
    set.
    """

    def test_unknown_before_first_inference(self):
        monitor = ConcludedMonitor(
            name="advance", decision_type=Advance, decision_slot=DecisionSlot()
        )
        assert monitor.on_tick(_context()) is ObservationStateValues.UNKNOWN

    def test_true_when_decision_present(self):
        slot = DecisionSlot()
        slot.publish(DecisionSet((Advance(),)))
        monitor = ConcludedMonitor(
            name="advance", decision_type=Advance, decision_slot=slot
        )
        assert monitor.on_tick(_context()) is ObservationStateValues.TRUE

    def test_false_when_decision_absent_after_inference(self):
        slot = DecisionSlot()
        slot.publish(DecisionSet((SetSpeed(0.3),)))
        monitor = ConcludedMonitor(
            name="advance", decision_type=Advance, decision_slot=slot
        )
        assert monitor.on_tick(_context()) is ObservationStateValues.FALSE


class TestDecisionBindingPolicy:
    """
    Whether the policy binds channels, refuses misuse, and applies parameter writes.
    """

    def test_validate_raises_for_an_unbound_decision_type(self):
        policy = DecisionBindingPolicy()
        theory = FixedTheory(
            decisions=DecisionSet(), declared_decision_types=frozenset({Advance})
        )
        with pytest.raises(UnboundDecisionTypeError):
            policy.validate(theory, FloatVariableData())

    def test_binding_a_type_twice_raises(self):
        policy = DecisionBindingPolicy()
        first = ConcludedMonitor(
            name="a", decision_type=Advance, decision_slot=DecisionSlot()
        )
        second = ConcludedMonitor(
            name="b", decision_type=Advance, decision_slot=DecisionSlot()
        )
        policy.activate(Advance, first)
        with pytest.raises(DecisionTypeAlreadyBoundError):
            policy.activate(Advance, second)

    def test_activating_a_parameter_decision_raises_channel_mismatch(self):
        policy = DecisionBindingPolicy()
        with pytest.raises(DecisionChannelMismatchError):
            policy.activate(
                SetSpeed,
                ConcludedMonitor(
                    name="x", decision_type=Advance, decision_slot=DecisionSlot()
                ),
            )

    def test_validate_raises_when_a_parameter_target_is_unregistered(self):
        policy = DecisionBindingPolicy()
        policy.parameterize(
            SetSpeed, lambda decision: decision.speed, sm.FloatVariable(name="v")
        )
        theory = FixedTheory(
            decisions=DecisionSet(), declared_decision_types=frozenset({SetSpeed})
        )
        with pytest.raises(UnregisteredFloatVariableTargetError):
            policy.validate(theory, FloatVariableData())

    def test_apply_parameters_writes_the_value_into_the_target(self):
        float_variable_data = FloatVariableData()
        target = sm.FloatVariable(name="speed")
        float_variable_data.register_expression(target)
        policy = DecisionBindingPolicy()
        policy.parameterize(SetSpeed, lambda decision: decision.speed, target)
        policy.apply_parameters(DecisionSet((SetSpeed(0.3),)), float_variable_data)
        assert float_variable_data.get_value(target) == 0.3


class TestSymbolicTheoryNode:
    """
    Whether the node drives both channels from a theory's decisions on each tick.
    """

    def _node_and_monitor(self):
        slot = DecisionSlot()
        target = sm.FloatVariable(name="speed")
        policy = DecisionBindingPolicy()
        policy.activate(
            Advance,
            ConcludedMonitor(name="advance", decision_type=Advance, decision_slot=slot),
        )
        policy.parameterize(SetSpeed, lambda decision: decision.speed, target)
        theory = FixedTheory(
            decisions=DecisionSet((Advance(), SetSpeed(0.3))),
            declared_decision_types=frozenset({Advance, SetSpeed}),
        )
        node = SymbolicTheoryNode(
            name="theory",
            grounding=FixedGrounding(situations=[object()]),
            theory=theory,
            binding_policy=policy,
            decision_slot=slot,
        )
        monitor = ConcludedMonitor(
            name="advance_reader", decision_type=Advance, decision_slot=slot
        )
        return node, monitor, target

    def test_tick_publishes_decisions_and_applies_parameters(self):
        node, monitor, target = self._node_and_monitor()
        context = _context()
        node.build(context)
        assert monitor.on_tick(context) is ObservationStateValues.UNKNOWN
        node.on_tick(context)
        assert monitor.on_tick(context) is ObservationStateValues.TRUE
        assert context.float_variable_data.get_value(target) == 0.3

    def test_build_validates_the_policy_against_the_theory(self):
        slot = DecisionSlot()
        policy = DecisionBindingPolicy()
        theory = FixedTheory(
            decisions=DecisionSet(), declared_decision_types=frozenset({Advance})
        )
        node = SymbolicTheoryNode(
            name="theory",
            grounding=FixedGrounding(situations=[]),
            theory=theory,
            binding_policy=policy,
            decision_slot=slot,
        )
        with pytest.raises(UnboundDecisionTypeError):
            node.build(_context())


@dataclass
class CountingTheory(SymbolicTheory):
    """
    A theory that records how often it was asked to infer.
    """

    decisions: DecisionSet
    declared_decision_types: frozenset
    inference_count: int = 0

    @property
    def decision_types(self) -> frozenset:
        return self.declared_decision_types

    def infer(self, situations) -> DecisionSet:
        self.inference_count += 1
        return self.decisions


class TestInferenceRate:
    """
    Whether the node reasons at its own rate rather than once per control cycle.
    """

    def _node(self, control_cycles_per_inference: int):
        slot = DecisionSlot()
        policy = DecisionBindingPolicy()
        policy.activate(
            Advance,
            ConcludedMonitor(name="advance", decision_type=Advance, decision_slot=slot),
        )
        theory = CountingTheory(
            decisions=DecisionSet((Advance(),)),
            declared_decision_types=frozenset({Advance}),
        )
        node = SymbolicTheoryNode(
            name="theory",
            grounding=FixedGrounding(situations=[object()]),
            theory=theory,
            binding_policy=policy,
            decision_slot=slot,
            control_cycles_per_inference=control_cycles_per_inference,
        )
        return node, theory

    def test_inference_runs_once_per_configured_number_of_cycles(self):
        node, theory = self._node(control_cycles_per_inference=5)
        context = _context()
        for _ in range(10):
            node.on_tick(context)
        assert theory.inference_count == 2

    def test_the_first_tick_always_infers_so_monitors_are_never_left_unknown(self):
        node, theory = self._node(control_cycles_per_inference=5)
        node.on_tick(_context())
        assert theory.inference_count == 1
        assert node.decision_slot.latest.contains_type(Advance)

    def test_decisions_persist_between_inferences(self):
        node, _theory = self._node(control_cycles_per_inference=5)
        context = _context()
        node.on_tick(context)
        node.on_tick(context)
        assert node.decision_slot.latest.contains_type(Advance)
