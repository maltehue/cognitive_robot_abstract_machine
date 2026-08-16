"""Assembling a statechart from the constraints its theories declare.

These tests pin the inversion the assembler exists for: the chart is built from the theory's
declarations, so gating monitors, start conditions and parameter bindings follow from the theory
rather than from whoever wired the chart — and a declaration the vocabulary does not cover is
rejected at assembly instead of silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from krrood.symbolic_math.float_variable_data import FloatVariableData
from krrood.symbolic_math.symbolic_math import FloatVariable

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.knowledge_servoing.chart_assembler import (
    PluggedTheory,
    TheoryChartAssembler,
)
from giskardpy.motion_statechart.knowledge_servoing.constraint_catalog import (
    ConstraintCatalog,
    ConstraintInstantiation,
)
from giskardpy.motion_statechart.knowledge_servoing.exceptions import (
    DuplicateConstraintFactoryError,
    MissingParameterTargetError,
    UnboundDecisionTypeError,
    UnknownConstraintKindError,
    UnknownParameterAttributeError,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
    ParameterChannel,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    DecisionSet,
    ParameterDecision,
    RegimeDecision,
)
from semantic_digital_twin.world import World

from .test_knowledge_servoing_binding import FixedGrounding, FixedTheory


# %% mimic vocabulary


@dataclass(frozen=True)
class Restrict(RegimeDecision):
    """A regime decision gating a declared constraint."""


@dataclass(frozen=True)
class SetLimit(ParameterDecision):
    """A parameter decision supplying a declared constraint's value."""

    limit: float
    """The value supplied."""


@dataclass(frozen=True)
class LimitDeclaration(ConstraintDeclaration):
    """A mimic constraint kind: some limit with a runtime value."""

    strength: float = 1.0
    """A numeric parameter of the declared constraint."""


@dataclass(frozen=True)
class UncoveredDeclaration(ConstraintDeclaration):
    """A constraint kind no factory covers, standing in for out-of-vocabulary output."""


@dataclass(eq=False, repr=False)
class StandInRemedyTask(Task):
    """A task standing in for whatever enforces a declared constraint."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts()


@dataclass
class DeclaringTheory(FixedTheory):
    """A fixed-conclusion theory that also declares the constraints it needs."""

    declarations: tuple = ()
    """The declared constraints."""

    @property
    def required_constraints(self) -> tuple:
        return self.declarations


# %% helpers


def _limit_factory(
    declaration: LimitDeclaration, world: World
) -> ConstraintInstantiation:
    """Builds the mimic remedy with a writable target."""
    return ConstraintInstantiation(
        node=StandInRemedyTask(name=declaration.identifier),
        parameter_target=FloatVariable(f"{declaration.identifier}_target"),
    )


def _catalog() -> ConstraintCatalog:
    catalog = ConstraintCatalog()
    catalog.register(LimitDeclaration, _limit_factory)
    return catalog


def _context() -> MotionStatechartContext:
    return MotionStatechartContext(
        world=World(), float_variable_data=FloatVariableData()
    )


def _assemble(theory, catalog=None):
    """Assembles one plugged theory into a fresh chart and returns both."""
    statechart = MotionStatechart()
    assembler = TheoryChartAssembler(catalog=catalog or _catalog(), world=World())
    [assembled] = assembler.assemble(
        [
            PluggedTheory(
                name="mimic",
                theory=theory,
                grounding=FixedGrounding(situations=[object()]),
            )
        ],
        statechart,
    )
    return assembled, statechart


# %% assembly wiring


class TestAssemblyFromDeclarations:
    """Whether the chart's wiring follows from what the theory declares."""

    def test_a_gated_declaration_gets_a_monitor_gating_its_node(self):
        theory = DeclaringTheory(
            decisions=DecisionSet((Restrict(),)),
            declared_decision_types=frozenset({Restrict}),
            declarations=(
                LimitDeclaration(identifier="limit", gating_decision_type=Restrict),
            ),
        )
        assembled, statechart = _assemble(theory)

        monitor = assembled.monitors["limit"]
        remedy = assembled.constraint_nodes["limit"]
        assert monitor in statechart.nodes
        assert monitor.observation_variable in remedy.start_condition.free_variables()

    def test_an_ungated_declaration_is_active_for_the_whole_motion(self):
        theory = DeclaringTheory(
            decisions=DecisionSet(()),
            declared_decision_types=frozenset(),
            declarations=(LimitDeclaration(identifier="limit"),),
        )
        assembled, _statechart = _assemble(theory)

        remedy = assembled.constraint_nodes["limit"]
        assert assembled.monitors == {}
        assert remedy.start_condition.free_variables() == []

    def test_a_parameter_channel_delivers_the_concluded_value(self):
        theory = DeclaringTheory(
            decisions=DecisionSet((Restrict(), SetLimit(0.7))),
            declared_decision_types=frozenset({Restrict, SetLimit}),
            declarations=(
                LimitDeclaration(
                    identifier="limit",
                    gating_decision_type=Restrict,
                    parameter_channel=ParameterChannel(
                        decision_type=SetLimit, attribute_name="limit"
                    ),
                ),
            ),
        )
        assembled, _statechart = _assemble(theory)
        context = _context()

        assembled.theory_node.build(context)
        assembled.theory_node.on_tick(context)

        [target] = [
            variable
            for variable in context.float_variable_data.variables
            if variable.name == "limit_target"
        ]
        assert context.float_variable_data.get_value(target) == 0.7

    def test_a_declared_decision_type_nothing_enacts_still_raises_at_build(self):
        theory = DeclaringTheory(
            decisions=DecisionSet((Restrict(),)),
            declared_decision_types=frozenset({Restrict}),
            declarations=(LimitDeclaration(identifier="limit"),),
        )
        assembled, _statechart = _assemble(theory)

        with pytest.raises(UnboundDecisionTypeError):
            assembled.theory_node.build(_context())


# %% vocabulary boundaries


class TestVocabularyBoundaries:
    """What assembly rejects rather than guesses about."""

    def test_an_uncovered_declaration_kind_is_rejected(self):
        theory = DeclaringTheory(
            decisions=DecisionSet(()),
            declared_decision_types=frozenset(),
            declarations=(UncoveredDeclaration(identifier="mystery"),),
        )
        with pytest.raises(UnknownConstraintKindError):
            _assemble(theory)

    def test_the_catalog_reports_its_coverage(self):
        catalog = _catalog()
        assert catalog.covers(LimitDeclaration)
        assert not catalog.covers(UncoveredDeclaration)

    def test_registering_a_second_factory_for_a_kind_raises(self):
        catalog = _catalog()
        with pytest.raises(DuplicateConstraintFactoryError):
            catalog.register(LimitDeclaration, _limit_factory)

    def test_a_channel_naming_a_missing_field_is_rejected(self):
        theory = DeclaringTheory(
            decisions=DecisionSet(()),
            declared_decision_types=frozenset({SetLimit}),
            declarations=(
                LimitDeclaration(
                    identifier="limit",
                    parameter_channel=ParameterChannel(
                        decision_type=SetLimit, attribute_name="speed"
                    ),
                ),
            ),
        )
        with pytest.raises(UnknownParameterAttributeError):
            _assemble(theory)

    def test_a_channel_without_a_target_is_rejected(self):
        catalog = ConstraintCatalog()
        catalog.register(
            LimitDeclaration,
            lambda declaration, world: ConstraintInstantiation(
                node=StandInRemedyTask(name=declaration.identifier)
            ),
        )
        theory = DeclaringTheory(
            decisions=DecisionSet(()),
            declared_decision_types=frozenset({SetLimit}),
            declarations=(
                LimitDeclaration(
                    identifier="limit",
                    parameter_channel=ParameterChannel(
                        decision_type=SetLimit, attribute_name="limit"
                    ),
                ),
            ),
        )
        with pytest.raises(MissingParameterTargetError):
            _assemble(theory, catalog=catalog)
