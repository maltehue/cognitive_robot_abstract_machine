"""
The framework's symbolic theory, backed by a
:class:`~krrood.ripple_down_rules.rdr.GeneralRDR`.

A general ripple-down-rules classifier composes one sub-classifier per decision family and re-runs
them to a fixpoint, so a rule in one family can condition on a conclusion another family reached.
That subsumes the single-family multi-class case, so it is the only engine the framework needs. This
adapter presents it behind the
:class:`~semantic_digital_twin.reasoning.knowledge_servoing.interfaces.SymbolicTheory` interface,
classifying each frozen situation through a mutable working copy that carries one accumulator per
family so the situation that crossed the thread boundary is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field, make_dataclass

from typing_extensions import Any, List, Sequence, Set, Tuple, Type

from krrood.ripple_down_rules.rdr import GeneralRDR
from krrood.ripple_down_rules.utils import make_set

from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    DecisionSet,
    Situation,
    SituationType,
    SymbolicTheory,
)


@dataclass
class GeneralRDRTheory(SymbolicTheory[SituationType]):
    """
    Presents a :class:`GeneralRDR` over situations as a pluggable symbolic theory.
    """

    rule_set: GeneralRDR
    """The general ripple-down-rules classifier composing one sub-classifier per decision family."""

    declared_decision_types: frozenset[Type[ControlDecision]]
    """
    The decision types the rules may conclude, declared for the binding policy's build-
    time checks.
    """

    constraint_declarations: Tuple[ConstraintDeclaration, ...] = ()
    """
    The constraints this theory requires the controller to enforce; the chart is
    assembled from them.
    """

    _working_case_type: Type = field(init=False, repr=False)
    """
    Mutable working-copy dataclass carrying the frozen situation and one accumulator per
    family.
    """

    @property
    def decision_types(self) -> frozenset[Type[ControlDecision]]:
        return self.declared_decision_types

    @property
    def required_constraints(self) -> Tuple[ConstraintDeclaration, ...]:
        return self.constraint_declarations

    def __post_init__(self) -> None:
        family_names = list(self.rule_set.start_rules_dict.keys())
        self._working_case_type = make_dataclass(
            "GeneralClassificationCase",
            [("situation", Situation)]
            + [(name, Set, field(default_factory=set)) for name in family_names],
        )

    def infer(self, situations: Sequence[SituationType]) -> DecisionSet:
        conclusions: List[Any] = []
        for situation in situations:
            working_copy = self._working_case_type(situation=situation)
            for family_conclusions in self.rule_set.classify(working_copy).values():
                conclusions.extend(make_set(family_conclusions))
        return DecisionSet.from_conclusions(conclusions)
