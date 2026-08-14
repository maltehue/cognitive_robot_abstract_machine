"""A symbolic theory backed by a :class:`~krrood.ripple_down_rules.rdr.MultiClassRDR`.

The engine natively supplies the defeasible machinery a control theory needs — several conclusions
in one pass, stop rules as defeaters, and intra-pass chaining. This adapter presents it behind the
framework's :class:`~semantic_digital_twin.reasoning.knowledge_servoing.interfaces.SymbolicTheory`
interface, classifying each frozen situation through a mutable working copy so the situation that
crossed the thread boundary is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Sequence, Set, Type

from krrood.ripple_down_rules.rdr import MultiClassRDR
from krrood.ripple_down_rules.utils import make_set

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    DecisionSet,
    Situation,
    SituationType,
    SymbolicTheory,
)

CONCLUSION_ATTRIBUTE_NAME = "conclusions"
"""Name of the working-copy attribute the engine accumulates conclusions into.

Rules read the situation's facts through ``case.situation`` and any earlier conclusions through
``case.conclusions``.
"""


@dataclass
class ClassificationCase:
    """Mutable working copy an :class:`MultiClassRDR` classifies.

    Wrapping the frozen situation in a mutable case gives every classification an isolated
    conclusion accumulator — ``copy_case`` clones a mutable object's mutable attributes rather than
    sharing them — so the frozen situation is never mutated and no conclusion leaks between passes.
    """

    situation: Situation
    """The frozen situation whose facts the rules read through ``case.situation``."""

    conclusions: Set[ControlDecision] = field(default_factory=set)
    """The conclusions accumulated so far in this classification pass."""


@dataclass
class MultiClassRDRTheory(SymbolicTheory[SituationType]):
    """Presents a :class:`MultiClassRDR` over situations as a pluggable symbolic theory."""

    rule_set: MultiClassRDR
    """The multi-class ripple-down-rules classifier authored over the situation's facts."""

    declared_decision_types: frozenset[Type[ControlDecision]]
    """The decision types the rules may conclude, declared for the binding policy's build-time checks."""

    @property
    def decision_types(self) -> frozenset[Type[ControlDecision]]:
        return self.declared_decision_types

    def infer(self, situations: Sequence[SituationType]) -> DecisionSet:
        decisions: list[ControlDecision] = []
        for situation in situations:
            working_copy = ClassificationCase(situation=situation)
            for conclusion in make_set(self.rule_set.classify(working_copy)):
                if isinstance(conclusion, ControlDecision):
                    decisions.append(conclusion)
        return DecisionSet(tuple(decisions))
