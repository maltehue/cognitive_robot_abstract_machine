"""Shared base for symbolic theories backed by a ripple-down-rules classifier.

Both the multi-class and general ripple-down-rules engines present the same contract to the
framework — situations in, decisions out — and declare the decision types they may conclude for the
binding policy's build-time checks. Only how they classify differs, so that stays in the subclasses.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from typing_extensions import Type

from krrood.ripple_down_rules.rdr import RippleDownRules

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    SituationType,
    SymbolicTheory,
)


@dataclass
class RippleDownRulesTheory(SymbolicTheory[SituationType], ABC):
    """A symbolic theory whose rules are a ripple-down-rules classifier over situations."""

    rule_set: RippleDownRules
    """The ripple-down-rules classifier authored over the situation's facts."""

    declared_decision_types: frozenset[Type[ControlDecision]]
    """The decision types the rules may conclude, declared for the binding policy's build-time checks."""

    @property
    def decision_types(self) -> frozenset[Type[ControlDecision]]:
        return self.declared_decision_types
