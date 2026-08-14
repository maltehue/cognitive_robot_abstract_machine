"""Domain-agnostic interfaces of the knowledge-servoing framework.

A pluggable symbolic theory grounds the world into immutable *situations* and infers *decisions*
addressed to one of two write channels into the running controller. These interfaces carry no
domain vocabulary: they are generic over the situation type a theory grounds and the decision types
it concludes, so a theory about pouring, cutting or contextual safety plugs in without editing them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from typing_extensions import Iterator, Sequence, Tuple, Type

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

SituationType = TypeVar("SituationType", bound="Situation")
"""The situation type a grounding produces and a theory reasons over."""


@dataclass(frozen=True)
class Situation(ABC):
    """One immutable snapshot of the facts a theory reasons over, for one subject of reasoning.

    Frozen so it can cross from the control thread to the reasoner thread without a shared-state
    race; the theory builds its own mutable working copy for classification.
    """


@dataclass(frozen=True)
class ControlDecision(ABC):
    """A conclusion a theory reached, addressed to exactly one of the two write channels."""


@dataclass(frozen=True)
class RegimeDecision(ControlDecision, ABC):
    """A decision that activates, pauses or ends constraints (channel 1)."""


@dataclass(frozen=True)
class ParameterDecision(ControlDecision, ABC):
    """A decision that supplies numeric values to registered float variables (channel 2)."""


@dataclass(frozen=True)
class DecisionSet:
    """The decisions a theory concluded in one inference cycle."""

    decisions: Tuple[ControlDecision, ...] = ()
    """The concluded decisions, in the order inference produced them."""

    def __iter__(self) -> Iterator[ControlDecision]:
        return iter(self.decisions)

    def __len__(self) -> int:
        return len(self.decisions)

    def of_type(
        self, decision_type: Type[ControlDecision]
    ) -> Tuple[ControlDecision, ...]:
        """Returns the concluded decisions that are instances of ``decision_type``."""
        return tuple(
            decision
            for decision in self.decisions
            if isinstance(decision, decision_type)
        )

    def contains_type(self, decision_type: Type[ControlDecision]) -> bool:
        """Whether any concluded decision is an instance of ``decision_type``."""
        return any(isinstance(decision, decision_type) for decision in self.decisions)


@dataclass
class SituationGrounding(Generic[SituationType], ABC):
    """Produces a theory's situations from the world; runs on the control thread."""

    @abstractmethod
    def ground(self, world: World) -> Sequence[SituationType]:
        """Grounds the current world state into immutable situations."""


@dataclass
class SymbolicTheory(Generic[SituationType], ABC):
    """A pluggable symbolic theory: situations in, decisions out. Runs off the control thread."""

    @property
    @abstractmethod
    def decision_types(self) -> frozenset[Type[ControlDecision]]:
        """The decision types this theory may conclude, declared for build-time binding checks."""

    @abstractmethod
    def infer(self, situations: Sequence[SituationType]) -> DecisionSet:
        """Infers the decisions the theory reaches for the given situations."""
