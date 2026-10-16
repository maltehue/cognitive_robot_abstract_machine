"""
Dynamic object state representation for the Semantic Digital Twin.

This module defines semantic state annotations that can be attached
to world entities and used for:
- Vision-Language Model (VLM) predictions
- Physical state tracking
- Temporal reasoning
- Neuro-symbolic fusion
- Task-level reasoning (via EQL Predicates)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TYPE_CHECKING

from krrood.entity_query_language.predicate import Predicate
from semantic_digital_twin.world_description.world_entity import (
    SemanticAnnotation,
    WorldEntity,
)

if TYPE_CHECKING:
    from .state_manager import DynamicStateManager


# ============================================================================
# State Enumerations
# ============================================================================


class FillLevel(str, Enum):
    """Possible fill levels for container-like objects."""

    EMPTY = "empty"
    FILLED = "filled"
    FULL = "full"


class CutCondition(str, Enum):
    """Possible cut conditions for cuttable objects."""

    CUT = "cut"
    UNCUT = "uncut"


# ============================================================================
# Base State Annotation
# ============================================================================


@dataclass(eq=False, kw_only=True)
class ObjectState(SemanticAnnotation, ABC):
    """
    Base class for all dynamic object state annotations.

    Stores metadata shared across all state observations regardless
    of whether they originate from perception, reasoning, simulation,
    or human input.
    """

    # Every state instance must strictly point to a physical world entity
    target: WorldEntity

    confidence: float = field(
        default=1.0,
        metadata={"description": "Belief score in range [0, 1]."},
    )

    source: str = field(
        default="SDT",
        metadata={
            "description": (
                "Origin of the state estimate "
                "(e.g. CLIP, VLM, Reasoner, FusionEngine)."
            )
        },
    )

    timestamp: Optional[float | int] = field(
        default=None,
        metadata={
            "description": "Unix timestamp or simulation frame tick associated with the observation."
        },
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in range [0, 1], " f"got {self.confidence}"
            )

    @property
    @abstractmethod
    def state_type(self) -> str:
        """
        Returns a unique string identifier for the state category.
        Enables rapid filtering and query serialization.
        """
        pass


# ============================================================================
# Concrete State Implementations
# ============================================================================


@dataclass(eq=False, kw_only=True)
class FillState(ObjectState):
    """Semantic annotation representing the fill level of a container."""

    state: FillLevel = FillLevel.EMPTY

    @property
    def state_type(self) -> str:
        return "fill"

    @property
    def is_empty(self) -> bool:
        return self.state == FillLevel.EMPTY

    @property
    def is_filled(self) -> bool:
        return self.state == FillLevel.FILLED

    @property
    def is_full(self) -> bool:
        return self.state == FillLevel.FULL


@dataclass(eq=False, kw_only=True)
class CutState(ObjectState):
    """Semantic annotation representing whether an object has been cut."""

    state: CutCondition = CutCondition.UNCUT

    @property
    def state_type(self) -> str:
        return "cut"

    @property
    def is_cut(self) -> bool:
        return self.state == CutCondition.CUT

    @property
    def is_uncut(self) -> bool:
        return self.state == CutCondition.UNCUT


# ============================================================================
# Entity Query Language (EQL) Predicates
# ============================================================================


@dataclass(eq=False)
class IsCut(Predicate):
    """EQL Predicate evaluating if an entity is cut."""

    manager: DynamicStateManager
    entity_id: str
    min_confidence: float = 0.8
    max_age_sec: Optional[float] = None

    def __call__(self) -> bool:
        state = self.manager.get_current_state_by_type(self.entity_id, CutState)
        self.manager.validate_state_safety(state, self.min_confidence, self.max_age_sec)
        return state.is_cut


@dataclass(eq=False)
class IsFilled(Predicate):
    """EQL Predicate evaluating if an entity container is filled or full."""

    manager: DynamicStateManager
    entity_id: str
    min_confidence: float = 0.8
    max_age_sec: Optional[float] = None

    def __call__(self) -> bool:
        state = self.manager.get_current_state_by_type(self.entity_id, FillState)
        self.manager.validate_state_safety(state, self.min_confidence, self.max_age_sec)
        return state.is_filled or state.is_full


@dataclass(eq=False)
class IsEmpty(Predicate):
    """EQL Predicate evaluating if an entity container is completely empty."""

    manager: DynamicStateManager
    entity_id: str
    min_confidence: float = 0.8
    max_age_sec: Optional[float] = None

    def __call__(self) -> bool:
        state = self.manager.get_current_state_by_type(self.entity_id, FillState)
        self.manager.validate_state_safety(state, self.min_confidence, self.max_age_sec)
        return state.is_empty
