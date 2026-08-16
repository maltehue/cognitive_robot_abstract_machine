"""The contextual-safety theory's decision vocabulary.

Every decision here gates constraints; none supplies a numeric value, because this theory owns no
effect model. That is the point of it: a theory can be useful to the controller without predicting
any physical quantity, by deciding which of the constraints the controller already knows how to
enforce should be active in the situation it finds itself in.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    RegimeDecision,
)


class CautionReason(Enum):
    """Why the situation calls for a more cautious motion."""

    SPILL_WOULD_REACH_SENSITIVE_OBJECT = auto()
    """Contents are in flight above an object that must not be spilled on."""

    CARRYING_CONTENTS_OVER_SENSITIVE_OBJECT = auto()
    """A filled container is being carried above an object that must not be spilled on."""


@dataclass(frozen=True)
class SafetyDecision(ControlDecision, ABC):
    """A conclusion the contextual-safety theory reached about the current scene."""


@dataclass(frozen=True)
class EnforceCaution(SafetyDecision, RegimeDecision):
    """Restrict the motion because the scene's semantics make a mistake expensive.

    The decision names its reason rather than its remedy: which constraint tightens is the binding
    policy's business, not the theory's.
    """

    reason: CautionReason
    """What about the scene warrants the restriction."""
