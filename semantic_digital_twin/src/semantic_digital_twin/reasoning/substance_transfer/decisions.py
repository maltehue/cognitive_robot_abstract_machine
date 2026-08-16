"""The substance-transfer theory's decision vocabulary.

Each decision is addressed to exactly one of the framework's two write channels, expressed as a
type rather than a convention: a regime decision gates constraints, a parameter decision supplies a
value to a registered float variable. The vocabulary deliberately contains no direction primitives
(move-left, increase-tilt); which way to move is what the optimizer solves once the right
constraints are active.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    ParameterDecision,
    RegimeDecision,
)


class TransferDefeat(Enum):
    """Why a transfer was defeated, recorded on :class:`AbandonTransfer`."""

    RECEIVER_WOULD_OVERFLOW = auto()
    """The receiver is at capacity, so further transfer would spill."""

    TRANSFER_STALLED = auto()
    """The source is tilted but nothing is flowing and the goal is not reached."""


@dataclass(frozen=True)
class TransferDecision(ControlDecision, ABC):
    """A conclusion the substance-transfer theory reached for one source/receiver pair."""


@dataclass(frozen=True)
class TransferRegimeDecision(TransferDecision, RegimeDecision, ABC):
    """A transfer decision that gates constraints rather than supplying a value."""


@dataclass(frozen=True)
class TransferParameterDecision(TransferDecision, ParameterDecision, ABC):
    """A transfer decision that supplies a value to a registered float variable."""


# %% regime activation (channel 1)


@dataclass(frozen=True)
class AlignSourceOverReceiver(TransferRegimeDecision):
    """Hold the pour geometry; gates the landing-point and rim-clearance constraints."""


@dataclass(frozen=True)
class PourIntoReceiver(TransferRegimeDecision):
    """Transfer substance now; gates the terminal-fill task driving the receiver to its goal."""


@dataclass(frozen=True)
class ConcludeTransfer(TransferRegimeDecision):
    """Finish a successful transfer; the pour regime ends and the source returns upright."""


@dataclass(frozen=True)
class AbandonTransfer(TransferRegimeDecision):
    """Abort the transfer because its affordance was defeated."""

    defeat: TransferDefeat
    """The defeater that removed the pour affordance."""


# %% parameterization (channel 2)


@dataclass(frozen=True)
class RetargetFillLevel(TransferParameterDecision):
    """Set the receiver's goal fill level, written into the terminal-fill goal variable.

    The goal is the reasoner's to set rather than a constant baked into the task at build time,
    which is what makes the numeric target part of the symbolic decision rather than a parameter of
    the controller.
    """

    goal_fill_level: float
    """Normalized target fill level in ``[0, 1]``."""
