"""
The original theory's motion-primitive vocabulary, kept for the replication arm.

These are the decisions the published system emitted: Boolean directions that a fixed-
gain bridge turns into a task-frame twist. They exist here so the *same* theory can
drive either bridge and the comparison between them isolates the bridge rather than the
reasoner, the scene or the robot.

Nothing in the regime vocabulary corresponds to these. That is the point: a regime
decision says which constraints hold and lets the optimizer solve the direction, whereas
a primitive *is* the direction, and the precision the comparison measures is lost in the
difference.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from semantic_digital_twin.reasoning.substance_transfer.decisions import (
    TransferRegimeDecision,
)


@dataclass(frozen=True)
class MotionPrimitive(TransferRegimeDecision, ABC):
    """
    One Boolean direction the fixed-gain bridge sums into a commanded twist.
    """


@dataclass(frozen=True)
class MoveForward(MotionPrimitive):
    """
    Move the source along world +x.
    """


@dataclass(frozen=True)
class MoveBack(MotionPrimitive):
    """
    Move the source along world −x.
    """


@dataclass(frozen=True)
class MoveLeft(MotionPrimitive):
    """
    Move the source along world +y.
    """


@dataclass(frozen=True)
class MoveRight(MotionPrimitive):
    """
    Move the source along world −y.
    """


@dataclass(frozen=True)
class IncreaseTilt(MotionPrimitive):
    """
    Tilt the source further over, increasing outflow.
    """


@dataclass(frozen=True)
class DecreaseTilt(MotionPrimitive):
    """
    Tilt the source back upright, reducing outflow.
    """
