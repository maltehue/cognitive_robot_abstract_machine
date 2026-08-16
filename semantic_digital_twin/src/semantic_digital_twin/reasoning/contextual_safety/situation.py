"""
The facts the contextual-safety theory reasons over.

Unlike the substance-transfer situation, none of these facts comes from a physical
model. They come from what the twin says objects *are* and where they are relative to
one another, which is why this theory needs no effect model and would work unchanged on
perceived facts.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import Situation
from semantic_digital_twin.semantic_annotations.mixins import LiquidSource


@dataclass(frozen=True)
class SafetySituation(Situation):
    """
    The scene's safety-relevant state around one carried container.
    """

    carried_container: LiquidSource
    """
    The container the robot is holding.
    """

    holds_contents: bool
    """
    Whether the carried container currently holds anything that could be spilled.
    """

    is_pouring_out: bool
    """
    Whether the container is tilted far enough for its contents to leave it.
    """

    above_sensitive_object: bool
    """
    Whether a body the twin marks as not-to-be-spilled-on lies below the container.
    """
