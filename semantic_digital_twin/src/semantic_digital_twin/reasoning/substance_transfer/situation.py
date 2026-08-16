"""
The qualitative facts the substance-transfer theory reasons over.

One situation is one source/receiver pair at one instant. Every field is a plain bool or
float resolved at grounding time, so the object the theory classifies is a value, not a
view onto the world: the rules cannot reach back into live state, and the same situation
replayed later yields the same conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import Situation
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource


@dataclass(frozen=True)
class TransferSituation(Situation):
    """
    One source/receiver pair's qualitative state at one control cycle.
    """

    source: LiquidSource
    """
    The container substance leaves; carried so decisions can name their subject.
    """

    receiver: HasFillLevel
    """
    The container substance enters.
    """

    requested_fill_level: float
    """
    The fill level the transfer was asked to reach, in ``[0, 1]``.
    """

    receiver_fill_level: float
    """
    The receiver's current normalized fill level.
    """

    near: bool
    """
    Whether the source is close enough to the receiver for a pour to be possible.
    """

    source_above_receiver: bool
    """
    Whether the source's pouring lip is above the receiver's rim.
    """

    opening_within: bool
    """
    Whether the pour's predicted landing point falls inside the receiver's opening.
    """

    is_tilted: bool
    """
    Whether the source is tilted far enough for substance to leave it.
    """

    pours_to: bool
    """
    Whether substance is measurably entering the receiver.
    """

    goal_reached: bool
    """
    Whether the receiver has reached the requested fill level.
    """

    receiver_overflowing: bool
    """
    Whether the receiver is at capacity, so further transfer would be lost.
    """

    @property
    def spill_risk(self) -> bool:
        """
        Whether substance may leave the source without landing in the receiver.

        This is predictive rather than observed: the analytic world conserves volume, so a spill can
        never be measured, only anticipated from the source being tilted while not aimed.
        """
        return self.is_tilted and not self.opening_within
