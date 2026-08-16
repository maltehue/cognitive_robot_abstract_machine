"""
Exceptions raised while grounding or reasoning about a substance transfer.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException

from semantic_digital_twin.semantic_annotations.mixins import LiquidSource


@dataclass
class SubstanceTransferError(DataclassException):
    """
    Base for errors in the substance-transfer theory and its grounding.
    """


@dataclass
class MissingExitSpeedForGroundingError(SubstanceTransferError):
    """
    Raised when the speed substance leaves the source cannot be determined.

    Without it the projectile arc, and therefore whether the pour would land in the
    receiver, cannot be predicted.
    """

    source: LiquidSource
    """The source whose outflow speed is unavailable."""

    def error_message(self) -> str:
        return (
            f"source {self.source.name} has no outflow velocity and its receiver's inflow is not "
            f"gated, so the pour's landing point cannot be predicted"
        )

    def suggest_correction(self) -> str:
        return (
            "Couple the receiver to the source with receive_outflow_from so the transfer gate "
            "supplies an exit speed."
        )
