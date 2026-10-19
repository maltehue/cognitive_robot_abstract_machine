"""
Uncertainty and timing of receiver measurements used by the hypothesis bank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# %% receiver observation contract


@dataclass
class InvalidReceiverEvidenceError(ValueError):
    """
    Raised for nonfinite or physically invalid receiver evidence settings.
    """


@dataclass(frozen=True)
class ReceiverEvidence:
    """
    Set measurement error scales and separate overlapping camera windows.
    """

    calibrated_noise: float
    """
    Error scale of calibrated measurements, in receiver-fill units.
    """

    geometric_noise: float
    """
    Error scale of geometric fallback measurements, in receiver-fill units.
    """

    delay: float
    """
    Effective observation delay used for comparison with particle history, in seconds.
    """

    interval: float
    """
    Minimum time between receiver likelihood updates, in seconds.
    """

    def __post_init__(self) -> None:
        """
        Require positive finite uncertainties and a finite causal time window.
        """
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.calibrated_noise,
                    self.geometric_noise,
                    self.delay,
                    self.interval,
                )
            )
            or self.calibrated_noise <= 0
            or self.geometric_noise <= 0
            or self.delay < 0
            or self.interval <= 0
        ):
            raise InvalidReceiverEvidenceError()


@dataclass(frozen=True)
class ReceiverObservation:
    """
    A receiver measurement and its uncertainty in bank grain units.
    """

    grains: float
    """
    Measured material already in the receiver.
    """

    noise: float
    """
    Measurement error scale in grains.
    """
