"""Matching camera calibration and receiver reporting for supported cups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from experiments.real_pouring.poured_material import PouredMaterial
from experiments.real_pouring.receiving_cup import ReceivingCup, ReceiverFillReport
from experiments.real_pouring.receiver_evidence import (
    ReceiverEvidence,
    InvalidReceiverEvidenceError,
)
from experiments.real_pouring.wall_fill import (
    WallFillCalibration,
    InvalidWallCalibrationError,
)
from krrood.adapters.json_serializer import from_json

# %% calibrated configurations


class PerceptionCalibration(StrEnum):
    """Packaged calibrations for a particular receiver and material."""

    TRANSLUCENT_CORN = "translucent_corn_wall.json"
    """
    Wall model fitted to the development pours of October 2026, with the receiver
    modelled standing on the table top (at 0.875 m in the world root).
    """

    RECEIVER_EVIDENCE = "translucent_corn_evidence.json"
    """Measurement error scales from cross-validating :attr:`TRANSLUCENT_CORN`."""

    TRANSLUCENT_CORN_LIFTED_RECEIVER = "translucent_corn_wall_lifted_receiver.json"
    """
    Wall model for the receiver modelled 5 cm above the table top (at 0.925 m), as in
    every bag recorded before 2026-10-25 14:31.

    .. note:: The wall reading depends on where the receiver is modelled: replaying such
        a bag with :attr:`TRANSLUCENT_CORN` reads about 0.09 low. To analyse those bags,
        have :meth:`PourPerception.for_cup` read this file and
        :attr:`RECEIVER_EVIDENCE_LIFTED_RECEIVER` instead.
    """

    RECEIVER_EVIDENCE_LIFTED_RECEIVER = "translucent_corn_evidence_lifted_receiver.json"
    """Measurement error scales from cross-validating :attr:`TRANSLUCENT_CORN_LIFTED_RECEIVER`."""


@dataclass(frozen=True)
class PourPerception:
    """Pair a receiver measurement with the reporting policy that consumes it."""

    fill_report: ReceiverFillReport
    """How measured contents are reported to the controller."""

    wall_calibration: WallFillCalibration | None = None
    """Cup-specific wall calibration, when available for the material."""

    receiver_evidence: ReceiverEvidence | None = None
    """Uncertainty and timing used by the internal receiver likelihood."""

    fill_tolerance: float = 0.01
    """Requested receiver-fill tolerance, in fractions of a full receiver."""

    @classmethod
    def for_cup(
        cls, receiver: ReceivingCup, material: PouredMaterial
    ) -> PourPerception:
        """Select calibrated wall reporting only for its supported cup and material."""
        if (
            receiver is not ReceivingCup.TRANSLUCENT_CUP
            or material is not PouredMaterial.CORN
        ):
            return cls(receiver.fill_report)
        calibration = from_json(
            json.loads(
                Path(__file__)
                .with_name(PerceptionCalibration.TRANSLUCENT_CORN)
                .read_text()
            )
        )
        if not isinstance(calibration, WallFillCalibration):
            raise InvalidWallCalibrationError()
        evidence = from_json(
            json.loads(
                Path(__file__)
                .with_name(PerceptionCalibration.RECEIVER_EVIDENCE)
                .read_text()
            )
        )
        if not isinstance(evidence, ReceiverEvidence):
            raise InvalidReceiverEvidenceError()
        return cls(ReceiverFillReport.MEASURED, calibration, evidence)
