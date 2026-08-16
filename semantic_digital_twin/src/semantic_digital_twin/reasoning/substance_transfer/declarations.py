"""The constraints a substance transfer requires, as declarations.

These are the transfer theory's statement of what the controller must enforce for it: aiming the
pour, keeping the rims clear, driving the receiver's fill to a runtime-supplied goal, returning the
source upright, and aborting on defeat. Each names the decision of the theory that gates it, so the
chart assembled from them enacts exactly the theory's vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Tuple

from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
    MotionAbortDeclaration,
    ParameterChannel,
)
from semantic_digital_twin.reasoning.substance_transfer.decisions import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
    RetargetFillLevel,
)


@dataclass(frozen=True)
class TransferPairDeclaration(ConstraintDeclaration):
    """Base for declarations about one source/receiver pair, both named."""

    source_name: str = field(kw_only=True)
    """Name of the annotation substance leaves."""

    receiver_name: str = field(kw_only=True)
    """Name of the annotation substance enters."""


@dataclass(frozen=True)
class AimedTransferDeclaration(TransferPairDeclaration):
    """Keep the pour's predicted landing point inside the receiver's opening."""


@dataclass(frozen=True)
class RimClearanceDeclaration(TransferPairDeclaration):
    """Keep the source's pouring lip above the receiver's rim."""

    minimum_clearance: float = field(default=0.08, kw_only=True)
    """Lower bound on the lip-above-rim clearance, in metres."""


@dataclass(frozen=True)
class TransferQuantityDeclaration(TransferPairDeclaration):
    """Drive the receiver's fill to a goal the theory supplies at runtime."""

    fill_level_tolerance: float = field(default=0.05, kw_only=True)
    """Band around the supplied goal within which the transfer counts as done."""


@dataclass(frozen=True)
class ReturnUprightDeclaration(ConstraintDeclaration):
    """Return a named container to upright, closing its pour."""

    subject_name: str = field(kw_only=True)
    """Name of the annotation whose body is returned upright."""


def transfer_constraint_declarations(
    source_name: str,
    receiver_name: str,
    minimum_rim_clearance: float = 0.08,
    fill_level_tolerance: float = 0.05,
) -> Tuple[ConstraintDeclaration, ...]:
    """The full set of constraints a substance transfer declares.

    :param source_name: Name of the annotation substance leaves.
    :param receiver_name: Name of the annotation substance enters.
    :param minimum_rim_clearance: Lower bound on the lip-above-rim clearance, in metres.
    :param fill_level_tolerance: Band around the runtime goal within which the transfer counts as
        done.
    :return: The declarations, ready to attach to the transfer theory.
    """
    return (
        AimedTransferDeclaration(
            identifier="aim",
            source_name=source_name,
            receiver_name=receiver_name,
            gating_decision_type=AlignSourceOverReceiver,
        ),
        RimClearanceDeclaration(
            identifier="rim_clearance",
            source_name=source_name,
            receiver_name=receiver_name,
            minimum_clearance=minimum_rim_clearance,
            gating_decision_type=AlignSourceOverReceiver,
        ),
        TransferQuantityDeclaration(
            identifier="quantity",
            source_name=source_name,
            receiver_name=receiver_name,
            fill_level_tolerance=fill_level_tolerance,
            gating_decision_type=PourIntoReceiver,
            parameter_channel=ParameterChannel(
                decision_type=RetargetFillLevel, attribute_name="goal_fill_level"
            ),
        ),
        ReturnUprightDeclaration(
            identifier="return_upright",
            subject_name=source_name,
            gating_decision_type=ConcludeTransfer,
        ),
        MotionAbortDeclaration(
            identifier="abort",
            reason="the transfer's affordance was defeated",
            gating_decision_type=AbandonTransfer,
        ),
    )


TRANSFER_DECLARATION_KINDS = {
    "aimed_transfer": AimedTransferDeclaration,
    "rim_clearance": RimClearanceDeclaration,
    "transfer_quantity": TransferQuantityDeclaration,
    "return_upright": ReturnUprightDeclaration,
}
"""The transfer domain's declaration kinds, by the name a specification declares them under."""
