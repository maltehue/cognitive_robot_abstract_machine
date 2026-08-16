"""
The catalog entries enforcing the transfer domain's constraint declarations.

These factories are where declaration data meets task construction: they resolve the
declaration's subject names in the world and build the task that enforces it. They live
here rather than in the controller's framework package because they know the domain's
tasks and annotations, which the framework deliberately does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException
from krrood.symbolic_math.symbolic_math import FloatVariable

from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import CancelMotion
from giskardpy.motion_statechart.knowledge_servoing.constraint_catalog import (
    ConstraintCatalog,
    ConstraintInstantiation,
)
from giskardpy.motion_statechart.tasks.align_planes import AlignPlanes
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianVelocityLimit
from giskardpy.motion_statechart.tasks.pouring import (
    FillByTransferTask,
    KeepProjectileInReceiver,
    KeepSourceRimAboveReceiverRim,
)
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    MotionAbortDeclaration,
    ToolSpeedLimitDeclaration,
)
from semantic_digital_twin.reasoning.substance_transfer.declarations import (
    AimedTransferDeclaration,
    ReturnUprightDeclaration,
    RimClearanceDeclaration,
    TransferQuantityDeclaration,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World


@dataclass
class DeclaredMotionAborted(DataclassException):
    """
    Raised by an assembled abort node when its gating decision is concluded.
    """

    reason: str
    """Why the theory aborted the motion."""

    def error_message(self) -> str:
        return f"The motion was aborted because {self.reason}"

    def suggest_correction(self) -> str:
        return (
            "Inspect the decision transcript for the defeater that concluded the abort"
        )


def _aimed_transfer(
    declaration: AimedTransferDeclaration, world: World
) -> ConstraintInstantiation:
    """
    Builds the landing-point constraint for a named source/receiver pair.
    """
    return ConstraintInstantiation(
        node=KeepProjectileInReceiver(
            name=declaration.identifier,
            receiver=world.get_semantic_annotation_by_name(declaration.receiver_name),
            source=world.get_semantic_annotation_by_name(declaration.source_name),
            weight=DefaultWeights.WEIGHT_MAXIMUM,
        )
    )


def _rim_clearance(
    declaration: RimClearanceDeclaration, world: World
) -> ConstraintInstantiation:
    """
    Builds the lip-above-rim clearance for a named source/receiver pair.
    """
    return ConstraintInstantiation(
        node=KeepSourceRimAboveReceiverRim(
            name=declaration.identifier,
            receiver=world.get_semantic_annotation_by_name(declaration.receiver_name),
            source=world.get_semantic_annotation_by_name(declaration.source_name),
            minimum_clearance=declaration.minimum_clearance,
        )
    )


def _transfer_quantity(
    declaration: TransferQuantityDeclaration, world: World
) -> ConstraintInstantiation:
    """
    Builds the terminal-fill task with a runtime-writable goal.
    """
    goal_fill_variable = FloatVariable(f"{declaration.identifier}_goal_fill_level")
    return ConstraintInstantiation(
        node=FillByTransferTask(
            name=declaration.identifier,
            receiver=world.get_semantic_annotation_by_name(declaration.receiver_name),
            goal_value=goal_fill_variable,
            fill_level_tolerance=declaration.fill_level_tolerance,
        ),
        parameter_target=goal_fill_variable,
    )


def _return_upright(
    declaration: ReturnUprightDeclaration, world: World
) -> ConstraintInstantiation:
    """
    Builds the return-to-upright alignment for a named container.
    """
    subject = world.get_semantic_annotation_by_name(declaration.subject_name)
    return ConstraintInstantiation(
        node=AlignPlanes(
            name=declaration.identifier,
            root_link=world.root,
            tip_link=subject.root,
            goal_normal=Vector3.Z(reference_frame=world.root),
            tip_normal=Vector3.Z(reference_frame=subject.root),
        )
    )


def _tool_speed_limit(
    declaration: ToolSpeedLimitDeclaration, world: World
) -> ConstraintInstantiation:
    """
    Builds the translational speed cap on a named annotation's body.
    """
    subject = world.get_semantic_annotation_by_name(declaration.subject_name)
    return ConstraintInstantiation(
        node=CartesianVelocityLimit(
            name=declaration.identifier,
            root_link=world.root,
            tip_link=subject.root,
            max_linear_velocity=declaration.maximum_speed,
        )
    )


def _motion_abort(
    declaration: MotionAbortDeclaration, world: World
) -> ConstraintInstantiation:
    """
    Builds the abort node raising when its gating decision is concluded.
    """
    return ConstraintInstantiation(
        node=CancelMotion(
            name=declaration.identifier,
            exception=DeclaredMotionAborted(reason=declaration.reason),
        )
    )


def build_transfer_catalog() -> ConstraintCatalog:
    """
    The constraint vocabulary the transfer demonstration's theories may declare from.

    :return: The catalog with every transfer-domain kind registered.
    """
    catalog = ConstraintCatalog()
    catalog.register(AimedTransferDeclaration, _aimed_transfer)
    catalog.register(RimClearanceDeclaration, _rim_clearance)
    catalog.register(TransferQuantityDeclaration, _transfer_quantity)
    catalog.register(ReturnUprightDeclaration, _return_upright)
    catalog.register(ToolSpeedLimitDeclaration, _tool_speed_limit)
    catalog.register(MotionAbortDeclaration, _motion_abort)
    return catalog
