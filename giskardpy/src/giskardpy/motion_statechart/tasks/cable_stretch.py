from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar

import krrood.symbolic_math.symbolic_math as sm
from krrood.exceptions import DataclassException
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    NodeArtifacts,
    Task,
)
from giskardpy.qp.constraint import LargeNumber
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% exceptions


@dataclass
class NonPositiveRestLengthError(DataclassException):
    """
    Raised when a cable stretch task is given a rest length that is not positive.
    """

    rest_length: float
    """The invalid rest length that was requested."""

    def error_message(self) -> str:
        return f"A cable rest length must be positive, got {self.rest_length}."

    def suggest_correction(self) -> str:
        return "Pass a rest_length greater than zero."


# %% cable stretch task


@dataclass(eq=False, repr=False)
class MinimizeCableStretch(Task):
    """
    Keep a routed cable within its rest length, minimizing its stretch during a motion.

    The cable is anchored on one body and guided past another. Its routed span is the
    distance between the two bodies, and the stretch is how far that span exceeds the
    cable's rest length. This task constrains the span to stay at or below the rest
    length, so the optimizer trades other objectives against over-stretching the cable.
    While the span stays below the rest length the cable is slack and the task is
    inactive.
    """

    cable_anchor: KinematicStructureEntity = field(kw_only=True)
    """
    Body where the cable is anchored.
    """

    wrist_guide: KinematicStructureEntity = field(kw_only=True)
    """
    Body the cable is guided past.
    """

    rest_length: float = field(kw_only=True)
    """
    Unstretched length of the cable in meters.
    """

    reference_velocity: float = field(default=0.2, kw_only=True)
    """
    Reference velocity for normalization in m/s.
    """

    threshold: float = field(default=0.005, kw_only=True)
    """
    Stretch below which the cable counts as within its rest length, in meters.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    Task priority relative to other tasks.
    """

    SPAN_COLOR: ClassVar[Color] = Color(R=1.0, G=0.5, B=0.0, A=1.0)
    """
    Color of the routed-span debug expression marker (orange).
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Build the constraint that keeps the routed span within the rest length.

        :param context: Provides access to the world model and kinematic expressions.
        :return: Node artifacts holding the stretch constraint and observation.
        :raises NonPositiveRestLengthError: If ``rest_length`` is not positive.
        """
        if self.rest_length <= 0:
            raise NonPositiveRestLengthError(rest_length=self.rest_length)

        artifacts = NodeArtifacts()

        root = context.world.root
        root_P_anchor = context.world.compose_forward_kinematics_expression(
            root, self.cable_anchor
        ).to_position()
        root_P_guide = context.world.compose_forward_kinematics_expression(
            root, self.wrist_guide
        ).to_position()
        span = root_P_anchor.euclidean_distance(root_P_guide)

        # The cable can shorten freely (going slack) but must not stretch past its rest
        # length, so only the upper side of the span is bounded.
        artifacts.constraints.add_inequality_constraint(
            name="stretch",
            task_expression=span,
            lower_error=sm.Scalar(-LargeNumber),
            upper_error=sm.Scalar(self.rest_length) - span,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )

        stretch = sm.max(sm.Scalar(0), span - self.rest_length)
        artifacts.observation = stretch < self.threshold

        artifacts.debug_expressions.append(
            DebugExpression(f"{self.name}/span", span, color=self.SPAN_COLOR)
        )
        return artifacts
