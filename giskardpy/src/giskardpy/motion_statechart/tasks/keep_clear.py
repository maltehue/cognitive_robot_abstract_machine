"""
Keeping a body horizontally clear of another.

One one-sided constraint on the planar distance between two bodies. Unlike
:class:`~giskardpy.motion_statechart.tasks.feature_functions.DistanceGoal` it adds no
motion-damping rows, so it can run alongside tasks that actively move the subject — it only pushes
back when the clearance is about to be violated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import krrood.symbolic_math.symbolic_math as sm

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.qp.constraint import LargeNumber
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class MaintainHorizontalClearance(Task):
    """
    Keeps the subject at least a minimum planar distance from an obstacle.

    The distance is measured in the root frame's x-y plane, so approaching from above is
    unrestricted; what the constraint forbids is carrying the subject over or beside the
    obstacle closer than the clearance.
    """

    root_link: Body = field(kw_only=True)
    """
    Root of both kinematic chains; the clearance lives in its x-y plane.
    """

    subject_link: Body = field(kw_only=True)
    """
    The body being kept clear.
    """

    obstacle_link: Body = field(kw_only=True)
    """
    The body being kept clear of.
    """

    minimum_clearance: float = field(kw_only=True)
    """
    Lower bound on the planar distance, in metres.
    """

    reference_velocity: float = field(default=0.1, kw_only=True)
    """
    Reference rate of change of the clearance, in metres per second.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the clearance.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the one-sided constraint holding the planar distance above the
        clearance.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        subject_position = context.world.compose_forward_kinematics_expression(
            self.root_link, self.subject_link
        ).to_position()
        obstacle_position = context.world.compose_forward_kinematics_expression(
            self.root_link, self.obstacle_link
        ).to_position()
        planar_distance = sm.sqrt(
            (subject_position.x - obstacle_position.x) ** 2
            + (subject_position.y - obstacle_position.y) ** 2
        )
        artifacts.constraints.add_inequality_constraint(
            name=f"{self.name}_clearance",
            reference_velocity=self.reference_velocity,
            lower_error=self.minimum_clearance - planar_distance,
            upper_error=LargeNumber,
            quadratic_weight=self.weight,
            task_expression=planar_distance,
        )
        artifacts.observation = planar_distance >= self.minimum_clearance
        return artifacts
