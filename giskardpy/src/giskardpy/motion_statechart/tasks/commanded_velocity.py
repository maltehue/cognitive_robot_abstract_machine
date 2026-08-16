"""
Tasks that command a velocity rather than bound one.

The controller's Cartesian tasks all take a goal and let the optimizer choose the velocity that
reaches it. These take the velocity itself, which is what a fixed-gain bridge from symbolic
primitives produces: the reasoner has already decided how fast to move, and the optimizer's only
remaining freedom is how to distribute that motion across the joints.

.. note:: This is the interface the knowledge-servoing comparison measures against, not one to
    prefer. A commanded velocity discards the optimizer's ability to anticipate, which is precisely
    the property the terminal-state prediction row supplies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import FloatVariable

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from semantic_digital_twin.physics.equations.pouring_equations import (
    tilt_expression_from_fk,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass(eq=False, repr=False)
class CommandedTranslationVelocity(Task):
    """
    Drives the tip's translational velocity to a commanded value along each world axis.

    The commanded values are registered float variables, so whatever writes them — here
    a bridge from symbolic primitives — retargets the motion each cycle without
    recompiling.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    """
    Root of the kinematic chain the velocity is expressed in.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    Tip of the kinematic chain whose velocity is commanded.
    """

    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Normalization scale for the commanded velocity, in metres per second.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for each commanded axis.
    """

    commanded_velocity: List[FloatVariable] = field(init=False, default_factory=list)
    """
    The registered x, y and z velocity commands, in metres per second.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates one velocity equality constraint per world axis.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        root_position_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        ).to_position()
        axes = (root_position_tip.x, root_position_tip.y, root_position_tip.z)
        for axis_name, axis_expression in zip("xyz", axes):
            command = FloatVariable(f"{self.name}_commanded_velocity_{axis_name}")
            context.float_variable_data.register_expression(command)
            self.commanded_velocity.append(command)
            artifacts.constraints.add_velocity_eq_constraint(
                name=f"{self.name}_velocity_{axis_name}",
                velocity_goal=command,
                task_expression=axis_expression,
                velocity_limit=self.maximum_velocity,
                quadratic_weight=self.weight,
            )
        return artifacts


@dataclass(eq=False, repr=False)
class CommandedTiltVelocity(Task):
    """
    Drives the rate of change of a container's tilt angle to a commanded value.

    Tilt is the quantity the original theory's tilt primitives act on, so commanding its
    rate is the faithful counterpart of those primitives — as opposed to commanding a
    tilt *goal*, which would hand the optimizer back the anticipation the comparison is
    about.
    """

    root_link: KinematicStructureEntity = field(kw_only=True)
    """
    Root of the kinematic chain; must be the world root for the tilt to be measured
    against vertical.
    """

    tip_link: KinematicStructureEntity = field(kw_only=True)
    """
    The container whose tilt rate is commanded.
    """

    maximum_velocity: float = field(default=0.5, kw_only=True)
    """
    Normalization scale for the commanded tilt rate, in radians per second.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the commanded tilt rate.
    """

    commanded_tilt_rate: FloatVariable = field(init=False, default=None)
    """
    The registered tilt-rate command, in radians per second.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the velocity equality constraint on the tilt angle.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        root_transform_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        self.commanded_tilt_rate = FloatVariable(f"{self.name}_commanded_tilt_rate")
        context.float_variable_data.register_expression(self.commanded_tilt_rate)
        artifacts.constraints.add_velocity_eq_constraint(
            name=f"{self.name}_tilt_rate",
            velocity_goal=self.commanded_tilt_rate,
            task_expression=tilt_expression_from_fk(root_transform_tip),
            velocity_limit=self.maximum_velocity,
            quadratic_weight=self.weight,
        )
        return artifacts
