"""
Grounds the safety-relevant state of a carried container in the twin.

The geometry here is deliberately coarse — is the container over that object, is it
tilted — because the facts a safety theory needs are qualitative. What matters is that
they come from the twin's semantics and the live kinematics rather than from a physical
model, which is what makes this grounding replaceable by perception without touching the
theory above it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from typing_extensions import List, Optional, Sequence

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import CompiledFunction, VariableParameters

from semantic_digital_twin.reasoning.contextual_safety.situation import SafetySituation
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    SituationGrounding,
)
from semantic_digital_twin.semantic_annotations.mixins import LiquidSource
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class SafetySituationGrounding(SituationGrounding[SafetySituation]):
    """
    Produces the safety theory's situation for one carried container.
    """

    carried_container: LiquidSource
    """The container the robot is holding."""

    sensitive_bodies: List[Body]
    """
    Bodies the twin marks as not-to-be-spilled-on.

    In a populated twin these come from the scene's semantics; the grounding takes them
    as an argument so what counts as sensitive is a property of the world rather than of
    this class.
    """

    contents_threshold: float = field(default=1e-3, kw_only=True)
    """
    Fill level above which the container counts as holding something spillable.
    """

    tilt_threshold: float = field(default=math.radians(15.0), kw_only=True)
    """
    Tilt angle above which contents count as leaving the container, in radians.
    """

    footprint_radius: float = field(default=0.25, kw_only=True)
    """
    Horizontal radius around a sensitive body that counts as being above it, in metres.
    """

    _tilt: Optional[CompiledFunction] = field(default=None, init=False, repr=False)
    """
    Compiled tilt angle of the carried container, in radians.
    """

    _horizontal_distances: List[CompiledFunction] = field(
        default_factory=list, init=False, repr=False
    )
    """
    Compiled horizontal distances from the container's lip to each sensitive body.
    """

    def ground(self, world: World) -> Sequence[SafetySituation]:
        """
        Grounds the current world state into a single safety situation.

        :param world: The world the container and the sensitive bodies live in.
        :return: A one-element sequence holding the carried container's situation.
        """
        self._compile_expressions(world)
        return [
            SafetySituation(
                carried_container=self.carried_container,
                holds_contents=self.carried_container.fill_level
                > self.contents_threshold,
                is_pouring_out=abs(float(self._tilt.evaluate()[0]))
                >= self.tilt_threshold,
                above_sensitive_object=any(
                    float(distance.evaluate()[0]) <= self.footprint_radius
                    for distance in self._horizontal_distances
                ),
            )
        ]

    def _compile_expressions(self, world: World) -> None:
        """
        Compiles the tilt and proximity expressions once, on the first grounding.

        :param world: The world providing the forward kinematics.
        """
        if self._tilt is not None:
            return
        self._tilt = self._bound_to_world_state(
            self.carried_container.pour_tilt_expression, world
        )
        container_lip = self.carried_container.liquid_exit_point(world)
        for body in self.sensitive_bodies:
            body_origin = world.compose_forward_kinematics_expression(
                world.root, body
            ).to_position()
            self._horizontal_distances.append(
                self._bound_to_world_state(
                    sm.sqrt(
                        (container_lip.x - body_origin.x) ** 2
                        + (container_lip.y - body_origin.y) ** 2
                    ),
                    world,
                )
            )

    @staticmethod
    def _bound_to_world_state(expression: sm.Scalar, world: World) -> CompiledFunction:
        """
        Compiles an expression so evaluating it reads the world's live state.

        :param expression: The symbolic expression to compile.
        :param world: The world whose state the expression reads.
        :return: The compiled function, bound to the world's position array.
        """
        compiled = expression.compile(
            parameters=VariableParameters.from_lists(
                world.state.position_float_variables
            ),
            sparse=False,
        )
        compiled.bind_args_to_memory_view(0, world.state.positions)
        return compiled
