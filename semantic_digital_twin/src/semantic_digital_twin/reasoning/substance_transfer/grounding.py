"""
Grounds a source/receiver pair in the twin into a :class:`TransferSituation`.

The geometry the theory reasons about already exists symbolically on the container
annotations — where the pour leaves, where it would land, how far the source is tilted —
so grounding compiles those expressions once and evaluates them per cycle rather than
reimplementing the geometry numerically. Thresholds turn each continuous quantity into
the qualitative fact the rules use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from typing_extensions import Optional, Sequence

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import CompiledFunction, VariableParameters

from semantic_digital_twin.physics.equations.pouring_equations import (
    GatedInflowEquation,
)
from semantic_digital_twin.reasoning.substance_transfer.exceptions import (
    MissingExitSpeedForGroundingError,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    SituationGrounding,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource
from semantic_digital_twin.world import World


@dataclass
class TransferSituationGrounding(SituationGrounding[TransferSituation]):
    """
    Produces the transfer theory's situation for one coupled source/receiver pair.
    """

    source: LiquidSource
    """The container substance leaves."""

    receiver: HasFillLevel
    """
    The container substance enters; must already be coupled to the source.
    """

    requested_fill_level: float
    """The fill level the transfer was asked to reach, in ``[0, 1]``."""

    fill_level_tolerance: float = field(default=0.05, kw_only=True)
    """
    Band around :attr:`requested_fill_level` within which the goal counts as nearly
    reached.

    The goal itself counts as reached only at or above :attr:`requested_fill_level`:
    withdrawing the transfer regime while the fill is still short of its target would
    remove the terminal-state constraint that eases the pour off, and the substance
    already in flight would overshoot.
    """

    tilt_threshold: float = field(default=math.radians(15.0), kw_only=True)
    """
    Tilt angle above which the source counts as tilted, in radians.
    """

    near_distance: float = field(default=0.5, kw_only=True)
    """
    Horizontal distance within which the source counts as near the receiver, in metres.
    """

    inflow_threshold: float = field(default=1e-4, kw_only=True)
    """
    Fill rate above which substance counts as measurably entering the receiver.
    """

    overflow_margin: float = field(default=1e-3, kw_only=True)
    """
    Distance below a full receiver at which it counts as overflowing.
    """

    _clearance: Optional[CompiledFunction] = field(default=None, init=False, repr=False)
    """
    Compiled source-lip-above-receiver-rim height, in metres.
    """

    _landing_distance: Optional[CompiledFunction] = field(
        default=None, init=False, repr=False
    )
    """
    Compiled horizontal distance from the predicted landing point to the opening centre.
    """

    _horizontal_distance: Optional[CompiledFunction] = field(
        default=None, init=False, repr=False
    )
    """
    Compiled horizontal distance between the source's lip and the receiver's opening.
    """

    _offset_forward: Optional[CompiledFunction] = field(
        default=None, init=False, repr=False
    )
    """
    Compiled signed lip-to-opening distance along world x, in metres.
    """

    _offset_left: Optional[CompiledFunction] = field(
        default=None, init=False, repr=False
    )
    """
    Compiled signed lip-to-opening distance along world y, in metres.
    """

    _tilt: Optional[CompiledFunction] = field(default=None, init=False, repr=False)
    """
    Compiled tilt angle of the source, in radians.
    """

    _inflow_rate: Optional[CompiledFunction] = field(
        default=None, init=False, repr=False
    )
    """
    Compiled fill rate of the receiver, in fill units per second.
    """

    def ground(self, world: World) -> Sequence[TransferSituation]:
        """
        Grounds the current world state into a single transfer situation.

        :param world: The world the containers live in.
        :return: A one-element sequence holding the pair's situation.
        """
        self._compile_expressions(world)
        receiver_fill_level = self.receiver.fill_level
        landing_distance = float(self._landing_distance.evaluate()[0])
        return [
            TransferSituation(
                source=self.source,
                receiver=self.receiver,
                requested_fill_level=self.requested_fill_level,
                receiver_fill_level=receiver_fill_level,
                near=float(self._horizontal_distance.evaluate()[0])
                <= self.near_distance,
                source_above_receiver=float(self._clearance.evaluate()[0]) > 0.0,
                opening_within=landing_distance < self.receiver.opening_radius,
                is_tilted=abs(float(self._tilt.evaluate()[0])) >= self.tilt_threshold,
                pours_to=float(self._inflow_rate.evaluate()[0]) > self.inflow_threshold,
                goal_reached=receiver_fill_level >= self.requested_fill_level,
                almost_goal_reached=receiver_fill_level
                >= self.requested_fill_level - self.fill_level_tolerance,
                receiver_offset_forward=float(self._offset_forward.evaluate()[0]),
                receiver_offset_left=float(self._offset_left.evaluate()[0]),
                receiver_overflowing=receiver_fill_level >= 1.0 - self.overflow_margin,
            )
        ]

    def _compile_expressions(self, world: World) -> None:
        """
        Compiles the geometry and flow expressions once, on the first grounding.

        :param world: The world providing the forward kinematics.
        """
        if self._clearance is not None:
            return
        self.receiver.ensure_inflow_coupling(world)
        source_lip = self.source.liquid_exit_point(world)
        receiver_opening = self.receiver.opening_point(world)
        landing_point = self.receiver.projectile_landing_point(
            self.source, world, self._exit_speed(world)
        )
        self._clearance = self._bound_to_world_state(
            source_lip.z - receiver_opening.z, world
        )
        self._landing_distance = self._bound_to_world_state(
            sm.sqrt(
                (landing_point.x - receiver_opening.x) ** 2
                + (landing_point.y - receiver_opening.y) ** 2
            ),
            world,
        )
        self._horizontal_distance = self._bound_to_world_state(
            sm.sqrt(
                (source_lip.x - receiver_opening.x) ** 2
                + (source_lip.y - receiver_opening.y) ** 2
            ),
            world,
        )
        self._offset_forward = self._bound_to_world_state(
            receiver_opening.x - source_lip.x, world
        )
        self._offset_left = self._bound_to_world_state(
            receiver_opening.y - source_lip.y, world
        )
        self._tilt = self._bound_to_world_state(self.source.pour_tilt_expression, world)
        inflow_equation = self.receiver.fill_connection.inflow_equation
        self._inflow_rate = self._bound_to_world_state(
            inflow_equation.symbolic_velocity(self.receiver.fill_connection), world
        )

    @staticmethod
    def _bound_to_world_state(expression: sm.Scalar, world: World) -> CompiledFunction:
        """
        Compiles an expression so evaluating it reads the world's live state.

        Binding the state array as the compiled function's argument buffer means each
        cycle's evaluation costs one call rather than a recompilation, which is what
        keeps grounding cheap enough to run on the control thread.

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

    def _exit_speed(self, world: World):
        """
        The speed substance leaves the source at, for the projectile prediction.

        :param world: The world providing the forward kinematics.
        :return: Symbolic exit speed in metres per second.
        """
        exit_speed = self.source.current_outflow_velocity(world)
        if exit_speed is not None:
            return exit_speed
        inflow_equation = self.receiver.fill_connection.inflow_equation
        if isinstance(inflow_equation, GatedInflowEquation):
            return inflow_equation.exit_speed
        raise MissingExitSpeedForGroundingError(source=self.source)
