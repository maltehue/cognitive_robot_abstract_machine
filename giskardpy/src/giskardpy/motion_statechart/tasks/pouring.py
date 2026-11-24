"""Tasks that pour liquid from a held container, or fill one container from
another."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, ClassVar

import krrood.symbolic_math.symbolic_math as sm
from cramph.context import ContextExtension
from cramph.data_types import ObservationStateValue
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
)
from krrood_spatial_math.spatial_types import Point3, Vector3
from semantic_digital_twin.physics.equations.pouring_equations import (
    GatedInflowEquation,
    PouringEquation,
    SymbolicFillContext,
    tilt_expression_from_fk,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource
from semantic_digital_twin.world_description.connections import LiquidConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.exceptions import (
    MissingExitSpeedError,
    MissingInflowEquationError,
    NonPositiveClearanceError,
    RootLinkNotWorldRootError,
)
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    DebugExpression,
    MotionNodeArtifacts,
    Task,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.qp.terminal_state_prediction_strategy import CoupledState

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


# %% fill level


class FillDebugValue(StrEnum):
    """What a fill task exposes as debug expressions, each named after the task
    and this value."""

    PREDICTED_FILL = "predicted_fill"
    """The fill the task's row predicts for the end of its window, the joints
    held."""

    FILL_RATE = "fill_rate"
    """The fill rate the task's model gives now, in fill per second."""

@dataclass(eq=False, repr=False, kw_only=True)
class TerminalFillConstraintTask(Task, ABC):
    """Base for tasks that drive a container's predicted terminal fill level to
    a goal.

    Subclasses resolve the fill connection and build the symbolic fill-
    velocity ODE; this base linearizes that ODE into a terminal-state
    prediction constraint and observes success once the fill reaches the
    goal and its rate has settled to zero.
    """

    goal_value: float
    """Target normalized fill level."""

    fill_level_tolerance: float
    """Tolerance around :attr:`goal_value` within which the goal counts as
    reached."""

    outflow_tolerance: float = 0.001
    """Tolerance around zero for the residual fill rate, in normalized fill per
    second."""

    cruise_velocity: float = 0.05
    """The rate the normalized fill level is driven at, in fill per second."""

    stopping_velocity: float | None = None
    """How fast the controller may take back fill that is predicted to arrive
    beyond the goal, in normalized fill per second; ``None`` stops no faster
    than :attr:`cruise_velocity` pours."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """QP constraint weight of the fill row."""

    prediction_duration: float = 1.5
    """How far ahead the fill level is predicted, in seconds.

    Long enough to cover the liquid that keeps flowing after the last
    commanded velocity; it also sets how quickly the tilt reacts to a
    predicted miss and is independent of the controller's prediction
    horizon.
    """

    fill_connection: LiquidConnection = field(init=False)
    """World-resident fill connection resolved while building."""

    fill_velocity_expression: Scalar = field(init=False)
    """Symbolic fill-velocity ODE built while building."""

    _compiled_fill_velocity: CompiledFunction = field(init=False, repr=False)
    """Compiled :attr:`fill_velocity_expression`, evaluated once per tick."""

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        return super().required_context_extensions + (MotionControlContext,)

    @abstractmethod
    def _resolve_fill_connection(self) -> LiquidConnection:
        """Resolves and validates the live fill connection whose DOF position
        the constraint drives.

        :return: The world-resident fill connection.
        """

    @abstractmethod
    def _fill_velocity(self) -> Scalar:
        """Builds the symbolic fill-velocity ODE to linearize;
        :attr:`fill_connection` is resolved.

        :return: Symbolic normalized fill velocity at the current
            operating point.
        """

    @abstractmethod
    def _fill_goal_reached(self, *, fill_level: float) -> bool:
        """
        :param fill_level: The current normalized fill level.
        :return: Whether the fill level has reached the goal in the task's fill
            direction.
        """

    def _compile_against_state(self, *, expression: Scalar) -> CompiledFunction:
        """
        :param expression: An expression of world positions and float variables.
        :return: The expression compiled to read both in place.
        """
        compiled = expression.compile(
            parameters=VariableParameters.from_lists(
                self.context.world.state.position_float_variables,
                self.context.float_variable_data.variables,
            ),
            sparse=False,
        )
        compiled.bind_args_to_memory_view(
            arg_idx=0, numpy_array=self.context.world.state.positions
        )
        self.context.float_variable_data.bind_argument(
            compiled_function=compiled, argument_index=1
        )
        return compiled

    def build_artifacts(self) -> MotionNodeArtifacts:
        """Linearizes the fill ODE into a single terminal-state prediction row
        over :attr:`prediction_duration`.

        Because the row couples earlier velocity decisions to a larger
        share of the predicted change, the optimizer eases off before
        overshooting rather than reacting late.

        :return: The artifacts of this task.
        """
        artifacts = MotionNodeArtifacts()
        self.fill_connection = self._resolve_fill_connection()
        self.fill_velocity_expression = self._fill_velocity()
        self._compiled_fill_velocity = self._compile_against_state(
            expression=self.fill_velocity_expression
        )
        row = artifacts.constraints.add_terminal_state_prediction_constraint(
            name=str(self.fill_connection.name),
            state_velocity=self.fill_velocity_expression,
            state_variable=self.fill_connection.dof.variables.position,
            goal_value=self.goal_value,
            quadratic_weight=self.weight,
            cruise_velocity=self.cruise_velocity,
            prediction_duration=self.prediction_duration,
            stopping_velocity=self.stopping_velocity,
            coupled_states=self._coupled_states(),
        )
        controller_config = self.context.require_extension(
            extension_type=MotionControlContext
        ).qp_controller_config
        artifacts.debug_expressions.extend(
            [
                DebugExpression(
                    name=f"{self.name}/{FillDebugValue.PREDICTED_FILL}",
                    expression=row.predicted_value(controller_config),
                ),
                DebugExpression(
                    name=f"{self.name}/{FillDebugValue.FILL_RATE}",
                    expression=self.fill_velocity_expression,
                ),
            ]
        )
        return artifacts

    def _coupled_states(self) -> list[CoupledState]:
        """
        :return: The passive fill levels the fill velocity depends on that keep changing
            over the prediction window; none, unless a task predicts them.
        """
        return []

    def on_tick(self) -> ObservationStateValue | None:
        """
        :return: True once the fill reaches the goal and the modelled fill rate has
            settled to zero.
        """
        fill_level = float(self.fill_connection.position)
        fill_rate = float(self._compiled_fill_velocity.evaluate()[0])
        rate_settled = -self.outflow_tolerance < fill_rate < self.outflow_tolerance
        if rate_settled and self._fill_goal_reached(fill_level=fill_level):
            return ObservationStateValue.TRUE
        return None


@dataclass(eq=False, repr=False, kw_only=True)
class PouringTask(TerminalFillConstraintTask):
    """
    Tilts a container the robot holds so its own fill level drains toward
    :attr:`goal_value`; the pouring ODE couples the controlled tilt to the passive fill
    DOF.
    """

    fill_equation: PouringEquation
    """Pouring ODE coupling tilt to the fill-level DOF."""

    fill_connection: LiquidConnection
    """Virtual DOF whose position encodes the fill level in [0, 1]."""

    root_link: Body
    """Root of the kinematic chain the cup tilt is derived along; must be the
    world root."""

    tip_link: Body
    """Tip of the kinematic chain, the cup body."""

    def _resolve_fill_connection(self) -> LiquidConnection:
        """
        :raises RootLinkNotWorldRootError: if ``root_link`` is not the world root, since
            the tilt expression is only valid relative to the vertical world-root frame.
        """
        world = self.context.world
        if self.root_link is not world.root:
            raise RootLinkNotWorldRootError(
                node=self, root_link=self.root_link, world_root=world.root
            )
        return world.get_connection(
            self.fill_connection.parent, self.fill_connection.child
        )

    def _fill_velocity(self) -> Scalar:
        root_T_tip = self.context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        return self.fill_equation.symbolic_velocity(
            SymbolicFillContext(
                tilt_expression=tilt_expression_from_fk(root_T_tip),
                fill_position=self.fill_connection.dof.variables.position,
            )
        )

    def _fill_goal_reached(self, *, fill_level: float) -> bool:
        return fill_level <= self.goal_value + self.fill_level_tolerance


@dataclass(eq=False, repr=False, kw_only=True)
class FillByTransferTask(TerminalFillConstraintTask):
    """Fills a receiver by tilting a separate source container.

    Unlike :class:`PouringTask`, the controlled degrees of freedom (the
    arm holding the source) do not belong to the container whose fill
    level is the goal. The receiver's inflow ODE depends symbolically on
    the source arm configuration through the source outflow, so driving
    the receiver's predicted terminal fill toward the goal makes the
    optimizer tilt the source.

    The transfer gate scales the predicted fill but is held as a value
    read once per tick, outside the row's gradient: its logistics never
    go flat, so a row that differentiated them would keep lifting the
    source to open the height gate a little more for the whole pour.
    Placing the source over the receiver is left to
    :class:`KeepProjectileInReceiver` and
    :class:`KeepSourceRimAboveReceiverRim`.
    """

    receiver: HasFillLevel
    """The container whose fill level is driven up to :attr:`goal_value`."""

    couple_source_fill: bool = True
    """Whether the prediction lets the source's fill level drain along with the
    receiver's, instead of holding the source's current outflow over the whole
    window."""

    _gate_value: FloatVariable = field(init=False, repr=False)
    """The transfer gate's value at the last tick, standing in for the symbolic
    gate in the fill velocity."""

    _compiled_gate: CompiledFunction = field(init=False, repr=False)
    """The symbolic transfer gate, compiled to be read once per tick."""

    def set_up(self) -> None:
        """Register the variable the transfer gate's value is read into each
        tick."""
        super().set_up()
        self._gate_value = FloatVariable(name=f"{self.name}_gate")
        self.context.float_variable_data.register_expression(
            expression=self._gate_value
        )

    def _resolve_fill_connection(self) -> LiquidConnection:
        """
        :raises MissingInflowEquationError: if the receiver has no inflow equation,
            meaning ``receive_outflow_from`` was not called to couple it to a source.
        """
        world = self.context.world
        self.receiver.ensure_inflow_coupling(world)
        fill_connection = world.get_connection(
            self.receiver.fill_connection.parent, self.receiver.fill_connection.child
        )
        if fill_connection.inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        return fill_connection

    def _fill_velocity(self) -> Scalar:
        inflow_equation = self.fill_connection.inflow_equation
        if not isinstance(inflow_equation, GatedInflowEquation):
            return inflow_equation.symbolic_velocity(self.fill_connection)
        self._compiled_gate = self._compile_against_state(
            expression=inflow_equation.gate
        )
        self._read_gate()
        return self._gate_value * inflow_equation.ungated_velocity(self.fill_connection)

    def _coupled_states(self) -> list[CoupledState]:
        """
        :return: The fill levels of the containers feeding the receiver, each changing by
            its own fill equations, so a source that empties is not predicted to pour on
            at its current rate.
        """
        if not self.couple_source_fill:
            return []
        rate_variables = {
            variable.id for variable in self.fill_velocity_expression.free_variables()
        }
        return [
            CoupledState(
                variable=connection.raw_dof.variables.position,
                velocity=connection.symbolic_fill_velocity(),
            )
            for connection in self.context.world.get_connections_by_type(
                LiquidConnection
            )
            if connection is not self.fill_connection
            and connection.raw_dof.variables.position.id in rate_variables
        ]

    def _read_gate(self) -> None:
        """Store the transfer gate's current value for the fill velocity to
        read."""
        self.context.float_variable_data.set_value(
            expression=self._gate_value,
            value=float(self._compiled_gate.evaluate()[0]),
        )

    def on_tick(self) -> ObservationStateValue | None:
        """Reads the transfer gate anew, then observes the fill as the base
        task does."""
        if isinstance(self.fill_connection.inflow_equation, GatedInflowEquation):
            self._read_gate()
        return super().on_tick()

    def _fill_goal_reached(self, *, fill_level: float) -> bool:
        return fill_level >= self.goal_value - self.fill_level_tolerance


# %% pour stream


@dataclass(eq=False, repr=False, kw_only=True)
class PourStreamTask(Task, ABC):
    """A task on the stream a source pours into a receiver, which leaves the
    source at its exit point and lands on the receiver's opening plane."""

    receiver: HasFillLevel
    """The container poured into; must already be coupled via
    ``receive_outflow_from``."""

    source: LiquidSource
    """The liquid source being poured from."""

    def landing_point(self, *, world: World) -> Point3:
        """
        :param world: The world providing the forward kinematics.
        :return: Where the stream lands on the receiver's opening plane, thrown at the
            source's live exit speed or, without one, at the receiver's nominal one.
        :raises MissingInflowEquationError: if the receiver is not coupled to a source.
        :raises MissingExitSpeedError: if neither the source nor the receiver's inflow
            equation provides an exit speed.
        """
        self.receiver.ensure_inflow_coupling(world)
        inflow_equation = self.receiver.fill_connection.inflow_equation
        if inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        exit_speed = self.source.current_outflow_velocity(world)
        if exit_speed is None:
            if not isinstance(inflow_equation, GatedInflowEquation):
                raise MissingExitSpeedError(node=self)
            exit_speed = inflow_equation.exit_speed
        return self.receiver.projectile_landing_point(self.source, world, exit_speed)


@dataclass(eq=False, repr=False, kw_only=True)
class KeepProjectileInReceiver(PourStreamTask, ConvergingTask):
    """Positions the source so the poured liquid's projectile lands in the
    receiver's opening.

    Drives the predicted landing point of the source's pour toward the
    receiver's opening centre, so as the source tilts the optimizer
    moves the gripper to keep the liquid landing inside the receiver.
    Its error is the distance between the two.
    """

    cruise_velocity: float = 0.2
    """The speed the landing point is moved at, in metres per second."""

    threshold: float = 0.02
    """Distance within which the landing point counts as inside the opening, in
    metres."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """QP constraint weight of the landing-point goal."""

    EXIT_POINT_COLOR: ClassVar[Color] = Color.from_rgba(
        red=0.0, green=0.6, blue=1.0, opacity=1.0
    )
    """Color of the exit-point marker."""

    LANDING_POINT_COLOR: ClassVar[Color] = Color.from_rgba(
        red=1.0, green=0.0, blue=0.0, opacity=1.0
    )
    """Color of the landing-point marker."""

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The constraint driving the pour's landing point to the receiver's
            opening, with the distance between them as error.
        """
        world = self.context.world
        artifacts = MotionNodeArtifacts()
        landing_point = self.landing_point(world=world)
        receiver_opening = self.receiver.opening_point(world)
        artifacts.geometry.add_point_goal_constraints(
            name=f"{self.receiver.root.name}_projectile",
            frame_P_goal=receiver_opening,
            frame_P_current=landing_point,
            cruise_velocity=self.cruise_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.debug_expressions.extend(
            [
                DebugExpression(
                    name=f"{self.name}/exit",
                    expression=self.source.liquid_exit_point(world),
                    color=self.EXIT_POINT_COLOR,
                ),
                DebugExpression(
                    name=f"{self.name}/landing",
                    expression=landing_point,
                    color=self.LANDING_POINT_COLOR,
                ),
            ]
        )
        artifacts.error = receiver_opening.euclidean_distance(landing_point)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class KeepSourceRimAboveReceiverRim(Task):
    """Keeps the pouring source's lip within a clearance band above the
    receiver's rim.

    The lip is derived from the live forward kinematics, so the
    constraint accounts for the lip descending as the source tilts, and
    the clearance is a true rim-to-rim gap rather than an offset on the
    cup origins.
    """

    receiver: HasFillLevel
    """The container whose rim the source's lip must stay above."""

    source: LiquidSource
    """The pouring source whose lip must stay above the receiver's rim."""

    minimum_clearance: float = 0.07
    """Lower end of the clearance band, in metres.

    Must be positive: the bound is enforced softly, so a band reaching down to zero
    would let the lip settle below the receiver's rim.
    """

    clearance_band: float = 0.05
    """Width of the clearance band above :attr:`minimum_clearance`, in metres.

    A band rather than a one-sided bound keeps the problem well conditioned: it is the
    only constraint pinning the source's height, because the landing point that
    :class:`KeepProjectileInReceiver` aims lies in the receiver's opening plane by
    construction and so carries no vertical error.
    """

    weight: float = DefaultWeights.WEIGHT_MAXIMUM
    """QP constraint weight of the clearance.

    The fill task pulls the lip down along the same direction this task
    holds it, so the clearance needs the top tier to keep its floor
    against it.
    """

    cruise_velocity: float = 0.2
    """The vertical speed the clearance is restored at, in metres per
    second."""

    @property
    def maximum_clearance(self) -> float:
        """Upper end of the clearance band, in metres."""
        return self.minimum_clearance + self.clearance_band

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The constraint keeping the source's lip in the band above the
            receiver's rim, observing whether it is inside the band.
        :raises NonPositiveClearanceError: if the band does not lie entirely above the
            rim.
        """
        if self.minimum_clearance <= 0.0:
            raise NonPositiveClearanceError(
                node=self, minimum_clearance=self.minimum_clearance
            )
        world = self.context.world
        artifacts = MotionNodeArtifacts()
        source_lip = self.source.liquid_exit_point(world)
        receiver_rim = self.receiver.opening_point(world)
        clearance = (source_lip - receiver_rim) @ Vector3.Z()
        artifacts.constraints.add_inequality_constraint(
            cruise_velocity=self.cruise_velocity,
            velocity_scale=self.cruise_velocity,
            lower_error=self.minimum_clearance - clearance,
            upper_error=self.maximum_clearance - clearance,
            quadratic_weight=self.weight,
            task_expression=clearance,
            name=f"{self.name}_clearance",
        )
        artifacts.observation = sm.logic_and(
            sm.if_less_eq(clearance, self.maximum_clearance, 1, 0),
            sm.if_greater_eq(clearance, self.minimum_clearance, 1, 0),
        )
        return artifacts
