"""Tasks that pour liquid from a held container, or fill one container from
another."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum

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
from semantic_digital_twin.semantic_annotations.pouring_stream import PouringStream
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel
from semantic_digital_twin.world_description.fill_level_connection import (
    FillLevelConnection,
)
from semantic_digital_twin.world_description.geometry import Color

from giskardpy.motion_statechart.context import MotionControlContext
from giskardpy.motion_statechart.graph_node import (
    ConvergingTask,
    DebugExpression,
    MotionNodeArtifacts,
    Task,
)
from giskardpy.motion_statechart.mixins import (
    HasCruiseVelocityParameter,
    HasWeightParameter,
)
from giskardpy.motion_statechart.tasks.exceptions import (
    MissingFillLevelError,
    MissingSourceError,
    NonPositiveClearanceError,
)
from giskardpy.motion_statechart.weights import DefaultWeights
from giskardpy.qp.terminal_state_prediction_strategy import CoupledState

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
class TerminalFillConstraintTask(
    Task, HasWeightParameter, HasCruiseVelocityParameter, ABC
):
    """Base for tasks that drive a container's predicted fill level at the end
    of :attr:`prediction_duration` to a goal, and observe success once the fill
    has reached it and stopped changing."""

    goal_value: float
    """Target normalized fill level."""

    fill_level_tolerance: float
    """Tolerance around :attr:`goal_value` within which the goal counts as
    reached."""

    fill_rate_tolerance: float = 0.001
    """Tolerance around zero for the residual fill rate, in normalized fill per
    second."""

    cruise_velocity: float = 0.05
    """The rate the normalized fill level is driven at, in fill per second."""

    stopping_velocity: float | None = None
    """How fast fill predicted beyond the goal may be taken back, in normalized
    fill per second; ``None`` takes it back no faster than
    :attr:`cruise_velocity`."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """QP constraint weight of the fill row."""

    prediction_duration: float = 1.5
    """How far ahead the fill level is predicted, in seconds, independent of
    the controller's prediction horizon."""

    fill_connection: FillLevelConnection = field(init=False)
    """World-resident fill connection resolved while building."""

    fill_velocity_expression: Scalar = field(init=False)
    """Symbolic fill velocity built while building."""

    _compiled_fill_velocity: CompiledFunction = field(init=False, repr=False)
    """Compiled :attr:`fill_velocity_expression`, evaluated once per tick."""

    @classproperty
    def required_context_extensions(cls) -> tuple[type[ContextExtension], ...]:
        """
        :return: The extensions of the base, and the motion control context the fill
            row is built in.
        """
        return super().required_context_extensions + (MotionControlContext,)

    @abstractmethod
    def _resolve_fill_connection(self) -> FillLevelConnection:
        """
        :return: The world-resident fill connection whose fill level the
            constraint drives.
        """

    @abstractmethod
    def _fill_velocity(self) -> Scalar:
        """
        :return: Symbolic normalized fill velocity of :attr:`fill_connection`,
            which is resolved when this is called.
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
        """
        :return: The artifacts of this task: one terminal-state prediction row on
            the fill level, and the predicted fill and fill rate as debug
            expressions.
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
            state_variable=self.fill_connection.fill_dof.variables.position,
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
                    expression=row.predicted_value(config=controller_config),
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
        fill_level = float(self.fill_connection.fill_level)
        fill_rate = float(self._compiled_fill_velocity.evaluate()[0])
        rate_settled = -self.fill_rate_tolerance < fill_rate < self.fill_rate_tolerance
        if rate_settled and self._fill_goal_reached(fill_level=fill_level):
            return ObservationStateValue.TRUE
        return None


@dataclass(eq=False, repr=False, kw_only=True)
class PouringTask(TerminalFillConstraintTask):
    """Tilts a container the robot holds so its own fill level drains toward
    :attr:`goal_value`, predicted with the drain its fill connection integrates."""

    container: HasFillLevel
    """The container poured from, whose fill level is driven down to
    :attr:`goal_value`."""

    def _resolve_fill_connection(self) -> FillLevelConnection:
        """
        :raises MissingFillLevelError: if the container has no fill level.
        """
        if self.container.fill_connection is None:
            raise MissingFillLevelError(node=self)
        return self.container.fill_connection

    def _fill_velocity(self) -> Scalar:
        return self.container.fill_velocity()

    def _fill_goal_reached(self, *, fill_level: float) -> bool:
        return fill_level <= self.goal_value + self.fill_level_tolerance


@dataclass(eq=False, repr=False, kw_only=True)
class FillByTransferTask(TerminalFillConstraintTask):
    """Fills a receiver by tilting a separate source container.

    The transfer gate scales the predicted fill but is read once per
    tick, outside the row's gradient, so the row only tilts; placing the
    source over the receiver is left to
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

    @property
    def source(self) -> HasFillLevel:
        """
        The container pouring into :attr:`receiver`.

        :raises MissingSourceError: if no container pours into the receiver.
        """
        source = self.receiver.poured_from()
        if source is None:
            raise MissingSourceError(node=self)
        return source

    def _resolve_fill_connection(self) -> FillLevelConnection:
        """
        :raises MissingSourceError: if no container pours into the receiver.
        """
        if self.receiver.poured_from() is None:
            raise MissingSourceError(node=self)
        return self.receiver.fill_connection

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The artifacts of the base task, with the transfer gate compiled and
            read once, so the fill row starts from its current value.
        """
        self._compile_gate()
        return super().build_artifacts()

    def _compile_gate(self) -> None:
        """
        Compile the receiver's transfer gate and read its current value.
        """
        stream = self.receiver.incoming_stream()
        if stream is None:
            raise MissingSourceError(node=self)
        self._compiled_gate = self._compile_against_state(expression=stream.gate())
        self._read_gate()

    def _fill_velocity(self) -> Scalar:
        return (
            self._gate_value
            * self.source.outflow_volume_rate()
            / self.receiver.capacity
        )

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
                variable=connection.fill_dof.variables.position,
                velocity=connection.symbolic_fill_velocity(),
            )
            for connection in self.context.world.get_connections_by_type(
                FillLevelConnection
            )
            if connection is not self.fill_connection
            and connection.fill_dof.variables.position.id in rate_variables
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
        self._read_gate()
        return super().on_tick()

    def _fill_goal_reached(self, *, fill_level: float) -> bool:
        return fill_level >= self.goal_value - self.fill_level_tolerance


# %% pour stream


class StreamDebugValue(StrEnum):
    """The points of the poured stream a task exposes as debug expressions,
    each named after the task and this value."""

    EXIT_POINT = "exit"
    """Where the stream leaves the source."""

    LANDING_POINT = "landing"
    """Where the stream lands on the receiver's opening plane."""


@dataclass(eq=False, repr=False, kw_only=True)
class PourStreamTask(Task, ABC):
    """A task on the stream pouring into a receiver, which leaves the source at its
    exit point and lands on the receiver's opening plane."""

    receiver: HasFillLevel
    """The container poured into."""

    @property
    def stream(self) -> PouringStream:
        """
        The stream pouring into :attr:`receiver`.

        :raises MissingSourceError: if no container pours into the receiver.
        """
        stream = self.receiver.incoming_stream()
        if stream is None:
            raise MissingSourceError(node=self)
        return stream

    @property
    def source(self) -> HasFillLevel:
        """
        The container pouring into :attr:`receiver`.

        :raises MissingSourceError: if no container pours into the receiver.
        """
        return self.stream.source

    def landing_point(self) -> Point3:
        """
        :return: Where the stream lands on the receiver's opening plane, thrown at the
            source's current exit speed.
        :raises MissingSourceError: if no container pours into the receiver.
        """
        return self.stream.landing_point()


@dataclass(eq=False, repr=False, kw_only=True)
class KeepProjectileInReceiver(
    PourStreamTask,
    ConvergingTask,
    HasWeightParameter,
    HasCruiseVelocityParameter,
):
    """Moves the source so the stream it pours lands on the receiver's opening
    centre; its error is the distance between the two."""

    cruise_velocity: float = 0.2
    """The speed the landing point is moved at, in metres per second."""

    threshold: float = 0.02
    """Distance within which the landing point counts as inside the opening, in
    metres."""

    weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """QP constraint weight of the landing-point goal."""

    @classproperty
    def exit_point_color(cls) -> Color:
        """Color of the exit-point marker."""
        return Color.from_rgba(red=0.0, green=0.6, blue=1.0, opacity=1.0)

    @classproperty
    def landing_point_color(cls) -> Color:
        """Color of the landing-point marker."""
        return Color.from_rgba(red=1.0, green=0.0, blue=0.0, opacity=1.0)

    def build_artifacts(self) -> MotionNodeArtifacts:
        """
        :return: The constraint driving the pour's landing point to the receiver's
            opening, with the distance between them as error.
        """
        artifacts = MotionNodeArtifacts()
        landing_point = self.landing_point()
        receiver_opening = self.receiver.opening_point()
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
                    name=f"{self.name}/{StreamDebugValue.EXIT_POINT}",
                    expression=self.source.liquid_exit_point(),
                    color=self.exit_point_color,
                ),
                DebugExpression(
                    name=f"{self.name}/{StreamDebugValue.LANDING_POINT}",
                    expression=landing_point,
                    color=self.landing_point_color,
                ),
            ]
        )
        artifacts.error = receiver_opening.euclidean_distance(landing_point)
        return artifacts


@dataclass(eq=False, repr=False, kw_only=True)
class KeepSourceRimAboveReceiverRim(
    Task, HasWeightParameter, HasCruiseVelocityParameter
):
    """Keeps the source's lip, taken from the live forward kinematics, within a
    clearance band above the receiver's rim."""

    receiver: HasFillLevel
    """The container whose rim the source's lip must stay above."""

    source: HasFillLevel
    """The container poured from, whose lip must stay above the receiver's rim."""

    minimum_clearance: float = 0.07
    """Lower end of the clearance band, in metres; must be positive, as the
    bound is enforced softly."""

    clearance_band: float = 0.05
    """Width of the clearance band above :attr:`minimum_clearance`, in
    metres."""

    weight: float = DefaultWeights.WEIGHT_MAXIMUM
    """QP constraint weight of the clearance."""

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
        artifacts = MotionNodeArtifacts()
        source_lip = self.source.liquid_exit_point()
        receiver_rim = self.receiver.opening_point()
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
