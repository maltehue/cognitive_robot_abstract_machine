from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    FloatVariable,
    Scalar,
    VariableParameters,
)
from semantic_digital_twin.physics.equations.pouring_equations import (
    GatedInflowEquation,
    PouringEquation,
    SymbolicFillContext,
    tilt_expression_from_fk,
)
from semantic_digital_twin.semantic_annotations.mixins import HasFillLevel, LiquidSource
from semantic_digital_twin.spatial_types.spatial_types import Point3, Vector3
from semantic_digital_twin.world_description.connections import LiquidConnection
from semantic_digital_twin.world_description.geometry import Color
from semantic_digital_twin.world_description.world_entity import Body

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    DefaultWeights,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    MissingExitSpeedError,
    MissingInflowEquationError,
    NonPositiveClearanceError,
    NonPositiveTiltRateError,
    RootLinkNotWorldRootError,
)
from giskardpy.motion_statechart.graph_node import (
    DebugExpression,
    MotionStatechartNode,
    NodeArtifacts,
    Task,
)


@dataclass(eq=False, repr=False)
class TerminalFillConstraintTask(Task, ABC):
    """
    Base for tasks that drive a container's predicted terminal fill level to a goal.

    Subclasses resolve the fill connection and build the symbolic fill-velocity ODE;
    this base linearizes that ODE into the terminal-state prediction constraint and
    reports convergence once the fill reaches the goal and its rate has settled to zero.
    """

    goal_value: float
    """
    Target fill level to achieve in terms of percentage.
    """

    fill_level_tolerance: float
    """
    Tolerance threshold around :attr:`goal_value`.
    """

    outflow_tolerance: float = field(default=0.001, kw_only=True)
    """
    Tolerance threshold around zero for the residual fill rate.
    """

    reference_velocity: float = field(default=0.05, kw_only=True)
    """
    Desired rate of change of the normalized fill level.
    """

    stopping_velocity: Optional[float] = field(default=None, kw_only=True)
    """
    How fast the controller may take back fill that is predicted to arrive beyond the
    goal, in normalized fill per second; ``None`` stops no faster than
    :attr:`reference_velocity` pours.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the fill-driving gradient.
    """

    prediction_duration: float = field(default=1.5, kw_only=True)
    """
    How far ahead the fill level is predicted, in seconds.

    Long enough to cover the liquid that keeps flowing after the last commanded
    velocity; it also sets how quickly the tilt reacts to a predicted miss and is
    independent of the controller's prediction horizon.
    """

    fill_connection: LiquidConnection = field(init=False)
    """
    World-resident fill connection resolved by :meth:`build`.
    """

    fill_velocity_expression: Scalar = field(init=False)
    """
    Symbolic fill-velocity ODE built by :meth:`build`.
    """

    _compiled_fill_velocity: CompiledFunction = field(init=False, repr=False)
    """
    Compiled :attr:`fill_velocity_expression`, evaluated once per tick.
    """

    @abstractmethod
    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        Resolves and validates the live fill connection whose DOF position the
        constraint drives.

        :param context: The build context.
        :return: The world-resident fill connection.
        """

    @abstractmethod
    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        """
        Builds the symbolic fill-velocity ODE to linearize; :attr:`fill_connection` is
        resolved.

        :param context: The build context.
        :return: Symbolic normalized fill velocity at the current operating point.
        """

    @abstractmethod
    def _fill_goal_reached(self, fill_level: float) -> bool:
        """
        Whether the fill level has reached the goal in the task's fill direction.

        :param fill_level: The current normalized fill level.
        """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Linearizes the fill ODE into a single terminal-state prediction row over
        :attr:`prediction_duration`.

        The fill ODE is linearized at the current operating point and its discrete-time
        recursion unrolled analytically, so the resulting QP row drives the MPC-
        predicted terminal fill toward :attr:`goal_value`.  Because the row couples
        earlier velocity decisions to a larger share of the predicted change, the
        optimizer eases off before overshooting rather than reacting late.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.fill_connection = self._resolve_fill_connection(context)
        self.fill_velocity_expression = self._fill_velocity(context)
        self._compiled_fill_velocity = self.fill_velocity_expression.compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_fill_velocity.bind_args_to_memory_view(
            0, context.world.state.positions
        )
        context.float_variable_data.bind_argument(self._compiled_fill_velocity, 1)
        artifacts.constraints.add_terminal_state_prediction_constraint(
            name=f"{self.fill_connection.name}",
            state_velocity=self.fill_velocity_expression,
            state_variable=self.fill_connection.dof.variables.position,
            goal_value=self.goal_value,
            quadratic_weight=self.weight,
            reference_velocity=self.reference_velocity,
            prediction_duration=self.prediction_duration,
            stopping_velocity=self.stopping_velocity,
        )
        return artifacts

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Reports success once the fill reaches the goal and its rate has settled to zero:
        the rate a sensor sees where one reports it, the model's prediction otherwise.

        :param context: The runtime context.
        :return: The observation state.
        """
        fill_level = float(self.fill_connection.position)
        observed_rate = self.fill_connection.observed_inflow
        fill_rate = (
            float(self._compiled_fill_velocity.evaluate()[0])
            if observed_rate is None
            else observed_rate
        )
        rate_settled = -self.outflow_tolerance < fill_rate < self.outflow_tolerance
        if rate_settled and self._fill_goal_reached(fill_level):
            return ObservationStateValues.TRUE
        return None


@dataclass(eq=False, repr=False)
class PouringTask(TerminalFillConstraintTask):
    """
    Motion Statechart task for controlling the tilt and fill level of a held container.

    Tilts a container the robot holds so its own fill level drains toward
    :attr:`goal_value`; the pouring ODE couples the controlled tilt to the passive fill
    DOF.
    """

    fill_equation: PouringEquation
    """
    Pouring ODE coupling tilt to the fill-level DOF.
    """

    fill_connection: LiquidConnection
    """
    Virtual DOF whose position encodes fill level in [0, 1].
    """

    root_link: Body = field(kw_only=True)
    """
    Root of the kinematic chain used to derive the cup tilt expression; must be the
    world root.
    """

    tip_link: Body = field(kw_only=True)
    """
    Tip of the kinematic chain (the cup body).
    """

    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        :raises RootLinkNotWorldRootError: if ``root_link`` is not the world root, since the tilt
            expression is only valid relative to the vertical world-root frame.
        """
        if self.root_link is not context.world.root:
            raise RootLinkNotWorldRootError(
                node=self, root_link=self.root_link, world_root=context.world.root
            )
        return context.world.get_connection(
            self.fill_connection.parent, self.fill_connection.child
        )

    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        root_T_tip = context.world.compose_forward_kinematics_expression(
            self.root_link, self.tip_link
        )
        tilt_expression = tilt_expression_from_fk(root_T_tip)
        return self.fill_equation.symbolic_velocity(
            SymbolicFillContext(
                tilt_expression=tilt_expression,
                fill_position=self.fill_connection.dof.variables.position,
                repose_correction=self.fill_connection.repose_correction,
                drain_scale=self.fill_connection.drain_scale,
                discharge_scale=self.fill_connection.discharge_scale,
            )
        )

    def _fill_goal_reached(self, fill_level: float) -> bool:
        return fill_level <= self.goal_value + self.fill_level_tolerance


@dataclass(eq=False, repr=False)
class FillByTransferTask(TerminalFillConstraintTask):
    """
    Motion Statechart task that fills a receiver by tilting a separate source container.

    Unlike :class:`PouringTask`, the controlled degrees of freedom (the arm holding the
    source) do not belong to the container whose fill level is the goal.  The receiver's
    inflow ODE depends symbolically on the source arm configuration through the source
    outflow, so driving the receiver's predicted terminal fill toward the goal makes the
    optimizer tilt the source.

    The transfer gate scales the predicted fill but is held as a value read once per
    tick, outside the row's gradient: its logistics never go flat, so a row that
    differentiated them would keep lifting the source to open the height gate a little
    more, against the clearance band, for the whole pour. Placing the source over the
    receiver is left to the aiming and clearance tasks.
    """

    receiver: HasFillLevel
    """
    The container whose fill level is driven up to :attr:`goal_value`.
    """

    _gate_value: FloatVariable = field(init=False, repr=False)
    """
    The transfer gate's value at the last tick, standing in for the symbolic gate in the
    fill velocity.
    """

    _compiled_gate: CompiledFunction = field(init=False, repr=False)
    """
    The symbolic transfer gate, compiled to be read once per tick.
    """

    def _resolve_fill_connection(
        self, context: MotionStatechartContext
    ) -> LiquidConnection:
        """
        :raises MissingInflowEquationError: if the receiver has no inflow equation, meaning
            ``receive_outflow_from`` was not called to couple it to a source.
        """
        self.receiver.ensure_inflow_coupling(context.world)
        fill_connection = context.world.get_connection(
            self.receiver.fill_connection.parent, self.receiver.fill_connection.child
        )
        if fill_connection.inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        return fill_connection

    def _fill_velocity(self, context: MotionStatechartContext) -> Scalar:
        inflow_equation = self.fill_connection.inflow_equation
        if not isinstance(inflow_equation, GatedInflowEquation):
            return inflow_equation.symbolic_velocity(self.fill_connection)
        self._gate_value = FloatVariable(name=f"{self.name}_gate")
        context.float_variable_data.register_expression(self._gate_value)
        self._compiled_gate = inflow_equation.gate.compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_gate.bind_args_to_memory_view(0, context.world.state.positions)
        context.float_variable_data.bind_argument(self._compiled_gate, 1)
        self._read_gate(context)
        return self._gate_value * inflow_equation.ungated_velocity(self.fill_connection)

    def _read_gate(self, context: MotionStatechartContext) -> None:
        """
        Store the transfer gate's current value for the fill velocity to read.

        :param context: The runtime context.
        """
        context.float_variable_data.set_value(
            self._gate_value, float(self._compiled_gate.evaluate()[0])
        )

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        Reads the transfer gate anew, then reports success as the base task does.

        :param context: The runtime context.
        :return: The observation state.
        """
        if isinstance(self.fill_connection.inflow_equation, GatedInflowEquation):
            self._read_gate(context)
        return super().on_tick(context)

    def _fill_goal_reached(self, fill_level: float) -> bool:
        return fill_level >= self.goal_value - self.fill_level_tolerance


@dataclass(eq=False, repr=False)
class KeepProjectileInReceiver(Task):
    """
    Positions the source so the poured liquid's projectile lands in the receiver
    opening.

    Drives the predicted projectile landing point of the source's pour toward the
    receiver's opening centre, so as the source tilts the optimizer moves the gripper to
    keep the liquid landing inside the receiver — the no-spill counterpart to
    :class:`FillByTransferTask`.
    """

    receiver: HasFillLevel
    """
    The container the liquid must land in; must already be coupled via
    ``receive_outflow_from``.
    """

    source: LiquidSource
    """
    The liquid source being poured from.
    """

    reference_velocity: float = field(default=0.2, kw_only=True)
    """
    Reference velocity for normalization in m/s.
    """

    threshold: float = field(default=0.02, kw_only=True)
    """
    Distance threshold for the landing point to count as inside the opening, in metres.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the landing-point goal.
    """

    EXIT_POINT_COLOR: ClassVar[Color] = Color(R=0.0, G=0.6, B=1.0, A=1.0)
    """
    Color of the exit-point marker (blue).
    """

    LANDING_POINT_COLOR: ClassVar[Color] = Color(R=1.0, G=0.0, B=0.0, A=1.0)
    """
    Color of the landing-point marker (red).
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraint driving the pour's projectile landing point to the
        receiver opening.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.receiver.ensure_inflow_coupling(context.world)
        inflow_equation = self.receiver.fill_connection.inflow_equation
        if inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        exit_speed = self.source.current_outflow_velocity(context.world)
        if exit_speed is None:
            if not isinstance(inflow_equation, GatedInflowEquation):
                raise MissingExitSpeedError(node=self)
            exit_speed = inflow_equation.exit_speed
        landing_point = self.receiver.projectile_landing_point(
            self.source, context.world, exit_speed
        )
        receiver_opening = self.receiver.opening_point(context.world)
        artifacts.geometry.add_point_goal_constraints(
            name=f"{self.receiver.root.name}_projectile",
            frame_P_goal=receiver_opening,
            frame_P_current=landing_point,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        artifacts.debug_expressions.extend(
            self._build_visualization_debug_expressions(context, landing_point)
        )
        artifacts.observation = (
            receiver_opening.euclidean_distance(landing_point) < self.threshold
        )
        return artifacts

    def _build_visualization_debug_expressions(
        self, context: MotionStatechartContext, landing_point: Point3
    ) -> list[DebugExpression]:
        """
        Build the debug expressions that visualize where the pour leaves and where it
        lands.

        :param context: The build context.
        :param landing_point: The projectile landing point on the receiver's opening
            plane.
        :return: Debug expressions for the exit point and the landing point.
        """
        exit_point = self.source.liquid_exit_point(context.world)
        return [
            DebugExpression(
                name="exit", expression=exit_point, color=self.EXIT_POINT_COLOR
            ),
            DebugExpression(
                name="landing", expression=landing_point, color=self.LANDING_POINT_COLOR
            ),
        ]


@dataclass(eq=False, repr=False)
class ShareAimWithReceiver(Task):
    """
    Moves the receiver horizontally towards the pour's predicted landing point, so the
    hand holding the receiver shares the aiming correction with the hand holding the
    source instead of leaving all of it to the source.

    :class:`KeepProjectileInReceiver` drives the source towards the receiver's current
    opening; only the source's side of that gap is differentiated into its constraint
    row, so the receiver never gets a reason to move. This task closes the same gap from
    the other side, by differentiating the receiver's opening instead and driving it
    towards the source's current landing point. Both targets are the other side's
    position at the start of the control cycle, re-evaluated every cycle, so the two
    sides converge on each other gradually at the control frequency rather than either
    one jumping to close the whole gap by itself.

    Meant for a receiver a robot itself holds and can therefore usefully move; a
    receiver resting on a support should not be given this task, since nothing then
    stops the optimizer from sliding it across that support.
    """

    receiver: HasFillLevel
    """
    The container being poured into; its opening is driven towards the landing point.
    """

    source: LiquidSource
    """
    The liquid source being poured from.
    """

    reference_velocity: float = field(default=0.05, kw_only=True)
    """
    Reference velocity for normalization in m/s.

    Below the aiming task's own reference velocity, so the source still does most of the
    correcting and the receiver only helps.
    """

    weight: float = field(
        default=DefaultWeights.WEIGHT_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP constraint weight for the receiver's share of the aiming correction.

    Below the aiming task's own weight, so the source's aim still takes priority.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Drive the receiver's opening towards the source's current predicted landing
        point.

        :param context: The build context.
        :return: The generated task artifacts.
        """
        artifacts = NodeArtifacts()
        self.receiver.ensure_inflow_coupling(context.world)
        inflow_equation = self.receiver.fill_connection.inflow_equation
        if inflow_equation is None:
            raise MissingInflowEquationError(node=self)
        exit_speed = self.source.current_outflow_velocity(context.world)
        if exit_speed is None:
            if not isinstance(inflow_equation, GatedInflowEquation):
                raise MissingExitSpeedError(node=self)
            exit_speed = inflow_equation.exit_speed
        landing_point = self.receiver.projectile_landing_point(
            self.source, context.world, exit_speed
        )
        receiver_opening = self.receiver.opening_point(context.world)
        artifacts.geometry.add_point_goal_constraints(
            name=f"{self.receiver.root.name}_share_aim",
            frame_P_goal=landing_point,
            frame_P_current=receiver_opening,
            reference_velocity=self.reference_velocity,
            quadratic_weight=self.weight,
        )
        return artifacts


@dataclass(eq=False, repr=False)
class KeepSourceRimAboveReceiverRim(Task):
    """
    Keeps the pouring source's rim above the receiver's rim so the rims never collide.

    Constrains the height of the source's pouring lip (its lowest rim point while
    tilting) above the receiver's rim to stay within a clearance band.  Because the lip
    is derived from the live forward kinematics, the constraint accounts for the lip
    descending as the source tilts, so the clearance is a true rim-to-rim gap rather
    than a hand-tuned offset on the cup origins.

    The task stores only the source and receiver, building the symbolic lip and rim on
    the target world, so it survives serialization to a standalone Giskard process
    (unlike a task that would carry a pre-built symbolic point).
    """

    receiver: HasFillLevel
    """
    The container whose rim the source's rim must stay above.
    """

    source: LiquidSource
    """
    The pouring source whose rim must stay above the receiver's rim.
    """

    minimum_clearance: float = field(default=0.07, kw_only=True)
    """
    Lower bound on the source-lip-above-receiver-rim clearance, in metres.

    Must be positive: a band reaching down to zero would ask the optimizer to hold the rims in
    contact, and since the bound is enforced softly the lip would settle below the receiver rim.
    Choose it above the clearance the optimizer actually tracks, not at the collision limit.
    """

    clearance_band: float = field(default=0.05, kw_only=True)
    """
    Width of the clearance band above :attr:`minimum_clearance`, in metres.

    A band, rather than a one-sided lower bound, keeps the optimization well-
    conditioned: it is the only constraint pinning the source's vertical position,
    because the landing point that :class:`KeepProjectileInReceiver` aims lies in the
    receiver's opening plane by construction and so carries no vertical error.
    """

    weight: float = field(default=DefaultWeights.WEIGHT_MAXIMUM, kw_only=True)
    """
    QP constraint weight for the clearance.

    The fill task pulls the lip down along the same direction this task holds it, so the
    clearance needs the top tier to keep its floor against it.
    """

    maximum_velocity: float = field(default=0.2, kw_only=True)
    """
    Maximum allowed vertical speed for the clearance motion, in metres per second.
    """

    @property
    def maximum_clearance(self) -> float:
        """
        Upper end of the clearance band, in metres.
        """
        return self.minimum_clearance + self.clearance_band

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        Creates the constraint keeping the source's pouring lip above the receiver's
        rim.

        :param context: The build context.
        :return: The generated task artifacts.
        :raises NonPositiveClearanceError: if the clearance band is not entirely above
            the rim, which would describe a physically impossible pour.
        """
        if self.minimum_clearance <= 0.0:
            raise NonPositiveClearanceError(
                node=self, minimum_clearance=self.minimum_clearance
            )
        artifacts = NodeArtifacts()
        source_lip = self.source.liquid_exit_point(context.world)
        receiver_rim = self.receiver.opening_point(context.world)
        clearance = (source_lip - receiver_rim) @ Vector3.Z()
        artifacts.constraints.add_inequality_constraint(
            reference_velocity=self.maximum_velocity,
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


CAP_NORMALIZATION_SHARE = 0.1
"""
The share of a tilt rate cap its QP row is normalized by.

A row's slack weight is its weight over the square of its normalization, so a cap
normalized by the cap itself would only trade off against a fill task's row (weight over
the square of its reference velocity); normalized by a tenth of the cap it outweighs
that row a hundred times over and holds.
"""


@dataclass(eq=False, repr=False)
class TiltPacing(Task):
    """
    Paces the source cup's tilt: kept turning at :attr:`approach_rate` while nothing
    else moves it, never faster than :attr:`maximum_rate` tilting further or
    :attr:`maximum_return_rate` tilting back, and never past :attr:`maximum_tilt`.

    The approach is a weak preference the fill task overrides as soon as its model
    predicts flow, and it stops once :attr:`receiver` holds :attr:`approach_until_fill`;
    the caps and the ceiling hold against every other task. Below
    :attr:`fast_until_tilt` the cup may turn at :attr:`fast_rate` instead, which covers
    the tilts at which nothing can leave the cup yet.
    """

    source: HasFillLevel
    """
    The cup whose tilt is paced.
    """

    approach_rate: float
    """
    How fast the cup keeps turning while no other task moves it, in radians per second.
    """

    maximum_rate: float
    """
    The fastest the cup may tilt further, in radians per second.
    """

    maximum_tilt: float
    """
    The tilt the cup is not turned past, in radians.
    """

    maximum_return_rate: float = field(default=math.radians(30.0), kw_only=True)
    """
    The fastest the cup may tilt back, in radians per second.
    """

    fast_rate: Optional[float] = field(default=None, kw_only=True)
    """
    The rate the cup approaches and may tilt at below :attr:`fast_until_tilt`, in
    radians per second, or ``None`` to pace the whole motion alike.
    """

    fast_until_tilt: float = field(default=0.0, kw_only=True)
    """
    The tilt below which :attr:`fast_rate` applies, in radians.
    """

    fast_blend_width: float = field(default=0.0, kw_only=True)
    """
    The span of tilt below :attr:`fast_until_tilt` over which the rate eases from
    :attr:`fast_rate` to the slow one, in radians; zero switches at once.
    """

    ceiling_slowdown_width: float = field(default=0.0, kw_only=True)
    """
    The span of tilt below :attr:`maximum_tilt` over which the rate the cup may tilt
    further eases to zero, in radians; zero stops it at the ceiling at once.
    """

    receiver: Optional[HasFillLevel] = field(default=None, kw_only=True)
    """
    The cup poured into, whose fill ends the approach, or ``None`` to approach until the
    ceiling.
    """

    approach_until_fill: float = field(default=1.0, kw_only=True)
    """
    The fill of :attr:`receiver` from which on the approach no longer turns the cup:

    once the receiver holds what was to be poured, nothing but the fill task may tilt it
    further.
    """

    cap_weight: float = field(default=DefaultWeights.WEIGHT_MAXIMUM, kw_only=True)
    """
    QP weight of the rate caps and the ceiling.
    """

    approach_weight: float = field(
        default=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE, kw_only=True
    )
    """
    QP weight of the approach preference: low, so any task with a reason to turn the cup
    otherwise wins.
    """

    @property
    def rates(self) -> tuple[float, ...]:
        """
        :return: Every rate this task is configured with.
        """
        rates = (self.approach_rate, self.maximum_rate, self.maximum_return_rate)
        if self.fast_rate is None:
            return rates
        return rates + (self.fast_rate,)

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :param context: The build context.
        :return: Two rows on the tilt's speed: the caps and the ceiling, and the approach
            preference. The task never holds the motion open.
        """
        for rate in self.rates:
            if rate <= 0.0:
                raise NonPositiveTiltRateError(node=self, rate=rate)
        tilt = self.source.pour_tilt_expression
        cap, approach = sm.Scalar(self.maximum_rate), sm.Scalar(self.approach_rate)
        if self.fast_rate is not None:
            fast_share = self._share_below(
                tilt, self.fast_until_tilt, self.fast_blend_width
            )
            cap = cap + fast_share * (self.fast_rate - self.maximum_rate)
            approach = approach + fast_share * (self.fast_rate - self.approach_rate)
        below_ceiling = self._share_below(
            tilt, self.maximum_tilt, self.ceiling_slowdown_width
        )
        if self.receiver is not None:
            approach = sm.if_less(
                self.receiver.fill_connection.dof.variables.position,
                sm.Scalar(self.approach_until_fill),
                approach,
                sm.Scalar(0.0),
            )
        artifacts = NodeArtifacts(observation=sm.Scalar.const_true())
        artifacts.constraints.add_velocity_constraint(
            lower_velocity_limit=-self.maximum_return_rate,
            upper_velocity_limit=cap * below_ceiling,
            quadratic_weight=self.cap_weight,
            task_expression=tilt,
            velocity_limit=self.maximum_rate * CAP_NORMALIZATION_SHARE,
            name=f"{self.name}_tilt_rate_cap",
        )
        artifacts.constraints.add_velocity_eq_constraint(
            velocity_goal=approach * below_ceiling,
            quadratic_weight=self.approach_weight,
            task_expression=tilt,
            velocity_limit=self.approach_rate,
            name=f"{self.name}_tilt_approach",
        )
        return artifacts

    @staticmethod
    def _share_below(tilt: sm.Scalar, limit: float, width: float) -> sm.Scalar:
        """
        :return: One where the tilt lies at least ``width`` below ``limit``, zero at and
            beyond it, and in between rising linearly; a step at the limit for no width.
        """
        if width <= 0.0:
            return sm.if_less(tilt, sm.Scalar(limit), sm.Scalar(1.0), sm.Scalar(0.0))
        return sm.limit((sm.Scalar(limit) - tilt) / width, 0.0, 1.0)


TILT_HOLD_TOLERANCE = math.radians(1.0)
"""
How close below the tilt a cup counts as held at it, in radians.
"""


@dataclass(eq=False, repr=False)
class HeldAtTilt(MotionStatechartNode):
    """
    Turns true once the source cup has stayed at :attr:`tilt` or beyond for
    :attr:`seconds` without a break: a cup held at its ceiling that long has poured all
    it will at that tilt, whatever the fill estimate says.
    """

    source: HasFillLevel
    """
    The cup whose tilt is watched.
    """

    tilt: float
    """
    The tilt watched for, in radians.
    """

    seconds: float
    """
    How long the cup has to stay there, in seconds.
    """

    tolerance: float = field(default=TILT_HOLD_TOLERANCE, kw_only=True)
    """
    How far below :attr:`tilt` the cup still counts as held at it, in radians.
    """

    _held_cycles: int = field(init=False, default=0)
    """
    Control cycles the cup has been held there in a row.
    """

    def build_artifacts(self, context: MotionStatechartContext) -> NodeArtifacts:
        """
        :param context: The build context.
        :return: No constraints: the node only watches the tilt.
        """
        self._compiled_tilt = self.source.pour_tilt_expression.compile(
            parameters=VariableParameters.from_lists(
                context.world.state.position_float_variables,
                context.float_variable_data.variables,
            ),
            sparse=False,
        )
        self._compiled_tilt.bind_args_to_memory_view(0, context.world.state.positions)
        self._compiled_tilt.bind_args_to_memory_view(
            1, context.float_variable_data.data
        )
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        self._held_cycles = 0

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        """
        :return: True once the cup has been held long enough, false until then.
        """
        tilt = float(self._compiled_tilt.evaluate()[0])
        if tilt < self.tilt - self.tolerance:
            self._held_cycles = 0
            return ObservationStateValues.FALSE
        self._held_cycles += 1
        held_for = self._held_cycles * context.qp_controller_config.control_dt
        if held_for >= self.seconds:
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
