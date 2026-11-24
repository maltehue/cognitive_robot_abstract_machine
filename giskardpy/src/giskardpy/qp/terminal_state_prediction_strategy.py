"""
Terminal-state prediction constraint and its enforcement strategy for linearized MPC.

Each constraint becomes one equality row that drives the value a scalar state ``ẋ = f(x, q)``
is predicted to reach at the end of a prediction window toward a goal, linear in the joint
velocities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import TYPE_CHECKING

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import FloatVariable, Matrix, Scalar, Vector

from giskardpy.qp.constraint import GiskardEqualityConstraint, LargeNumber
from giskardpy.qp.dof_limits import DirectLimits
from giskardpy.qp.enforcement_strategy import IntegralStrategy, normalize_slack_weight

if TYPE_CHECKING:
    from giskardpy.qp.qp_controller_config import QPControllerConfig


def _geometric_series(*, base: Scalar, n: int) -> Scalar:
    """
    :param base: Symbolic base value λ.
    :param n: Highest power (inclusive); a negative value yields an empty sum of zero.
    :return: Symbolic ``1 + base + base² + … + base^n``.
    """
    power = sm.Scalar(1.0)
    total = sm.Scalar(0.0)
    for _ in range(n + 1):
        total = total + power
        power = power * base
    return total


def _compute_power(*, base: Scalar, n: int) -> Scalar:
    """
    :param base: Symbolic base value.
    :param n: Non-negative integer exponent.
    :return: Symbolic ``base^n``.
    """
    result = sm.Scalar(1.0)
    for _ in range(n):
        result = result * base
    return result


def window_normalized_weights(
    *,
    weights: list[Scalar],
    prediction_steps: int,
    minimum_weight_sum: float = 1e-9,
) -> list[Scalar]:
    """Rescales lookahead weights so their sum has the magnitude
    ``prediction_steps``, keeping each weight's sign, which keeps the row at
    the scale of the reactive integral row.

    :param weights: Lookahead weights from :meth:`LinearizedScalarStateModel.lookahead_weights`.
    :param prediction_steps: Number of predicted state steps the sum's magnitude is scaled to.
    :param minimum_weight_sum: Smallest magnitude of the weight sum that is still rescaled;
        weights summing to less are kept as they are.
    :return: The rescaled weights.
    """
    total = sm.Scalar(0.0)
    for weight in weights:
        total = total + weight
    window = sm.Scalar(float(prediction_steps))
    magnitude = abs(total)
    return [
        sm.if_less(
            magnitude,
            sm.Scalar(minimum_weight_sum),
            weight,
            weight * window / magnitude,
        )
        for weight in weights
    ]


@dataclass(kw_only=True)
class LinearizedScalarStateModel:
    """
    Discrete-time linearization ``x_{k+1} = λ·x_k + c + dt·a·δu_k`` of a first-order scalar
    ODE at the current operating point, with ``λ = 1 + dt·(∂f/∂x)``. Control is commanded
    during the control horizon only; the joints rest for the remaining predicted steps.
    """

    state_value: Scalar
    """Current state value x₀ (passive DOF position)."""

    state_velocity: Scalar
    """Current ODE rate f₀ = f(x₀, q₀)."""

    state_sensitivity: Scalar
    """Partial derivative ∂f/∂x at the operating point."""

    time_step: float
    """MPC discretization step dt in seconds."""

    control_horizon: int
    """Number of velocity decision steps M over which commands are applied."""

    prediction_steps: int
    """Number of state steps N the terminal state is predicted over."""

    @property
    def decay(self) -> Scalar:
        """Linearized state decay factor ``λ = 1 + dt·(∂f/∂x)``."""
        return sm.Scalar(1.0) + sm.Scalar(self.time_step) * self.state_sensitivity

    def free_response(self) -> Scalar:
        """
        :return: The predicted terminal state with the joints held at their current
            positions; the state keeps evolving at the held control.
        """
        decay_to_horizon = _compute_power(base=self.decay, n=self.prediction_steps)
        series = _geometric_series(base=self.decay, n=self.prediction_steps - 1)
        return decay_to_horizon * self.state_value + series * sm.Scalar(
            self.time_step
        ) * (self.state_velocity - self.state_sensitivity * self.state_value)

    def lookahead_weights(self) -> list[Scalar]:
        """
        :return: The geometric weight ``G_{N-2-i}`` of each velocity block ``i``, earliest
            first: the sensitivity of the terminal state to that block's control. A block
            no state step follows has weight zero.
        """
        return [
            _geometric_series(base=self.decay, n=self.prediction_steps - 2 - block)
            for block in range(self.control_horizon)
        ]


@dataclass(kw_only=True)
class CoupledState:
    """A passive state the predicted state's rate depends on, evolving by its
    own rate while the joints are held."""

    variable: FloatVariable
    """Symbolic current value of the coupled state (passive DOF position
    variable)."""

    velocity: Scalar
    """Symbolic rate of the coupled state."""


@dataclass(kw_only=True)
class LinearizedCoupledFreeResponse:
    """
    Held-joint terminal value of a state together with the passive states its rate depends
    on, from their joint ODE ``x_{k+1} = (I + dt·J)·x_k + dt·(f₀ − J·x₀)`` linearized at the
    current operating point.
    """

    state_variables: list[FloatVariable]
    """Current values; the predicted state first, then its coupled states."""

    state_velocities: list[Scalar]
    """Rates, in the order of :attr:`state_variables`."""

    time_step: float
    """MPC discretization step dt in seconds."""

    prediction_steps: int
    """Number of state steps the terminal value is predicted over."""

    def terminal_value(self) -> Scalar:
        """
        :return: The predicted state's value after :attr:`prediction_steps` held-joint
            steps.
        """
        jacobian = sm.Vector(self.state_velocities).jacobian(
            variables=self.state_variables
        )
        current = sm.Vector(self.state_variables)
        transition = (
            sm.Matrix.eye(size=len(self.state_variables)) + jacobian * self.time_step
        )
        offset = (
            sm.Vector(self.state_velocities) - jacobian @ current
        ) * self.time_step
        state = current
        for _ in range(self.prediction_steps):
            state = transition @ state + offset
        return state[0]


@dataclass(kw_only=True)
class TerminalStatePredictionConstraint(GiskardEqualityConstraint):
    """Equality constraint on the value a scalar state is predicted to reach at
    the end of a prediction window.

    Its ``expression`` is the state rate ``f(x₀, q₀)`` the
    :class:`TerminalStatePredictionStrategy` linearizes.
    """

    state_variable: FloatVariable = field(kw_only=True)
    """Symbolic current state x₀, the position variable of a passive degree of
    freedom; the rate is differentiated with respect to it."""

    goal_value: float = field(kw_only=True)
    """Target state value at the end of the prediction window."""

    prediction_duration: float | None = field(default=None, kw_only=True)
    """Length of the prediction window in seconds; ``None`` predicts to the end
    of the control horizon."""

    stopping_velocity: float | None = field(default=None, kw_only=True)
    """How fast the state may be brought back against the change it makes on
    its own, in state units per second; ``None`` uses the cruise velocity."""

    coupled_states: list[CoupledState] = field(default_factory=list, kw_only=True)
    """Passive states the rate depends on that keep evolving while the joints
    are held; empty holds every other state at its current value over the
    window."""

    bound: Scalar = field(default_factory=lambda: sm.Scalar(0.0), kw_only=True)
    """Unused inherited equality bound; the strategy computes the terminal
    bound ``goal − x_free`` instead."""

    def prediction_steps(self, *, config: QPControllerConfig) -> int:
        """
        :param config: The controller the constraint runs in.
        :return: Number of state steps the prediction window spans, at least one; the
            control horizon when the constraint names no window.
        """
        if self.prediction_duration is None:
            return config.control_horizon
        return max(
            1,
            round(self.prediction_duration / config.control_time_step.total_seconds()),
        )

    def state_model(self, *, config: QPControllerConfig) -> LinearizedScalarStateModel:
        """
        :param config: The controller the constraint runs in.
        :return: The state's own ODE linearized at the current operating point.
        """
        return LinearizedScalarStateModel(
            state_value=self.state_variable,
            state_velocity=self.expression,
            state_sensitivity=self.expression.jacobian(variables=[self.state_variable])[
                0, 0
            ],
            time_step=config.control_time_step.total_seconds(),
            control_horizon=config.control_horizon,
            prediction_steps=self.prediction_steps(config=config),
        )

    def predicted_value(self, *, config: QPControllerConfig) -> Scalar:
        """
        :param config: The controller the constraint runs in.
        :return: The state's value at the end of the prediction window with the joints
            held, its coupled states evolving with it.
        """
        if not self.coupled_states:
            return self.state_model(config=config).free_response()
        return LinearizedCoupledFreeResponse(
            state_variables=[self.state_variable]
            + [state.variable for state in self.coupled_states],
            state_velocities=[self.expression]
            + [state.velocity for state in self.coupled_states],
            time_step=config.control_time_step.total_seconds(),
            prediction_steps=self.prediction_steps(config=config),
        ).terminal_value()


@dataclass(kw_only=True)
class TerminalStatePredictionRow:
    """The QP row of one terminal-state constraint in one controller."""

    constraint: TerminalStatePredictionConstraint
    """The constraint the row enforces."""

    config: QPControllerConfig
    """The controller the row is built for."""

    @cached_property
    def prediction_steps(self) -> int:
        """Number of state steps the constraint's prediction window spans."""
        return self.constraint.prediction_steps(config=self.config)

    @cached_property
    def state_model(self) -> LinearizedScalarStateModel:
        """The constraint's ODE linearized at the current operating point."""
        return self.constraint.state_model(config=self.config)

    @property
    def window_duration(self) -> float:
        """Length of the prediction window in seconds."""
        return self.config.control_time_step.total_seconds() * self.prediction_steps

    def matrix(self, *, position_variables: Vector, jerk_columns: int) -> Matrix:
        """
        :param position_variables: The joint position variables of the velocity columns.
        :param jerk_columns: Number of jerk columns, padded with zeros.
        :return: The row: ``dt·∂f/∂q`` per velocity block, scaled by the block's
            :func:`window_normalized_weights` lookahead weight.
        """
        jacobian = (
            sm.Vector([self.constraint.expression]).jacobian(
                variables=position_variables
            )
            * self.config.control_time_step.total_seconds()
        )
        weights = window_normalized_weights(
            weights=self.state_model.lookahead_weights(),
            prediction_steps=self.prediction_steps,
        )
        return sm.Matrix.hstack(
            list_of_matrices=[jacobian * weight for weight in weights]
            + [sm.Matrix.zeros(rows=jacobian.shape[0], columns=jerk_columns)]
        )

    def bound(self) -> Scalar:
        """
        :return: The terminal prediction error ``goal − x_free``, capped to what the
            cruise velocity reaches within the window; undoing the state's own change
            ``x_free − x₀`` is capped by the stopping velocity instead.
        """
        free_response = self.constraint.predicted_value(config=self.config)
        error = sm.Scalar(self.constraint.goal_value) - free_response
        driving_reach = sm.Scalar(
            self.constraint.cruise_velocity * self.window_duration
        )
        stopping_velocity = (
            self.constraint.cruise_velocity
            if self.constraint.stopping_velocity is None
            else self.constraint.stopping_velocity
        )
        stopping_reach = sm.Scalar(stopping_velocity * self.window_duration)
        free_change = free_response - self.state_model.state_value
        own_change = abs(free_change)
        reach_against_own_change = sm.min(own_change, stopping_reach) + sm.max(
            sm.Scalar(0.0), driving_reach - own_change
        )
        lower = sm.if_less(
            free_change, sm.Scalar(0.0), -driving_reach, -reach_against_own_change
        )
        upper = sm.if_less(
            free_change, sm.Scalar(0.0), reach_against_own_change, driving_reach
        )
        return sm.limit(error, lower, upper)

    def slack_weight(self) -> Scalar:
        """
        :return: The quadratic slack weight, normalized over the prediction window like
            the bound.
        """
        return normalize_slack_weight(
            self.constraint.quadratic_weight,
            self.constraint.velocity_scale,
            self.prediction_steps,
        )


@dataclass(kw_only=True)
class TerminalStatePredictionStrategy(IntegralStrategy):
    """Enforces each terminal-state constraint of its block by its own
    :class:`TerminalStatePredictionRow`.

    Where the control sensitivity ``∂f/∂q`` vanishes the row vanishes
    and the state evolves on its own.
    """

    @cached_property
    def _rows(self) -> list[TerminalStatePredictionRow]:
        """The row of each constraint, in constraint order.

        :raises ConstraintTypeMismatchError: If a constraint is not a
            terminal-state constraint.
        """
        self._require_constraint_type(expected=TerminalStatePredictionConstraint)
        return [
            TerminalStatePredictionRow(
                constraint=constraint, config=self.qp_controller_config
            )
            for constraint in self.constraints
        ]

    def create_matrix(self) -> Matrix:
        """
        :return: One row per constraint over the velocity and jerk columns.
        """
        return sm.Matrix.vstack(
            list_of_matrices=[
                row.matrix(
                    position_variables=self.position_variables,
                    jerk_columns=self.number_of_jerk_columns,
                )
                for row in self._rows
            ]
        )

    def create_equality_bounds(self) -> Vector:
        """
        :return: The capped terminal prediction error of each constraint.
        """
        return sm.Vector([row.bound() for row in self._rows])

    def create_slack_variables(self) -> DirectLimits:
        """
        :return: One unbounded slack per constraint, weighted over its prediction window.
        """
        return DirectLimits(
            lower_bounds=Vector([-LargeNumber for _ in self._rows]),
            upper_bounds=Vector([LargeNumber for _ in self._rows]),
            quadratic_weights=Vector([row.slack_weight() for row in self._rows]),
            linear_weights=Vector([sm.Scalar(0.0) for _ in self._rows]),
            names=[row.constraint.name for row in self._rows],
        )
