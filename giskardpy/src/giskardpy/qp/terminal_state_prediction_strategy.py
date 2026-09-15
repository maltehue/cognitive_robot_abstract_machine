"""
Terminal-state prediction constraint and enforcement strategy for linearized MPC.

The :class:`TerminalStatePredictionStrategy` builds a single equality constraint row that drives
the MPC-predicted value of a scalar state at the end of a prediction window to a target value.
The prediction is obtained by linearizing the state's first-order ODE at the current operating
point and unrolling the discrete-time recursion analytically, so the terminal state is a linear
function of the joint velocity decision variables — compatible with the existing PIQP quadratic
solver.

The prediction window may be longer than the control horizon the velocities span: the joints
are taken to rest after the last commanded block, which is what the controller's own dynamics
assume, and the state keeps evolving until the window ends. The window is therefore a property
of the constraint, while the control horizon stays a property of the controller.

The mechanism is domain-agnostic: any scalar state governed by ``ẋ = f(x, q)`` whose terminal
value should reach a goal (e.g. a container fill level driven by a tilt or a valve angle) can use
it by supplying the symbolic state velocity and the passive state variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import krrood.symbolic_math.symbolic_math as sm
from krrood.symbolic_math.symbolic_math import Matrix, Scalar, Vector

from giskardpy.qp.constraint import GiskardEqualityConstraint, LargeNumber
from giskardpy.qp.dof_limits import DirectLimits
from giskardpy.qp.enforcement_strategy import IntegralStrategy, normalize_slack_weight
from giskardpy.qp.exceptions import MultipleTerminalStateConstraintsError

WEIGHT_SUM_MAGNITUDE_EPSILON: float = 1e-9
"""
Smallest lookahead-weight sum magnitude that is still normalized.

A stiff or oscillatory linearization can nearly cancel the raw weight sum; dividing by such
a sum would inject enormous or non-finite coefficients into the QP matrix, so below this
magnitude the raw weights are kept instead.
"""


def _geometric_series(base: Scalar, n: int) -> Scalar:
    """
    Computes ``1 + base + base² + … + base^n`` symbolically.

    :param base: Symbolic base value λ.
    :param n: Highest power (inclusive). A negative value yields an empty sum of zero.
    :return: Symbolic geometric series sum.
    """
    power = sm.Scalar(1.0)
    total = sm.Scalar(0.0)
    for _ in range(n + 1):
        total = total + power
        power = power * base
    return total


def _compute_power(base: Scalar, n: int) -> Scalar:
    """
    Computes ``base^n`` symbolically via repeated multiplication.

    :param base: Symbolic base value.
    :param n: Non-negative integer exponent.
    :return: Symbolic ``base^n``.
    """
    result = sm.Scalar(1.0)
    for _ in range(n):
        result = result * base
    return result


def window_normalized_weights(
    weights: list[Scalar], prediction_steps: int
) -> list[Scalar]:
    """
    Rescales lookahead weights to sum to the number of predicted state steps.

    The raw terminal-state sensitivity carries an extra ``dt`` per block, making the matrix
    coefficients far smaller than the proven reactive integral and ill-conditioning the QP.
    Rescaling so the weights sum to the prediction steps preserves their relative lookahead
    emphasis and asks a velocity held over the control horizon to close the terminal error at
    the rate that would close it over the prediction window, so the row's gain follows the
    window rather than the controller's horizon.  This is a solver-conditioning concern, not a
    property of the linearized dynamics, so it lives with the strategy rather than the model.

    :param weights: Geometric lookahead weights from
        :meth:`LinearizedScalarStateModel.lookahead_weights`.
    :param prediction_steps: Number of predicted state steps the weight sum is rescaled to.
    :return: The rescaled weights, summing to ``prediction_steps``.
    """
    total = sm.Scalar(0.0)
    for weight in weights:
        total = total + weight
    window = sm.Scalar(float(prediction_steps))
    return [
        sm.if_less(
            abs(total),
            sm.Scalar(WEIGHT_SUM_MAGNITUDE_EPSILON),
            weight,
            weight * window / total,
        )
        for weight in weights
    ]


@dataclass
class LinearizedScalarStateModel:
    """
    Discrete-time linearization of a first-order scalar ODE about the current operating point.

    Encodes the recursion ``x_{k+1} = λ·x_k + c + dt·a·δu_k`` with ``λ = 1 + dt·(∂f/∂x)`` and
    ``c = dt·(f₀ − (∂f/∂x)·x₀)``.  Solving it over the predicted state steps splits the terminal
    state into a control-independent free response and a control contribution whose per-step
    control deviations are weighted by a geometric series. Control is commanded during the
    control horizon only; the joints rest for the remaining predicted steps.
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

    prediction_steps: int | None = None
    """
    Number of state steps N the terminal state is predicted over; defaults to the
    control horizon when not given.
    """

    def __post_init__(self):
        if self.prediction_steps is None:
            self.prediction_steps = self.control_horizon

    @property
    def decay(self) -> Scalar:
        """Linearized state decay factor ``λ = 1 + dt·(∂f/∂x)``."""
        return sm.Scalar(1.0) + sm.Scalar(self.time_step) * self.state_sensitivity

    def free_response(self) -> Scalar:
        """
        Predicted terminal state if the control is held constant (zero joint velocity).

        This is the autonomous evolution of the linearized system, not a frozen-state assumption:
        the state keeps evolving at the held control.
        """
        decay_to_horizon = _compute_power(self.decay, self.prediction_steps)
        series = _geometric_series(self.decay, self.prediction_steps - 1)
        return decay_to_horizon * self.state_value + series * sm.Scalar(
            self.time_step
        ) * (self.state_velocity - self.state_sensitivity * self.state_value)

    def lookahead_weights(self) -> list[Scalar]:
        """
        Geometric lookahead weight ``G_{N-2-i}`` of each velocity block.

        Block ``i`` (earliest first) raises the control for the remaining ``N-1-i`` state steps,
        so the terminal state is more sensitive to early decisions. A block no state step follows
        has weight zero.
        """
        return [
            _geometric_series(self.decay, self.prediction_steps - 2 - block)
            for block in range(self.control_horizon)
        ]


@dataclass
class TerminalStatePredictionConstraint(GiskardEqualityConstraint):
    """
    Equality constraint carrying the operating point that
    :class:`TerminalStatePredictionStrategy` linearizes into a terminal-state prediction.

    The inherited ``expression`` holds the state rate ``f(x₀, q₀)``: its free variables register the
    joint variables with the QP, and its jacobian w.r.t. them gives the per-step control sensitivity
    the strategy weights across the horizon.  The strategy derives ``∂f/∂x`` and the predicted
    terminal bound from this expression and :attr:`state_variable`, so no further linearization
    fields are stored here.
    """

    state_variable: Scalar = field(kw_only=True)
    """Symbolic current state x₀ (passive DOF position variable); the rate is differentiated w.r.t. it."""

    goal_value: float = field(kw_only=True)
    """Target state value at the end of the prediction window."""

    prediction_duration: float | None = field(default=None, kw_only=True)
    """
    Length of the prediction window in seconds; the state is predicted this far ahead.
    ``None`` predicts to the end of the control horizon.
    """

    bound: Scalar = field(default_factory=lambda: sm.Scalar(0.0), kw_only=True)
    """Unused inherited equality bound; the strategy computes the terminal bound ``goal − x_free`` instead."""


@dataclass
class TerminalStatePredictionStrategy(IntegralStrategy):
    """
    Enforcement strategy for the terminal-state prediction constraint.

    The constraint couples each velocity decision to the predicted terminal state with the
    relative emphasis derived from the linearized recursion: the state-rate jacobian
    ``∂f/∂q·dt`` is scaled per horizon block by :func:`window_normalized_weights`, so an earlier
    velocity — which keeps the control applied for more of the remaining window — affects the
    terminal state more than a later one.  The weights are normalized to the prediction window so
    the row stays at the well-conditioned scale of the plain reactive integral.

    The bound ``goal − x_free`` supplies the proactive terminal prediction error.  Where the
    control sensitivity ``∂f/∂q → 0`` the whole row vanishes, the QP regularizes velocities to
    zero, and the state self-corrects.
    """

    @cached_property
    def _constraint(self) -> TerminalStatePredictionConstraint:
        """
        Validates and returns the single constraint this strategy enforces.

        :raises ConstraintTypeMismatchError: If a constraint is not a terminal-state constraint.
        :raises MultipleTerminalStateConstraintsError: If more than one constraint was grouped
            into this block; the single-row terminal prediction cannot represent that.
        """
        self._require_constraint_type(TerminalStatePredictionConstraint)
        if len(self.constraints) != 1:
            raise MultipleTerminalStateConstraintsError(
                constraint_names=[constraint.name for constraint in self.constraints]
            )
        return self.constraints[0]

    @cached_property
    def _state_model(self) -> LinearizedScalarStateModel:
        """
        Linearizes the single constraint's ODE at the current operating point, built once per solve.

        The state sensitivity ``∂f/∂x`` is differentiated here from the constraint's state-rate
        expression w.r.t. its own state variable, keeping all jacobian computation in one layer.
        """
        constraint = self._constraint
        state_sensitivity = constraint.expression.jacobian([constraint.state_variable])[
            0, 0
        ]
        return LinearizedScalarStateModel(
            state_value=constraint.state_variable,
            state_velocity=constraint.expression,
            state_sensitivity=state_sensitivity,
            time_step=self.qp_controller_config.model_predictive_control_time_step,
            control_horizon=self.qp_controller_config.control_horizon,
            prediction_steps=self._prediction_steps,
        )

    @cached_property
    def _prediction_steps(self) -> int:
        """
        Number of state steps the constraint's prediction window spans, at least one;
        the control horizon when the constraint names no window.
        """
        duration = self._constraint.prediction_duration
        if duration is None:
            return self.qp_controller_config.control_horizon
        return max(
            1,
            round(
                duration / self.qp_controller_config.model_predictive_control_time_step
            ),
        )

    def create_matrix(self) -> Matrix:
        """
        Builds the constraint row by scaling the state-rate jacobian per horizon block with the
        geometric velocity weights, padding the jerk columns with zeros.

        Scaling contract: the jacobian carries a single ``time_step`` factor and the lookahead
        weights are normalized to the prediction window by :func:`window_normalized_weights`, so
        the row lives at the same scale as the reactive
        :class:`~giskardpy.qp.enforcement_strategy.IntegralStrategy` row and its effective gain
        follows the constraint's prediction duration rather than the controller's horizon.
        """
        constraint = self._constraint
        time_step = self.qp_controller_config.model_predictive_control_time_step
        jacobian = (
            sm.Vector([constraint.expression]).jacobian(self.position_variables)
            * time_step
        )
        weights = window_normalized_weights(
            self._state_model.lookahead_weights(), self._prediction_steps
        )
        blocks = [jacobian * weight for weight in weights]
        return sm.hstack(
            blocks + [sm.Matrix.zeros(jacobian.shape[0], self.number_of_jerk_columns)]
        )

    def create_equality_bounds(self) -> Vector:
        """
        Computes the capped equality bound ``goal − x_free``.

        ``x_free`` is the predicted terminal state under zero joint velocity, so the bound is the
        terminal prediction error the QP drives to zero. It is capped to the state change
        reachable within the prediction window.

        .. note:: This relies on a prediction window long enough for ``x_free`` to span the
            terminal overshoot; with a very short window the prediction is too myopic to ease off
            in time and the state overshoots the goal.
        """
        constraint = self._constraint
        bound = sm.Scalar(constraint.goal_value) - self._state_model.free_response()
        capped = self.capped_bound(
            bound,
            self.qp_controller_config.model_predictive_control_time_step,
            constraint.normalization_factor,
            self._prediction_steps,
        )
        return sm.Vector([capped])

    def create_slack_variables(self) -> DirectLimits:
        """
        Creates one slack variable for the single terminal-state constraint, normalized over
        the prediction window like its bound.
        """
        constraint = self._constraint
        return DirectLimits(
            lower_bounds=Vector([-LargeNumber]),
            upper_bounds=Vector([LargeNumber]),
            quadratic_weights=Vector(
                [
                    normalize_slack_weight(
                        constraint.quadratic_weight,
                        constraint.normalization_factor,
                        self._prediction_steps,
                    )
                ]
            ),
            linear_weights=Vector([sm.Scalar(0.0)]),
            names=[constraint.name],
        )
