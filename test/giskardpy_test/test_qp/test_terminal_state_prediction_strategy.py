from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import Callable

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import GiskardEqualityConstraint, LargeNumber
from giskardpy.qp.enforcement_strategy import normalize_slack_weight
from giskardpy.qp.exceptions import (
    ConstraintTypeMismatchError,
    MultipleTerminalStateConstraintsError,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.terminal_state_prediction_strategy import (
    LinearizedScalarStateModel,
    TerminalStatePredictionConstraint,
    TerminalStatePredictionStrategy,
    window_normalized_weights,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.pouring_equations import (
    ArticulatedPouringEquation,
    SymbolicFillContext,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% operating point constants

_JOINT_POSITION = 0.2
"""
Position of the controlled joint at the operating point.
"""

_STATE_POSITION = 0.5
"""
Position of the passive state DOF at the operating point.
"""

_JOINT_SENSITIVITY = 0.1
"""
Constant ``df/dq`` of the linear state-rate expression.
"""

_STATE_SENSITIVITY = -0.5
"""
Constant ``df/dx`` of the linear state-rate expression.
"""

_GOAL_VALUE = 0.6
"""
Terminal state goal within the reachable cap.
"""

# %% fixtures


@dataclass
class LinearStateRateSetup:
    """
    A world with one controlled joint and one passive state DOF, plus a strategy whose
    single constraint carries the linear state rate ``f = df/dq * q + df/dx * x``.
    """

    world: World
    """
    The world owning both degrees of freedom.
    """

    joint_degree_of_freedom: DegreeOfFreedom
    """
    The controlled joint DOF the strategy optimizes.
    """

    state_degree_of_freedom: DegreeOfFreedom
    """
    The passive state DOF whose terminal value is constrained.
    """

    strategy: TerminalStatePredictionStrategy
    """
    The strategy under test, built over the single terminal constraint.
    """

    config: QPControllerConfig
    """
    The controller configuration the strategy was built with.
    """


def _world_with_joint_and_state() -> tuple[World, DegreeOfFreedom, DegreeOfFreedom]:
    """
    Builds a world with a controlled joint DOF and a passive state DOF.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        joint_body = Body(name=PrefixedName("joint_body"))
        state_body = Body(name=PrefixedName("state_body"))
        world.add_body(map_body)
        world.add_body(joint_body)
        world.add_body(state_body)
        joint_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=map_body, child=joint_body, axis=Vector3.X()
        )
        world.add_connection(joint_connection)
        state_connection = PrismaticConnection.create_with_dofs(
            world=world, parent=map_body, child=state_body, axis=Vector3.X()
        )
        world.add_connection(state_connection)
    JointState.from_mapping(
        {joint_connection: _JOINT_POSITION, state_connection: _STATE_POSITION}
    ).apply_to(world)
    return world, joint_connection.dof, state_connection.dof


def _terminal_constraint(
    name: str,
    joint_degree_of_freedom: DegreeOfFreedom,
    state_degree_of_freedom: DegreeOfFreedom,
    goal_value: float = _GOAL_VALUE,
    prediction_duration: float | None = None,
) -> TerminalStatePredictionConstraint:
    """
    Builds a terminal constraint over the linear state rate of the two DOFs.
    """
    state_velocity = joint_degree_of_freedom.variables.position * sm.Scalar(
        _JOINT_SENSITIVITY
    ) + state_degree_of_freedom.variables.position * sm.Scalar(_STATE_SENSITIVITY)
    return TerminalStatePredictionConstraint(
        name=name,
        expression=state_velocity,
        quadratic_weight=1.0,
        normalization_factor=1.0,
        enforcement_strategy=TerminalStatePredictionStrategy,
        state_variable=state_degree_of_freedom.variables.position,
        goal_value=goal_value,
        prediction_duration=prediction_duration,
    )


def _strategy(
    constraints: list[GiskardEqualityConstraint],
    joint_degree_of_freedom: DegreeOfFreedom,
) -> TerminalStatePredictionStrategy:
    """
    Builds the strategy under test over the given constraints and controlled DOF.
    """
    return TerminalStatePredictionStrategy(
        degrees_of_freedom=[joint_degree_of_freedom],
        constraints=constraints,
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )


@pytest.fixture
def linear_state_rate_setup() -> LinearStateRateSetup:
    """
    Single-constraint strategy over a real joint and state DOF.
    """
    world, joint_degree_of_freedom, state_degree_of_freedom = (
        _world_with_joint_and_state()
    )
    constraint = _terminal_constraint(
        "fill_goal", joint_degree_of_freedom, state_degree_of_freedom
    )
    strategy = _strategy([constraint], joint_degree_of_freedom)
    return LinearStateRateSetup(
        world=world,
        joint_degree_of_freedom=joint_degree_of_freedom,
        state_degree_of_freedom=state_degree_of_freedom,
        strategy=strategy,
        config=strategy.qp_controller_config,
    )


# %% happy path


class TestTerminalStatePredictionRow:
    """
    Validates the single QP row the strategy builds from a linear state rate.
    """

    def test_state_model_differentiates_rate_by_state_variable(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The linearization must extract ``df/dx`` from the rate expression's jacobian.
        """
        model = linear_state_rate_setup.strategy._state_model

        assert model.state_sensitivity.evaluate()[0] == pytest.approx(
            _STATE_SENSITIVITY
        )
        assert model.state_value.evaluate()[0] == pytest.approx(_STATE_POSITION)
        assert model.state_velocity.evaluate()[0] == pytest.approx(
            _JOINT_SENSITIVITY * _JOINT_POSITION + _STATE_SENSITIVITY * _STATE_POSITION
        )

    def test_matrix_scales_jacobian_per_block_with_normalized_weights(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The row must hold ``dt * df/dq`` scaled by the normalized lookahead weight of
        each velocity block, followed by zero-padded jerk columns.
        """
        strategy = linear_state_rate_setup.strategy
        config = linear_state_rate_setup.config

        matrix = np.array(strategy.create_matrix().evaluate()).flatten()

        expected_weights = [
            weight.evaluate()[0]
            for weight in window_normalized_weights(
                strategy._state_model.lookahead_weights(), config.control_horizon
            )
        ]
        time_step = config.model_predictive_control_time_step
        expected_velocity_blocks = [
            time_step * _JOINT_SENSITIVITY * weight for weight in expected_weights
        ]
        assert matrix.shape == (config.control_horizon + config.prediction_horizon,)
        assert matrix[: config.control_horizon] == pytest.approx(
            expected_velocity_blocks
        )
        assert matrix[config.control_horizon :] == pytest.approx(
            [0.0] * config.prediction_horizon
        )

    def test_equality_bound_is_goal_minus_free_response(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The bound must equal the terminal prediction error under zero joint velocity.
        """
        strategy = linear_state_rate_setup.strategy

        bound = strategy.create_equality_bounds().evaluate()[0]

        free_response = strategy._state_model.free_response().evaluate()[0]
        assert bound == pytest.approx(_GOAL_VALUE - free_response)

    def test_equality_bound_is_capped_to_reachable_change(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A goal far beyond the horizon's reach must be capped, not passed through.
        """
        config = linear_state_rate_setup.config
        constraint = _terminal_constraint(
            "far_goal",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            goal_value=5.0,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        bound = strategy.create_equality_bounds().evaluate()[0]

        reachable_change = (
            constraint.normalization_factor
            * config.model_predictive_control_time_step
            * config.control_horizon
        )
        assert bound == pytest.approx(reachable_change)

    def test_slack_variable_is_single_and_normalized(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The strategy must add exactly one unbounded slack with the normalized weight.
        """
        strategy = linear_state_rate_setup.strategy
        config = linear_state_rate_setup.config
        constraint = strategy.constraints[0]

        slack = strategy.create_slack_variables()

        assert slack.names == ["fill_goal"]
        assert slack.lower_bounds.evaluate().flatten().tolist() == [-LargeNumber]
        assert slack.upper_bounds.evaluate().flatten().tolist() == [LargeNumber]
        expected_weight = normalize_slack_weight(
            sm.Scalar(constraint.quadratic_weight),
            constraint.normalization_factor,
            config.control_horizon,
        ).evaluate()[0]
        assert slack.quadratic_weights.evaluate()[0] == pytest.approx(expected_weight)
        assert slack.linear_weights.evaluate()[0] == pytest.approx(0.0)


# %% predicting beyond the control window

_PREDICTION_DURATION = 0.75
"""
Prediction window of the far-looking constraint in seconds, longer than the control
window of the simulation defaults.
"""


class TestPredictionWindow:
    """
    The row predicts the state over the constraint's prediction duration, not over the
    control horizon the velocities span.
    """

    def test_prediction_steps_default_to_the_control_horizon(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        model = linear_state_rate_setup.strategy._state_model

        assert model.prediction_steps == linear_state_rate_setup.config.control_horizon

    def test_prediction_duration_is_converted_to_time_steps(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        config = linear_state_rate_setup.config
        constraint = _terminal_constraint(
            "far_looking",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            prediction_duration=_PREDICTION_DURATION,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        model = strategy._state_model

        assert model.prediction_steps == round(
            _PREDICTION_DURATION / config.model_predictive_control_time_step
        )
        assert model.prediction_steps > config.control_horizon
        assert model.control_horizon == config.control_horizon

    def test_matrix_keeps_one_block_per_control_step(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A longer prediction window adds no decision variables: the row still has one
        velocity block per control step and the final block now carries weight, since
        state steps follow it.
        """
        config = linear_state_rate_setup.config
        constraint = _terminal_constraint(
            "far_looking",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            prediction_duration=_PREDICTION_DURATION,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        matrix = np.array(strategy.create_matrix().evaluate()).flatten()

        assert matrix.shape == (config.control_horizon + config.prediction_horizon,)
        assert matrix[config.control_horizon - 1] > 0.0

    def test_velocity_blocks_sum_to_the_prediction_window(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A velocity held over the control horizon is asked to close the terminal error at
        the rate that closes it over the prediction window, so the row's gain follows
        the window and not the controller's horizon.
        """
        config = linear_state_rate_setup.config
        constraint = _terminal_constraint(
            "far_looking",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            prediction_duration=_PREDICTION_DURATION,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        matrix = np.array(strategy.create_matrix().evaluate()).flatten()

        assert matrix[: config.control_horizon].sum() == pytest.approx(
            config.model_predictive_control_time_step
            * _JOINT_SENSITIVITY
            * strategy._state_model.prediction_steps
        )

    def test_slack_weight_is_normalized_over_the_prediction_window(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        constraint = _terminal_constraint(
            "far_looking",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            prediction_duration=_PREDICTION_DURATION,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        slack = strategy.create_slack_variables()

        expected_weight = normalize_slack_weight(
            sm.Scalar(constraint.quadratic_weight),
            constraint.normalization_factor,
            strategy._state_model.prediction_steps,
        ).evaluate()[0]
        assert slack.quadratic_weights.evaluate()[0] == pytest.approx(expected_weight)

    def test_equality_bound_is_capped_to_the_prediction_window(
        self, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The bound is a state error over the prediction window, so its cap is what the
        state can change within that window rather than within the control window.
        """
        config = linear_state_rate_setup.config
        constraint = _terminal_constraint(
            "far_goal",
            linear_state_rate_setup.joint_degree_of_freedom,
            linear_state_rate_setup.state_degree_of_freedom,
            goal_value=5.0,
            prediction_duration=_PREDICTION_DURATION,
        )
        strategy = _strategy(
            [constraint], linear_state_rate_setup.joint_degree_of_freedom
        )

        bound = strategy.create_equality_bounds().evaluate()[0]

        assert bound == pytest.approx(
            constraint.normalization_factor
            * config.model_predictive_control_time_step
            * strategy._state_model.prediction_steps
        )


# %% validation


class TestTerminalStateConstraintValidation:
    """
    Validates rejection of constraint sets the single-row strategy cannot represent.
    """

    def test_two_terminal_constraints_raise_dedicated_error(self) -> None:
        """
        Grouping two terminal-state constraints into one block must fail loudly, naming
        both.
        """
        _world, joint_degree_of_freedom, state_degree_of_freedom = (
            _world_with_joint_and_state()
        )
        strategy = _strategy(
            [
                _terminal_constraint(
                    "fill_goal", joint_degree_of_freedom, state_degree_of_freedom
                ),
                _terminal_constraint(
                    "second_goal", joint_degree_of_freedom, state_degree_of_freedom
                ),
            ],
            joint_degree_of_freedom,
        )

        with pytest.raises(MultipleTerminalStateConstraintsError) as error_info:
            strategy.create_equality_bounds()

        assert error_info.value.constraint_names == ["fill_goal", "second_goal"]

    def test_plain_equality_constraint_raises_type_mismatch(self) -> None:
        """
        A non-terminal-state equality constraint must be rejected, reporting the types.
        """
        _world, joint_degree_of_freedom, _state_degree_of_freedom = (
            _world_with_joint_and_state()
        )
        plain_constraint = GiskardEqualityConstraint(
            name="plain",
            expression=sm.Scalar(0.0),
            quadratic_weight=1.0,
            normalization_factor=1.0,
            enforcement_strategy=TerminalStatePredictionStrategy,
            bound=sm.Scalar(0.0),
        )
        strategy = _strategy([plain_constraint], joint_degree_of_freedom)

        with pytest.raises(ConstraintTypeMismatchError) as error_info:
            strategy.create_equality_bounds()

        assert error_info.value.expected_type is TerminalStatePredictionConstraint
        assert error_info.value.actual_type is GiskardEqualityConstraint
        assert error_info.value.constraint_name == "plain"


# %% weight normalization guard


class TestHorizonNormalizedWeightGuard:
    def test_zero_weight_sum_does_not_produce_nan(self) -> None:
        """
        A weight set that cancels to zero must fall back to the raw weights instead of
        dividing by zero.
        """
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0)]
        normalized = window_normalized_weights(weights, prediction_steps=2)
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == [1.0, -1.0]

    def test_near_cancelling_weight_sum_keeps_raw_weights(self) -> None:
        """
        A sum below the magnitude epsilon must also fall back instead of exploding.
        """
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0 + 1e-12)]
        normalized = window_normalized_weights(weights, prediction_steps=2)
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == pytest.approx([1.0, -1.0 + 1e-12])


# %% linearized state model behind the row

_TIME_STEP = 0.05
"""
Time step of the linearized model.
"""

_CONTROL_HORIZON = 5
"""
Control horizon of the linearized model.
"""

ModelFactory = Callable[..., LinearizedScalarStateModel]
"""
Builds a linearized model at the flowing operating point for a given state velocity and,
optionally, a number of predicted state steps.
"""


def _make_flowing_setup() -> tuple[ArticulatedPouringEquation, float, float]:
    """
    Builds a pouring equation and an operating point at which liquid is actively
    flowing.

    The tilt angle is chosen well above the geometric spill threshold so the outflow gap
    is inside the smooth region of ``max(0, ...)`` and both ODE partials are non-zero.
    """
    equation = ArticulatedPouringEquation(
        container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
    )
    tilt_angle = 1.3
    fill_level = 0.8
    return equation, tilt_angle, fill_level


def _ode_value(
    equation: ArticulatedPouringEquation, tilt_angle: float, fill_level: float
) -> float:
    """
    Evaluates the nonlinear fill velocity at a concrete operating point.
    """
    return equation.symbolic_velocity(
        SymbolicFillContext(sm.Scalar(tilt_angle), sm.Scalar(fill_level))
    ).evaluate()[0]


def _ode_partials(
    equation: ArticulatedPouringEquation, tilt_angle: float, fill_level: float
) -> tuple[float, float]:
    """
    Evaluates ``(df/dtilt, df/dfill)`` at a concrete operating point.
    """
    df_dtilt, df_dfill = equation.symbolic_ode_jacobians(
        sm.Scalar(tilt_angle), sm.Scalar(fill_level)
    )
    return df_dtilt.evaluate()[0], df_dfill.evaluate()[0]


_PREDICTION_STEPS = 12
"""
State steps of the far-looking linearized model, more than its control horizon.
"""


def _nonlinear_rollout(
    equation: ArticulatedPouringEquation,
    tilt_angle: float,
    fill_level: float,
    tilt_velocity: float,
    steps: int = _CONTROL_HORIZON,
) -> float:
    """
    Brute-force forward-Euler rollout of the true nonlinear pouring ODE.
    """
    fill = fill_level
    tilt = tilt_angle
    for _ in range(steps):
        fill += _TIME_STEP * _ode_value(equation, tilt, fill)
        tilt += tilt_velocity * _TIME_STEP
    return fill


def _expected_lookahead_weights(
    decay: float, control_horizon: int, prediction_steps: int | None = None
) -> list[float]:
    """
    Computes the geometric lookahead weight of every velocity block numerically.

    Block ``i`` carries weight ``sum_{k=0}^{N-2-i} decay^k`` over the ``N`` predicted
    state steps; with ``N`` equal to the control horizon the final block has weight zero
    because no state step follows it.
    """
    if prediction_steps is None:
        prediction_steps = control_horizon
    return [
        sum(decay**power for power in range(prediction_steps - 1 - block))
        for block in range(control_horizon)
    ]


@pytest.fixture
def flowing_model_factory() -> ModelFactory:
    """
    Factory for :class:`LinearizedScalarStateModel` at the flowing operating point.

    The fill sensitivity is derived from the pouring equation; only the state velocity
    varies between tests.
    """
    equation, tilt_angle, fill_level = _make_flowing_setup()
    _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)

    def _build(
        state_velocity: float, prediction_steps: int | None = None
    ) -> LinearizedScalarStateModel:
        return LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(state_velocity),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
            prediction_steps=prediction_steps,
        )

    return _build


class TestLinearizedScalarStateModel:
    """
    Validates the linearized fill-prediction math used by the QP constraint.
    """

    def test_decay_is_one_plus_time_step_times_sensitivity(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        The decay factor must equal ``1 + dt * df/dfill`` at the operating point.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        _, fill_sensitivity = _ode_partials(equation, tilt_angle, fill_level)
        model = flowing_model_factory(0.0)

        assert model.decay.evaluate()[0] == pytest.approx(
            1.0 + _TIME_STEP * fill_sensitivity
        )

    def test_free_response_matches_held_tilt_rollout(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        The free response must equal a nonlinear rollout in which the tilt is held
        constant, confirming it is the zero-control prediction rather than a frozen-fill
        assumption.
        """
        equation, tilt_angle, fill_level = _make_flowing_setup()
        outflow_rate = _ode_value(equation, tilt_angle, fill_level)
        model = flowing_model_factory(outflow_rate)

        predicted = model.free_response().evaluate()[0]
        held_tilt = _nonlinear_rollout(equation, tilt_angle, fill_level, 0.0)
        assert predicted == pytest.approx(held_tilt, abs=1e-3)

    def test_lookahead_weights_equal_geometric_series_of_decay(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        Every velocity block must carry exactly the geometric series of the decay over
        its remaining state steps; earlier blocks therefore carry strictly larger weight
        and the final block carries zero because no fill step follows it.
        """
        model = flowing_model_factory(0.0)
        decay = model.decay.evaluate()[0]

        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]
        assert weights == pytest.approx(
            _expected_lookahead_weights(decay, _CONTROL_HORIZON)
        )
        assert weights[-1] == pytest.approx(0.0)
        assert all(earlier > later for earlier, later in zip(weights, weights[1:]))

    def test_normalized_weights_sum_to_the_prediction_window(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        Normalizing the lookahead weights must keep their decreasing shape while summing
        to the predicted state steps, so the matrix stays at the calibrated reactive
        scale.
        """
        model = flowing_model_factory(0.0, _PREDICTION_STEPS)

        weights = [
            weight.evaluate()[0]
            for weight in window_normalized_weights(
                model.lookahead_weights(), _PREDICTION_STEPS
            )
        ]
        assert len(weights) == _CONTROL_HORIZON
        assert sum(weights) == pytest.approx(_PREDICTION_STEPS)
        assert weights[0] > weights[-1] > 0.0

    def test_single_step_horizon_predicts_one_euler_step(self) -> None:
        """
        With a single-step control horizon the free response is one Euler step of the
        ODE and the only velocity block carries zero weight, since no state step can
        follow it.
        """
        fill_level = 0.8
        outflow_rate = -0.2
        fill_sensitivity = -0.5
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=1,
        )

        predicted = model.free_response().evaluate()[0]
        assert predicted == pytest.approx(fill_level + _TIME_STEP * outflow_rate)
        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]
        assert weights == pytest.approx([0.0])

    def test_negative_sensitivity_free_response_matches_linear_recursion(self) -> None:
        """
        With a negative state sensitivity the free response must follow the contracting
        linearized recursion ``x_{k+1} = x_k + dt * (f0 + a * (x_k - x0))``.
        """
        fill_level = 0.8
        outflow_rate = -0.2
        fill_sensitivity = -2.0
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )
        assert model.decay.evaluate()[0] == pytest.approx(
            1.0 + _TIME_STEP * fill_sensitivity
        )

        expected_fill = fill_level
        for _ in range(_CONTROL_HORIZON):
            expected_fill += _TIME_STEP * (
                outflow_rate + fill_sensitivity * (expected_fill - fill_level)
            )
        assert model.free_response().evaluate()[0] == pytest.approx(expected_fill)


class TestPredictionBeyondTheControlWindow:
    """
    With more predicted state steps than control steps the model predicts the state
    after the joints have come to rest, so the row sees what still flows after the last
    command.
    """

    def test_free_response_matches_held_tilt_rollout_over_the_prediction_window(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        equation, tilt_angle, fill_level = _make_flowing_setup()
        outflow_rate = _ode_value(equation, tilt_angle, fill_level)
        model = flowing_model_factory(outflow_rate, _PREDICTION_STEPS)

        predicted = model.free_response().evaluate()[0]

        held_tilt = _nonlinear_rollout(
            equation, tilt_angle, fill_level, 0.0, steps=_PREDICTION_STEPS
        )
        assert predicted == pytest.approx(held_tilt, abs=1e-2)
        assert predicted != pytest.approx(
            _nonlinear_rollout(equation, tilt_angle, fill_level, 0.0), abs=1e-3
        )

    def test_free_response_follows_the_linear_recursion_over_the_prediction_window(
        self,
    ) -> None:
        fill_level = 0.8
        outflow_rate = -0.2
        fill_sensitivity = -2.0
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
            prediction_steps=_PREDICTION_STEPS,
        )

        expected_fill = fill_level
        for _ in range(_PREDICTION_STEPS):
            expected_fill += _TIME_STEP * (
                outflow_rate + fill_sensitivity * (expected_fill - fill_level)
            )
        assert model.free_response().evaluate()[0] == pytest.approx(expected_fill)

    def test_lookahead_weights_count_the_state_steps_after_each_block(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        """
        Every control block still gets one weight, and every weight now counts the
        predicted state steps that follow the block, so the last block is no longer
        weightless.
        """
        model = flowing_model_factory(0.0, _PREDICTION_STEPS)
        decay = model.decay.evaluate()[0]

        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]

        assert len(weights) == _CONTROL_HORIZON
        assert weights == pytest.approx(
            _expected_lookahead_weights(decay, _CONTROL_HORIZON, _PREDICTION_STEPS)
        )
        assert weights[-1] > 0.0

    def test_prediction_steps_default_to_the_control_horizon(
        self, flowing_model_factory: ModelFactory
    ) -> None:
        model = flowing_model_factory(0.0)

        assert model.prediction_steps == _CONTROL_HORIZON


class TestIncreasingFillLinearization:
    """
    Validates the linearization for a container that is filling (inflow goal) rather
    than draining.

    For a pure inflow the fill velocity does not depend on the receiver's own fill
    level, so the fill sensitivity is zero and the linearized model reduces to a well-
    conditioned integrator.
    """

    def test_free_response_is_a_pure_integrator_when_filling(self) -> None:
        """
        With zero fill sensitivity the free response is the fill plus the projected
        inflow.
        """
        fill_level = 0.0
        inflow_rate = 0.1
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(inflow_rate),
            state_sensitivity=sm.Scalar(0.0),
            time_step=_TIME_STEP,
            control_horizon=_CONTROL_HORIZON,
        )
        predicted = model.free_response().evaluate()[0]
        assert predicted == pytest.approx(
            fill_level + _CONTROL_HORIZON * _TIME_STEP * inflow_rate
        )
        assert predicted > fill_level
