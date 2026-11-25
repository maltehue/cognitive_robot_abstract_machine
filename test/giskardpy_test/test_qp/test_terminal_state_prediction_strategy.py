"""
Tests of the terminal-state prediction constraint, the row its strategy builds and the
linearized state model behind that row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import GiskardEqualityConstraint, LargeNumber
from giskardpy.qp.enforcement_strategy import normalize_slack_weight
from giskardpy.qp.exceptions import ConstraintTypeMismatchError
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.terminal_state_prediction_strategy import (
    CoupledState,
    LinearizedScalarStateModel,
    TerminalStatePredictionConstraint,
    TerminalStatePredictionStrategy,
)
from krrood_spatial_math.spatial_types import Vector3
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.physics.equations.pouring_equations import (
    PouringEquation,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% a linear state rate over a joint and a passive state


def _prismatic_connection(*, world: World, name: str) -> PrismaticConnection:
    """
    :return: A new prismatic connection below the world's root.
    """
    with world.modify_world():
        body = Body(name=PrefixedName(name))
        world.add_body(body)
        connection = PrismaticConnection.create_with_dofs(
            world=world, parent=world.root, child=body, axis=Vector3.X()
        )
        world.add_connection(connection)
    return connection


@dataclass(kw_only=True)
class LinearStateRateSetup:
    """
    A world with one controlled joint and one passive state, whose state rate is
    ``f = df/dq·q + df/dx·x`` at the given operating point.
    """

    world: World
    """
    The world owning both degrees of freedom.
    """

    joint_degree_of_freedom: DegreeOfFreedom
    """
    The controlled joint the strategy optimizes.
    """

    state_degree_of_freedom: DegreeOfFreedom
    """
    The passive state whose terminal value is constrained.
    """

    config: QPControllerConfig
    """
    The controller the strategies are built for.
    """

    joint_position: float
    """
    Position of the controlled joint at the operating point.
    """

    state_position: float
    """
    Position of the passive state at the operating point.
    """

    joint_sensitivity: float
    """
    Constant ``df/dq`` of the state rate.
    """

    state_sensitivity: float
    """
    Constant ``df/dx`` of the state rate.
    """

    goal_value: float
    """
    Terminal state goal within the reachable cap.
    """

    far_prediction_duration: float
    """
    A prediction window in seconds longer than the control window.
    """

    @property
    def state_rate(self) -> sm.Scalar:
        """
        The linear state rate ``f``.
        """
        return self.joint_degree_of_freedom.variables.position * sm.Scalar(
            self.joint_sensitivity
        ) + self.state_degree_of_freedom.variables.position * sm.Scalar(
            self.state_sensitivity
        )

    @property
    def resting_state_rate(self) -> sm.Scalar:
        """
        A state rate that is zero at the operating point and independent of the state.
        """
        return (
            self.joint_degree_of_freedom.variables.position
            - sm.Scalar(self.joint_position)
        ) * sm.Scalar(self.joint_sensitivity)

    def constraint(
        self,
        *,
        name: str = "fill_goal",
        goal_value: float | None = None,
        prediction_duration: float | None = None,
        cruise_velocity: float = 1.0,
        stopping_velocity: float | None = None,
        state_rate: sm.Scalar | None = None,
    ) -> TerminalStatePredictionConstraint:
        """
        :return: A terminal constraint on the state; on :attr:`goal_value` over
            :attr:`state_rate` unless given otherwise.
        """
        return TerminalStatePredictionConstraint(
            name=name,
            expression=self.state_rate if state_rate is None else state_rate,
            quadratic_weight=1.0,
            cruise_velocity=cruise_velocity,
            velocity_scale=cruise_velocity,
            enforcement_strategy=TerminalStatePredictionStrategy,
            state_variable=self.state_degree_of_freedom.variables.position,
            goal_value=self.goal_value if goal_value is None else goal_value,
            prediction_duration=prediction_duration,
            stopping_velocity=stopping_velocity,
        )

    def strategy(
        self, *, constraints: list[GiskardEqualityConstraint]
    ) -> TerminalStatePredictionStrategy:
        """
        :return: The strategy under test over the given constraints and the joint.
        """
        return TerminalStatePredictionStrategy(
            degrees_of_freedom=[self.joint_degree_of_freedom],
            constraints=constraints,
            qp_controller_config=self.config,
        )

    def held_joint_recursion(self, *, steps: int) -> float:
        """
        :return: The state after the given Euler steps of its rate with the joint held.
        """
        state = self.state_position
        for _ in range(steps):
            state += self.config.control_time_step.total_seconds() * (
                self.joint_sensitivity * self.joint_position
                + self.state_sensitivity * state
            )
        return state

    def reach(self, *, velocity: float, steps: int) -> float:
        """
        :return: How far a velocity moves the state within the given state steps.
        """
        return velocity * self.config.control_time_step.total_seconds() * steps


@pytest.fixture
def linear_state_rate_setup() -> LinearStateRateSetup:
    """
    A joint and a passive state at their operating point.
    """
    joint_position = 0.2
    state_position = 0.5
    world = World()
    with world.modify_world():
        world.add_body(Body(name=PrefixedName("map")))
    joint_connection = _prismatic_connection(world=world, name="joint_body")
    state_connection = _prismatic_connection(world=world, name="state_body")
    JointState.from_mapping(
        {joint_connection: joint_position, state_connection: state_position}
    ).apply_to(world)
    return LinearStateRateSetup(
        world=world,
        joint_degree_of_freedom=joint_connection.dof,
        state_degree_of_freedom=state_connection.dof,
        config=QPControllerConfig.create_with_simulation_defaults(),
        joint_position=joint_position,
        state_position=state_position,
        joint_sensitivity=0.1,
        state_sensitivity=-0.5,
        goal_value=0.6,
        far_prediction_duration=0.75,
    )


# %% the row of a single constraint


class TestTerminalStatePredictionRow:
    """
    The QP row the strategy builds from a linear state rate.
    """

    def test_state_model_differentiates_rate_by_state_variable(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The linearization extracts ``df/dx`` from the rate's jacobian.
        """
        setup = linear_state_rate_setup
        model = setup.constraint().state_model(config=setup.config)

        assert model.state_sensitivity.evaluate()[0] == pytest.approx(
            setup.state_sensitivity
        )
        assert model.state_value.evaluate()[0] == pytest.approx(setup.state_position)
        assert model.state_velocity.evaluate()[0] == pytest.approx(
            setup.joint_sensitivity * setup.joint_position
            + setup.state_sensitivity * setup.state_position
        )

    def test_matrix_scales_jacobian_per_block_with_normalized_weights(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The row holds ``dt·df/dq`` scaled by each velocity block's normalized lookahead
        weight, followed by zero-padded jerk columns.
        """
        setup = linear_state_rate_setup
        config = setup.config
        constraint = setup.constraint()

        matrix = np.array(
            setup.strategy(constraints=[constraint]).create_matrix().evaluate()
        ).flatten()

        expected_weights = [
            weight.evaluate()[0]
            for weight in LinearizedScalarStateModel.window_normalized_weights(
                weights=constraint.state_model(config=config).lookahead_weights(),
                prediction_steps=config.control_horizon,
            )
        ]
        time_step = config.control_time_step.total_seconds()
        assert matrix.shape == (config.control_horizon + config.prediction_horizon,)
        assert matrix[: config.control_horizon] == pytest.approx(
            [
                time_step * setup.joint_sensitivity * weight
                for weight in expected_weights
            ]
        )
        assert matrix[config.control_horizon :] == pytest.approx(
            [0.0] * config.prediction_horizon
        )

    def test_equality_bound_is_goal_minus_free_response(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The bound is the terminal prediction error with the joint held.
        """
        setup = linear_state_rate_setup

        bound = (
            setup.strategy(constraints=[setup.constraint()])
            .create_equality_bounds()
            .evaluate()[0]
        )

        assert bound == pytest.approx(
            setup.goal_value
            - setup.held_joint_recursion(steps=setup.config.control_horizon)
        )

    def test_equality_bound_is_capped_to_reachable_change(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A goal far beyond the window's reach is capped, not passed through.
        """
        setup = linear_state_rate_setup
        constraint = setup.constraint(name="far_goal", goal_value=5.0)

        bound = (
            setup.strategy(constraints=[constraint])
            .create_equality_bounds()
            .evaluate()[0]
        )

        assert bound == pytest.approx(
            setup.reach(
                velocity=constraint.cruise_velocity,
                steps=setup.config.control_horizon,
            )
        )

    def test_slack_variable_is_single_and_normalized(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The strategy adds exactly one unbounded slack with the normalized weight.
        """
        setup = linear_state_rate_setup
        constraint = setup.constraint()

        slack = setup.strategy(constraints=[constraint]).create_slack_variables()

        assert slack.names == [constraint.name]
        assert slack.lower_bounds.evaluate().flatten().tolist() == [-LargeNumber]
        assert slack.upper_bounds.evaluate().flatten().tolist() == [LargeNumber]
        expected_weight = normalize_slack_weight(
            sm.Scalar(constraint.quadratic_weight),
            constraint.velocity_scale,
            setup.config.control_horizon,
        ).evaluate()[0]
        assert slack.quadratic_weights.evaluate()[0] == pytest.approx(expected_weight)
        assert slack.linear_weights.evaluate()[0] == pytest.approx(0.0)


# %% several constraints in one block


class TestSeveralTerminalConstraints:
    """
    Each terminal constraint of a block gets its own row.
    """

    def test_each_constraint_gets_the_row_it_gets_alone(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        Two constraints give two rows, each with the matrix row, bound and slack weight
        a strategy over that constraint alone builds.
        """
        setup = linear_state_rate_setup
        constraints = [
            setup.constraint(name="first_goal"),
            setup.constraint(
                name="second_goal",
                goal_value=0.4,
                prediction_duration=setup.far_prediction_duration,
                cruise_velocity=0.5,
            ),
        ]
        combined = setup.strategy(constraints=constraints)
        alone = [setup.strategy(constraints=[constraint]) for constraint in constraints]

        matrix = np.array(combined.create_matrix().evaluate())
        bounds = combined.create_equality_bounds().evaluate().flatten()
        slack = combined.create_slack_variables()

        assert matrix.shape[0] == len(constraints)
        assert slack.names == [constraint.name for constraint in constraints]
        for index, single in enumerate(alone):
            assert matrix[index] == pytest.approx(
                np.array(single.create_matrix().evaluate()).flatten()
            )
            assert bounds[index] == pytest.approx(
                single.create_equality_bounds().evaluate()[0]
            )
            assert slack.quadratic_weights.evaluate().flatten()[index] == pytest.approx(
                single.create_slack_variables().quadratic_weights.evaluate()[0]
            )


# %% predicting beyond the control window


class TestPredictionWindow:
    """
    The row predicts the state over the constraint's prediction duration, not over the
    control horizon the velocities span.
    """

    def test_prediction_steps_default_to_the_control_horizon(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        Without a prediction duration the window is the control horizon.
        """
        setup = linear_state_rate_setup

        assert (
            setup.constraint().prediction_steps(config=setup.config)
            == setup.config.control_horizon
        )

    def test_prediction_duration_is_converted_to_time_steps(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The prediction duration is counted in control time steps.
        """
        setup = linear_state_rate_setup
        config = setup.config
        constraint = setup.constraint(prediction_duration=setup.far_prediction_duration)

        model = constraint.state_model(config=config)

        assert model.prediction_steps == round(
            setup.far_prediction_duration / config.control_time_step.total_seconds()
        )
        assert model.prediction_steps > config.control_horizon
        assert model.control_horizon == config.control_horizon

    def test_matrix_keeps_one_block_per_control_step(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A longer window adds no decision variables, and the final block now carries
        weight, since state steps follow it.
        """
        setup = linear_state_rate_setup
        config = setup.config
        constraint = setup.constraint(prediction_duration=setup.far_prediction_duration)

        matrix = np.array(
            setup.strategy(constraints=[constraint]).create_matrix().evaluate()
        ).flatten()

        assert matrix.shape == (config.control_horizon + config.prediction_horizon,)
        assert matrix[config.control_horizon - 1] > 0.0

    def test_velocity_blocks_sum_to_the_prediction_window(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The row's gain follows the prediction window, not the controller's horizon.
        """
        setup = linear_state_rate_setup
        config = setup.config
        constraint = setup.constraint(prediction_duration=setup.far_prediction_duration)

        matrix = np.array(
            setup.strategy(constraints=[constraint]).create_matrix().evaluate()
        ).flatten()

        assert matrix[: config.control_horizon].sum() == pytest.approx(
            config.control_time_step.total_seconds()
            * setup.joint_sensitivity
            * constraint.prediction_steps(config=config)
        )

    def test_slack_weight_is_normalized_over_the_prediction_window(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The slack weight is spread over the prediction window's steps.
        """
        setup = linear_state_rate_setup
        constraint = setup.constraint(prediction_duration=setup.far_prediction_duration)

        slack = setup.strategy(constraints=[constraint]).create_slack_variables()

        expected_weight = normalize_slack_weight(
            sm.Scalar(constraint.quadratic_weight),
            constraint.velocity_scale,
            constraint.prediction_steps(config=setup.config),
        ).evaluate()[0]
        assert slack.quadratic_weights.evaluate()[0] == pytest.approx(expected_weight)

    def test_equality_bound_is_capped_to_the_prediction_window(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        The cap is what the state can change within the prediction window.
        """
        setup = linear_state_rate_setup
        constraint = setup.constraint(
            name="far_goal",
            goal_value=5.0,
            prediction_duration=setup.far_prediction_duration,
        )

        bound = (
            setup.strategy(constraints=[constraint])
            .create_equality_bounds()
            .evaluate()[0]
        )

        assert bound == pytest.approx(
            setup.reach(
                velocity=constraint.cruise_velocity,
                steps=constraint.prediction_steps(config=setup.config),
            )
        )


# %% stopping the state's own change


@dataclass(kw_only=True)
class StoppingVelocities:
    """
    Cruise and stopping velocities relative to how much the linear setup's state changes
    on its own within the control window.
    """

    slow_cruise_velocity: float
    """
    A cruise velocity whose cap lies far below the state's own change.
    """

    fast_stopping_velocity: float
    """
    A stopping velocity whose reach covers the state's whole own change.
    """

    moderate_stopping_velocity: float
    """
    A stopping velocity whose reach lies between the cruise cap and the state's own
    change.
    """

    slower_stopping_velocity: float
    """
    A stopping velocity below the slow cruise velocity.
    """


@pytest.fixture
def stopping_velocities() -> StoppingVelocities:
    """
    Velocities around the linear setup's own change of the state.
    """
    return StoppingVelocities(
        slow_cruise_velocity=0.01,
        fast_stopping_velocity=1.0,
        moderate_stopping_velocity=0.1,
        slower_stopping_velocity=0.005,
    )


class TestStoppingTheStatesOwnChange:
    """
    The cruise velocity caps how fast the controller drives the state; a stopping
    velocity caps, separately, how fast it brings the state back against the change the
    state makes on its own.
    """

    def _bound(
        self,
        *,
        setup: LinearStateRateSetup,
        goal_value: float,
        cruise_velocity: float,
        stopping_velocity: float | None,
    ) -> float:
        """
        :return: The capped bound of a constraint with the given velocities.
        """
        constraint = setup.constraint(
            name="slow_goal",
            goal_value=goal_value,
            cruise_velocity=cruise_velocity,
            stopping_velocity=stopping_velocity,
        )
        return (
            setup.strategy(constraints=[constraint])
            .create_equality_bounds()
            .evaluate()[0]
        )

    def _reach(self, *, setup: LinearStateRateSetup, velocity: float) -> float:
        """
        :return: How far a velocity moves the state within the control window.
        """
        return setup.reach(velocity=velocity, steps=setup.config.control_horizon)

    def _free_change(self, *, setup: LinearStateRateSetup) -> float:
        """
        :return: How far the state moves on its own within the control window.
        """
        return (
            setup.held_joint_recursion(steps=setup.config.control_horizon)
            - setup.state_position
        )

    def test_the_setup_changes_on_its_own_by_more_than_a_moderate_stop_reaches(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        The state falls on its own by more than the moderate stop reaches.
        """
        setup = linear_state_rate_setup
        assert self._free_change(setup=setup) < -self._reach(
            setup=setup, velocity=stopping_velocities.moderate_stopping_velocity
        )

    def test_without_a_stopping_velocity_stopping_is_capped_like_driving(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        Without a stopping velocity the stop is capped by the cruise velocity.
        """
        setup = linear_state_rate_setup
        bound = self._bound(
            setup=setup,
            goal_value=setup.state_position,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=None,
        )
        assert bound == pytest.approx(
            self._reach(setup=setup, velocity=stopping_velocities.slow_cruise_velocity)
        )

    def test_a_stop_within_reach_undoes_the_states_whole_change(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        A fast stop asks to undo the whole own change.
        """
        setup = linear_state_rate_setup
        bound = self._bound(
            setup=setup,
            goal_value=setup.state_position,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=stopping_velocities.fast_stopping_velocity,
        )
        assert bound == pytest.approx(-self._free_change(setup=setup))

    def test_a_stop_beyond_reach_is_capped_by_the_stopping_velocity(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        A moderate stop is capped by what the stopping velocity reaches.
        """
        setup = linear_state_rate_setup
        bound = self._bound(
            setup=setup,
            goal_value=setup.state_position,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=stopping_velocities.moderate_stopping_velocity,
        )
        assert bound == pytest.approx(
            self._reach(
                setup=setup, velocity=stopping_velocities.moderate_stopping_velocity
            )
        )

    def test_a_stopping_velocity_below_the_cruise_velocity_caps_the_stop(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        A stopping velocity slower than the cruise velocity caps the stop below the
        cruise cap.
        """
        setup = linear_state_rate_setup
        bound = self._bound(
            setup=setup,
            goal_value=setup.state_position,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=stopping_velocities.slower_stopping_velocity,
        )
        assert bound == pytest.approx(
            self._reach(
                setup=setup, velocity=stopping_velocities.slower_stopping_velocity
            )
        )

    def test_a_resting_state_is_driven_either_way_at_the_cruise_cap(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        A state that does not change on its own has nothing to stop, so a goal on either
        side is driven at the cruise cap, whatever the stopping velocity.
        """
        setup = linear_state_rate_setup
        cruise_velocity = 1.0
        bounds = [
            setup.strategy(
                constraints=[
                    setup.constraint(
                        goal_value=goal_value,
                        cruise_velocity=cruise_velocity,
                        stopping_velocity=stopping_velocities.slower_stopping_velocity,
                        state_rate=setup.resting_state_rate,
                    )
                ]
            )
            .create_equality_bounds()
            .evaluate()[0]
            for goal_value in (setup.state_position - 5.0, setup.state_position + 5.0)
        ]
        reach = self._reach(setup=setup, velocity=cruise_velocity)
        assert bounds == pytest.approx([-reach, reach])

    def test_a_goal_behind_the_state_asks_at_once_only_to_stop_it(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        A goal against the own change asks only to stop the state.
        """
        setup = linear_state_rate_setup
        bound = self._bound(
            setup=setup,
            goal_value=5.0,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=stopping_velocities.fast_stopping_velocity,
        )
        assert bound == pytest.approx(-self._free_change(setup=setup))

    def test_driving_the_state_further_its_own_way_keeps_the_cruise_cap(
        self,
        *,
        linear_state_rate_setup: LinearStateRateSetup,
        stopping_velocities: StoppingVelocities,
    ) -> None:
        """
        A goal along the own change is capped by the cruise velocity.
        """
        setup = linear_state_rate_setup
        bound = self._bound(
            setup=setup,
            goal_value=-5.0,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=stopping_velocities.fast_stopping_velocity,
        )
        assert bound == pytest.approx(
            -self._reach(setup=setup, velocity=stopping_velocities.slow_cruise_velocity)
        )


# %% validation


class TestTerminalStateConstraintValidation:
    """
    Rejection of constraints the strategy cannot enforce.
    """

    def test_plain_equality_constraint_raises_type_mismatch(
        self, *, linear_state_rate_setup: LinearStateRateSetup
    ) -> None:
        """
        A non-terminal-state equality constraint is rejected, reporting the types.
        """
        plain_constraint = GiskardEqualityConstraint(
            name="plain",
            expression=sm.Scalar(0.0),
            quadratic_weight=1.0,
            cruise_velocity=1.0,
            velocity_scale=1.0,
            enforcement_strategy=TerminalStatePredictionStrategy,
            bound=sm.Scalar(0.0),
        )
        strategy = linear_state_rate_setup.strategy(constraints=[plain_constraint])

        with pytest.raises(ConstraintTypeMismatchError) as error_info:
            strategy.create_equality_bounds()

        assert error_info.value.expected_type is TerminalStatePredictionConstraint
        assert error_info.value.actual_type is GiskardEqualityConstraint
        assert error_info.value.constraint_name == plain_constraint.name


# %% window-normalized weights


class TestWindowNormalizedWeights:
    """
    Rescaling the lookahead weights to the prediction window.
    """

    def test_zero_weight_sum_does_not_produce_nan(self) -> None:
        """
        Weights that cancel to zero are kept instead of divided by zero.
        """
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0)]
        normalized = LinearizedScalarStateModel.window_normalized_weights(
            weights=weights, prediction_steps=2
        )
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == [1.0, -1.0]

    def test_near_cancelling_weight_sum_keeps_raw_weights(self) -> None:
        """
        A sum below the minimum weight sum is kept instead of exploding.
        """
        weights = [sm.Scalar(1.0), sm.Scalar(-1.0 + 1e-12)]
        normalized = LinearizedScalarStateModel.window_normalized_weights(
            weights=weights, prediction_steps=2
        )
        values = [weight.evaluate()[0] for weight in normalized]
        assert all(math.isfinite(value) for value in values)
        assert values == pytest.approx([1.0, -1.0 + 1e-12])

    def test_a_negative_weight_sum_keeps_every_weights_sign(self) -> None:
        """
        A stiff state (decay below minus one) has a negative weight sum; the rescaled
        weights keep the raw weights' signs, with their sum at minus the prediction
        steps.
        """
        prediction_steps = 5
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(0.8),
            state_velocity=sm.Scalar(-0.2),
            state_sensitivity=sm.Scalar(-80.0),
            time_step=0.05,
            control_horizon=prediction_steps,
            prediction_steps=prediction_steps,
        )
        raw = [weight.evaluate()[0] for weight in model.lookahead_weights()]
        assert sum(raw) < 0.0

        normalized = [
            weight.evaluate()[0]
            for weight in LinearizedScalarStateModel.window_normalized_weights(
                weights=model.lookahead_weights(), prediction_steps=prediction_steps
            )
        ]

        assert np.sign(normalized).tolist() == np.sign(raw).tolist()
        assert sum(normalized) == pytest.approx(-prediction_steps)


# %% linearized state model behind the row


@dataclass(kw_only=True)
class FlowingPourSetup:
    """
    A pouring equation at an operating point at which liquid flows, and the
    discretization its linearized model uses.
    """

    equation: PouringEquation
    """
    The drain of the pouring container.
    """

    tilt: float
    """
    Tilt at the operating point, well above the onset so both partials are non-zero.
    """

    fill: float
    """
    Fill level at the operating point.
    """

    time_step: float
    """
    Time step of the linearized model.
    """

    control_horizon: int
    """
    Control horizon of the linearized model.
    """

    prediction_steps: int
    """
    State steps of the far-looking model, more than its control horizon.
    """

    def rate(self, *, tilt: float, fill: float) -> float:
        """
        :return: The nonlinear fill velocity at the given point.
        """
        return self.equation.fill_velocity(
            tilt=sm.Scalar(tilt), fill=sm.Scalar(fill)
        ).evaluate()[0]

    @property
    def fill_sensitivity(self) -> float:
        """
        ``df/dfill`` at the operating point.
        """
        fill = sm.FloatVariable("fill")
        velocity = self.equation.fill_velocity(tilt=sm.Scalar(self.tilt), fill=fill)
        return (
            velocity.jacobian(variables=[fill])[0, 0]
            .substitute(old_variables=[fill], new_variables=[sm.Scalar(self.fill)])
            .evaluate()[0]
        )

    def model(
        self, *, state_velocity: float, prediction_steps: int
    ) -> LinearizedScalarStateModel:
        """
        :return: The linearized model at the operating point with the given rate.
        """
        return LinearizedScalarStateModel(
            state_value=sm.Scalar(self.fill),
            state_velocity=sm.Scalar(state_velocity),
            state_sensitivity=sm.Scalar(self.fill_sensitivity),
            time_step=self.time_step,
            control_horizon=self.control_horizon,
            prediction_steps=prediction_steps,
        )

    def nonlinear_rollout(self, *, tilt_velocity: float, steps: int) -> float:
        """
        :return: The fill after a forward-Euler rollout of the nonlinear equation.
        """
        fill = self.fill
        tilt = self.tilt
        for _ in range(steps):
            fill += self.time_step * self.rate(tilt=tilt, fill=fill)
            tilt += tilt_velocity * self.time_step
        return fill

    def expected_lookahead_weights(
        self, *, decay: float, prediction_steps: int
    ) -> list[float]:
        """
        :return: ``sum_{k=0}^{N-2-i} decay^k`` for every velocity block ``i``.
        """
        return [
            sum(decay**power for power in range(prediction_steps - 1 - block))
            for block in range(self.control_horizon)
        ]


@pytest.fixture
def flowing_pour_setup() -> FlowingPourSetup:
    """
    A container pouring at a tilt well above its onset.
    """
    return FlowingPourSetup(
        equation=PouringEquation(
            container_height=0.2, container_width=0.08, outflow_rate_constant=1.0
        ),
        tilt=1.3,
        fill=0.8,
        time_step=0.05,
        control_horizon=5,
        prediction_steps=12,
    )


class TestLinearizedScalarStateModel:
    """
    The linearized fill prediction behind the row.
    """

    def test_decay_is_one_plus_time_step_times_sensitivity(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        The decay is ``1 + dt·df/dfill`` at the operating point.
        """
        setup = flowing_pour_setup
        model = setup.model(state_velocity=0.0, prediction_steps=setup.control_horizon)

        assert model.decay.evaluate()[0] == pytest.approx(
            1.0 + setup.time_step * setup.fill_sensitivity
        )

    def test_free_response_matches_held_tilt_rollout(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        The free response is the nonlinear rollout with the tilt held.
        """
        setup = flowing_pour_setup
        model = setup.model(
            state_velocity=setup.rate(tilt=setup.tilt, fill=setup.fill),
            prediction_steps=setup.control_horizon,
        )

        assert model.free_response().evaluate()[0] == pytest.approx(
            setup.nonlinear_rollout(tilt_velocity=0.0, steps=setup.control_horizon),
            abs=1e-3,
        )

    def test_lookahead_weights_equal_geometric_series_of_decay(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        Each block carries the geometric series of the decay over its remaining state
        steps, so the final block carries zero.
        """
        setup = flowing_pour_setup
        model = setup.model(state_velocity=0.0, prediction_steps=setup.control_horizon)
        decay = model.decay.evaluate()[0]

        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]

        assert weights == pytest.approx(
            setup.expected_lookahead_weights(
                decay=decay, prediction_steps=setup.control_horizon
            )
        )
        assert weights[-1] == pytest.approx(0.0)
        assert all(earlier > later for earlier, later in zip(weights, weights[1:]))

    def test_normalized_weights_sum_to_the_prediction_window(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        Normalized weights keep their decreasing shape and sum to the predicted state
        steps.
        """
        setup = flowing_pour_setup
        model = setup.model(state_velocity=0.0, prediction_steps=setup.prediction_steps)

        weights = [
            weight.evaluate()[0]
            for weight in LinearizedScalarStateModel.window_normalized_weights(
                weights=model.lookahead_weights(),
                prediction_steps=setup.prediction_steps,
            )
        ]

        assert len(weights) == setup.control_horizon
        assert sum(weights) == pytest.approx(setup.prediction_steps)
        assert weights[0] > weights[-1] > 0.0

    def test_single_step_horizon_predicts_one_euler_step(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        With one step the free response is one Euler step and the only block carries
        zero weight.
        """
        time_step = flowing_pour_setup.time_step
        fill_level = 0.8
        outflow_rate = -0.2
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(-0.5),
            time_step=time_step,
            control_horizon=1,
            prediction_steps=1,
        )

        assert model.free_response().evaluate()[0] == pytest.approx(
            fill_level + time_step * outflow_rate
        )
        assert [weight.evaluate()[0] for weight in model.lookahead_weights()] == (
            pytest.approx([0.0])
        )

    @pytest.mark.parametrize("prediction_steps", [5, 12])
    def test_negative_sensitivity_free_response_matches_linear_recursion(
        self, *, flowing_pour_setup: FlowingPourSetup, prediction_steps: int
    ) -> None:
        """With a negative sensitivity the free response follows the contracting
        recursion ``x_{k+1} = x_k + dt·(f₀ + a·(x_k − x₀))``, within and beyond the
        control window."""
        time_step = flowing_pour_setup.time_step
        fill_level = 0.8
        outflow_rate = -0.2
        fill_sensitivity = -2.0
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(fill_level),
            state_velocity=sm.Scalar(outflow_rate),
            state_sensitivity=sm.Scalar(fill_sensitivity),
            time_step=time_step,
            control_horizon=flowing_pour_setup.control_horizon,
            prediction_steps=prediction_steps,
        )

        expected_fill = fill_level
        for _ in range(prediction_steps):
            expected_fill += time_step * (
                outflow_rate + fill_sensitivity * (expected_fill - fill_level)
            )
        assert model.free_response().evaluate()[0] == pytest.approx(expected_fill)

    def test_free_response_is_a_pure_integrator_when_filling(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        A pure inflow does not depend on the receiver's own fill, so the free response
        is the fill plus the projected inflow.
        """
        setup = flowing_pour_setup
        inflow_rate = 0.1
        model = LinearizedScalarStateModel(
            state_value=sm.Scalar(0.0),
            state_velocity=sm.Scalar(inflow_rate),
            state_sensitivity=sm.Scalar(0.0),
            time_step=setup.time_step,
            control_horizon=setup.control_horizon,
            prediction_steps=setup.control_horizon,
        )

        assert model.free_response().evaluate()[0] == pytest.approx(
            setup.control_horizon * setup.time_step * inflow_rate
        )


class TestPredictionBeyondTheControlWindow:
    """
    With more predicted state steps than control steps the model predicts the state
    after the joints have come to rest.
    """

    def test_free_response_matches_held_tilt_rollout_over_the_prediction_window(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        The free response follows the held-tilt rollout over the whole window, not only
        over the control window.
        """
        setup = flowing_pour_setup
        model = setup.model(
            state_velocity=setup.rate(tilt=setup.tilt, fill=setup.fill),
            prediction_steps=setup.prediction_steps,
        )

        predicted = model.free_response().evaluate()[0]

        assert predicted == pytest.approx(
            setup.nonlinear_rollout(tilt_velocity=0.0, steps=setup.prediction_steps),
            abs=1e-2,
        )
        assert predicted != pytest.approx(
            setup.nonlinear_rollout(tilt_velocity=0.0, steps=setup.control_horizon),
            abs=1e-3,
        )

    def test_lookahead_weights_count_the_state_steps_after_each_block(
        self, *, flowing_pour_setup: FlowingPourSetup
    ) -> None:
        """
        Every weight counts the predicted state steps after its block, so the last block
        is no longer weightless.
        """
        setup = flowing_pour_setup
        model = setup.model(state_velocity=0.0, prediction_steps=setup.prediction_steps)
        decay = model.decay.evaluate()[0]

        weights = [weight.evaluate()[0] for weight in model.lookahead_weights()]

        assert weights == pytest.approx(
            setup.expected_lookahead_weights(
                decay=decay, prediction_steps=setup.prediction_steps
            )
        )
        assert weights[-1] > 0.0


# %% a source the predicted receiver feeds on


@dataclass(kw_only=True)
class FedReceiverSetup:
    """
    The linear setup's state as a receiver fed by a source that drains with the joint:
    ``ṡ = -k·q·s`` and ``ẋ = k·q·s``.
    """

    joint_and_receiver: LinearStateRateSetup
    """
    The joint and the receiver's state.
    """

    source_variable: sm.FloatVariable
    """
    The source's fill, a passive state the receiver's rate depends on.
    """

    source_position: float
    """
    Fill of the source at the operating point.
    """

    drain_constant: float
    """
    Share of the source draining per second and per unit of joint position.
    """

    @property
    def source_velocity(self) -> sm.Scalar:
        """
        The source's own fill rate ``-k·q·s``.
        """
        return (
            -sm.Scalar(self.drain_constant)
            * self.joint_and_receiver.joint_degree_of_freedom.variables.position
            * self.source_variable
        )

    @property
    def coupled_source(self) -> CoupledState:
        """
        The source as a state coupled to the receiver's prediction.
        """
        return CoupledState(
            variable=self.source_variable, velocity=self.source_velocity
        )

    def constraint(
        self, *, coupled_states: list[CoupledState], goal_value: float
    ) -> TerminalStatePredictionConstraint:
        """
        :return: A terminal constraint on the receiver over its rate ``k·q·s``.
        """
        return TerminalStatePredictionConstraint(
            name="receiver_goal",
            expression=-self.source_velocity,
            quadratic_weight=1.0,
            cruise_velocity=1.0,
            velocity_scale=1.0,
            enforcement_strategy=TerminalStatePredictionStrategy,
            state_variable=self.joint_and_receiver.state_degree_of_freedom.variables.position,
            goal_value=goal_value,
            coupled_states=coupled_states,
        )

    def held_joint_recursion(self, *, steps: int) -> float:
        """
        :return: The receiver's fill after the given held-joint Euler steps, the
            source draining along.
        """
        setup = self.joint_and_receiver
        time_step = setup.config.control_time_step.total_seconds()
        source, receiver = self.source_position, setup.state_position
        rate = self.drain_constant * setup.joint_position
        for _ in range(steps):
            source, receiver = (
                source - time_step * rate * source,
                receiver + time_step * rate * source,
            )
        return receiver


@pytest.fixture
def fed_receiver_setup(
    linear_state_rate_setup: LinearStateRateSetup,
) -> FedReceiverSetup:
    """
    The linear setup's joint and state, with a source the state is fed from.
    """
    source_position = 0.8
    source_connection = _prismatic_connection(
        world=linear_state_rate_setup.world, name="source_body"
    )
    JointState.from_mapping({source_connection: source_position}).apply_to(
        linear_state_rate_setup.world
    )
    return FedReceiverSetup(
        joint_and_receiver=linear_state_rate_setup,
        source_variable=source_connection.dof.variables.position,
        source_position=source_position,
        drain_constant=2.0,
    )


class TestCoupledSourceInThePrediction:
    """
    A receiver's prediction lets the source it feeds on drain over the window instead of
    holding the source's current outflow.
    """

    def test_coupled_free_response_follows_both_states(
        self, *, fed_receiver_setup: FedReceiverSetup
    ) -> None:
        """
        The coupled system is linear in its states here, so the prediction equals the
        held-joint recursion of source and receiver together.
        """
        setup = fed_receiver_setup
        config = setup.joint_and_receiver.config
        constraint = setup.constraint(
            coupled_states=[setup.coupled_source],
            goal_value=setup.joint_and_receiver.goal_value,
        )

        predicted = constraint.predicted_value(config=config).evaluate()[0]

        assert predicted == pytest.approx(
            setup.held_joint_recursion(steps=config.control_horizon)
        )

    def test_without_coupled_states_the_source_rate_is_held(
        self, *, fed_receiver_setup: FedReceiverSetup
    ) -> None:
        """
        Without the coupling the receiver gains the source's current outflow over the
        whole window.
        """
        setup = fed_receiver_setup
        receiver = setup.joint_and_receiver
        config = receiver.config
        constraint = setup.constraint(coupled_states=[], goal_value=receiver.goal_value)

        predicted = constraint.predicted_value(config=config).evaluate()[0]

        assert predicted == pytest.approx(
            receiver.state_position
            + receiver.reach(
                velocity=setup.drain_constant
                * receiver.joint_position
                * setup.source_position,
                steps=config.control_horizon,
            )
        )

    def test_equality_bound_uses_the_coupled_prediction(
        self, *, fed_receiver_setup: FedReceiverSetup
    ) -> None:
        """
        The bound is the goal minus the prediction with the source draining along.
        """
        setup = fed_receiver_setup
        receiver = setup.joint_and_receiver
        predicted = setup.held_joint_recursion(steps=receiver.config.control_horizon)
        goal = predicted + 0.001
        strategy = receiver.strategy(
            constraints=[
                setup.constraint(coupled_states=[setup.coupled_source], goal_value=goal)
            ]
        )

        bound = strategy.create_equality_bounds().evaluate()[0]

        assert bound == pytest.approx(goal - predicted)


# %% the prediction as the constraint states it


def test_the_constraint_predicts_the_held_joint_recursion(
    *, linear_state_rate_setup: LinearStateRateSetup
) -> None:
    """
    The prediction the constraint states is the state after the window's held-joint
    steps.
    """
    setup = linear_state_rate_setup
    constraint = setup.constraint(prediction_duration=setup.far_prediction_duration)

    predicted = constraint.predicted_value(config=setup.config).evaluate()[0]

    assert predicted == pytest.approx(
        setup.held_joint_recursion(
            steps=constraint.prediction_steps(config=setup.config)
        )
    )
