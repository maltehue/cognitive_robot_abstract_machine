"""
Tests of the terminal-state prediction constraint, the row its strategy builds and the
linearized state model behind that row.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm
import numpy as np
import pytest
from giskardpy.qp.constraint import GiskardEqualityConstraint
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
from semantic_digital_twin.world_description.connections import ScrewAxisConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% a linear state rate over a joint and a passive state


def _prismatic_connection(*, world: World, name: str) -> ScrewAxisConnection:
    """
    :return: A new prismatic connection below the world's root.
    """
    with world.modify_world():
        body = Body(name=PrefixedName(name))
        world.add_body(body)
        connection = ScrewAxisConnection.create_with_dofs(
            world=world,
            parent=world.root,
            child=body,
            screw_axis=ScrewAxisConnection.create_prismatic_screw_axis(
                axis=Vector3.X()
            ),
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


# %% predicting beyond the control window


class TestPredictionWindow:
    """
    The row predicts the state over the constraint's prediction duration, not over the
    control horizon the velocities span.
    """

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


@pytest.fixture
def stopping_velocities() -> StoppingVelocities:
    """
    Velocities around the linear setup's own change of the state.
    """
    return StoppingVelocities(
        slow_cruise_velocity=0.01,
        fast_stopping_velocity=1.0,
        moderate_stopping_velocity=0.1,
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
        moderate_reach = self._reach(
            setup=setup, velocity=stopping_velocities.moderate_stopping_velocity
        )
        assert self._free_change(setup=setup) < -moderate_reach, (
            "the state must fall on its own by more than the moderate stop reaches"
        )
        bound = self._bound(
            setup=setup,
            goal_value=setup.state_position,
            cruise_velocity=stopping_velocities.slow_cruise_velocity,
            stopping_velocity=stopping_velocities.moderate_stopping_velocity,
        )
        assert bound == pytest.approx(moderate_reach)

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
