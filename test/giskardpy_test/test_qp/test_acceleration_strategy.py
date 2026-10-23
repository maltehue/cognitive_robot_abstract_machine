"""
Rows that limit how fast an expression's rate may change over the horizon.
"""

from __future__ import annotations

import numpy as np
import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import GiskardInequalityConstraint
from giskardpy.qp.enforcement_strategy import AccelerationStrategy
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% one joint driving a scaled expression

_GAIN = 2.0
"""
Rate of the expression per unit of joint velocity.
"""

_JOINT_VELOCITY = 0.3
"""
The joint's current velocity.
"""

_ACCELERATION_LIMIT = 0.5
"""
How fast the expression's rate may change, per second.
"""


@pytest.fixture
def joint() -> DegreeOfFreedom:
    """
    A prismatic joint moving at :data:`_JOINT_VELOCITY`.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        child = Body(name=PrefixedName("child"))
        world.add_body(root)
        world.add_body(child)
        connection = PrismaticConnection.create_with_dofs(
            world=world, parent=root, child=child, axis=Vector3.X()
        )
        world.add_connection(connection)
    JointState.from_mapping({connection: 0.0}).apply_to(world)
    connection.velocity = _JOINT_VELOCITY
    return connection.dof


@pytest.fixture
def strategy(joint: DegreeOfFreedom) -> AccelerationStrategy:
    """
    The strategy over one constraint on ``_GAIN * q``.
    """
    return AccelerationStrategy(
        degrees_of_freedom=[joint],
        constraints=[
            GiskardInequalityConstraint(
                name="rate_change",
                expression=joint.variables.position * sm.Scalar(_GAIN),
                lower_bound=-_ACCELERATION_LIMIT,
                upper_bound=_ACCELERATION_LIMIT,
                quadratic_weight=1.0,
                normalization_factor=_ACCELERATION_LIMIT,
                enforcement_strategy=AccelerationStrategy,
            )
        ],
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )


class TestAccelerationRows:
    """
    Each row bounds the change of the expression's rate from one step to the next.
    """

    def test_each_row_takes_the_rate_minus_the_previous_steps_rate(
        self, strategy: AccelerationStrategy
    ) -> None:
        """
        Row ``k`` reads the velocity of step ``k`` and, after the first, subtracts the
        velocity of step ``k - 1``.
        """
        config = strategy.qp_controller_config
        time_step = config.model_predictive_control_time_step
        matrix = np.asarray(strategy.create_matrix().evaluate())
        velocities = matrix[:, : config.control_horizon]
        expected = (
            _GAIN
            * time_step
            * (np.eye(config.control_horizon) - np.eye(config.control_horizon, k=-1))
        )
        assert velocities == pytest.approx(expected)
        assert matrix[:, config.control_horizon :] == pytest.approx(0.0)

    def test_the_first_step_is_measured_against_the_current_rate(
        self, strategy: AccelerationStrategy
    ) -> None:
        """
        The first step may differ from the expression's current rate by one step of the
        limit; later steps from their predecessor by the same.
        """
        config = strategy.qp_controller_config
        time_step = config.model_predictive_control_time_step
        step_change = _ACCELERATION_LIMIT * time_step * time_step
        current = _GAIN * _JOINT_VELOCITY * time_step
        lower = np.asarray(strategy.create_lower_bounds().evaluate()).reshape(-1)
        upper = np.asarray(strategy.create_upper_bounds().evaluate()).reshape(-1)
        assert lower[0] == pytest.approx(current - step_change)
        assert upper[0] == pytest.approx(current + step_change)
        assert lower[1:] == pytest.approx(-step_change)
        assert upper[1:] == pytest.approx(step_change)
