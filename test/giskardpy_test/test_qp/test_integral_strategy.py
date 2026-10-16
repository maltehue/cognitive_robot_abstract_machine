from __future__ import annotations

import pytest

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.qp.constraint import GiskardInequalityConstraint, LargeNumber
from giskardpy.qp.enforcement_strategy import IntegralStrategy
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import PrismaticConnection
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.world_entity import Body

# %% operating point constants

_LOWER_BOUND = -0.1
"""
Lowest allowed value of the constrained expression.
"""

_UPPER_BOUND = 0.1
"""
Highest allowed value of the constrained expression.
"""

_ALLOWED_VIOLATION = 0.02
"""
Slack limit handed to the constraint, bounding how far its bound may be violated.
"""

# %% fixtures


def _world_with_joint() -> tuple[World, DegreeOfFreedom]:
    """
    Builds a world with a single controlled prismatic joint.
    """
    world = World()
    with world.modify_world():
        map_body = Body(name=PrefixedName("map"))
        joint_body = Body(name=PrefixedName("joint_body"))
        world.add_body(map_body)
        world.add_body(joint_body)
        connection = PrismaticConnection.create_with_dofs(
            world=world, parent=map_body, child=joint_body, axis=Vector3.X()
        )
        world.add_connection(connection)
    return world, connection.dof


def _strategy_over(constraint: GiskardInequalityConstraint) -> IntegralStrategy:
    """
    Builds the strategy under test over one inequality constraint.
    """
    _world, degree_of_freedom = _world_with_joint()
    return IntegralStrategy(
        degrees_of_freedom=[degree_of_freedom],
        constraints=[constraint],
        qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )


def _inequality_constraint(
    degree_of_freedom: DegreeOfFreedom,
    lower_slack_limit: sm.ScalarData = -LargeNumber,
    upper_slack_limit: sm.ScalarData = LargeNumber,
) -> GiskardInequalityConstraint:
    """
    Builds an inequality constraint over the joint position with the given slack limits.
    """
    return GiskardInequalityConstraint(
        name="clearance",
        expression=degree_of_freedom.variables.position,
        quadratic_weight=1.0,
        normalization_factor=1.0,
        enforcement_strategy=IntegralStrategy,
        lower_bound=sm.Scalar(_LOWER_BOUND),
        upper_bound=sm.Scalar(_UPPER_BOUND),
        lower_slack_limit=lower_slack_limit,
        upper_slack_limit=upper_slack_limit,
    )


# %% slack limits


class TestBoundedViolationReachesTheSlackVariable:
    """
    A constraint that bounds how far it may be violated must have that bound applied to
    its slack variable, otherwise the violation it was given is unlimited and the
    constraint is enforced by price alone.
    """

    def test_configured_slack_limits_bound_the_slack_variable(self) -> None:
        """
        The limits the constraint carries are the slack variable's bounds.
        """
        world, degree_of_freedom = _world_with_joint()
        constraint = _inequality_constraint(
            degree_of_freedom,
            lower_slack_limit=-_ALLOWED_VIOLATION,
            upper_slack_limit=_ALLOWED_VIOLATION,
        )
        strategy = IntegralStrategy(
            degrees_of_freedom=[degree_of_freedom],
            constraints=[constraint],
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

        slack = strategy.create_slack_variables()

        assert slack.lower_bounds.evaluate().flatten().tolist() == [-_ALLOWED_VIOLATION]
        assert slack.upper_bounds.evaluate().flatten().tolist() == [_ALLOWED_VIOLATION]

    def test_unset_slack_limits_leave_the_violation_unbounded(self) -> None:
        """
        A constraint that sets no limits keeps the unbounded slack the solver defaults
        to, so bounding the violation stays opt-in.
        """
        world, degree_of_freedom = _world_with_joint()
        constraint = _inequality_constraint(degree_of_freedom)
        strategy = IntegralStrategy(
            degrees_of_freedom=[degree_of_freedom],
            constraints=[constraint],
            qp_controller_config=QPControllerConfig.create_with_simulation_defaults(),
        )

        slack = strategy.create_slack_variables()

        assert slack.lower_bounds.evaluate().flatten().tolist() == [-LargeNumber]
        assert slack.upper_bounds.evaluate().flatten().tolist() == [LargeNumber]
