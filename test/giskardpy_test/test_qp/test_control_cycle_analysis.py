import numpy as np
import pytest

from giskardpy.qp.control_cycle_analysis import ControlCycleAnalysis
from giskardpy.qp.control_cycle_recording import (
    ConstraintProblemStructure,
    ControlCycleRecording,
)

TIME_STEP = 0.5
CONTROL_HORIZON = 2


def build_opposing_constraints_recording() -> ControlCycleRecording:
    """
    Two constraints on two joints over two velocity blocks, with hand-picked numbers.

    The first row pulls along the first joint, the second row pulls against it, so the
    two rows are in direct conflict.
    """
    structure = ConstraintProblemStructure(
        row_names=["pull#0/0", "push#1/0"],
        degree_of_freedom_names=["first_joint", "second_joint"],
        number_of_equality_rows=1,
        control_horizon=CONTROL_HORIZON,
        model_predictive_control_time_step=TIME_STEP,
    )
    return ControlCycleRecording(
        structure=structure,
        times=np.array([0.0, TIME_STEP]),
        row_sensitivities=np.array(
            [[[2.0, 0.0], [-2.0, 0.0]], [[2.0, 0.0], [0.0, 1.0]]]
        ),
        row_contributions=np.array(
            [[[6.0, 0.0], [-6.0, 0.0]], [[2.0, 0.0], [0.0, 1.0]]]
        ),
        row_lower_bounds=np.array([[3.0, -1.0], [2.0, 0.0]]),
        row_upper_bounds=np.array([[3.0, 1.0], [2.0, 1.0]]),
        row_weights=np.array([[2.0, 4.0], [2.0, 0.0]]),
        row_slacks=np.array([[0.5, -1.5], [0.0, np.nan]]),
        velocities=np.array([[2.0, 3.0], [1.0, 1.0]]),
        velocity_lower_limits=np.full((2, 2), -4.0),
        velocity_upper_limits=np.full((2, 2), 4.0),
        world_degree_of_freedom_ids=[
            "6f1d0e6e-0000-4000-8000-000000000001",
            "6f1d0e6e-0000-4000-8000-000000000002",
        ],
        world_positions=np.array([[0.25, 0.5], [0.75, 1.0]]),
    )


@pytest.fixture()
def analysis() -> ControlCycleAnalysis:
    return ControlCycleAnalysis(build_opposing_constraints_recording())


# %% what the commands achieved


def test_achieved_change_sums_the_horizon(analysis):
    np.testing.assert_allclose(analysis.achieved_changes[0], [6.0, -6.0])


def test_contributions_split_the_achieved_change_over_the_joints(analysis):
    np.testing.assert_allclose(
        analysis.degree_of_freedom_contributions(0), [[6.0, 0.0], [-6.0, 0.0]]
    )


def test_contributions_of_a_cycle_add_up_to_its_achieved_change(analysis):
    np.testing.assert_allclose(
        analysis.degree_of_freedom_contributions(1).sum(axis=1),
        analysis.achieved_changes[1],
    )


def test_violation_measures_the_distance_to_the_nearest_bound(analysis):
    np.testing.assert_allclose(analysis.bound_violations[0], [3.0, 5.0])


def test_satisfied_constraints_are_not_violated(analysis):
    np.testing.assert_allclose(analysis.bound_violations[1], [0.0, 0.0])


def test_cost_weights_the_squared_violation_the_optimizer_allowed(analysis):
    np.testing.assert_allclose(analysis.slack_costs[0], [0.5, 9.0])


def test_dropped_rows_have_no_cost(analysis):
    assert np.isnan(analysis.slack_costs[1, 1])


def test_a_row_without_weight_is_inactive(analysis):
    np.testing.assert_array_equal(analysis.is_active[1], [True, False])


# %% how the constraints relate


def test_sensitivity_is_the_gradient_folded_over_the_horizon(analysis):
    np.testing.assert_allclose(
        analysis.horizon_sensitivities(0), [[2.0, 0.0], [-2.0, 0.0]]
    )


def test_opposing_constraints_have_opposing_gradients(analysis):
    np.testing.assert_allclose(analysis.conflict_matrix(0), [[1.0, -1.0], [-1.0, 1.0]])


def test_perpendicular_constraints_do_not_conflict(analysis):
    np.testing.assert_allclose(analysis.conflict_matrix(1), [[1.0, 0.0], [0.0, 1.0]])


def test_the_opposing_pair_is_reported_as_the_conflict(analysis):
    assert analysis.most_conflicting_rows(0) == (0, 1)


# %% how hard the joints work


def test_saturation_compares_the_command_to_the_joint_limit(analysis):
    np.testing.assert_allclose(analysis.velocity_saturation(0), [0.5, 0.75])
