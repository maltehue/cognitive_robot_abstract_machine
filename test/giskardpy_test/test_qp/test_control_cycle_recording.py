import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPosition,
    CartesianPositionVelocityLimit,
)
from giskardpy.qp.control_cycle_recording import (
    ControlCycleRecorder,
    ControlCycleRecording,
    RECORDING_FORMAT_VERSION,
)
from giskardpy.qp.exceptions import (
    EmptyControlCycleRecordingError,
    UnknownRecordingFormatVersionError,
)
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.spatial_types.spatial_types import Point3

MAXIMUM_LINEAR_VELOCITY = 0.05


def _run_recorded_motion(world) -> ControlCycleRecording:
    """
    Move the PR2 hand to a point while its linear speed is capped, and record it.

    The two tasks give the recording both an equality block, from the point goal, and an
    inequality block enforced per time step, from the velocity limit.
    """
    root_link = world.root
    tip_link = world.get_body_by_name("r_gripper_tool_frame")
    reach = CartesianPosition(
        root_link=root_link,
        tip_link=tip_link,
        goal_point=Point3(0.6, -0.3, 1.0, reference_frame=root_link),
    )
    speed_limit = CartesianPositionVelocityLimit(
        root_link=root_link,
        tip_link=tip_link,
        max_linear_velocity=MAXIMUM_LINEAR_VELOCITY,
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(reach)
    motion_statechart.add_node(speed_limit)
    motion_statechart.add_node(EndMotion.when_true(reach))

    recorder = ControlCycleRecorder()
    executor = Executor(
        MotionStatechartContext(world=world), control_cycle_recorder=recorder
    )
    executor.compile(motion_statechart=motion_statechart)
    executor.tick_until_end()
    return recorder.build_recording()


@pytest.fixture()
def recording(pr2_world_state_reset) -> ControlCycleRecording:
    return _run_recorded_motion(pr2_world_state_reset)


# %% row layout


def test_rows_are_named_after_the_nodes_that_created_them(recording):
    node_names = set(recording.structure.node_names)

    assert any(name.startswith(CartesianPosition.__name__) for name in node_names)
    assert any(
        name.startswith(CartesianPositionVelocityLimit.__name__) for name in node_names
    )


def test_point_goal_contributes_one_equality_row_per_axis(recording):
    structure = recording.structure
    equality_nodes = [
        node_name
        for node_name, is_equality in zip(
            structure.node_names, structure.row_is_equality
        )
        if is_equality
    ]

    assert (
        sum(name.startswith(CartesianPosition.__name__) for name in equality_nodes) == 3
    )


def test_velocity_limit_contributes_one_inequality_row_per_control_step(recording):
    structure = recording.structure
    inequality_nodes = [
        node_name
        for node_name, is_equality in zip(
            structure.node_names, structure.row_is_equality
        )
        if not is_equality
    ]

    assert (
        sum(
            name.startswith(CartesianPositionVelocityLimit.__name__)
            for name in inequality_nodes
        )
        == structure.control_horizon
    )


def test_recorded_arrays_share_the_layout_of_the_problem(recording):
    structure = recording.structure
    expected_row_matrix_shape = (
        recording.number_of_cycles,
        structure.number_of_rows,
        structure.number_of_degrees_of_freedom,
    )
    expected_row_shape = (recording.number_of_cycles, structure.number_of_rows)

    assert recording.row_sensitivities.shape == expected_row_matrix_shape
    assert recording.row_contributions.shape == expected_row_matrix_shape
    assert recording.row_lower_bounds.shape == expected_row_shape
    assert recording.row_upper_bounds.shape == expected_row_shape
    assert recording.row_weights.shape == expected_row_shape
    assert recording.row_slacks.shape == expected_row_shape
    assert recording.velocities.shape == (
        recording.number_of_cycles,
        structure.number_of_degrees_of_freedom,
    )


def test_an_equality_row_is_recorded_as_the_quadratic_program_states_it(recording):
    """
    Every equality row of the quadratic program reads ``coefficients @ velocities +
    dt * slack = bound``.  Recording that identity back to within solver tolerance shows
    the folded gradients, the commands, the bounds and the slack all refer to the same
    row.
    """
    structure = recording.structure
    is_equality = structure.row_is_equality
    achieved = recording.row_contributions.sum(axis=2)[:, is_equality]
    slack = recording.row_slacks[:, is_equality]

    np.testing.assert_allclose(
        achieved + structure.model_predictive_control_time_step * slack,
        recording.row_lower_bounds[:, is_equality],
        atol=1e-6,
    )


def test_contributions_only_credit_joints_the_row_is_sensitive_to(recording):
    """
    A joint can only change a constraint's expression if the constraint has a gradient
    on it, so a contribution without a sensitivity would mean the two were misaligned.
    """
    has_no_sensitivity = recording.row_sensitivities == 0

    np.testing.assert_allclose(recording.row_contributions[has_no_sensitivity], 0)


def test_equality_rows_share_one_bound(recording):
    is_equality = recording.structure.row_is_equality

    np.testing.assert_allclose(
        recording.row_lower_bounds[:, is_equality],
        recording.row_upper_bounds[:, is_equality],
    )


# %% recorded values


def test_every_control_cycle_of_the_motion_is_recorded(recording):
    assert recording.number_of_cycles > 1
    assert np.all(np.diff(recording.times) > 0)


def test_each_cycle_records_its_own_numbers(recording):
    """
    The compiled problem evaluates into reused buffers, so a recording that stored views
    instead of values would report the last cycle for every cycle.
    """
    assert not np.allclose(
        recording.row_sensitivities[0], recording.row_sensitivities[-1]
    )


def test_running_rows_carry_a_weight(recording):
    assert np.all(recording.row_weights[0] > 0)


# %% replaying the pose


def test_the_pose_of_the_whole_world_is_recorded(recording):
    """
    A replay puts the objects the robot works on back too, so the recording covers every
    degree of freedom of the world rather than only the ones the optimizer moves.
    """
    number_of_world_degrees_of_freedom = len(recording.world_degree_of_freedom_ids)

    assert recording.world_positions.shape == (
        recording.number_of_cycles,
        number_of_world_degrees_of_freedom,
    )
    assert (
        number_of_world_degrees_of_freedom
        > recording.structure.number_of_degrees_of_freedom
    )


def test_the_recorded_pose_follows_the_motion(recording):
    assert not np.allclose(recording.world_positions[0], recording.world_positions[-1])


def test_recorder_without_a_cycle_cannot_build_a_recording():
    with pytest.raises(EmptyControlCycleRecordingError):
        ControlCycleRecorder().build_recording()


# %% storing


def test_stored_recording_reads_back_unchanged(recording, tmp_path):
    file_path = str(tmp_path / "goal_0.npz")

    recording.save(file_path)
    loaded = ControlCycleRecording.load(file_path)

    assert loaded.structure == recording.structure
    np.testing.assert_allclose(loaded.times, recording.times)
    np.testing.assert_allclose(loaded.row_sensitivities, recording.row_sensitivities)
    np.testing.assert_allclose(loaded.row_contributions, recording.row_contributions)
    np.testing.assert_allclose(loaded.row_lower_bounds, recording.row_lower_bounds)
    np.testing.assert_allclose(loaded.row_upper_bounds, recording.row_upper_bounds)
    np.testing.assert_allclose(loaded.row_weights, recording.row_weights)
    np.testing.assert_allclose(loaded.row_slacks, recording.row_slacks, equal_nan=True)
    np.testing.assert_allclose(loaded.velocities, recording.velocities)
    np.testing.assert_allclose(loaded.world_positions, recording.world_positions)
    assert loaded.world_degree_of_freedom_ids == recording.world_degree_of_freedom_ids


def test_recording_of_an_unknown_format_version_is_rejected(recording, tmp_path):
    file_path = str(tmp_path / "goal_0.npz")
    recording.save(file_path)
    archive = dict(np.load(file_path, allow_pickle=False))
    archive["metadata"] = np.array(
        str(archive["metadata"].item()).replace(
            f'"format_version": {RECORDING_FORMAT_VERSION}', '"format_version": 0'
        )
    )
    np.savez_compressed(file_path, **archive)

    with pytest.raises(UnknownRecordingFormatVersionError):
        ControlCycleRecording.load(file_path)
