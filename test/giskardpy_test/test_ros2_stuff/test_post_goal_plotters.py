from pathlib import Path

from giskardpy.middleware.ros2.post_goal_plotters import (
    GoalControlCycleRecorder,
    GoalGanttChartPlotter,
    GoalTrajectoryPlotter,
)

from .test_motion_server import create_executor

# %% recording is opt in


def test_creating_a_trajectory_plotter_does_not_record_yet():
    executor = create_executor()

    GoalTrajectoryPlotter(executor=executor)

    assert executor.trajectory_plotter is None


def test_start_recording_hands_the_trajectory_plotter_to_the_executor():
    executor = create_executor()
    plotter = GoalTrajectoryPlotter(executor=executor)

    plotter.start_recording()

    assert executor.trajectory_plotter is plotter.trajectory_plotter


def test_a_plotter_without_own_recording_leaves_the_executor_alone():
    executor = create_executor()

    GoalGanttChartPlotter(executor=executor).start_recording()

    assert executor.trajectory_plotter is None


def test_creating_a_control_cycle_recorder_does_not_record_yet():
    executor = create_executor()

    GoalControlCycleRecorder(executor=executor)

    assert executor.control_cycle_recorder is None


def test_start_recording_hands_the_control_cycle_recorder_to_the_executor():
    executor = create_executor()
    recorder = GoalControlCycleRecorder(executor=executor)

    recorder.start_recording()

    assert executor.control_cycle_recorder is recorder.control_cycle_recorder


# %% writing


def test_a_goal_without_control_cycles_writes_no_recording(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    recorder = GoalControlCycleRecorder(executor=create_executor())
    recorder.start_recording()

    recorder.plot(goal_id=0)

    assert list(Path(tmp_path).rglob("*.npz")) == []


def test_a_recording_is_named_after_its_goal(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    recorder = GoalControlCycleRecorder(executor=create_executor())

    file_name = recorder.create_file_name("control_cycles", 7, extension=".npz")

    assert Path(file_name) == Path(tmp_path) / "control_cycles" / "goal_7.npz"
