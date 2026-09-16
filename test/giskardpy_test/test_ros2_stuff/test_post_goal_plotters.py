from pathlib import Path

from giskardpy.executor import Executor, NoPacing
from giskardpy.middleware.ros2.post_goal_plotters import (
    GoalControlCycleRecorder,
)
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from semantic_digital_twin.spatial_types import Point3

from .test_motion_server import create_executor

# %% recording is opt in


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


def test_a_recording_with_cycles_is_written(
    init_rospy, pr2_world_state_reset, tmp_path, monkeypatch
):
    """
    A goal that ran control cycles ends with its recording on disk and the save reported
    on the ros node.
    """
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    world = pr2_world_state_reset
    executor = Executor(MotionStatechartContext(world=world), pacer=NoPacing())
    recorder = GoalControlCycleRecorder(executor=executor)
    recorder.start_recording()
    reach = CartesianPosition(
        root_link=world.root,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
        goal_point=Point3(0.6, -0.3, 1.0, reference_frame=world.root),
    )
    motion_statechart = MotionStatechart()
    motion_statechart.add_node(reach)
    motion_statechart.add_node(EndMotion.when_true(reach))
    executor.compile(motion_statechart=motion_statechart)
    executor.tick()

    recorder.plot(goal_id=3)

    assert (Path(tmp_path) / "control_cycles" / "goal_3.npz").exists()
