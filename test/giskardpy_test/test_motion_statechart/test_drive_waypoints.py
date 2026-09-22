"""
Differential-drive headings for paths compiled before driving.
"""

from __future__ import annotations

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.goals.cartesian_goals import DifferentialDriveBaseGoal
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World

# %% segment headings


@pytest.mark.parametrize("relative", [False, True])
def test_segment_heading_uses_its_planned_start(
    cylinder_bot_diff_world: World, relative: bool
) -> None:
    """
    A later path segment faces from its preceding waypoint toward its goal.

    :param cylinder_bot_diff_world: World with one annotated differential drive.
    :param relative: Whether to express the start in the drive child's frame.
    """
    world = cylinder_bot_diff_world
    from_frame = world.connections[-1].child if relative else world.root
    start = Pose.from_xyz_rpy(1, 0, reference_frame=from_frame)
    goal = Pose.from_xyz_rpy(1, 1, reference_frame=world.root)
    segment = DifferentialDriveBaseGoal(goal_pose=goal, start_pose=start)
    chart = MotionStatechart()
    chart.add_node(segment)
    executor = Executor(context=MotionStatechartContext(world=world))
    executor.compile(chart)
    expected = Pose.from_xyz_rpy(
        yaw=np.pi / 2, reference_frame=world.root
    ).to_rotation_matrix()
    np.testing.assert_allclose(
        segment.nodes[0].goal_orientation.to_np(), expected.to_np(), atol=1e-10
    )
    chart.cleanup_nodes(executor.context)
    executor.context.cleanup()
