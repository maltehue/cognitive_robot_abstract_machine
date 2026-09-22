"""
Stop infeasible reachability candidates when the native controller stalls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from unittest.mock import patch

from coraplex.datastructures.dataclasses import Context
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.locations.pose_validator import AreReachableBy
from giskardpy.executor import Executor
from giskardpy.motion_statechart.exceptions import NoProgressError
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World


# %% native execution observations
@dataclass
class ReachabilityFailureTrace:
    """
    Observe controller termination without changing the candidate execution.
    """

    execute: Callable[[Executor, int], None]
    """Original native execution loop."""
    failures: list[Exception] = field(default_factory=list)
    """
    Reasons the original loop stopped unsuccessfully.
    """

    cycles: int = 0
    """
    Actual controller cycles before termination.
    """

    limit: int = 0
    """
    Maximum cycles requested by the validator.
    """

    def record(self, executor: Executor, timeout: int) -> None:
        """
        Forward unchanged execution and retain its termination evidence.

        :param executor: Native executor performing the candidate trajectory.
        :param timeout: Tick limit supplied by the validator.
        """
        self.limit = timeout
        try:
            self.execute(executor, timeout)
        except (TimeoutError, NoProgressError) as error:
            self.failures.append(error)
            raise
        finally:
            self.cycles = executor.control_cycles


# %% unreachable native goal
def test_stalled_candidate_is_rejected_before_its_tick_limit(
    pr2_world_copy: World,
) -> None:
    """
    A stalled arm candidate returns False before consuming the full tick budget.

    :param pr2_world_copy: Existing isolated native robot fixture.
    """
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    robot.mobile_base.full_body_controlled = False
    validator = AreReachableBy(
        context=Context(world=world, robot=robot),
        pose_sequence=[Pose.from_xyz_rpy(5, 0, 1, reference_frame=world.root)],
        tip_link=robot.left_arm.end_effector.tool_frame,
    )
    trace = ReachabilityFailureTrace(Executor.tick_until_end)
    with (
        simulated_robot_advanced,
        patch.object(
            Executor, "tick_until_end", autospec=True, side_effect=trace.record
        ),
    ):
        assert validator() is False

    [failure] = trace.failures
    assert isinstance(failure, NoProgressError)
    assert trace.cycles < trace.limit
