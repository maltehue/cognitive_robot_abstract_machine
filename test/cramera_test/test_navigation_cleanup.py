"""
Failed controllers release collision consumers and pending motion commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import NoReturn

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.motions.navigation import MoveMotion
from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_controller import QPController
from giskardpy.qp.qp_controller_config import QPControllerConfig
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World

# %% failed controller resources


@dataclass(eq=False, repr=False)
class CleanupTrackedPose(CartesianPose):
    """
    A Cartesian goal recording whether its cleanup hook ran.
    """

    cleaned: bool = field(default=False, init=False)
    """
    Whether the executable released this goal's resources.
    """

    def cleanup(self, context: MotionStatechartContext) -> None:
        """
        Record cleanup while preserving the inherited hook.

        :param context: The simulation context being released.
        """
        super().cleanup(context)
        self.cleaned = True


@dataclass
class SimulationFailureProbe:
    """
    Inject a controller failure after observing active resources.
    """

    world: World
    """
    World used by the real executor.
    """

    error: InfeasibleException
    """
    Exception whose identity must survive cleanup.
    """

    def fail_compile(self, controller_config: QPControllerConfig) -> None:
        """
        Fail after chart compilation registers collision consumers.

        :param controller_config: Configuration supplied by the executor.
        """
        self.raise_failure()

    def fail_control(
        self,
        world_state: np.ndarray,
        life_cycle_state: np.ndarray,
        float_variables: np.ndarray,
    ) -> np.ndarray:
        """
        Fail at the controller's command boundary.

        :param world_state: Current state array.
        :param life_cycle_state: Compiled lifecycle array.
        :param float_variables: Current symbolic values.
        """
        self.raise_failure()

    def raise_failure(self) -> NoReturn:
        """
        Leave registered resources and pending commands for cleanup.
        """
        assert self.world.collision_manager.collision_consumers
        self.world.state.velocities[:] = 0.2
        self.world.state.accelerations[:] = 0.3
        self.world.state.jerks[:] = 0.4
        raise self.error


@pytest.mark.parametrize("failure_stage", ["compile", "control"])
def test_simulation_cleans_resources_after_failure(
    cylinder_bot_world: World, monkeypatch: pytest.MonkeyPatch, failure_stage: str
) -> None:
    """
    Release collision consumers and commands when compilation or control fails.

    :param cylinder_bot_world: Existing mobile-robot fixture.
    :param monkeypatch: Restores the injected controller failure.
    :param failure_stage: Boundary at which the failure is injected.
    """
    world = cylinder_bot_world
    context = Context.from_world(world)
    target = Pose.from_xyz_rpy(0.2, reference_frame=world.root)
    motion_node = execute_single(MoveMotion(target), context=context)
    task = CleanupTrackedPose(
        root_link=world.root, tip_link=context.robot.root, goal_pose=target
    )
    executable = GiskardExecutable(motion_mappings={motion_node: task}, context=context)
    failure = InfeasibleException(solver_status="cleanup regression")
    probe = SimulationFailureProbe(world, failure)
    if failure_stage == "compile":
        monkeypatch.setattr(Executor, "_compile_qp_controller", probe.fail_compile)
    else:
        monkeypatch.setattr(QPController, "compute_command", probe.fail_control)
    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        with pytest.raises(InfeasibleException) as raised:
            executable.execute()
    assert raised.value is failure
    assert task.cleaned
    assert not world.collision_manager.collision_consumers
    np.testing.assert_array_equal(world.state.velocities, 0)
    np.testing.assert_array_equal(world.state.accelerations, 0)
    np.testing.assert_array_equal(world.state.jerks, 0)
