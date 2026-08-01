from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex_mcp.validation import SimulationValidator
from semantic_digital_twin.spatial_types.spatial_types import Pose


class TestSimulatedValidation:
    """
    Reporting the outcome of performing a capability in the simulated robot.
    """

    def test_navigation_reaches_the_goal(self, simulated_context: Context):
        target = Pose.from_xyz_quaternion(
            1.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            reference_frame=simulated_context.world.root,
        )
        result = SimulationValidator().validate(
            NavigateAction(target_location=target), simulated_context
        )
        assert result.status is TaskStatus.SUCCEEDED
        assert result.succeeded is True
        assert result.reason is None
