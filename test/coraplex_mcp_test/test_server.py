from dataclasses import dataclass

from coraplex.datastructures.dataclasses import Context
from coraplex_mcp.server import ControlFlow, RobotControlServer, build_mcp_server
from coraplex_mcp.world_provider import WorldProvider


@dataclass
class FixedContextWorldProvider(WorldProvider):
    """
    Hands out a context prepared in advance, so a session can be opened without loading
    a robot from disk.
    """

    context: Context
    """
    The context every session created through this provider receives.
    """

    def create_context(self) -> Context:
        return self.context


class TestControlFlowSelection:
    """
    Selecting the plan-language construct a plan is composed with.
    """

    def test_construct_parsed_from_name(self):
        assert ControlFlow("try_all") is ControlFlow.TRY_ALL

    def test_all_constructs_are_named(self):
        assert {construct.value for construct in ControlFlow} == {
            "sequential",
            "parallel",
            "try_in_order",
            "try_all",
        }

    def test_server_is_named(self):
        server = build_mcp_server(RobotControlServer())
        assert server.name == "coraplex-robot-control"


class TestSimulatedProgram:
    """
    Opening a session and performing actions against a world.
    """

    def test_open_session_reports_the_robot(self, simulated_context: Context):
        server = RobotControlServer(
            world_provider=FixedContextWorldProvider(simulated_context)
        )
        opened = server.open_session()
        assert opened["robot"] == "PR2"

    def test_listed_capabilities_include_navigation(self, simulated_context: Context):
        server = RobotControlServer(
            world_provider=FixedContextWorldProvider(simulated_context)
        )
        session_id = server.open_session()["session_id"]
        names = [
            capability["name"] for capability in server.list_capabilities(session_id)
        ]
        assert "NavigateAction" in names

    def test_perform_navigation_succeeds(self, simulated_context: Context):
        server = RobotControlServer(
            world_provider=FixedContextWorldProvider(simulated_context)
        )
        session_id = server.open_session()["session_id"]
        result = server.perform_action(
            session_id, "NavigateAction", {"target_location": {"x": 1.0, "y": 0.5}}
        )
        assert result["succeeded"] is True

    def test_authored_capability_is_performable(self, simulated_context: Context):
        server = RobotControlServer(
            world_provider=FixedContextWorldProvider(simulated_context)
        )
        session_id = server.open_session()["session_id"]
        server.author_capability(
            session_id,
            {
                "name": "DriveThenLook",
                "documentation": "Drive to a pose, then look at it.",
                "parameters": [{"name": "goal", "type": "Pose"}],
                "steps": [
                    {
                        "capability": "NavigateAction",
                        "arguments": {"target_location": {"from_field": "goal"}},
                    },
                    {
                        "capability": "LookAtAction",
                        "arguments": {"target": {"from_field": "goal"}},
                    },
                ],
            },
        )
        result = server.perform_action(
            session_id, "DriveThenLook", {"goal": {"x": 1.0, "y": 0.5}}
        )
        assert result["succeeded"] is True
