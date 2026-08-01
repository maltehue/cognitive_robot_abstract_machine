from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum

from mcp.server.mcpserver import MCPServer
from typing_extensions import Any, Callable, Dict, List

from coraplex.plans.factories import (
    execute_single,
    parallel,
    sequential,
    try_all,
    try_in_order,
)
from coraplex.plans.plan_node import PlanNode
from coraplex_mcp.authoring import CompositeCapabilitySpec
from coraplex_mcp.exceptions import SessionLimitReached
from coraplex_mcp.results import tool_boundary
from coraplex_mcp.sessions import RobotSession, SessionRegistry
from coraplex_mcp.validation import SimulationValidator
from coraplex_mcp.world_provider import (
    Pr2WorldProvider,
    WorldProvider,
    world_provider_from_environment,
)

# %% control flow


class ControlFlow(Enum):
    """
    The plan-language constructs a plan of actions can be composed with.
    """

    SEQUENTIAL = "sequential"
    """
    Perform the actions one after another, failing on the first failure.
    """

    PARALLEL = "parallel"
    """
    Perform the actions concurrently.
    """

    TRY_IN_ORDER = "try_in_order"
    """
    Perform the actions in order until one succeeds.
    """

    TRY_ALL = "try_all"
    """
    Perform the actions concurrently and succeed if any one does.
    """


_CONTROL_FLOW_FACTORIES: Dict[ControlFlow, Callable] = {
    ControlFlow.SEQUENTIAL: sequential,
    ControlFlow.PARALLEL: parallel,
    ControlFlow.TRY_IN_ORDER: try_in_order,
    ControlFlow.TRY_ALL: try_all,
}
"""
The plan-language factory for each control-flow construct.
"""


# %% server


@dataclass
class RobotControlServer:
    """
    Binds the CoraPlex capability catalogue, authoring and simulation to MCP tools, so
    an agent can compose robot programs and author new perception and manipulation
    capabilities for them.

    All performances run in the simulated robot; the real robot is not driven. Operations
    are serialized, so the global execution state CoraPlex uses stays consistent when a
    client issues overlapping calls.
    """

    world_provider: WorldProvider = field(default_factory=Pr2WorldProvider)
    """
    The source of the world and robot each new session operates on.
    """

    registry: SessionRegistry = field(default_factory=SessionRegistry)
    """
    The open robot sessions.
    """

    validator: SimulationValidator = field(default_factory=SimulationValidator)
    """
    The simulator used to perform capabilities and plans.
    """

    max_sessions: int = 32
    """
    The maximum number of sessions open at once, bounding memory use.
    """

    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )
    """
    Serializes operations so overlapping calls do not race on shared execution state.
    """

    def open_session(self) -> Dict[str, Any]:
        """
        Open a session on a fresh world and robot.

        :return: The session identifier and the robot's name.
        :raises SessionLimitReached: If the session limit is already reached.
        """
        with self._lock:
            if len(self.registry.identifiers()) >= self.max_sessions:
                raise SessionLimitReached(self.max_sessions)
            session = self.registry.open_session(
                uuid.uuid4().hex, self.world_provider.create_context()
            )
            return {
                "session_id": session.identifier,
                "robot": type(session.context.robot).__name__,
            }

    def list_capabilities(self, session_id: str) -> List[Dict[str, Any]]:
        """
        :param session_id: The session whose capabilities are listed.
        :return: The schema of every capability the session can construct.
        """
        with self._lock:
            session = self.registry.session(session_id)
            return [schema.to_dict() for schema in session.catalogue.schemas()]

    def describe_capability(self, session_id: str, name: str) -> Dict[str, Any]:
        """
        :param session_id: The session the capability belongs to.
        :param name: The capability name.
        :return: The schema of the capability.
        """
        with self._lock:
            return self.registry.session(session_id).catalogue.schema(name).to_dict()

    def perform_action(
        self, session_id: str, action_type: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Construct and perform one action in the simulated robot.

        :param session_id: The session to perform in.
        :param action_type: The capability name to construct.
        :param parameters: The capability arguments keyed by field name.
        :return: The outcome of the performance.
        """
        with self._lock:
            session = self.registry.session(session_id)
            action = session.construct_capability(action_type, parameters)
            node = execute_single(action, context=session.context)
            return self.validator.validate_plan(node).to_dict()

    def run_plan(
        self, session_id: str, control_flow: str, steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compose actions with a control-flow construct and perform them in simulation.

        :param session_id: The session to perform in.
        :param control_flow: The control-flow construct name.
        :param steps: The actions to compose, each ``{"action_type", "parameters"}``.
        :return: The outcome of the performance.
        """
        with self._lock:
            session = self.registry.session(session_id)
            node = self._build_plan(session, ControlFlow(control_flow), steps)
            return self.validator.validate_plan(node).to_dict()

    def author_capability(
        self, session_id: str, specification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Author a new action from existing capabilities and register it in the session.

        :param session_id: The session to author into.
        :param specification: The declarative capability specification.
        :return: The schema of the authored capability.
        """
        with self._lock:
            session = self.registry.session(session_id)
            session.factory.define(CompositeCapabilitySpec.from_dict(specification))
            return session.catalogue.schema(specification["name"]).to_dict()

    def world_state(self, session_id: str) -> Dict[str, Any]:
        """
        :param session_id: The session whose world is inspected.
        :return: The names of the bodies in the session's world.
        """
        with self._lock:
            session = self.registry.session(session_id)
            return {"bodies": [body.name.name for body in session.context.world.bodies]}

    def close_session(self, session_id: str) -> Dict[str, Any]:
        """
        :param session_id: The session to close.
        :return: A confirmation the session was closed.
        """
        with self._lock:
            self.registry.close_session(session_id)
            return {"closed": session_id}

    def _build_plan(
        self,
        session: RobotSession,
        control_flow: ControlFlow,
        steps: List[Dict[str, Any]],
    ) -> PlanNode:
        """
        :param session: The session the actions are constructed against.
        :param control_flow: The control-flow construct composing the actions.
        :param steps: The actions to compose.
        :return: The root node of the composed plan.
        """
        actions = [
            session.construct_capability(
                step["action_type"], step.get("parameters", {})
            )
            for step in steps
        ]
        return _CONTROL_FLOW_FACTORIES[control_flow](actions, context=session.context)


_TOOL_DESCRIPTIONS: Dict[str, str] = {
    "open_session": (
        "Open a session on a world and robot and return its 'session_id'. Call this "
        "first; pass the id to every other tool."
    ),
    "list_capabilities": (
        "List every capability the session can construct, each with its name, kind and "
        "typed parameters."
    ),
    "describe_capability": "Return one capability's name, kind and typed parameters.",
    "perform_action": (
        "Construct and perform one action in simulation. 'parameters' maps each field "
        "to a value: enums by name ('RIGHT'), a pose as {x, y, z, qx, qy, qz, qw}, a "
        "body by its name. Returns the performance status."
    ),
    "run_plan": (
        "Compose actions and perform them. 'control_flow' is one of sequential, "
        "parallel, try_in_order, try_all. 'steps' is a list of "
        "{action_type, parameters}."
    ),
    "author_capability": (
        "Author a new action from existing capabilities. 'specification' has name, "
        "documentation, parameters ([{name, type}]) and steps ([{capability, "
        "arguments}]); each argument is {from_field: <name>} or {value: <literal>}. The "
        "authored capability is then usable like a built-in."
    ),
    "world_state": "List the names of the bodies in the session's world.",
    "close_session": "Close a session and release its world.",
}
"""
The client-facing description of each tool, keyed by the backing method name.
"""


def build_mcp_server(server: RobotControlServer) -> MCPServer:
    """
    Register the robot-control tools on an MCP server.

    Each tool is wrapped so it returns a success or failure envelope and never raises, so
    malformed input is reported rather than crashing the server.

    :param server: The robot-control server whose methods back the tools.
    :return: The configured MCP server.
    """
    mcp = MCPServer(
        name="coraplex-robot-control",
        instructions=(
            "Compose CoraPlex robot programs and author new perception and "
            "manipulation capabilities against a simulated robot. Open a session, list "
            "capabilities, then perform actions, run plans, or author capabilities that "
            "slot into those plans. Every tool returns {ok, data} on success or "
            "{ok, error} on failure."
        ),
    )
    for method_name, description in _TOOL_DESCRIPTIONS.items():
        operation = getattr(server, method_name)
        mcp.tool(description=description)(tool_boundary(operation))
    return mcp


def main() -> None:
    """
    Run the robot-control MCP server over standard input and output.

    The world sessions design against is resolved from the environment, so the server
    can be pointed at an existing belief without code changes.
    """
    provider = world_provider_from_environment(os.environ)
    build_mcp_server(RobotControlServer(world_provider=provider)).run("stdio")


if __name__ == "__main__":
    main()
