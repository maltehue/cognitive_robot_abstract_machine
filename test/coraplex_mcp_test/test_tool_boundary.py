import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex_mcp.exceptions import SessionLimitReached, UnknownSession
from coraplex_mcp.results import tool_boundary
from coraplex_mcp.server import RobotControlServer
from coraplex_mcp.world_provider import WorldProvider


@dataclass
class StubWorldProvider(WorldProvider):
    """
    Hands out a minimal stand-in context, so session bookkeeping can be tested without
    building a world.
    """

    def create_context(self) -> Context:
        return SimpleNamespace(robot=SimpleNamespace(), world=SimpleNamespace())


class TestToolBoundary:
    """
    Turning tool outcomes into envelopes that never raise.
    """

    def test_success_is_wrapped_in_an_envelope(self):
        assert tool_boundary(lambda: {"value": 1})() == {
            "ok": True,
            "data": {"value": 1},
        }

    def test_known_failure_reports_message_and_suggestion(self):
        def operation():
            raise UnknownSession("missing")

        result = tool_boundary(operation)()
        assert result["ok"] is False
        assert result["error"]["type"] == "UnknownSession"
        assert result["error"]["suggestion"] == "Open a session before referring to it."

    def test_unexpected_failure_is_contained(self):
        def operation():
            raise ValueError("boom")

        result = tool_boundary(operation)()
        assert result["ok"] is False
        assert result["error"]["type"] == "InternalError"
        assert result["error"]["message"] == "boom"

    def test_known_failure_is_logged(self, caplog):
        def operation():
            raise UnknownSession("missing")

        with caplog.at_level(logging.WARNING, logger="coraplex_mcp"):
            tool_boundary(operation)()
        assert any(record.levelno == logging.WARNING for record in caplog.records)


class TestSessionCap:
    """
    Bounding the number of open sessions.
    """

    def test_open_session_returns_an_identifier(self):
        server = RobotControlServer(world_provider=StubWorldProvider())
        assert "session_id" in server.open_session()

    def test_further_sessions_are_rejected_at_the_limit(self):
        server = RobotControlServer(world_provider=StubWorldProvider(), max_sessions=2)
        server.open_session()
        server.open_session()
        with pytest.raises(SessionLimitReached):
            server.open_session()

    def test_limit_surfaces_as_a_failure_envelope(self):
        server = RobotControlServer(world_provider=StubWorldProvider(), max_sessions=1)
        tool_boundary(server.open_session)()
        result = tool_boundary(server.open_session)()
        assert result["error"]["type"] == "SessionLimitReached"
