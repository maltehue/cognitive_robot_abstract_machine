import pytest

from coraplex_mcp.exceptions import MalformedWorldEntryPoint
from coraplex_mcp.world_provider import (
    CallableWorldProvider,
    Pr2WorldProvider,
    world_provider_from_environment,
)


def _return_non_belief() -> object:
    """
    :return: A value that is neither a world nor a context, to check the provider
        rejects it.
    """
    return object()


class TestBeliefBinding:
    """
    Resolving the world a session designs against from the environment.
    """

    def test_absent_entry_point_uses_fresh_world(self):
        provider = world_provider_from_environment({})
        assert isinstance(provider, Pr2WorldProvider)

    def test_entry_point_binds_the_referenced_callable(self):
        provider = world_provider_from_environment(
            {"CORAPLEX_MCP_WORLD": "builtins:list"}
        )
        assert isinstance(provider, CallableWorldProvider)
        assert provider.context_factory is list

    def test_entry_point_without_colon_is_rejected(self):
        with pytest.raises(MalformedWorldEntryPoint):
            world_provider_from_environment({"CORAPLEX_MCP_WORLD": "not_a_reference"})

    def test_unknown_module_is_rejected(self):
        with pytest.raises(MalformedWorldEntryPoint):
            world_provider_from_environment(
                {"CORAPLEX_MCP_WORLD": "no_such_module:factory"}
            )

    def test_unknown_attribute_is_rejected(self):
        with pytest.raises(MalformedWorldEntryPoint):
            world_provider_from_environment(
                {"CORAPLEX_MCP_WORLD": "builtins:missing_factory"}
            )

    def test_non_belief_result_is_rejected(self):
        provider = CallableWorldProvider(_return_non_belief)
        with pytest.raises(MalformedWorldEntryPoint):
            provider.create_context()
