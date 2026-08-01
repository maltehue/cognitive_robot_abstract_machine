import pytest

from coraplex_mcp.catalogue import CapabilityCatalogue, CapabilityKind
from coraplex_mcp.exceptions import UnknownCapability


class TestCapabilityDiscovery:
    """
    Reflecting the installed capabilities into serializable schemas.
    """

    def test_navigate_action_parameters(self, catalogue: CapabilityCatalogue):
        schema = catalogue.schema("NavigateAction")
        parameters = {parameter.name: parameter for parameter in schema.parameters}
        assert schema.kind is CapabilityKind.ACTION
        assert parameters["target_location"].type_name == "Pose"
        assert parameters["target_location"].required is True
        assert parameters["keep_joint_states"].type_name == "bool"
        assert parameters["keep_joint_states"].required is False

    def test_motion_is_classified_as_motion(self, catalogue: CapabilityCatalogue):
        assert catalogue.schema("MoveMotion").kind is CapabilityKind.MOTION

    def test_builtin_actions_are_listed(self, catalogue: CapabilityCatalogue):
        names = catalogue.names()
        assert "NavigateAction" in names
        assert "LookAtAction" in names

    def test_unknown_capability_is_rejected(self, catalogue: CapabilityCatalogue):
        with pytest.raises(UnknownCapability):
            catalogue.schema("NotACapability")
