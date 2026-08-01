import pytest
from typing_extensions import Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex_mcp.exceptions import UnmarshalableValue
from coraplex_mcp.marshaling import ValueMarshaller
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


class TestEnumMarshaling:
    """
    Resolving enum members from their names.
    """

    def test_member_resolved_by_name(self):
        assert ValueMarshaller().marshal("RIGHT", Arms, None) is Arms.RIGHT

    def test_optional_none_stays_none(self):
        assert ValueMarshaller().marshal(None, Optional[Arms], None) is None

    def test_optional_member_resolved_by_name(self):
        assert ValueMarshaller().marshal("LEFT", Optional[Arms], None) is Arms.LEFT

    def test_unknown_member_is_rejected(self):
        with pytest.raises(UnmarshalableValue):
            ValueMarshaller().marshal("SIDEWAYS", Arms, None)


class TestPrimitiveMarshaling:
    """
    Passing scalar arguments through unchanged.
    """

    def test_boolean_passed_through(self):
        assert ValueMarshaller().marshal(True, bool, None) is True


class TestReferenceMarshaling:
    """
    Resolving pose and body references against a world.
    """

    def test_pose_built_in_world_frame(self, simulated_context: Context):
        world = simulated_context.world
        pose = ValueMarshaller().marshal({"x": 1.0, "y": 2.0, "z": 0.0}, Pose, world)
        assert isinstance(pose, Pose)

    def test_body_resolved_by_name(self, simulated_context: Context):
        world = simulated_context.world
        expected = world.bodies[0]
        resolved = ValueMarshaller().marshal(expected.name, Body, world)
        assert resolved is expected
