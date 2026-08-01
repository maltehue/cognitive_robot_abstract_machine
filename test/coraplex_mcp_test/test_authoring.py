import pytest

from coraplex.robot_plans.actions.base import ActionDescription
from coraplex_mcp.authoring import (
    CapabilityParameter,
    CapabilityStep,
    CompositeCapabilityFactory,
    CompositeCapabilitySpec,
    FieldReference,
    LiteralValue,
)
from coraplex_mcp.catalogue import CapabilityCatalogue
from coraplex_mcp.exceptions import (
    DuplicateCapability,
    MalformedBinding,
    UnknownCapability,
    UnknownParameterType,
)


def _navigate_then_look() -> CompositeCapabilitySpec:
    """
    :return: A specification composing a navigation and a look-at into one action.
    """
    return CompositeCapabilitySpec(
        name="NavigateThenLook",
        documentation="Drive to a pose, then look at a target.",
        parameters=[
            CapabilityParameter("drive_to", "Pose"),
            CapabilityParameter("look_at", "Pose"),
        ],
        steps=[
            CapabilityStep(
                "NavigateAction", {"target_location": FieldReference("drive_to")}
            ),
            CapabilityStep("LookAtAction", {"target": FieldReference("look_at")}),
        ],
    )


class TestCompositeCapabilitySynthesis:
    """
    Synthesizing a new action from existing capabilities.
    """

    def test_synthesized_class_is_an_action(self, catalogue: CapabilityCatalogue):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        synthesized = factory.define(_navigate_then_look())
        assert issubclass(synthesized, ActionDescription)

    def test_synthesized_fields_match_specification(
        self, catalogue: CapabilityCatalogue
    ):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        synthesized = factory.define(_navigate_then_look())
        field_types = {field.name: field.type.__name__ for field in synthesized.fields}
        assert field_types == {"drive_to": "Pose", "look_at": "Pose"}

    def test_authored_capability_is_registered(self, catalogue: CapabilityCatalogue):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        factory.define(_navigate_then_look())
        assert "NavigateThenLook" in factory.catalogue.names()

    def test_action_plan_composes_the_steps(self, catalogue: CapabilityCatalogue):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        synthesized = factory.define(_navigate_then_look())
        instance = synthesized(drive_to="pose_a", look_at="pose_b")
        step_designators = [
            child.designator for child in instance._action_plan.children
        ]
        step_types = [type(designator).__name__ for designator in step_designators]
        assert step_types == ["NavigateAction", "LookAtAction"]

    def test_field_reference_binds_the_owning_field(
        self, catalogue: CapabilityCatalogue
    ):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        synthesized = factory.define(_navigate_then_look())
        instance = synthesized(drive_to="pose_a", look_at="pose_b")
        navigate = next(
            child.designator
            for child in instance._action_plan.children
            if type(child.designator).__name__ == "NavigateAction"
        )
        assert navigate.target_location == "pose_a"

    def test_duplicate_name_is_rejected(self, catalogue: CapabilityCatalogue):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        factory.define(_navigate_then_look())
        with pytest.raises(DuplicateCapability):
            factory.define(_navigate_then_look())

    def test_unknown_step_capability_is_rejected(self, catalogue: CapabilityCatalogue):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        specification = CompositeCapabilitySpec(
            "Broken", "", [], [CapabilityStep("NotACapability", {})]
        )
        with pytest.raises(UnknownCapability):
            factory.define(specification)

    def test_unknown_parameter_type_is_rejected(self, catalogue: CapabilityCatalogue):
        factory = CompositeCapabilityFactory(catalogue=catalogue.copy())
        specification = CompositeCapabilitySpec(
            "Broken", "", [CapabilityParameter("p", "NotAType")], []
        )
        with pytest.raises(UnknownParameterType):
            factory.define(specification)


class TestSpecificationParsing:
    """
    Parsing a declarative specification into bindings.
    """

    def test_field_reference_is_parsed(self):
        specification = CompositeCapabilitySpec.from_dict(
            {
                "name": "Example",
                "steps": [
                    {
                        "capability": "NavigateAction",
                        "arguments": {"target_location": {"from_field": "goal"}},
                    }
                ],
            }
        )
        binding = specification.steps[0].arguments["target_location"]
        assert binding == FieldReference("goal")

    def test_literal_value_is_parsed(self):
        specification = CompositeCapabilitySpec.from_dict(
            {
                "name": "Example",
                "steps": [
                    {
                        "capability": "MoveTorsoAction",
                        "arguments": {"torso_state": {"value": "HIGH"}},
                    }
                ],
            }
        )
        binding = specification.steps[0].arguments["torso_state"]
        assert binding == LiteralValue("HIGH")

    def test_malformed_binding_is_rejected(self):
        with pytest.raises(MalformedBinding):
            CompositeCapabilitySpec.from_dict(
                {
                    "name": "Example",
                    "steps": [{"capability": "NavigateAction", "arguments": {"x": {}}}],
                }
            )
