from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, make_dataclass

from typing_extensions import Any, Dict, List, Type

from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex_mcp.catalogue import CapabilityCatalogue
from coraplex_mcp.exceptions import (
    DuplicateCapability,
    MalformedBinding,
    UnknownParameterType,
)
from coraplex_mcp.marshaling import ValueMarshaller
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body

# %% bindings


class Binding(ABC):
    """
    Supplies one argument to a step of an authored capability, resolved when the
    capability builds its plan.
    """

    @abstractmethod
    def resolve(
        self,
        action: ActionDescription,
        field_type: Type[Any],
        marshaller: ValueMarshaller,
    ) -> Any:
        """
        :param action: The authored-capability instance being expanded.
        :param field_type: The type the step field expects.
        :param marshaller: The marshaller used to convert literal values.
        :return: The argument value for the step field.
        """


@dataclass
class FieldReference(Binding):
    """
    Binds a step argument to one of the authored capability's own fields, so the value
    is supplied when the capability is performed.
    """

    field_name: str
    """
    The authored-capability field whose value is forwarded to the step.
    """

    def resolve(
        self,
        action: ActionDescription,
        field_type: Type[Any],
        marshaller: ValueMarshaller,
    ) -> Any:
        return getattr(action, self.field_name)


@dataclass
class LiteralValue(Binding):
    """
    Binds a step argument to a constant value fixed when the capability is authored.
    """

    value: Any
    """
    The raw value, marshaled to the step field type when the capability is performed.
    """

    def resolve(
        self,
        action: ActionDescription,
        field_type: Type[Any],
        marshaller: ValueMarshaller,
    ) -> Any:
        return marshaller.marshal(self.value, field_type, action.world)


def binding_from_dict(description: Dict[str, Any]) -> Binding:
    """
    :param description: A binding as ``{"from_field": <name>}`` or ``{"value": <raw>}``.
    :return: The binding it describes.
    :raises MalformedBinding: If the description is neither form.
    """
    if "from_field" in description:
        return FieldReference(description["from_field"])
    if "value" in description:
        return LiteralValue(description["value"])
    raise MalformedBinding(description)


# %% specification


@dataclass
class CapabilityParameter:
    """
    One field of an authored capability.
    """

    name: str
    """
    The field name.
    """

    type_name: str
    """
    The name of the field type, resolved against the factory's authorable types.
    """


@dataclass
class CapabilityStep:
    """
    One step in the plan of an authored capability.
    """

    capability: str
    """
    The name of the existing capability this step performs.
    """

    arguments: Dict[str, Binding]
    """
    The bindings supplying the step's arguments, keyed by the step field name.
    """


@dataclass
class CompositeCapabilitySpec:
    """
    The declarative description of a new action composed from existing capabilities.
    """

    name: str
    """
    The class name of the authored capability.
    """

    documentation: str
    """
    The docstring of the authored capability.
    """

    parameters: List[CapabilityParameter]
    """
    The fields of the authored capability.
    """

    steps: List[CapabilityStep]
    """
    The ordered capabilities the authored capability performs.
    """

    @classmethod
    def from_dict(cls, description: Dict[str, Any]) -> CompositeCapabilitySpec:
        """
        :param description: A specification with ``name``, ``documentation``,
            ``parameters`` and ``steps`` entries.
        :return: The specification it describes.
        """
        parameters = [
            CapabilityParameter(parameter["name"], parameter["type"])
            for parameter in description.get("parameters", [])
        ]
        steps = [
            CapabilityStep(
                step["capability"],
                {
                    name: binding_from_dict(binding)
                    for name, binding in step.get("arguments", {}).items()
                },
            )
            for step in description.get("steps", [])
        ]
        return cls(
            name=description["name"],
            documentation=description.get("documentation", ""),
            parameters=parameters,
            steps=steps,
        )


# %% factory


def default_authorable_types() -> Dict[str, Type[Any]]:
    """
    :return: The field types an authored capability may declare, keyed by name.
    """
    return {
        "Pose": Pose,
        "Body": Body,
        "Arms": Arms,
        "GripperState": GripperState,
        "TorsoState": TorsoState,
        "GraspDescription": GraspDescription,
        "bool": bool,
        "int": int,
        "float": float,
        "str": str,
    }


@dataclass
class CompositeCapabilityFactory:
    """
    Synthesizes a new action from a declarative specification.

    The synthesized class is an ordinary :class:`ActionDescription` subclass registered
    in the catalogue, so an authored capability is discovered, constructed and performed
    exactly like a built-in one, and can itself be composed into a plan.
    """

    catalogue: CapabilityCatalogue
    """
    The catalogue the authored capability is registered in and whose capabilities its
    steps reference.
    """

    marshaller: ValueMarshaller = field(default_factory=ValueMarshaller)
    """
    The marshaller used to convert literal step arguments when the capability performs.
    """

    authorable_types: Dict[str, Type[Any]] = field(
        default_factory=default_authorable_types
    )
    """
    The field types an authored capability may declare.
    """

    def define(self, specification: CompositeCapabilitySpec) -> Type[ActionDescription]:
        """
        Synthesize and register the action described by ``specification``.

        :param specification: The declarative description of the new action.
        :return: The synthesized action class.
        :raises DuplicateCapability: If the name is already registered.
        :raises UnknownCapability: If a step references an unknown capability.
        :raises UnknownParameterType: If a field declares an unresolvable type.
        """
        if specification.name in self.catalogue.names():
            raise DuplicateCapability(specification.name)
        for step in specification.steps:
            self.catalogue.capability_type(step.capability)
        parameter_fields = [
            (parameter.name, self._resolve_type(parameter.type_name))
            for parameter in specification.parameters
        ]
        synthesized = make_dataclass(
            specification.name,
            parameter_fields,
            bases=(ActionDescription,),
            namespace={
                "_action_plan": property(self._plan_builder(specification)),
                "__doc__": specification.documentation,
                "__module__": __name__,
            },
        )
        self.catalogue.register_action(synthesized)
        return synthesized

    def _resolve_type(self, type_name: str) -> Type[Any]:
        """
        :param type_name: The declared field type name.
        :return: The type it names.
        :raises UnknownParameterType: If the name is not authorable.
        """
        if type_name not in self.authorable_types:
            raise UnknownParameterType(type_name, list(self.authorable_types))
        return self.authorable_types[type_name]

    def _plan_builder(self, specification: CompositeCapabilitySpec):
        """
        :param specification: The description whose steps the plan performs.
        :return: A property getter that builds the authored capability's sub-plan.
        """
        factory = self

        def action_plan(action: ActionDescription) -> PlanNode:
            return sequential(
                [factory._build_step(step, action) for step in specification.steps]
            )

        return action_plan

    def _build_step(
        self, step: CapabilityStep, action: ActionDescription
    ) -> ActionDescription:
        """
        :param step: The step to instantiate.
        :param action: The authored-capability instance being expanded.
        :return: The capability instance the step performs, with its arguments bound.
        """
        step_type = self.catalogue.capability_type(step.capability)
        field_types = {f.name: f.type for f in step_type.fields}
        arguments = {
            name: binding.resolve(action, field_types[name], self.marshaller)
            for name, binding in step.arguments.items()
        }
        return step_type(**arguments)
