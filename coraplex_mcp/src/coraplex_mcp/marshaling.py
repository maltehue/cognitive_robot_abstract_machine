from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from typing_extensions import Any, Dict, List, Type, Union, get_args, get_origin

from coraplex.plans.designator import Designator
from coraplex_mcp.exceptions import UnmarshalableValue
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

# %% converters


class TypeConverter(ABC):
    """
    Converts a JSON tool argument into one CoraPlex value type.
    """

    @abstractmethod
    def handles(self, target_type: Type[Any]) -> bool:
        """
        :param target_type: The type a capability field expects.
        :return: Whether this converter produces values of that type.
        """

    @abstractmethod
    def convert(self, value: Any, target_type: Type[Any], world: World) -> Any:
        """
        :param value: The JSON argument to convert.
        :param target_type: The type the capability field expects.
        :param world: The world the value is resolved against.
        :return: The converted CoraPlex value.
        """


@dataclass
class PrimitiveConverter(TypeConverter):
    """
    Passes through the JSON scalar types that CoraPlex fields accept directly.
    """

    _primitives: tuple = (bool, int, float, str)
    """
    The scalar types handled without conversion.
    """

    def handles(self, target_type: Type[Any]) -> bool:
        return target_type in self._primitives

    def convert(self, value: Any, target_type: Type[Any], world: World) -> Any:
        return value


@dataclass
class EnumConverter(TypeConverter):
    """
    Resolves an enum member from its name, so agents pass enums as readable strings.
    """

    def handles(self, target_type: Type[Any]) -> bool:
        return isinstance(target_type, type) and issubclass(target_type, Enum)

    def convert(self, value: Any, target_type: Type[Any], world: World) -> Any:
        if isinstance(value, target_type):
            return value
        if value in target_type.__members__:
            return target_type[value]
        raise UnmarshalableValue("enum", target_type, value)


@dataclass
class PoseConverter(TypeConverter):
    """
    Builds a pose in the world frame from its position and orientation components.
    """

    def handles(self, target_type: Type[Any]) -> bool:
        return target_type is Pose

    def convert(self, value: Any, target_type: Type[Any], world: World) -> Any:
        if isinstance(value, Pose):
            return value
        if not isinstance(value, dict):
            raise UnmarshalableValue("pose", target_type, value)
        return Pose.from_xyz_quaternion(
            value.get("x", 0.0),
            value.get("y", 0.0),
            value.get("z", 0.0),
            value.get("qx", 0.0),
            value.get("qy", 0.0),
            value.get("qz", 0.0),
            value.get("qw", 1.0),
            reference_frame=world.root,
        )


@dataclass
class BodyConverter(TypeConverter):
    """
    Resolves a world body from its name, so agents refer to objects by name.
    """

    def handles(self, target_type: Type[Any]) -> bool:
        return target_type is Body

    def convert(self, value: Any, target_type: Type[Any], world: World) -> Any:
        if isinstance(value, Body):
            return value
        if not isinstance(value, (str, PrefixedName)):
            raise UnmarshalableValue("body", target_type, value)
        return world.get_body_by_name(value)


_DEFAULT_CONVERTERS: List[TypeConverter] = [
    PrimitiveConverter(),
    EnumConverter(),
    PoseConverter(),
    BodyConverter(),
]
"""
The converters a marshaller uses when none are supplied, ordered so the cheapest checks
run first.
"""


# %% marshaller


@dataclass
class ValueMarshaller:
    """
    Converts JSON tool arguments into the CoraPlex value types a capability constructor
    expects, resolving object and pose references against a world.
    """

    converters: List[TypeConverter] = field(
        default_factory=lambda: list(_DEFAULT_CONVERTERS)
    )
    """
    The converters consulted in order for each argument.
    """

    def marshal(self, value: Any, target_type: Type[Any], world: World) -> Any:
        """
        :param value: The JSON argument to convert.
        :param target_type: The type the capability field expects.
        :param world: The world object and pose references are resolved against.
        :return: The converted CoraPlex value.
        :raises UnmarshalableValue: If no converter handles ``target_type``.
        """
        resolved_type = _unwrap_optional(target_type)
        if value is None and resolved_type is not target_type:
            return None
        for converter in self.converters:
            if converter.handles(resolved_type):
                return converter.convert(value, resolved_type, world)
        raise UnmarshalableValue("value", resolved_type, value)

    def marshal_parameters(
        self, capability: Type[Designator], parameters: Dict[str, Any], world: World
    ) -> Dict[str, Any]:
        """
        :param capability: The capability whose fields the arguments are converted for.
        :param parameters: The JSON arguments keyed by field name.
        :param world: The world references are resolved against.
        :return: The arguments converted to the field types, ready for construction.
        """
        type_by_name = {f.name: f.type for f in capability.fields}
        return {
            name: self.marshal(value, type_by_name[name], world)
            for name, value in parameters.items()
        }


def _unwrap_optional(target_type: Type[Any]) -> Type[Any]:
    """
    :param target_type: A possibly-optional type annotation.
    :return: The single non-``None`` member of an ``Optional`` annotation, or the type
        itself when it is not optional.
    """
    if get_origin(target_type) is not Union:
        return target_type
    non_none = [
        argument for argument in get_args(target_type) if argument is not type(None)
    ]
    if len(non_none) == 1:
        return non_none[0]
    return target_type
