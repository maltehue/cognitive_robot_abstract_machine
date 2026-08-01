from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
from dataclasses import MISSING, dataclass, field
from enum import Enum, auto

from typing_extensions import Any, Dict, List, Optional, Type

from coraplex.plans.designator import Designator
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex_mcp.exceptions import UnknownCapability

# %% capability kinds


class CapabilityKind(Enum):
    """
    Distinguishes the two tiers of robot capability an agent can reference.
    """

    ACTION = auto()
    """
    A high-level action that expands into a sub-plan of motions or other actions.
    """

    MOTION = auto()
    """
    A low-level motion that maps to a single motion-statechart goal.
    """


# %% serializable schemas


@dataclass
class ParameterSchema:
    """
    The machine-readable description of one capability parameter.
    """

    name: str
    """
    The parameter name, matching the capability field.
    """

    type_name: str
    """
    The human-readable name of the type the parameter expects.
    """

    required: bool
    """
    Whether the parameter must be supplied because it has no default.
    """

    default: Optional[str]
    """
    The ``repr`` of the parameter default, or ``None`` when the parameter is required.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: A JSON-serializable view of this parameter.
        """
        return {
            "name": self.name,
            "type": self.type_name,
            "required": self.required,
            "default": self.default,
        }


@dataclass
class CapabilitySchema:
    """
    The machine-readable description of one robot capability an agent can construct.
    """

    name: str
    """
    The capability class name.
    """

    kind: CapabilityKind
    """
    Whether the capability is an action or a motion.
    """

    documentation: str
    """
    The capability class docstring.
    """

    parameters: List[ParameterSchema]
    """
    The constructable parameters of the capability.
    """

    def to_dict(self) -> Dict[str, Any]:
        """
        :return: A JSON-serializable view of this capability.
        """
        return {
            "name": self.name,
            "kind": self.kind.name,
            "documentation": self.documentation,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


# %% catalogue


_CAPABILITY_PACKAGES = (
    "coraplex.robot_plans.actions",
    "coraplex.robot_plans.motions",
)
"""
The packages whose modules are imported so their capability classes register as
subclasses.
"""


def _schema_for(capability: Type[Designator], kind: CapabilityKind) -> CapabilitySchema:
    """
    :param capability: The capability class to describe.
    :param kind: The tier the capability belongs to.
    :return: The serializable schema of the capability.
    """
    parameters = []
    for capability_field in capability.fields:
        required = (
            capability_field.default is MISSING
            and capability_field.default_factory is MISSING
        )
        default = None if required else repr(_field_default(capability_field))
        parameters.append(
            ParameterSchema(
                name=capability_field.name,
                type_name=getattr(
                    capability_field.type, "__name__", str(capability_field.type)
                ),
                required=required,
                default=default,
            )
        )
    return CapabilitySchema(
        name=capability.__name__,
        kind=kind,
        documentation=inspect.getdoc(capability) or "",
        parameters=parameters,
    )


def _field_default(capability_field: Any) -> Any:
    """
    :param capability_field: The dataclass field to read.
    :return: The field's default value, resolving a default factory when present.
    """
    if capability_field.default is not MISSING:
        return capability_field.default
    return capability_field.default_factory()


@dataclass
class CapabilityCatalogue:
    """
    The set of robot capabilities an agent can discover and construct.

    Authored capabilities register here alongside the built-in ones, so a synthesized
    capability is discoverable and constructable through the same interface.
    """

    _actions: Dict[str, Type[ActionDescription]] = field(default_factory=dict)
    """
    Concrete action capabilities keyed by class name.
    """

    _motions: Dict[str, Type[BaseMotion]] = field(default_factory=dict)
    """
    Concrete motion capabilities keyed by class name.
    """

    import_errors: Dict[str, str] = field(default_factory=dict)
    """
    Capability modules that failed to import, keyed by module name, with the error
    message. Populated during discovery so an unimportable capability module is
    observable rather than silently dropped.
    """

    @classmethod
    def from_installed_capabilities(cls) -> CapabilityCatalogue:
        """
        Import the capability packages and collect every concrete action and motion.

        A module that fails to import is recorded in :attr:`import_errors` and does not
        prevent discovery of the remaining capabilities.

        :return: A catalogue populated with the installed capabilities.
        """
        catalogue = cls()
        for package_name in _CAPABILITY_PACKAGES:
            catalogue.import_errors.update(_import_submodules(package_name))
        for action in _concrete_subclasses(ActionDescription):
            catalogue._actions[action.__name__] = action
        for motion in _concrete_subclasses(BaseMotion):
            catalogue._motions[motion.__name__] = motion
        return catalogue

    def register_action(self, action: Type[ActionDescription]) -> None:
        """
        :param action: The action capability to add to the catalogue.
        """
        self._actions[action.__name__] = action

    def copy(self) -> CapabilityCatalogue:
        """
        :return: An independent catalogue holding the same capabilities, so authored
            capabilities added to the copy do not affect the original.
        """
        return CapabilityCatalogue(
            dict(self._actions), dict(self._motions), dict(self.import_errors)
        )

    def capability_type(self, name: str) -> Type[Designator]:
        """
        :param name: The capability class name.
        :return: The capability class registered under ``name``.
        :raises UnknownCapability: If no capability is registered under ``name``.
        """
        if name in self._actions:
            return self._actions[name]
        if name in self._motions:
            return self._motions[name]
        raise UnknownCapability(name, self.names())

    def names(self) -> List[str]:
        """
        :return: The names of every registered capability.
        """
        return list(self._actions) + list(self._motions)

    def schema(self, name: str) -> CapabilitySchema:
        """
        :param name: The capability class name.
        :return: The schema of the capability registered under ``name``.
        :raises UnknownCapability: If no capability is registered under ``name``.
        """
        if name in self._actions:
            return _schema_for(self._actions[name], CapabilityKind.ACTION)
        if name in self._motions:
            return _schema_for(self._motions[name], CapabilityKind.MOTION)
        raise UnknownCapability(name, self.names())

    def schemas(self) -> List[CapabilitySchema]:
        """
        :return: The schema of every registered capability, actions before motions.
        """
        return [
            _schema_for(action, CapabilityKind.ACTION)
            for action in self._actions.values()
        ] + [
            _schema_for(motion, CapabilityKind.MOTION)
            for motion in self._motions.values()
        ]


def _import_submodules(package_name: str) -> Dict[str, str]:
    """
    Import every module under a package so its capability classes register as
    subclasses.

    :param package_name: The dotted package to import.
    :return: The modules that failed to import, keyed by module name, with the error
        message.
    """
    errors: Dict[str, str] = {}

    def record(failed_module_name: str) -> None:
        errors[failed_module_name] = repr(sys.exc_info()[1])

    package = importlib.import_module(package_name)
    for module in pkgutil.walk_packages(
        package.__path__, package.__name__ + ".", onerror=record
    ):
        try:
            importlib.import_module(module.name)
        except Exception as error:
            errors[module.name] = repr(error)
    return errors


def _concrete_subclasses(base: Type[Designator]) -> List[Type[Designator]]:
    """
    :param base: The capability base class.
    :return: The concrete (non-abstract) descendants of ``base``.
    """
    result = []
    for subclass in base.__subclasses__():
        result.extend(_concrete_subclasses(subclass))
        if not inspect.isabstract(subclass):
            result.append(subclass)
    return result
