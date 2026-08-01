from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List

from krrood.exceptions import DataclassException


@dataclass
class UnknownCapability(DataclassException):
    """
    Raised when a capability name is requested that the catalogue does not contain.
    """

    name: str
    """
    The requested capability name.
    """

    available: List[str]
    """
    The capability names the catalogue currently knows.
    """

    def error_message(self) -> str:
        return f"Unknown capability {self.name!r}."

    def suggest_correction(self) -> str:
        return f"Choose one of the registered capabilities: {sorted(self.available)}."


@dataclass
class UnknownSession(DataclassException):
    """
    Raised when a session identifier is used that the registry does not hold.
    """

    session_id: str
    """
    The requested session identifier.
    """

    def error_message(self) -> str:
        return f"Unknown session {self.session_id!r}."

    def suggest_correction(self) -> str:
        return "Open a session before referring to it."


@dataclass
class UnmarshalableValue(DataclassException):
    """
    Raised when a tool argument cannot be converted into the type a capability field
    expects.
    """

    field_name: str
    """
    The capability field the value was meant for.
    """

    expected_type: type
    """
    The type the field expects.
    """

    value: object
    """
    The value that could not be converted.
    """

    def error_message(self) -> str:
        return (
            f"Cannot convert {self.value!r} for field {self.field_name!r} into "
            f"{self.expected_type!r}."
        )

    def suggest_correction(self) -> str:
        return "Supply the argument in the shape the capability schema declares."


@dataclass
class UnknownParameterType(DataclassException):
    """
    Raised when an authored capability declares a field type the factory cannot resolve.
    """

    type_name: str
    """
    The requested type name.
    """

    available: List[str]
    """
    The type names the factory can resolve.
    """

    def error_message(self) -> str:
        return f"Unknown parameter type {self.type_name!r}."

    def suggest_correction(self) -> str:
        return f"Choose one of the authorable types: {sorted(self.available)}."


@dataclass
class MalformedBinding(DataclassException):
    """
    Raised when a step argument is neither a field reference nor a literal value.
    """

    argument: object
    """
    The malformed binding description.
    """

    def error_message(self) -> str:
        return f"Binding {self.argument!r} is neither a field reference nor a literal."

    def suggest_correction(self) -> str:
        return "Provide either {'from_field': <name>} or {'value': <literal>}."


@dataclass
class DuplicateCapability(DataclassException):
    """
    Raised when an authored capability reuses the name of an existing capability.
    """

    name: str
    """
    The capability name that already exists.
    """

    def error_message(self) -> str:
        return f"Capability {self.name!r} already exists and cannot be redefined."

    def suggest_correction(self) -> str:
        return "Author the capability under a new, unused name."
