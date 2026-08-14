"""Build-time errors raised while binding a theory's decisions to the controller's write channels."""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Type

from krrood.exceptions import DataclassException


@dataclass
class DecisionBindingError(DataclassException):
    """Base for errors raised while validating a :class:`DecisionBindingPolicy`."""


@dataclass
class UnboundDecisionTypeError(DecisionBindingError):
    """Raised when a theory can conclude a decision type the policy binds to neither channel."""

    decision_type: Type
    """The decision type the theory declares but the policy does not bind."""

    def error_message(self) -> str:
        return (
            f"Decision type '{self.decision_type.__name__}' is concluded by the theory but bound "
            f"to neither the activation nor the parameterization channel"
        )

    def suggest_correction(self) -> str:
        return "Bind it with activate(...) or parameterize(...), or remove it from the theory"


@dataclass
class DecisionTypeAlreadyBoundError(DecisionBindingError):
    """Raised when a decision type is bound a second time.

    A decision addresses exactly one channel, so binding the same type again — to the same channel or
    to the other one — is ambiguous.
    """

    decision_type: Type
    """The decision type that was already bound."""

    def error_message(self) -> str:
        return (
            f"Decision type '{self.decision_type.__name__}' is already bound; a decision addresses "
            f"exactly one channel once"
        )

    def suggest_correction(self) -> str:
        return "Bind each decision type exactly once"


@dataclass
class DecisionChannelMismatchError(DecisionBindingError):
    """Raised when a decision type is bound to a channel its base class does not address."""

    decision_type: Type
    """The decision type whose base class does not match the channel."""

    expected_base: Type
    """The decision base the channel requires."""

    def error_message(self) -> str:
        return (
            f"Decision type '{self.decision_type.__name__}' is not a "
            f"'{self.expected_base.__name__}' and cannot be bound to this channel"
        )

    def suggest_correction(self) -> str:
        return f"Bind only '{self.expected_base.__name__}' subclasses to this channel"


@dataclass
class UnregisteredFloatVariableTargetError(DecisionBindingError):
    """Raised when a parameter decision is bound to a float variable that was never registered."""

    decision_type: Type
    """The parameter decision type whose target is unregistered."""

    def error_message(self) -> str:
        return (
            f"The float-variable target bound to '{self.decision_type.__name__}' is not registered "
            f"with the float-variable data, so its value could never reach the solver"
        )

    def suggest_correction(self) -> str:
        return (
            "Register the target before validating, e.g. in the theory node's build()"
        )
