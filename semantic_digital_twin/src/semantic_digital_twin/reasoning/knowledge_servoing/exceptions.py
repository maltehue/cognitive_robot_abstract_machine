"""
Errors raised while validating and compiling a theory specification.

Every error here is a rejection before execution: a specification that parses, validates
and compiles yields a well-formed theory by construction, so anything a synthesizer gets
wrong is reported at this boundary rather than misbehaving inside a running controller.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import FrozenSet, Tuple

from krrood.exceptions import DataclassException


@dataclass
class TheorySpecificationError(DataclassException):
    """
    Base for errors raised while parsing, validating or compiling a theory
    specification.
    """


@dataclass
class UnknownSpecificationFieldError(TheorySpecificationError):
    """
    Raised when a specification object carries a field the format does not define.
    """

    field_name: str
    """The unknown field."""

    allowed_fields: Tuple[str, ...]
    """
    The fields the object may carry.
    """

    def error_message(self) -> str:
        return f"Unknown specification field '{self.field_name}'"

    def suggest_correction(self) -> str:
        return f"Use only the fields {sorted(self.allowed_fields)}"


@dataclass
class MalformedConditionError(TheorySpecificationError):
    """
    Raised when a condition is not parseable as a Python expression.
    """

    source: str
    """The condition that failed to parse."""

    def error_message(self) -> str:
        return f"Condition is not a parseable expression: {self.source!r}"

    def suggest_correction(self) -> str:
        return "Write a boolean expression over case.situation.<fact>"


@dataclass
class ForbiddenConditionSyntaxError(TheorySpecificationError):
    """
    Raised when a condition uses syntax outside the validated grammar.

    The grammar is deliberately small — boolean logic, comparisons, arithmetic, numeric
    constants and ``case.situation.<fact>`` reads — because everything in it can be
    checked before the condition is ever evaluated.
    """

    source: str
    """The condition carrying the forbidden construct."""

    construct: str
    """
    The syntactic construct that is not allowed.
    """

    def error_message(self) -> str:
        return f"Condition {self.source!r} uses forbidden syntax: {self.construct}"

    def suggest_correction(self) -> str:
        return (
            "Use only boolean logic, comparisons, arithmetic, numeric constants and "
            "case.situation.<fact>"
        )


@dataclass
class UnknownSituationFactError(TheorySpecificationError):
    """
    Raised when a condition reads a fact the situation type does not have.
    """

    source: str
    """The condition reading the unknown fact."""

    fact_name: str
    """
    The fact the situation type does not define.
    """

    allowed_facts: FrozenSet[str]
    """The facts the situation type defines."""

    def error_message(self) -> str:
        return f"Condition {self.source!r} reads unknown fact '{self.fact_name}'"

    def suggest_correction(self) -> str:
        return f"Read one of {sorted(self.allowed_facts)}"


@dataclass
class UnknownDeclarationKindError(TheorySpecificationError):
    """
    Raised when a specification declares a constraint of a kind the registry does not
    know.
    """

    kind: str
    """The unknown kind."""

    known_kinds: FrozenSet[str]
    """
    The kinds the registry maps to declarations.
    """

    def error_message(self) -> str:
        return f"Unknown constraint kind '{self.kind}'"

    def suggest_correction(self) -> str:
        return f"Declare one of {sorted(self.known_kinds)}"


@dataclass
class InvalidDeclarationParametersError(TheorySpecificationError):
    """
    Raised when a constraint's parameters do not match its declaration kind's fields.
    """

    kind: str
    """The declaration kind the parameters were meant for."""

    unexpected: Tuple[str, ...]
    """
    Parameters the kind does not define.
    """

    missing: Tuple[str, ...]
    """Required parameters the specification did not supply."""

    def error_message(self) -> str:
        return (
            f"Parameters for constraint kind '{self.kind}' do not fit: "
            f"unexpected {sorted(self.unexpected)}, missing {sorted(self.missing)}"
        )

    def suggest_correction(self) -> str:
        return "Supply exactly the declaration kind's fields"


@dataclass
class UnknownDecisionNameError(TheorySpecificationError):
    """
    Raised when a rule concludes or requires a decision no constraint refers to.
    """

    name: str
    """The decision name nothing refers to."""

    known_names: FrozenSet[str]
    """
    The decision names the constraints define via gating and value channels.
    """

    def error_message(self) -> str:
        return f"Unknown decision name '{self.name}'"

    def suggest_correction(self) -> str:
        return (
            f"Conclude one of {sorted(self.known_names)}; a decision exists only if a "
            f"constraint is gated by it or takes its value from it"
        )


@dataclass
class DecisionRoleConflictError(TheorySpecificationError):
    """
    Raised when one decision name is used both to gate a constraint and to supply a
    value.
    """

    name: str
    """The decision name with conflicting roles."""

    def error_message(self) -> str:
        return f"Decision '{self.name}' both gates a constraint and supplies a value"

    def suggest_correction(self) -> str:
        return "A decision addresses exactly one channel; use two names"


@dataclass
class InvalidRuleValueError(TheorySpecificationError):
    """
    Raised when a rule's value does not fit the channel its decision addresses.
    """

    concludes: str
    """The decision the rule concludes."""

    reason: str
    """
    Why the value does not fit.
    """

    def error_message(self) -> str:
        return f"Rule concluding '{self.concludes}' has an invalid value: {self.reason}"

    def suggest_correction(self) -> str:
        return "Regime rules carry no value; parameter rules must carry one"


@dataclass
class UnconcludableDecisionError(TheorySpecificationError):
    """
    Raised when a constraint refers to a decision no rule can ever conclude.

    A constraint gated by such a decision would never activate, and a value channel fed
    by one would never receive a value — almost certainly a specification bug, so it is
    rejected.
    """

    name: str
    """The decision name no rule concludes."""

    def error_message(self) -> str:
        return (
            f"No rule concludes decision '{self.name}', so its constraint can never act"
        )

    def suggest_correction(self) -> str:
        return "Add a rule concluding it, or remove the constraint's reference to it"
