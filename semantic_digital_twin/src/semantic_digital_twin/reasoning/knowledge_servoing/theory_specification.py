"""
The specification a theory can be compiled from — the synthesis target.

A specification says, as plain data, what a hand-written theory says in code: which
constraints the controller must enforce, which decisions gate them and supply their
values, and the rules that conclude those decisions from a situation's facts. It is
deliberately the *smallest* such format: conditions are expressions over
``case.situation.<fact>``, decisions are names, constraints are a kind plus parameters.
Everything else — decision types, rule chaining, channel bindings — is the compiler's
job, so a synthesizer never touches the parts that are easy to get structurally wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Mapping, Optional, Tuple

from semantic_digital_twin.reasoning.knowledge_servoing.exceptions import (
    UnknownSpecificationFieldError,
)

_CONSTRAINT_FIELDS = ("identifier", "kind", "parameters", "gated_by", "value_from")
_RULE_FIELDS = ("concludes", "condition", "requires_concluded", "defeated_by", "value")
_THEORY_FIELDS = ("constraints", "rules")


def _reject_unknown_fields(data: Mapping[str, Any], allowed: Tuple[str, ...]) -> None:
    """
    Rejects any field the format does not define, so a synthesizer's guess fails loudly.

    :param data: One specification object as parsed JSON.
    :param allowed: The fields the object may carry.
    :raises UnknownSpecificationFieldError: on the first unknown field.
    """
    for field_name in data:
        if field_name not in allowed:
            raise UnknownSpecificationFieldError(
                field_name=field_name, allowed_fields=allowed
            )


@dataclass(frozen=True)
class ConstraintSpecification:
    """
    One constraint the specified theory requires, by kind and parameters.
    """

    identifier: str
    """
    Names the constraint within the theory.
    """

    kind: str
    """
    The declaration kind, resolved against a registry at compile time.
    """

    parameters: Mapping[str, Any] = field(default_factory=dict)
    """
    The declaration kind's fields: subject names and numeric parameters.
    """

    gated_by: Optional[str] = None
    """
    Name of the regime decision that activates the constraint; ``None`` keeps it always
    active.
    """

    value_from: Optional[str] = None
    """
    Name of the parameter decision that supplies the constraint's runtime value.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ConstraintSpecification:
        """
        Parses one constraint from its JSON object.

        :param data: The parsed JSON object.
        :return: The constraint specification.
        """
        _reject_unknown_fields(data, _CONSTRAINT_FIELDS)
        return cls(
            identifier=data["identifier"],
            kind=data["kind"],
            parameters=dict(data.get("parameters", {})),
            gated_by=data.get("gated_by"),
            value_from=data.get("value_from"),
        )


@dataclass(frozen=True)
class RuleSpecification:
    """
    One rule concluding a decision from a situation's facts.
    """

    concludes: str
    """
    Name of the decision the rule concludes.
    """

    condition: str
    """
    Boolean expression over ``case.situation.<fact>`` the rule fires on.
    """

    requires_concluded: Tuple[str, ...] = ()
    """
    Regime decisions that must already be concluded this cycle for the rule to fire.
    """

    defeated_by: Tuple[str, ...] = ()
    """
    Conditions that withdraw the conclusion even when the rule fires.
    """

    value: Optional[str] = None
    """
    Value expression for a parameter decision; regime rules carry none.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> RuleSpecification:
        """
        Parses one rule from its JSON object.

        :param data: The parsed JSON object.
        :return: The rule specification.
        """
        _reject_unknown_fields(data, _RULE_FIELDS)
        value = data.get("value")
        return cls(
            concludes=data["concludes"],
            condition=data["condition"],
            requires_concluded=tuple(data.get("requires_concluded", ())),
            defeated_by=tuple(data.get("defeated_by", ())),
            value=None if value is None else str(value),
        )


@dataclass(frozen=True)
class TheorySpecification:
    """
    A whole theory as data: its constraints and the rules concluding their decisions.
    """

    constraints: Tuple[ConstraintSpecification, ...]
    """
    The constraints the theory requires the controller to enforce.
    """

    rules: Tuple[RuleSpecification, ...]
    """
    The rules concluding the decisions the constraints refer to.
    """

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> TheorySpecification:
        """
        Parses a whole specification from its JSON object.

        :param data: The parsed JSON object.
        :return: The theory specification.
        """
        _reject_unknown_fields(data, _THEORY_FIELDS)
        return cls(
            constraints=tuple(
                ConstraintSpecification.from_json(constraint)
                for constraint in data.get("constraints", ())
            ),
            rules=tuple(
                RuleSpecification.from_json(rule) for rule in data.get("rules", ())
            ),
        )
