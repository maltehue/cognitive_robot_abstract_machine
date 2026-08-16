"""
Validates rule conditions before anything evaluates them.

A rule condition is Python source evaluated against the classification case, which makes
it exactly as expressive as the engine needs and exactly as dangerous as arbitrary code.
This validator closes that gap with a whitelist grammar: boolean logic, comparisons,
arithmetic, numeric constants, and reads of ``case.situation.<fact>`` where the fact is
one the situation type actually defines. A condition that passes cannot call anything,
reach anything but the situation's facts, or read a fact that does not exist — which is
the well-formedness half of trusting a synthesized theory.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields

from typing_extensions import FrozenSet, Type

from semantic_digital_twin.reasoning.knowledge_servoing.exceptions import (
    ForbiddenConditionSyntaxError,
    MalformedConditionError,
    UnknownSituationFactError,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import Situation

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.Eq,
    ast.NotEq,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Attribute,
    ast.Name,
    ast.Constant,
    ast.Load,
)
"""
The whole grammar a condition may use.
"""

_SCALAR_ANNOTATIONS = {"bool", "float", "int", bool, float, int}
"""
The annotations that make a field or property a readable fact.
"""


def situation_fact_names(situation_type: Type[Situation]) -> FrozenSet[str]:
    """
    The scalar facts a situation type exposes: its bool and numeric fields and
    properties.

    Object-valued fields — the situation's subjects — are deliberately not facts: a
    condition reasons about the situation's state, and reaching into a subject would be
    a read of live world objects the validator could not bound.

    :param situation_type: The situation type conditions read from.
    :return: The readable fact names.
    """
    names = {
        declared_field.name
        for declared_field in dataclass_fields(situation_type)
        if declared_field.type in _SCALAR_ANNOTATIONS
    }
    for klass in situation_type.__mro__:
        for attribute_name, attribute in vars(klass).items():
            if isinstance(attribute, property) and attribute.fget is not None:
                return_annotation = attribute.fget.__annotations__.get("return")
                if return_annotation in _SCALAR_ANNOTATIONS:
                    names.add(attribute_name)
    return frozenset(names)


@dataclass
class ConditionValidator:
    """
    Checks rule conditions against the whitelist grammar and a situation type's facts.
    """

    situation_type: Type[Situation]
    """
    The situation type whose facts conditions may read.
    """

    _allowed_facts: FrozenSet[str] = field(init=False)
    """
    The fact names the situation type exposes.
    """

    def __post_init__(self) -> None:
        self._allowed_facts = situation_fact_names(self.situation_type)

    def validate(self, source: str) -> None:
        """
        Rejects a condition unless it fits the grammar and reads only existing facts.

        :param source: The condition as Python expression source.
        :raises MalformedConditionError: if the source does not parse as an expression.
        :raises ForbiddenConditionSyntaxError: if it uses syntax outside the grammar.
        :raises UnknownSituationFactError: if it reads a fact the situation type lacks.
        """
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as error:
            raise MalformedConditionError(source=source) from error
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODE_TYPES):
                raise ForbiddenConditionSyntaxError(
                    source=source, construct=type(node).__name__
                )
            if isinstance(node, ast.Constant) and not isinstance(
                node.value, (bool, int, float)
            ):
                raise ForbiddenConditionSyntaxError(
                    source=source, construct=f"constant {node.value!r}"
                )
        _StructureCheck(source=source, allowed_facts=self._allowed_facts).visit(tree)


@dataclass
class _StructureCheck(ast.NodeVisitor):
    """
    Enforces that every name access is exactly ``case.situation.<known fact>``.
    """

    source: str
    """
    The condition being checked, for error reporting.
    """

    allowed_facts: FrozenSet[str]
    """
    The facts the situation type exposes.
    """

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            if node.value.id != "case" or node.attr != "situation":
                raise ForbiddenConditionSyntaxError(
                    source=self.source,
                    construct=f"attribute access on '{node.value.id}'",
                )
            return
        if isinstance(node.value, ast.Attribute):
            inner = node.value
            inner_is_situation = (
                isinstance(inner.value, ast.Name)
                and inner.value.id == "case"
                and inner.attr == "situation"
            )
            if not inner_is_situation:
                raise ForbiddenConditionSyntaxError(
                    source=self.source, construct="nested attribute access"
                )
            if node.attr not in self.allowed_facts:
                raise UnknownSituationFactError(
                    source=self.source,
                    fact_name=node.attr,
                    allowed_facts=self.allowed_facts,
                )
            return
        raise ForbiddenConditionSyntaxError(
            source=self.source, construct="attribute access on an expression"
        )

    def visit_Name(self, node: ast.Name) -> None:
        raise ForbiddenConditionSyntaxError(
            source=self.source, construct=f"bare name '{node.id}'"
        )
