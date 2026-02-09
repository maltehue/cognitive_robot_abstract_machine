from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import assert_never, Any, Tuple

from random_events.interval import Bound
from random_events.interval import SimpleInterval
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Integer, Symbolic, Variable
from sqlalchemy import inspect, Column
from sqlalchemy.orm import Relationship
from typing_extensions import List, Optional

from krrood.adapters.json_serializer import list_like_classes
from krrood.class_diagrams.class_diagram import ClassDiagram, WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ormatic.dao import DataAccessObject, get_dao_class
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)

SKIPPED_FIELD_TYPES = (datetime,)


@dataclass
class Parameterizer:

    variables: List[Variable] = field(default_factory=list)
    simple_event: SimpleEvent = field(default_factory=lambda: SimpleEvent({}))

    def parameterize_dao(
        self, dao: DataAccessObject, prefix: str
    ) -> Tuple[List[Variable], Optional[SimpleEvent]]:
        """
        Create variables for all fields of a DataAccessObject.

        :return: A list of random event variables and a SimpleEvent containing the values.
        """

        original_class = dao.original_class()
        wrapped_class = WrappedClass(original_class)
        mapper = inspect(dao).mapper

        for wrapped_field in wrapped_class.fields:
            if wrapped_field.type_endpoint in SKIPPED_FIELD_TYPES:
                continue

            for column in mapper.columns:
                vars, vals = self._process_column(column, wrapped_field, dao, prefix)

                for val, var in zip(vals, vars):
                    if var is None:
                        continue
                    self.variables.append(var)
                    if val is None:
                        continue

                    event = self._create_simple_event_singleton_from_set_attribute(
                        var, val
                    )
                    self.simple_event.update(event)

            for relationship in mapper.relationships:
                self._process_relationship(relationship, wrapped_field, dao, prefix)

        self.simple_event.fill_missing_variables(self.variables)
        return self.variables, self.simple_event

    def _process_column(
        self,
        column: Column,
        wrapped_field: WrappedField,
        dao: DataAccessObject,
        prefix: str,
    ) -> Tuple[List[Variable], List[Any]]:
        attribute_name = self.column_attribute_name(column)
        if not self.is_attribute_of_interest(attribute_name, wrapped_field):
            return [], []

        # one to one relationships are handled through relationships, they should never appear here
        if wrapped_field.is_one_to_one_relationship and not (
            wrapped_field.is_enum or wrapped_field.type_endpoint is uuid.UUID
        ):
            return [], []

        attribute = getattr(dao, attribute_name)
        if wrapped_field.is_optional and attribute is None:
            return [], []

        if wrapped_field.is_collection_of_builtins:
            variables = [
                self._create_variable_from_wrapped_field(
                    wrapped_field, f"{prefix}.{value}"
                )
                for value in attribute
            ]
            return variables, attribute

        if attribute is None:
            if wrapped_field.type_endpoint is str:
                return [], []
            var = self._create_variable_from_wrapped_field(
                wrapped_field, f"{prefix}.{attribute_name}"
            )
            return [var], [None]
        elif isinstance(attribute, list_like_classes):
            # skip attributes that are not None, and not list-like. those are already set correctly, and by not
            # adding the variable we dont clutter the model
            return [], []
        else:
            var = self._create_variable_from_wrapped_field(
                wrapped_field, f"{prefix}.{attribute_name}"
            )
            return [var], [attribute]

    def _create_simple_event_singleton_from_set_attribute(self, variable, attribute):
        if isinstance(attribute, bool) or isinstance(attribute, enum.Enum):
            return SimpleEvent({variable: Set.from_iterable([attribute])})
        else:
            return SimpleEvent(
                {
                    variable: SimpleInterval(
                        attribute, attribute, Bound.CLOSED, Bound.CLOSED
                    )
                }
            )

    def _process_relationship(
        self,
        relationship: Relationship,
        wrapped_field: WrappedField,
        dao: DataAccessObject,
        prefix: str,
    ) -> Tuple[List[Variable], Optional[SimpleEvent]]:
        attribute_name = relationship.key
        attribute_dao = getattr(dao, attribute_name)

        if not self.is_attribute_of_interest(attribute_name, wrapped_field):
            return [], None
        elif wrapped_field.is_one_to_many_relationship:
            one_to_many_variables = []
            one_to_many_simple_event = SimpleEvent({})
            for value in attribute_dao:
                variables, simple_event = self.parameterize_dao(
                    dao=value, prefix=f"{prefix}.{attribute_name}"
                )
                one_to_many_variables.extend(variables)
                one_to_many_simple_event.update(simple_event)
            return one_to_many_variables, one_to_many_simple_event

        elif wrapped_field.is_one_to_one_relationship:

            if wrapped_field.is_optional and attribute_dao is None:
                return [], None

            if attribute_dao is None:
                attribute_dao = get_dao_class(wrapped_field.type_endpoint)()
            variables, simple_event = self.parameterize_dao(
                dao=attribute_dao,
                prefix=f"{prefix}.{attribute_name}",
            )
            return variables, simple_event
        else:
            assert_never(wrapped_field)

    def is_attribute_of_interest(
        self, attribute_name: Optional[str], wrapped_field: WrappedField
    ):
        """
        If it's of the same name as the field, we are interested.
        """
        return (
            attribute_name
            and wrapped_field.public_name == attribute_name
            and not wrapped_field.type_endpoint is uuid.UUID
        )

    def column_attribute_name(self, column: Column) -> Optional[str]:
        if (
            column.key == "polymorphic_type"
            or column.primary_key
            or column.foreign_keys
        ):
            return None

        return column.name

    def _create_variable_from_wrapped_field(
        self, wrapped_field: WrappedField, name: str
    ) -> Variable:
        """
        Create a random event variable from a WrappedField based on its type.

        :return: A random event variable or raise error if the type is not supported.
        """
        type_endpoint = wrapped_field.type_endpoint

        if wrapped_field.is_enum:
            return Symbolic(name, Set.from_iterable(list(type_endpoint)))
        elif type_endpoint is int:
            return Integer(name)
        elif type_endpoint is float:
            return Continuous(name)
        elif type_endpoint is bool:
            return Symbolic(name, Set.from_iterable([True, False]))
        else:
            raise NotImplementedError(
                f"No conversion between {type_endpoint} and random_events.Variable is known."
            )

    @classmethod
    def create_fully_factorized_distribution(
        cls,
        variables: List[Variable],
    ) -> ProbabilisticCircuit:
        """
        Create a fully factorized probabilistic circuit over the given variables.

        :return: A fully factorized probabilistic circuit.
        """
        distribution_variables = [v for v in variables if not isinstance(v, Integer)]

        return fully_factorized(
            distribution_variables,
            means={v: 0.0 for v in distribution_variables if isinstance(v, Continuous)},
            variances={
                v: 1.0 for v in distribution_variables if isinstance(v, Continuous)
            },
        )
