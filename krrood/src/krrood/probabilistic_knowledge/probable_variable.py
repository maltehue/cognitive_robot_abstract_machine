import operator
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby
from typing import assert_never, List, Dict

import numpy as np

from random_events.interval import open_closed, closed_open, closed

from random_events.product_algebra import Event, SimpleEvent

from .object_access_variable import ObjectAccessVariable, AttributeAccessLike
from ..entity_query_language.core.base_expressions import Selectable
from ..entity_query_language.core.variable import Variable, Literal
from ..entity_query_language.operators.comparator import Comparator
from ..entity_query_language.query.query import Entity
from ..entity_query_language.query_graph import QueryGraph
from ..ormatic.dao import get_dao_class


@dataclass
class QueryToRandomEventTranslator:

    query: Entity

    def translate(self) -> Event:
        self.query.build()

        result = SimpleEvent()

        # check that it is always a comparison between a variable and a literal
        for variable, comparators in self.comparators_grouped_by_variable.items():
            self._translate_comparators(variable, comparators, result)

        return result

    def _object_access_variable_from_comparator(self, comparator: Comparator):
        assert isinstance(comparator.left, AttributeAccessLike)
        return ObjectAccessVariable.from_attribute_access_and_type(
            comparator.left, comparator.left._type_
        )

    @cached_property
    def comparators_grouped_by_variable(
        self,
    ) -> Dict[ObjectAccessVariable, List[Comparator]]:
        # Get the Where expression
        where_expr = self.query._where_expression_

        # Get all descendants of the Where expression that are Comparators
        comparators = [
            expr for expr in where_expr._descendants_ if isinstance(expr, Comparator)
        ]

        comparators_grouped_by_variable = groupby(
            comparators, key=lambda c: self._object_access_variable_from_comparator(c)
        )

        return {k: list(v) for k, v in comparators_grouped_by_variable}

    def _translate_comparators(
        self,
        variable: ObjectAccessVariable,
        comparators: List[Comparator],
        result: SimpleEvent,
    ) -> None:

        result[variable.variable] = variable.variable.domain
        for comparator in comparators:

            match comparator.operation:
                case operator.eq:
                    self._translate_eq(comparator, variable, result)
                case operator.ne:
                    self._translate_ne(comparator, variable, result)
                case operator.gt:
                    self._translate_gt(comparator, variable, result)
                case operator.lt:
                    self._translate_lt(comparator, variable, result)
                case operator.ge:
                    self._translate_ge(comparator, variable, result)
                case operator.le:
                    self._translate_le(comparator, variable, result)
                case _:
                    assert_never(comparator.operation)

    def _translate_eq(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ) -> None:
        result[
            object_access_variable.variable
        ] &= object_access_variable.variable.make_value(comparator.right._domain_[0])

    def _translate_ne(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ):
        result[
            object_access_variable.variable
        ] &= object_access_variable.variable.make_value(
            comparator.right._domain_[0]
        ).complement()

    def _translate_gt(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ):
        result[object_access_variable.variable] &= open_closed(
            comparator.right._domain_[0], np.inf
        )

    def _translate_lt(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ):
        result[object_access_variable.variable] &= closed_open(
            -np.inf,
            comparator.right._domain_[0],
        )

    def _translate_le(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ):
        result[object_access_variable.variable] &= closed(
            -np.inf,
            comparator.right._domain_[0],
        )

    def _translate_ge(
        self,
        comparator: Comparator,
        object_access_variable: ObjectAccessVariable,
        result: SimpleEvent,
    ):
        result[object_access_variable.variable] &= closed(
            comparator.right._domain_[0],
            np.inf,
        )
