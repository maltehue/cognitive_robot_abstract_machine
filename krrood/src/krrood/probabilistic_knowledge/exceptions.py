from dataclasses import dataclass

from ..entity_query_language.query.operations import Where
from ..utils import DataclassException


@dataclass
class WhereExpressionNotInDisjunctiveNormalForm(DataclassException):
    """
    Raised when a `Where` expression is not in disjunctive normal form.
    """

    where_expression: Where

    def __post_init__(self):
        self.message = f"The where expression {self.where_expression} is not in disjunctive normal form."
