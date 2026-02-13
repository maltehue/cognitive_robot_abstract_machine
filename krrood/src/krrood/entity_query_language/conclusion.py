from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache, cached_property

from typing_extensions import Any, Optional, List, Dict, Iterable

from .enums import RDREdge
from .rxnode import ColorLegend
from .symbolic import (
    SymbolicExpression,
    Variable,
    OperationResult,
    ResultQuantifier,
    Selectable,
    Bindings,
)
from .utils import T


@dataclass(eq=False)
class Conclusion(SymbolicExpression, ABC):
    """
    Base for side-effecting/action clauses that adjust outputs (e.g., Set, Add).

    :ivar var: The variable being affected by the conclusion.
    :ivar value: The value or expression used by the conclusion.
    """

    var: Selectable
    value: Any

    def __post_init__(self):

        self.var, self.value = self._update_children_(self.var, self.value)

        self.value._is_inferred_ = True

        current_parent = SymbolicExpression._current_parent_in_context_stack_()
        if current_parent is None:
            current_parent = self._conditions_root_
        self._parent_ = current_parent
        self._parent_._add_conclusion_(self)

    @cached_property
    def _all_variable_instances_(self) -> List[Selectable]:
        return self.var._all_variable_instances_ + self.value._all_variable_instances_

    @property
    def _name_(self) -> str:
        value_str = (
            self.value._type_.__name__
            if isinstance(self.value, Variable)
            else str(self.value)
        )
        return f"{self.__class__.__name__}({self.var._var_._name_}, {value_str})"

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("Conclusion", "#8cf2ff")


@dataclass(eq=False)
class Set(Conclusion):
    """Set the value of a variable in the current solution binding."""

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        if self.var._binding_id_ not in sources:
            parent_value = next(iter(self.var._evaluate_(sources, parent=self)))[
                self.var._binding_id_
            ]
            sources[self.var._binding_id_] = parent_value
        sources[self.var._binding_id_] = next(
            iter(self.value._evaluate_(sources, parent=self))
        )[self.value._binding_id_]
        yield OperationResult(sources, False, self)


@dataclass(eq=False)
class Add(Conclusion):
    """Add a new value to the domain of a variable."""

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        v = next(iter(self.value._evaluate_(sources, parent=self)))[
            self.value._binding_id_
        ]
        sources[self.var._binding_id_] = v
        yield OperationResult(sources, False, self)
