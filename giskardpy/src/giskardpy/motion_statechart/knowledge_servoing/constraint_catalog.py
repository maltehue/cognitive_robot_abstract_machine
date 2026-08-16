"""The vocabulary of constraints a declared theory may ask for.

The catalog maps each declaration kind onto a factory that builds the task enforcing it. It is the
finite part of the pluggability claim: a theory may declare any constraint of a kind someone
implemented, and a declaration outside the vocabulary is rejected at assembly rather than silently
dropped — which is what turns coverage from an assertion into something measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Callable, Dict, Optional, Type

from krrood.symbolic_math.symbolic_math import FloatVariable

from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
)
from semantic_digital_twin.world import World

from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.knowledge_servoing.exceptions import (
    DuplicateConstraintFactoryError,
    UnknownConstraintKindError,
)


@dataclass
class ConstraintInstantiation:
    """A declared constraint made concrete: the node enforcing it and its writable target."""

    node: MotionStatechartNode
    """The task or monitor enforcing the declared constraint."""

    parameter_target: Optional[FloatVariable] = None
    """The float variable a parameter channel writes into, if the constraint has one."""


ConstraintFactory = Callable[[ConstraintDeclaration, World], ConstraintInstantiation]
"""Builds the node enforcing a declaration, resolving its subject names in the world."""


@dataclass
class ConstraintCatalog:
    """Maps declaration kinds onto the factories that enforce them."""

    _factories: Dict[Type[ConstraintDeclaration], ConstraintFactory] = field(
        default_factory=dict, init=False
    )
    """One factory per declaration kind."""

    def register(
        self,
        declaration_type: Type[ConstraintDeclaration],
        factory: ConstraintFactory,
    ) -> None:
        """Registers the factory enforcing a declaration kind.

        :param declaration_type: The declaration kind the factory enforces.
        :param factory: Builds the enforcing node from a declaration and the world.
        :raises DuplicateConstraintFactoryError: if the kind already has a factory.
        """
        if declaration_type in self._factories:
            raise DuplicateConstraintFactoryError(declaration_type=declaration_type)
        self._factories[declaration_type] = factory

    def covers(self, declaration_type: Type[ConstraintDeclaration]) -> bool:
        """Whether a declaration kind is within the catalog's vocabulary.

        :param declaration_type: The declaration kind to check.
        """
        return declaration_type in self._factories

    def instantiate(
        self, declaration: ConstraintDeclaration, world: World
    ) -> ConstraintInstantiation:
        """Builds the node enforcing a declaration.

        :param declaration: The constraint a theory declared.
        :param world: The world its subject names are resolved in.
        :return: The enforcing node and its writable target, if any.
        :raises UnknownConstraintKindError: if no factory covers the declaration's kind.
        """
        factory = self._factories.get(type(declaration))
        if factory is None:
            raise UnknownConstraintKindError(declaration_type=type(declaration))
        return factory(declaration, world)
