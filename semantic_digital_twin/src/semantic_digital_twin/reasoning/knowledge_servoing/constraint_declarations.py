"""
What a theory needs from the controller, stated as data.

A theory that only chooses among pre-wired constraints is not pluggable: whoever
assembles the statechart has to know every remedy every theory might conclude. A
constraint declaration inverts the dependency — the theory states which constraints it
requires, and the chart is assembled from those declarations, so plugging a theory in
means adding one object rather than rewiring the chart.

Subjects are named rather than referenced, so a declaration can be written — by hand or
by a synthesizer — without holding world objects; the catalog resolves the names when
the chart is assembled.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from typing_extensions import Optional, Type

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ParameterDecision,
    RegimeDecision,
)


@dataclass(frozen=True)
class ParameterChannel:
    """
    How a parameter decision's value reaches a declared constraint (channel 2).
    """

    decision_type: Type[ParameterDecision]
    """
    The parameter decision whose conclusion supplies the value.
    """

    attribute_name: str
    """
    Name of the field on the decision that carries the value.
    """


@dataclass(frozen=True)
class ConstraintDeclaration(ABC):
    """
    One constraint a theory requires the controller to enforce.

    Concrete declaration kinds carry the constraint's numeric parameters and subject
    names; a catalog maps each kind onto the task that enforces it.
    """

    identifier: str
    """
    Names this constraint within its theory; also names the nodes assembled for it.
    """

    gating_decision_type: Optional[Type[RegimeDecision]] = field(
        default=None, kw_only=True
    )
    """
    The regime decision whose conclusion activates the constraint (channel 1).

    ``None`` means the constraint is active for the whole motion.
    """

    parameter_channel: Optional[ParameterChannel] = field(default=None, kw_only=True)
    """
    How the constraint's runtime value is supplied, if it has one.
    """


# %% domain-free declaration kinds


@dataclass(frozen=True)
class ToolSpeedLimitDeclaration(ConstraintDeclaration):
    """Cap the translational speed of a named annotation's body."""

    subject_name: str = field(kw_only=True)
    """Name of the annotation whose body the cap applies to."""

    maximum_speed: float = field(kw_only=True)
    """Maximum allowed linear speed, in metres per second."""


@dataclass(frozen=True)
class MotionAbortDeclaration(ConstraintDeclaration):
    """Abort the whole motion when the gating decision is concluded.

    The one declaration kind that is a remedy of last resort rather than a constraint: it exists so
    a theory's defeat decisions have a declared enactment instead of a hand-wired one.
    """

    reason: str = field(kw_only=True)
    """Why the motion is aborted, reported by the raised error."""
