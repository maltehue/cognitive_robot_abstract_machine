"""Mimic case objects and a toy ``MultiClassRDR`` for the knowledge-servoing spike.

The spike (`doc/knowledge_servoing/implementation_plan.md` §4.1.1) needs to know how the
`MultiClassRDR` engine behaves before a substance-transfer theory is authored on top of it. These
mimics stand in for that theory's case object and decision vocabulary with a domain-free toy: three
boolean facts and three "regime" conclusions. Keeping the mimics here honours ``krrood``'s
self-containment rule — the spike never imports from another workspace package.

.. note::
    Three case shapes exist on purpose. They differ only in whether the case is frozen and whether it
    carries a mutable conclusion accumulator, which is exactly the axis that decides whether a frozen
    dataclass survives use as an RDR ``Case``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.rdr import MultiClassRDR
from krrood.ripple_down_rules.rules import MultiClassStopRule, MultiClassTopRule

CONCLUSION_ATTRIBUTE_NAME = "conclusions"
"""Name of the case attribute the engine accumulates conclusions into."""


# %% regime conclusions (mimic a control-decision vocabulary)


@dataclass(frozen=True)
class EngageRegime:
    """A regime a top rule concludes when engagement is requested and not defeated."""


@dataclass(frozen=True)
class RestrictRegime:
    """A regime a second, independent top rule concludes when restriction is requested."""


@dataclass(frozen=True)
class EscalateRegime:
    """A regime concluded only by chaining: a top rule that fires on an earlier conclusion."""


# %% case shapes (differ only in frozen-ness and the accumulator)


@dataclass(frozen=True)
class FrozenSituationWithoutAccumulator:
    """A fully frozen case with no place to accumulate conclusions."""

    engagement_requested: bool
    """Whether the toy scene asks for the engage regime."""

    restriction_requested: bool
    """Whether the toy scene asks for the restrict regime."""

    defeated: bool
    """Whether the engage regime's defeater holds."""


@dataclass(frozen=True)
class FrozenSituationWithAccumulator:
    """A frozen case that nonetheless carries a mutable conclusion accumulator."""

    engagement_requested: bool
    """Whether the toy scene asks for the engage regime."""

    restriction_requested: bool
    """Whether the toy scene asks for the restrict regime."""

    defeated: bool
    """Whether the engage regime's defeater holds."""

    conclusions: Set = field(default_factory=set)
    """Mutable accumulator the engine writes firing conclusions into."""


@dataclass
class MutableSituation:
    """A non-frozen working case that copies cleanly for classification."""

    engagement_requested: bool
    """Whether the toy scene asks for the engage regime."""

    restriction_requested: bool
    """Whether the toy scene asks for the restrict regime."""

    defeated: bool
    """Whether the engage regime's defeater holds."""

    conclusions: Set = field(default_factory=set)
    """Mutable accumulator the engine writes firing conclusions into."""


# %% toy theory construction


def _condition(source: str) -> CallableExpression:
    """Build a boolean rule condition from a source expression over ``case``."""
    return CallableExpression(
        user_input=source,
        scope={"EngageRegime": EngageRegime},
    )


def _conclusion(decision_type: type) -> CallableExpression:
    """Build a rule conclusion that returns a single instance of ``decision_type``."""
    return CallableExpression(
        user_input=f"{decision_type.__name__}()",
        scope={decision_type.__name__: decision_type},
        conclusion_type=(decision_type,),
    )


def build_regime_multi_class_rdr() -> MultiClassRDR:
    """Assemble the toy ``MultiClassRDR``.

    The tree has three top rules: engage (with a stop-rule defeater), restrict, and an escalate rule
    that conditions on the engage conclusion so it can only fire by intra-pass chaining.
    """
    engage_rule = MultiClassTopRule(
        conditions=_condition("case.engagement_requested"),
        conclusion=_conclusion(EngageRegime),
        conclusion_name=CONCLUSION_ATTRIBUTE_NAME,
    )
    defeater = MultiClassStopRule(conditions=_condition("case.defeated"))
    defeater.top_rule = engage_rule
    engage_rule.refinement = defeater

    restrict_rule = MultiClassTopRule(
        conditions=_condition("case.restriction_requested"),
        conclusion=_conclusion(RestrictRegime),
        conclusion_name=CONCLUSION_ATTRIBUTE_NAME,
    )
    escalate_rule = MultiClassTopRule(
        conditions=_condition(
            "any(isinstance(decision, EngageRegime) for decision in case.conclusions)"
        ),
        conclusion=_conclusion(EscalateRegime),
        conclusion_name=CONCLUSION_ATTRIBUTE_NAME,
    )
    engage_rule.alternative = restrict_rule
    restrict_rule.alternative = escalate_rule

    reasoner = MultiClassRDR(start_rule=engage_rule)
    reasoner.case_type = MutableSituation
    reasoner.case_name = MutableSituation.__name__
    return reasoner
