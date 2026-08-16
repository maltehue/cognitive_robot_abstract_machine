"""
A qualitative theory of when a manipulation warrants extra caution.

Two rules, both of the form "this scene makes a mistake expensive, so restrict the
motion". Neither predicts a physical quantity: the theory reasons about what the objects
*are*, and leaves what to do about it to the binding policy and to constraints the
controller already knows how to enforce.

Its purpose in the architecture is to be a second theory — a different situation type, a
different vocabulary, no effect model of its own — running alongside the substance-
transfer theory and proving the framework is not shaped around either of them.
"""

from __future__ import annotations

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.rdr import GeneralRDR, MultiClassRDR
from krrood.ripple_down_rules.rules import MultiClassTopRule

from semantic_digital_twin.reasoning.contextual_safety.decisions import (
    CautionReason,
    EnforceCaution,
)
from semantic_digital_twin.reasoning.contextual_safety.situation import SafetySituation
from semantic_digital_twin.reasoning.knowledge_servoing.general_rdr_theory import (
    GeneralRDRTheory,
)

CAUTION_FAMILY = "caution_decisions"
"""
Attribute name of the caution decision family.
"""


def _condition(source: str) -> CallableExpression:
    """
    Builds a boolean rule condition over ``case``.

    :param source: Python expression evaluated against the classification case.
    :return: The condition as a callable expression.
    """
    return CallableExpression(user_input=source, scope={})


def _caution(reason: CautionReason) -> CallableExpression:
    """
    Builds a conclusion enforcing caution for a given reason.

    :param reason: What about the scene warrants the restriction.
    :return: The conclusion as a callable expression.
    """
    return CallableExpression(
        user_input=f"EnforceCaution(CautionReason.{reason.name})",
        scope={"EnforceCaution": EnforceCaution, "CautionReason": CautionReason},
        conclusion_type=(EnforceCaution,),
    )


def build_contextual_safety_theory() -> GeneralRDRTheory[SafetySituation]:
    """
    Assembles the contextual-safety theory.

    Caution is warranted while contents are in flight above something that must not be
    spilled on, and — more weakly but for the same reason — while a filled container is
    merely carried above one. Pouring is the sharper case, so its rule comes first and
    its reason is the one reported when both hold.

    :return: The theory, ready to plug into a symbolic theory node.
    """
    pouring_over_sensitive_object = MultiClassTopRule(
        conditions=_condition(
            "case.situation.above_sensitive_object and case.situation.is_pouring_out"
        ),
        conclusion=_caution(CautionReason.SPILL_WOULD_REACH_SENSITIVE_OBJECT),
        conclusion_name=CAUTION_FAMILY,
    )
    carrying_over_sensitive_object = MultiClassTopRule(
        conditions=_condition(
            "case.situation.above_sensitive_object "
            "and case.situation.holds_contents "
            "and not case.situation.is_pouring_out"
        ),
        conclusion=_caution(CautionReason.CARRYING_CONTENTS_OVER_SENSITIVE_OBJECT),
        conclusion_name=CAUTION_FAMILY,
    )
    pouring_over_sensitive_object.alternative = carrying_over_sensitive_object

    rule_set = GeneralRDR()
    caution_family = MultiClassRDR(start_rule=pouring_over_sensitive_object)
    caution_family.name = CAUTION_FAMILY
    rule_set.add_rdr(caution_family)
    return GeneralRDRTheory(
        rule_set=rule_set, declared_decision_types=frozenset({EnforceCaution})
    )
