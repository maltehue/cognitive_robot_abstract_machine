"""
The qualitative theory of substance transfer, as a ripple-down rule set.

The rules below are the pouring fragment of the case-study theory, with one change: no
rule concludes a direction to move in. Where the original theory picked between move-
left, move-right, move-up and move-down to bring the openings into alignment, this one
concludes a single *align* regime and lets the optimizer solve the direction from the
landing-point constraint's gradient. What survives unchanged is the derivational part —
when a pour is possible, when it is defeated, when it is finished.

Two decision families are composed so a rule in one may condition on a conclusion the
other reached: the regime family decides which constraints are active, the parameter
family supplies the numeric goal.
"""

from __future__ import annotations

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.rdr import GeneralRDR, MultiClassRDR
from krrood.ripple_down_rules.rules import MultiClassStopRule, MultiClassTopRule

from semantic_digital_twin.reasoning.knowledge_servoing.general_rdr_theory import (
    GeneralRDRTheory,
)
from semantic_digital_twin.reasoning.substance_transfer.decisions import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
    RetargetFillLevel,
    TransferDefeat,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)

REGIME_FAMILY = "regime_decisions"
"""
Attribute name of the regime decision family (channel 1).
"""

PARAMETER_FAMILY = "parameter_decisions"
"""
Attribute name of the parameter decision family (channel 2).
"""


def _condition(source: str, scope: dict | None = None) -> CallableExpression:
    """
    Builds a boolean rule condition over ``case``.

    :param source: Python expression evaluated against the classification case.
    :param scope: Names the expression may reference besides ``case``.
    :return: The condition as a callable expression.
    """
    return CallableExpression(user_input=source, scope=scope or {})


def _conclusion(source: str, decision_type: type, scope: dict) -> CallableExpression:
    """
    Builds a rule conclusion returning one decision instance.

    :param source: Python expression constructing the decision.
    :param decision_type: The type the expression constructs.
    :param scope: Names the expression may reference besides ``case``.
    :return: The conclusion as a callable expression.
    """
    return CallableExpression(
        user_input=source, scope=scope, conclusion_type=(decision_type,)
    )


def _family(start_rule: MultiClassTopRule, family: str) -> MultiClassRDR:
    """
    Wraps a top rule in a family-named classifier for composition in a general rule set.

    :param start_rule: First top rule of the family's chain.
    :param family: Attribute name the family's conclusions accumulate under.
    :return: The named classifier.
    """
    rule_set = MultiClassRDR(start_rule=start_rule)
    rule_set.name = family
    return rule_set


def build_substance_transfer_theory() -> GeneralRDRTheory[TransferSituation]:
    """
    Assembles the substance-transfer theory.

    Regime rules, in top-rule order so each may see what the previous concluded:

    - *align* whenever the source is near the receiver and the goal is not yet reached;
    - *pour* once the pour would land in the opening, defeated while the receiver is overflowing;
    - *conclude* once the goal is reached;
    - *abandon* while the receiver is overflowing.

    The parameter family then supplies the requested fill level as the terminal-fill goal whenever
    pouring is active, so the numeric target is a conclusion of the theory rather than a constant
    compiled into the task.

    :return: The theory, ready to plug into a symbolic theory node.
    """
    align = MultiClassTopRule(
        conditions=_condition(
            "case.situation.near and not case.situation.goal_reached"
        ),
        conclusion=_conclusion(
            "AlignSourceOverReceiver()",
            AlignSourceOverReceiver,
            {"AlignSourceOverReceiver": AlignSourceOverReceiver},
        ),
        conclusion_name=REGIME_FAMILY,
    )

    pour = MultiClassTopRule(
        conditions=_condition(
            "case.situation.opening_within "
            "and case.situation.source_above_receiver "
            "and not case.situation.goal_reached"
        ),
        conclusion=_conclusion(
            "PourIntoReceiver()",
            PourIntoReceiver,
            {"PourIntoReceiver": PourIntoReceiver},
        ),
        conclusion_name=REGIME_FAMILY,
    )
    overflow_defeater = MultiClassStopRule(
        conditions=_condition("case.situation.receiver_overflowing")
    )
    overflow_defeater.top_rule = pour
    pour.refinement = overflow_defeater
    align.alternative = pour

    conclude = MultiClassTopRule(
        conditions=_condition("case.situation.goal_reached"),
        conclusion=_conclusion(
            "ConcludeTransfer()",
            ConcludeTransfer,
            {"ConcludeTransfer": ConcludeTransfer},
        ),
        conclusion_name=REGIME_FAMILY,
    )
    pour.alternative = conclude

    abandon = MultiClassTopRule(
        conditions=_condition("case.situation.receiver_overflowing"),
        conclusion=_conclusion(
            "AbandonTransfer(TransferDefeat.RECEIVER_WOULD_OVERFLOW)",
            AbandonTransfer,
            {"AbandonTransfer": AbandonTransfer, "TransferDefeat": TransferDefeat},
        ),
        conclusion_name=REGIME_FAMILY,
    )
    conclude.alternative = abandon

    retarget = MultiClassTopRule(
        conditions=_condition(
            f"any(isinstance(decision, PourIntoReceiver) "
            f"for decision in case.{REGIME_FAMILY})",
            {"PourIntoReceiver": PourIntoReceiver},
        ),
        conclusion=_conclusion(
            "RetargetFillLevel(case.situation.requested_fill_level)",
            RetargetFillLevel,
            {"RetargetFillLevel": RetargetFillLevel},
        ),
        conclusion_name=PARAMETER_FAMILY,
    )

    rule_set = GeneralRDR()
    rule_set.add_rdr(_family(align, REGIME_FAMILY))
    rule_set.add_rdr(_family(retarget, PARAMETER_FAMILY))
    return GeneralRDRTheory(
        rule_set=rule_set,
        declared_decision_types=frozenset(
            {
                AlignSourceOverReceiver,
                PourIntoReceiver,
                ConcludeTransfer,
                AbandonTransfer,
                RetargetFillLevel,
            }
        ),
    )
