"""
The replication arm's rule set: the same facts, concluding directions instead of
regimes.

This and :func:`~semantic_digital_twin.reasoning.substance_transfer.theory.\
build_substance_transfer_theory` reason over the same :class:`TransferSituation`, produced by the
same grounding from the same world. They differ only in what they conclude — directions here,
constraint regimes there — which is what makes a comparison between them a comparison of the
bridge rather than of the reasoner, the scene or the robot.

They are separate objects rather than one theory with two families because a binding policy
validates that every decision type a theory declares is bound to a channel, and neither arm should
be obliged to bind the other's vocabulary.

The superiority pair the paper reports (near-goal decrease-tilt overriding slow-flow increase-tilt)
is encoded here as a stop rule refining the increase-tilt rule, with decrease-tilt taking over. It
is the interaction that the terminal-state row makes redundant when its horizon outlasts the
actuation, so it belongs to this arm and not to the other.
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
from semantic_digital_twin.reasoning.substance_transfer.motion_primitives import (
    DecreaseTilt,
    IncreaseTilt,
    MoveBack,
    MoveForward,
    MoveLeft,
    MoveRight,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)

PRIMITIVE_FAMILY = "motion_primitives"
"""
Attribute name of the motion-primitive decision family.
"""

ALIGNMENT_TOLERANCE = 0.01
"""
Offset below which the source counts as aligned on an axis, in metres.
"""


def _condition(source: str) -> CallableExpression:
    """
    Builds a boolean rule condition over ``case``.

    :param source: Python expression evaluated against the classification case.
    :return: The condition as a callable expression.
    """
    return CallableExpression(user_input=source, scope={})


def _primitive(primitive_type: type) -> CallableExpression:
    """
    Builds a conclusion returning one motion primitive.

    :param primitive_type: The primitive to conclude.
    :return: The conclusion as a callable expression.
    """
    return CallableExpression(
        user_input=f"{primitive_type.__name__}()",
        scope={primitive_type.__name__: primitive_type},
        conclusion_type=(primitive_type,),
    )


def _top_rule(condition: str, primitive_type: type) -> MultiClassTopRule:
    """
    Builds one primitive-concluding top rule.

    :param condition: Python expression the rule fires on.
    :param primitive_type: The primitive it concludes.
    :return: The rule.
    """
    return MultiClassTopRule(
        conditions=_condition(condition),
        conclusion=_primitive(primitive_type),
        conclusion_name=PRIMITIVE_FAMILY,
    )


def build_motion_primitive_theory() -> GeneralRDRTheory[TransferSituation]:
    """
    Assembles the replication arm's rule set.

    While the pour is not aimed, one rule per axis direction moves the source toward
    alignment — the family the regime vocabulary replaces with a single align decision.
    Once aimed, tilt increases until the goal is near, at which point the near-goal rule
    defeats it and tilt decreases instead.

    :return: The theory, ready to plug into a symbolic theory node.
    """
    unaimed = f"not case.situation.opening_within"
    move_forward = _top_rule(
        f"{unaimed} and case.situation.receiver_offset_forward > {ALIGNMENT_TOLERANCE}",
        MoveForward,
    )
    move_back = _top_rule(
        f"{unaimed} and case.situation.receiver_offset_forward < {-ALIGNMENT_TOLERANCE}",
        MoveBack,
    )
    move_left = _top_rule(
        f"{unaimed} and case.situation.receiver_offset_left > {ALIGNMENT_TOLERANCE}",
        MoveLeft,
    )
    move_right = _top_rule(
        f"{unaimed} and case.situation.receiver_offset_left < {-ALIGNMENT_TOLERANCE}",
        MoveRight,
    )
    increase_tilt = _top_rule(
        "case.situation.opening_within and not case.situation.goal_reached",
        IncreaseTilt,
    )
    near_goal_defeater = MultiClassStopRule(
        conditions=_condition("case.situation.almost_goal_reached")
    )
    near_goal_defeater.top_rule = increase_tilt
    increase_tilt.refinement = near_goal_defeater
    decrease_tilt = _top_rule(
        "case.situation.almost_goal_reached or case.situation.goal_reached",
        DecreaseTilt,
    )

    move_forward.alternative = move_back
    move_back.alternative = move_left
    move_left.alternative = move_right
    move_right.alternative = increase_tilt
    increase_tilt.alternative = decrease_tilt

    rule_set = GeneralRDR()
    primitive_family = MultiClassRDR(start_rule=move_forward)
    primitive_family.name = PRIMITIVE_FAMILY
    rule_set.add_rdr(primitive_family)
    return GeneralRDRTheory(
        rule_set=rule_set,
        declared_decision_types=frozenset(
            {MoveForward, MoveBack, MoveLeft, MoveRight, IncreaseTilt, DecreaseTilt}
        ),
    )
