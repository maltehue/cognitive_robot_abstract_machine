"""A domain-free mimic theory for the knowledge-servoing framework.

Stands in for a real :class:`~semantic_digital_twin.reasoning.knowledge_servoing.interfaces.\
SymbolicTheory` with two boolean facts and a small regime/parameter decision vocabulary. It proves
the framework runs a theory it has never heard of and keeps any domain (pouring) vocabulary out of
the framework, mirroring the repository's established mimic-class pattern.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.ripple_down_rules.datastructures.callable_expression import (
    CallableExpression,
)
from krrood.ripple_down_rules.rdr import GeneralRDR, MultiClassRDR
from krrood.ripple_down_rules.rules import MultiClassStopRule, MultiClassTopRule

from semantic_digital_twin.reasoning.knowledge_servoing.general_rdr_theory import (
    GeneralRDRTheory,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ParameterDecision,
    RegimeDecision,
    Situation,
)
from semantic_digital_twin.reasoning.knowledge_servoing.multi_class_rdr_theory import (
    CONCLUSION_ATTRIBUTE_NAME,
    MultiClassRDRTheory,
)

REGIME_FAMILY = "regime_decisions"
"""Attribute name of the regime decision family in the general-RDR mimic."""

PARAMETER_FAMILY = "parameter_decisions"
"""Attribute name of the parameter decision family in the general-RDR mimic."""


@dataclass(frozen=True)
class GaugeSituation(Situation):
    """A toy situation: whether a gauge reads high and whether an alarm is latched."""

    reads_high: bool
    """Whether the monitored gauge is above its threshold."""

    alarm_latched: bool
    """Whether a latched alarm defeats acting on the gauge."""


@dataclass(frozen=True)
class OpenValve(RegimeDecision):
    """Regime decision to open the valve; concluded when the gauge reads high and no alarm holds."""


@dataclass(frozen=True)
class RaiseAlert(RegimeDecision):
    """Regime decision chained on :class:`OpenValve` to escalate within the same pass."""


@dataclass(frozen=True)
class Throttle(ParameterDecision):
    """Parameter decision supplying a throttle fraction to a registered float variable."""

    fraction: float
    """Normalized throttle fraction in ``[0, 1]``."""


def _condition(source: str, scope: dict | None = None) -> CallableExpression:
    """Builds a boolean rule condition from a source expression over ``case``."""
    return CallableExpression(user_input=source, scope=scope or {})


def _conclusion(
    decision_source: str, decision_type: type, scope: dict
) -> CallableExpression:
    """Builds a rule conclusion that returns a single decision instance."""
    return CallableExpression(
        user_input=decision_source, scope=scope, conclusion_type=(decision_type,)
    )


def build_gauge_theory() -> MultiClassRDRTheory[GaugeSituation]:
    """Assembles the mimic theory as a :class:`MultiClassRDRTheory`.

    Three top rules exercise both write channels, a defeater and intra-pass chaining: open-valve
    (regime, with an alarm defeater), throttle (parameter, carrying a value) and raise-alert (regime,
    chained on the open-valve conclusion).
    """
    open_valve = MultiClassTopRule(
        conditions=_condition("case.situation.reads_high"),
        conclusion=_conclusion("OpenValve()", OpenValve, {"OpenValve": OpenValve}),
        conclusion_name=CONCLUSION_ATTRIBUTE_NAME,
    )
    alarm_defeater = MultiClassStopRule(
        conditions=_condition("case.situation.alarm_latched")
    )
    alarm_defeater.top_rule = open_valve
    open_valve.refinement = alarm_defeater

    throttle = MultiClassTopRule(
        conditions=_condition("case.situation.reads_high"),
        conclusion=_conclusion("Throttle(0.5)", Throttle, {"Throttle": Throttle}),
        conclusion_name=CONCLUSION_ATTRIBUTE_NAME,
    )
    raise_alert = MultiClassTopRule(
        conditions=_condition(
            "any(isinstance(decision, OpenValve) for decision in case.conclusions)",
            {"OpenValve": OpenValve},
        ),
        conclusion=_conclusion("RaiseAlert()", RaiseAlert, {"RaiseAlert": RaiseAlert}),
        conclusion_name=CONCLUSION_ATTRIBUTE_NAME,
    )
    open_valve.alternative = throttle
    throttle.alternative = raise_alert

    rule_set = MultiClassRDR(start_rule=open_valve)
    return MultiClassRDRTheory(
        rule_set=rule_set,
        declared_decision_types=frozenset({OpenValve, RaiseAlert, Throttle}),
    )


def _named_multi_class_rdr(start_rule: MultiClassTopRule, family: str) -> MultiClassRDR:
    """Wraps a top rule in a family-named ``MultiClassRDR`` for composition in a ``GeneralRDR``."""
    rule_set = MultiClassRDR(start_rule=start_rule)
    rule_set.name = family
    return rule_set


def build_gauge_general_theory() -> GeneralRDRTheory[GaugeSituation]:
    """Assembles the mimic theory as a :class:`GeneralRDRTheory` of two decision families.

    The regime family concludes open-valve (with an alarm defeater) and chains raise-alert on it; the
    parameter family concludes throttle by chaining across families on the regime's open-valve
    conclusion. It shares the decision vocabulary of :func:`build_gauge_theory`, so the two engines
    reach the same decisions for the same situation.
    """
    open_valve = MultiClassTopRule(
        conditions=_condition("case.situation.reads_high"),
        conclusion=_conclusion("OpenValve()", OpenValve, {"OpenValve": OpenValve}),
        conclusion_name=REGIME_FAMILY,
    )
    alarm_defeater = MultiClassStopRule(
        conditions=_condition("case.situation.alarm_latched")
    )
    alarm_defeater.top_rule = open_valve
    open_valve.refinement = alarm_defeater
    raise_alert = MultiClassTopRule(
        conditions=_condition(
            f"any(isinstance(decision, OpenValve) for decision in case.{REGIME_FAMILY})",
            {"OpenValve": OpenValve},
        ),
        conclusion=_conclusion("RaiseAlert()", RaiseAlert, {"RaiseAlert": RaiseAlert}),
        conclusion_name=REGIME_FAMILY,
    )
    open_valve.alternative = raise_alert

    throttle = MultiClassTopRule(
        conditions=_condition(
            f"any(isinstance(decision, OpenValve) for decision in case.{REGIME_FAMILY})",
            {"OpenValve": OpenValve},
        ),
        conclusion=_conclusion("Throttle(0.5)", Throttle, {"Throttle": Throttle}),
        conclusion_name=PARAMETER_FAMILY,
    )

    rule_set = GeneralRDR()
    rule_set.add_rdr(_named_multi_class_rdr(open_valve, REGIME_FAMILY))
    rule_set.add_rdr(_named_multi_class_rdr(throttle, PARAMETER_FAMILY))
    return GeneralRDRTheory(
        rule_set=rule_set,
        declared_decision_types=frozenset({OpenValve, RaiseAlert, Throttle}),
    )
