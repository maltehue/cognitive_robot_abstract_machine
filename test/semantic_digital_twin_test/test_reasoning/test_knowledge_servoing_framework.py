"""The knowledge-servoing framework runs a pluggable theory it has never heard of.

Exercises the domain-agnostic reasoning interface through the mimic gauge theory: situations in,
decisions out, with both write channels, a defeater, intra-pass chaining, and the guarantee that the
frozen situation crossing the thread boundary is never mutated by classification.
"""

from __future__ import annotations

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    ParameterDecision,
    RegimeDecision,
)

from .knowledge_servoing_mimic import (
    CONCLUSION_ATTRIBUTE_NAME,
    GaugeSituation,
    OpenValve,
    RaiseAlert,
    Throttle,
    build_gauge_theory,
)


class TestMultiClassRdrTheory:
    """Whether the MCRDR-backed theory infers decisions from situations through the interface."""

    def test_theory_infers_both_channels_and_chains(self):
        theory = build_gauge_theory()
        decisions = theory.infer([GaugeSituation(reads_high=True, alarm_latched=False)])
        assert set(decisions) == {OpenValve(), Throttle(0.5), RaiseAlert()}

    def test_defeater_blocks_a_regime_decision_and_its_chain(self):
        theory = build_gauge_theory()
        decisions = theory.infer([GaugeSituation(reads_high=True, alarm_latched=True)])
        assert set(decisions) == {Throttle(0.5)}

    def test_no_facts_yield_no_decisions(self):
        theory = build_gauge_theory()
        decisions = theory.infer(
            [GaugeSituation(reads_high=False, alarm_latched=False)]
        )
        assert set(decisions) == set()

    def test_inference_aggregates_across_situations(self):
        theory = build_gauge_theory()
        decisions = theory.infer(
            [
                GaugeSituation(reads_high=True, alarm_latched=False),
                GaugeSituation(reads_high=False, alarm_latched=False),
            ]
        )
        assert set(decisions) == {OpenValve(), Throttle(0.5), RaiseAlert()}

    def test_declared_decision_types_are_exposed(self):
        theory = build_gauge_theory()
        assert theory.decision_types == frozenset({OpenValve, RaiseAlert, Throttle})

    def test_inference_does_not_mutate_the_frozen_situation(self):
        theory = build_gauge_theory()
        situation = GaugeSituation(reads_high=True, alarm_latched=False)
        theory.infer([situation])
        assert not hasattr(situation, CONCLUSION_ATTRIBUTE_NAME)


class TestDecisionSet:
    """Whether the returned decision set separates the two write channels by type."""

    def test_of_type_selects_by_channel(self):
        theory = build_gauge_theory()
        decisions = theory.infer([GaugeSituation(reads_high=True, alarm_latched=False)])
        assert set(decisions.of_type(RegimeDecision)) == {OpenValve(), RaiseAlert()}
        assert set(decisions.of_type(ParameterDecision)) == {Throttle(0.5)}

    def test_contains_type_reflects_membership(self):
        theory = build_gauge_theory()
        decisions = theory.infer([GaugeSituation(reads_high=True, alarm_latched=True)])
        assert decisions.contains_type(Throttle) is True
        assert decisions.contains_type(OpenValve) is False

    def test_parameter_decision_carries_its_value(self):
        theory = build_gauge_theory()
        decisions = theory.infer([GaugeSituation(reads_high=True, alarm_latched=False)])
        (throttle,) = decisions.of_type(Throttle)
        assert throttle.fraction == 0.5
        assert isinstance(throttle, ControlDecision)
