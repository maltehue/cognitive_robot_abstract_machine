"""
The contextual-safety theory's rules, exercised on hand-built situations.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.reasoning.contextual_safety import (
    CautionReason,
    EnforceCaution,
    SafetySituation,
    build_contextual_safety_theory,
)


def situation(**overrides) -> SafetySituation:
    """
    Builds a situation defaulting to a filled container held clear of anything
    sensitive.
    """
    facts = {
        "carried_container": None,
        "holds_contents": True,
        "is_pouring_out": False,
        "above_sensitive_object": False,
    }
    facts.update(overrides)
    return SafetySituation(**facts)


@pytest.fixture
def theory():
    return build_contextual_safety_theory()


class TestCautionRules:
    """
    When the scene's semantics warrant restricting the motion.
    """

    def test_a_clear_workspace_warrants_no_caution(self, theory):
        assert not theory.infer([situation()]).contains_type(EnforceCaution)

    def test_carrying_contents_over_a_sensitive_object_warrants_caution(self, theory):
        [caution] = theory.infer([situation(above_sensitive_object=True)]).of_type(
            EnforceCaution
        )
        assert caution.reason is CautionReason.CARRYING_CONTENTS_OVER_SENSITIVE_OBJECT

    def test_pouring_over_a_sensitive_object_reports_the_sharper_reason(self, theory):
        [caution] = theory.infer(
            [situation(above_sensitive_object=True, is_pouring_out=True)]
        ).of_type(EnforceCaution)
        assert caution.reason is CautionReason.SPILL_WOULD_REACH_SENSITIVE_OBJECT

    def test_an_empty_container_over_a_sensitive_object_warrants_no_caution(
        self, theory
    ):
        decisions = theory.infer(
            [situation(above_sensitive_object=True, holds_contents=False)]
        )
        assert not decisions.contains_type(EnforceCaution)

    def test_being_sensitive_object_free_is_enough_to_stay_unrestricted_while_pouring(
        self, theory
    ):
        decisions = theory.infer([situation(is_pouring_out=True)])
        assert not decisions.contains_type(EnforceCaution)
