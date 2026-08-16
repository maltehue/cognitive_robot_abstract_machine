"""
Compiling a theory from its specification.

The golden test is the one that matters: a specification mirroring the hand-written
transfer theory must conclude the same decisions on the same situations, so a
synthesizer targeting the specification format produces theories indistinguishable from
hand-written ones. The rest pins the validation boundary — everything a synthesizer can
get wrong must be rejected before execution, with the specific error naming what to fix.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.reasoning.knowledge_servoing.condition_validation import (
    ConditionValidator,
)
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    GENERIC_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.knowledge_servoing.exceptions import (
    DecisionRoleConflictError,
    ForbiddenConditionSyntaxError,
    InvalidDeclarationParametersError,
    InvalidRuleValueError,
    MalformedConditionError,
    UnconcludableDecisionError,
    UnknownDecisionNameError,
    UnknownDeclarationKindError,
    UnknownSituationFactError,
    UnknownSpecificationFieldError,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_compiler import (
    TheoryCompiler,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_specification import (
    TheorySpecification,
)
from semantic_digital_twin.reasoning.substance_transfer.declarations import (
    TRANSFER_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)
from semantic_digital_twin.reasoning.substance_transfer.theory import (
    build_substance_transfer_theory,
)

from .test_substance_transfer_theory import situation

DECLARATION_KINDS = {**GENERIC_DECLARATION_KINDS, **TRANSFER_DECLARATION_KINDS}

TRANSFER_SPECIFICATION = {
    "constraints": [
        {
            "identifier": "aim",
            "kind": "aimed_transfer",
            "parameters": {
                "source_name": "source_cup",
                "receiver_name": "receiving_cup",
            },
            "gated_by": "AlignSourceOverReceiver",
        },
        {
            "identifier": "rim_clearance",
            "kind": "rim_clearance",
            "parameters": {
                "source_name": "source_cup",
                "receiver_name": "receiving_cup",
            },
            "gated_by": "AlignSourceOverReceiver",
        },
        {
            "identifier": "quantity",
            "kind": "transfer_quantity",
            "parameters": {
                "source_name": "source_cup",
                "receiver_name": "receiving_cup",
            },
            "gated_by": "PourIntoReceiver",
            "value_from": "RetargetFillLevel",
        },
        {
            "identifier": "return_upright",
            "kind": "return_upright",
            "parameters": {"subject_name": "source_cup"},
            "gated_by": "ConcludeTransfer",
        },
        {
            "identifier": "abort",
            "kind": "motion_abort",
            "parameters": {"reason": "the receiver would overflow"},
            "gated_by": "AbandonTransfer",
        },
    ],
    "rules": [
        {
            "concludes": "AlignSourceOverReceiver",
            "condition": "case.situation.near and not case.situation.goal_reached",
        },
        {
            "concludes": "PourIntoReceiver",
            "condition": (
                "case.situation.opening_within "
                "and case.situation.source_above_receiver "
                "and not case.situation.goal_reached"
            ),
            "defeated_by": ["case.situation.receiver_overflowing"],
        },
        {"concludes": "ConcludeTransfer", "condition": "case.situation.goal_reached"},
        {
            "concludes": "AbandonTransfer",
            "condition": "case.situation.receiver_overflowing",
        },
        {
            "concludes": "RetargetFillLevel",
            "condition": "True",
            "requires_concluded": ["PourIntoReceiver"],
            "value": "case.situation.requested_fill_level",
        },
    ],
}
"""
The hand-written transfer theory, restated as a specification.
"""


def _compile(specification_data: dict):
    compiler = TheoryCompiler(
        declaration_kinds=DECLARATION_KINDS, situation_type=TransferSituation
    )
    return compiler.compile(TheorySpecification.from_json(specification_data))


def _with_rules(rules: list) -> dict:
    """
    A minimal specification with one gated constraint and the given rules.
    """
    return {
        "constraints": [
            {
                "identifier": "abort",
                "kind": "motion_abort",
                "parameters": {"reason": "testing"},
                "gated_by": "Abort",
            }
        ],
        "rules": rules,
    }


class TestGoldenEquivalenceWithTheHandWrittenTheory:
    """
    A compiled specification concludes exactly what the hand-written theory concludes.
    """

    SCENARIOS = [
        situation(),
        situation(opening_within=False),
        situation(source_above_receiver=False),
        situation(receiver_fill_level=0.7, goal_reached=True),
        situation(receiver_fill_level=1.0, receiver_overflowing=True),
        situation(near=False, opening_within=False, is_tilted=False),
    ]

    @pytest.fixture(scope="class")
    def compiled_theory(self):
        return _compile(TRANSFER_SPECIFICATION)

    @pytest.fixture(scope="class")
    def hand_written_theory(self):
        return build_substance_transfer_theory()

    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_the_same_decisions_are_concluded(
        self, compiled_theory, hand_written_theory, scenario
    ):
        compiled_names = {
            type(decision).__name__ for decision in compiled_theory.infer([scenario])
        }
        hand_written_names = {
            type(decision).__name__
            for decision in hand_written_theory.infer([scenario])
        }
        assert compiled_names == hand_written_names

    def test_the_supplied_fill_goal_matches_the_situation(self, compiled_theory):
        pouring = situation()
        [retarget] = [
            decision
            for decision in compiled_theory.infer([pouring])
            if type(decision).__name__ == "RetargetFillLevel"
        ]
        assert retarget.value == pouring.requested_fill_level

    def test_the_declarations_mirror_the_constraints(self, compiled_theory):
        identifiers = [
            declaration.identifier
            for declaration in compiled_theory.required_constraints
        ]
        assert identifiers == [
            "aim",
            "rim_clearance",
            "quantity",
            "return_upright",
            "abort",
        ]
        quantity = compiled_theory.required_constraints[2]
        assert quantity.gating_decision_type.__name__ == "PourIntoReceiver"
        assert quantity.parameter_channel.attribute_name == "value"


class TestConditionValidation:
    """
    What the condition grammar rejects.
    """

    @pytest.fixture
    def validator(self):
        return ConditionValidator(situation_type=TransferSituation)

    def test_a_fact_read_with_comparisons_and_logic_passes(self, validator):
        validator.validate(
            "case.situation.near and case.situation.receiver_fill_level < 0.5"
        )

    def test_a_derived_property_counts_as_a_fact(self, validator):
        validator.validate("case.situation.spill_risk")

    def test_a_call_is_rejected(self, validator):
        with pytest.raises(ForbiddenConditionSyntaxError):
            validator.validate("__import__('os').system('rm -rf /')")

    def test_a_bare_name_is_rejected(self, validator):
        with pytest.raises(ForbiddenConditionSyntaxError):
            validator.validate("near")

    def test_an_attribute_outside_the_situation_is_rejected(self, validator):
        with pytest.raises(ForbiddenConditionSyntaxError):
            validator.validate("case.world")

    def test_a_deep_attribute_path_is_rejected(self, validator):
        with pytest.raises(ForbiddenConditionSyntaxError):
            validator.validate("case.situation.source.fill_level")

    def test_an_unknown_fact_is_rejected_with_the_allowed_ones(self, validator):
        with pytest.raises(UnknownSituationFactError) as error:
            validator.validate("case.situation.fill_velocity")
        assert "near" in error.value.allowed_facts

    def test_unparseable_source_is_rejected(self, validator):
        with pytest.raises(MalformedConditionError):
            validator.validate("case.situation.near and")

    def test_a_string_constant_is_rejected(self, validator):
        with pytest.raises(ForbiddenConditionSyntaxError):
            validator.validate("'near'")


class TestSpecificationRejection:
    """
    What the compiler rejects before anything could run.
    """

    def test_an_unknown_specification_field_is_rejected(self):
        with pytest.raises(UnknownSpecificationFieldError):
            _compile({"constraints": [], "rules": [], "goals": []})

    def test_an_unknown_constraint_kind_is_rejected(self):
        with pytest.raises(UnknownDeclarationKindError):
            _compile(
                {
                    "constraints": [
                        {"identifier": "x", "kind": "levitate", "parameters": {}}
                    ],
                    "rules": [],
                }
            )

    def test_wrong_declaration_parameters_are_rejected(self):
        with pytest.raises(InvalidDeclarationParametersError) as error:
            _compile(
                {
                    "constraints": [
                        {
                            "identifier": "x",
                            "kind": "motion_abort",
                            "parameters": {"speed": 1.0},
                        }
                    ],
                    "rules": [],
                }
            )
        assert error.value.unexpected == ("speed",)
        assert error.value.missing == ("reason",)

    def test_a_rule_concluding_an_unknown_decision_is_rejected(self):
        with pytest.raises(UnknownDecisionNameError):
            _compile(
                _with_rules(
                    [
                        {"concludes": "Abort", "condition": "True"},
                        {"concludes": "Levitate", "condition": "True"},
                    ]
                )
            )

    def test_a_gating_decision_no_rule_concludes_is_rejected(self):
        with pytest.raises(UnconcludableDecisionError):
            _compile(_with_rules([]))

    def test_a_decision_gating_and_supplying_at_once_is_rejected(self):
        specification = _with_rules([{"concludes": "Abort", "condition": "True"}])
        specification["constraints"][0]["value_from"] = "Abort"
        with pytest.raises(DecisionRoleConflictError):
            _compile(specification)

    def test_a_regime_rule_carrying_a_value_is_rejected(self):
        with pytest.raises(InvalidRuleValueError):
            _compile(
                _with_rules(
                    [{"concludes": "Abort", "condition": "True", "value": "0.5"}]
                )
            )
