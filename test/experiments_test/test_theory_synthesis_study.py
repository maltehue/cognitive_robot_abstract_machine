"""The sentence study's mechanics, with the generator mocked.

The study must report faithfully in all three of its outcome shapes: a valid proposal executes and
its run is recorded, an empty specification counts as the model declining, and an invalid proposal
is recorded as the typed rejection — never silently guessed around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from experiments.theory_synthesis.generator import SpecificationGenerator
from experiments.theory_synthesis.study import (
    STUDY_SENTENCES,
    SentenceExpectation,
    StudySentence,
    SynthesisStudy,
)
from semantic_digital_twin.utils import tracy_installed

SELF_CONTAINED_SPECIFICATION = {
    "constraints": [
        {
            "identifier": "aim",
            "kind": "aimed_transfer",
            "parameters": {
                "source_name": "source_cup",
                "receiver_name": "receiving_cup",
            },
            "gated_by": "Align",
        },
        {
            "identifier": "rim_clearance",
            "kind": "rim_clearance",
            "parameters": {
                "source_name": "source_cup",
                "receiver_name": "receiving_cup",
            },
            "gated_by": "Align",
        },
        {
            "identifier": "quantity",
            "kind": "transfer_quantity",
            "parameters": {
                "source_name": "source_cup",
                "receiver_name": "receiving_cup",
            },
            "gated_by": "Pour",
            "value_from": "SetGoal",
        },
        {
            "identifier": "return_upright",
            "kind": "return_upright",
            "parameters": {"subject_name": "source_cup"},
            "gated_by": "Finish",
        },
    ],
    "rules": [
        {
            "concludes": "Align",
            "condition": "case.situation.near and case.situation.receiver_fill_level < 0.4",
        },
        {
            "concludes": "Pour",
            "condition": (
                "case.situation.opening_within and case.situation.source_above_receiver "
                "and case.situation.receiver_fill_level < 0.4"
            ),
        },
        {
            "concludes": "Finish",
            "condition": "case.situation.receiver_fill_level >= 0.4",
        },
        {
            "concludes": "SetGoal",
            "condition": "True",
            "requires_concluded": ["Pour"],
            "value": "0.4",
        },
    ],
}
"""A self-contained specification of the shape the prompt asks for, with its goal in its rules."""


@dataclass
class OneAnswerGenerator(SpecificationGenerator):
    """A generator that always proposes the same text."""

    answer: str
    """The proposal returned for every sentence."""

    def propose(self, system_prompt: str, instruction_prompt: str) -> str:
        return self.answer

    def revise(self, feedback: str) -> str:
        return self.answer


def _study(answer: str) -> SynthesisStudy:
    return SynthesisStudy(generator_factory=lambda: OneAnswerGenerator(answer=answer))


SENTENCE = StudySentence(
    identifier="probe",
    instruction="Pour 40 ml into the flask.",
    expectation=SentenceExpectation.EXECUTES,
)


class TestStudyOutcomeShapes:
    """The three outcome shapes the study must report faithfully."""

    def test_a_valid_proposal_executes_and_reaches_its_own_goal(self):
        if not tracy_installed():
            pytest.skip("Tracy not installed")
        outcome = _study(json.dumps(SELF_CONTAINED_SPECIFICATION)).run_sentence(
            SENTENCE
        )
        assert outcome.executed
        assert outcome.ended_by_theory
        assert outcome.final_fill_level == pytest.approx(0.4, abs=0.06)
        assert "quantity" in outcome.constraint_identifiers

    def test_an_empty_specification_counts_as_declined(self):
        outcome = _study('{"constraints": [], "rules": []}').run_sentence(SENTENCE)
        assert outcome.declined
        assert not outcome.executed
        assert outcome.rejection is None

    def test_an_invalid_proposal_is_recorded_as_its_typed_rejection(self):
        proposal = {
            "constraints": [{"identifier": "x", "kind": "levitate", "parameters": {}}],
            "rules": [],
        }
        outcome = _study(json.dumps(proposal)).run_sentence(SENTENCE)
        assert outcome.rejection is not None
        assert "UnknownDeclarationKindError" in outcome.rejection
        assert not outcome.executed


class TestStudySentenceSet:
    """The properties the sentence set was designed around."""

    def test_exactly_one_sentence_is_outside_the_vocabulary(self):
        declined = [
            sentence
            for sentence in STUDY_SENTENCES
            if sentence.expectation is SentenceExpectation.DECLINED
        ]
        assert len(declined) == 1

    def test_the_set_is_large_enough_to_not_be_anecdotal(self):
        assert len(STUDY_SENTENCES) >= 6
