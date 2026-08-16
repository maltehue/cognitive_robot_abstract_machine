"""
Synthesizing theories from instruction sentences, with the generator mocked.

The pipeline's contract is pinned against a mimic generator: a good proposal compiles
into a chart-ready theory, a bad one is rejected with the typed error the repair loop
would relay, and prose around the JSON is tolerated. The live generator is exercised by
one smoke test that only runs when explicitly requested, so the suite never depends on a
model being reachable.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

import pytest

from experiments.theory_synthesis.generator import (
    ClaudeCommandLineGenerator,
    SpecificationGenerator,
)
from experiments.theory_synthesis.prompting import (
    ContainerDescription,
    SceneDescription,
    SynthesisPromptBuilder,
)
from experiments.theory_synthesis.synthesis import (
    TheorySynthesis,
    UnparseableSpecificationTextError,
    extract_json_object,
)
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    GENERIC_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.knowledge_servoing.exceptions import (
    UnknownDeclarationKindError,
    UnknownSituationFactError,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_compiler import (
    TheoryCompiler,
)
from semantic_digital_twin.reasoning.substance_transfer.declarations import (
    TRANSFER_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)

from ..semantic_digital_twin_test.test_reasoning.test_theory_compilation import (
    TRANSFER_SPECIFICATION,
)

DECLARATION_KINDS = {**GENERIC_DECLARATION_KINDS, **TRANSFER_DECLARATION_KINDS}

SCENE = SceneDescription(
    containers=(
        ContainerDescription(
            annotation_name="source_cup",
            description="the cup the robot holds",
            capacity_milliliters=100.0,
        ),
        ContainerDescription(
            annotation_name="receiving_cup",
            description="the flask on the table",
            capacity_milliliters=100.0,
        ),
    ),
    other_objects={"laptop": "a laptop beside the flask"},
)

INSTRUCTION = "Transfer 40 ml into the flask."


@dataclass
class CannedGenerator(SpecificationGenerator):
    """
    A generator answering with prepared responses, standing in for the model.
    """

    responses: list
    """
    The responses, consumed in order across propose and revise.
    """

    received_prompts: list = field(default_factory=list)
    """
    Every prompt the generator was given, for assertions.
    """

    def propose(self, system_prompt: str, instruction_prompt: str) -> str:
        self.received_prompts.append(instruction_prompt)
        return self.responses.pop(0)

    def revise(self, feedback: str) -> str:
        self.received_prompts.append(feedback)
        return self.responses.pop(0)


def _synthesis(responses: list) -> TheorySynthesis:
    return TheorySynthesis(
        generator=CannedGenerator(responses=responses),
        compiler=TheoryCompiler(
            declaration_kinds=DECLARATION_KINDS, situation_type=TransferSituation
        ),
        prompt_builder=SynthesisPromptBuilder(
            situation_type=TransferSituation, declaration_kinds=DECLARATION_KINDS
        ),
    )


class TestSynthesisPipeline:
    """
    What the pipeline makes of a generator's proposals.
    """

    def test_a_valid_proposal_compiles_into_a_chart_ready_theory(self):
        synthesis = _synthesis([json.dumps(TRANSFER_SPECIFICATION)])
        synthesized = synthesis.synthesize(SCENE, INSTRUCTION)
        assert len(synthesized.theory.required_constraints) == 5
        assert {
            decision.__name__ for decision in synthesized.theory.decision_types
        } == {
            "AlignSourceOverReceiver",
            "PourIntoReceiver",
            "ConcludeTransfer",
            "AbandonTransfer",
            "RetargetFillLevel",
        }

    def test_prose_around_the_json_is_tolerated(self):
        wrapped = f"Here is the specification:\n```json\n{json.dumps(TRANSFER_SPECIFICATION)}\n```"
        synthesized = _synthesis([wrapped]).synthesize(SCENE, INSTRUCTION)
        assert len(synthesized.specification.rules) == 5

    def test_a_proposal_without_json_is_rejected(self):
        with pytest.raises(UnparseableSpecificationTextError):
            _synthesis(["I cannot help with that."]).synthesize(SCENE, INSTRUCTION)

    def test_an_out_of_vocabulary_kind_is_rejected_with_the_known_kinds(self):
        proposal = {
            "constraints": [
                {"identifier": "x", "kind": "levitate_object", "parameters": {}}
            ],
            "rules": [],
        }
        with pytest.raises(UnknownDeclarationKindError) as error:
            _synthesis([json.dumps(proposal)]).synthesize(SCENE, INSTRUCTION)
        assert "transfer_quantity" in error.value.known_kinds

    def test_a_hallucinated_fact_is_rejected_by_the_condition_grammar(self):
        proposal = {
            "constraints": [
                {
                    "identifier": "abort",
                    "kind": "motion_abort",
                    "parameters": {"reason": "testing"},
                    "gated_by": "Abort",
                }
            ],
            "rules": [
                {"concludes": "Abort", "condition": "case.situation.is_corrosive"}
            ],
        }
        with pytest.raises(UnknownSituationFactError):
            _synthesis([json.dumps(proposal)]).synthesize(SCENE, INSTRUCTION)

    def test_a_revision_compiles_like_a_proposal(self):
        synthesis = _synthesis(["not json", json.dumps(TRANSFER_SPECIFICATION)])
        with pytest.raises(UnparseableSpecificationTextError):
            synthesis.synthesize(SCENE, INSTRUCTION)
        revised = synthesis.revise("Answer with a single JSON object only.")
        assert len(revised.specification.constraints) == 5


class TestPromptContent:
    """
    Whether the prompt offers exactly what the compiler accepts.
    """

    @pytest.fixture
    def builder(self):
        return SynthesisPromptBuilder(
            situation_type=TransferSituation, declaration_kinds=DECLARATION_KINDS
        )

    def test_every_scalar_fact_is_offered(self, builder):
        system_prompt = builder.system_prompt()
        assert "case.situation.receiver_overflowing" in system_prompt
        assert "case.situation.spill_risk" in system_prompt

    def test_subject_objects_are_not_offered_as_facts(self, builder):
        assert "case.situation.source:" not in builder.system_prompt()

    def test_every_declaration_kind_is_offered_with_its_parameters(self, builder):
        system_prompt = builder.system_prompt()
        for kind_name in DECLARATION_KINDS:
            assert kind_name in system_prompt
        assert "parameter fill_level_tolerance" in system_prompt

    def test_the_instruction_prompt_carries_scene_and_sentence(self, builder):
        prompt = builder.instruction_prompt(SCENE, INSTRUCTION)
        assert "source_cup" in prompt
        assert "100 ml" in prompt
        assert INSTRUCTION in prompt


class TestJsonExtraction:
    """
    Parsing the object out of proposal text.
    """

    def test_a_bare_object_parses(self):
        assert extract_json_object('{"constraints": []}') == {"constraints": []}

    def test_a_fenced_object_parses(self):
        assert extract_json_object('```json\n{"rules": []}\n```') == {"rules": []}

    def test_text_without_an_object_raises(self):
        with pytest.raises(UnparseableSpecificationTextError):
            extract_json_object("no object here")


@pytest.mark.skipif(
    shutil.which("claude") is None or not os.environ.get("RUN_LIVE_LLM_TESTS"),
    reason="live generator smoke runs only with RUN_LIVE_LLM_TESTS set and the claude "
    "command line installed",
)
class TestLiveGeneratorSmoke:
    """
    One live round trip through the command-line generator; mirrors the mocked pipeline.
    """

    def test_the_flagship_sentence_synthesizes(self):
        synthesis = TheorySynthesis(
            generator=ClaudeCommandLineGenerator(),
            compiler=TheoryCompiler(
                declaration_kinds=DECLARATION_KINDS, situation_type=TransferSituation
            ),
            prompt_builder=SynthesisPromptBuilder(
                situation_type=TransferSituation, declaration_kinds=DECLARATION_KINDS
            ),
        )
        synthesized = synthesis.synthesize(
            SCENE,
            "Transfer 40 ml into the flask; it is corrosive, so keep well clear of "
            "the laptop and pour gently.",
        )
        kinds = {
            constraint.kind for constraint in synthesized.specification.constraints
        }
        assert "transfer_quantity" in kinds
