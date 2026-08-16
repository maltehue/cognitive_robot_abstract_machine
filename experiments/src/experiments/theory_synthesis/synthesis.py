"""
From an instruction sentence to a compiled, chart-ready theory.

The pipeline is generator → parse → validate → compile, and its value is where it fails:
a wrong proposal is rejected by the specification format, the condition grammar or the
compiler, each with an error naming the fix — which is exactly the feedback a repair
loop sends back. A proposal that gets through is a theory indistinguishable from a hand-
written one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from typing_extensions import Any, Dict, Optional

from krrood.exceptions import DataclassException

from experiments.theory_synthesis.generator import SpecificationGenerator
from experiments.theory_synthesis.prompting import (
    SceneDescription,
    SynthesisPromptBuilder,
)
from semantic_digital_twin.reasoning.knowledge_servoing.general_rdr_theory import (
    GeneralRDRTheory,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_compiler import (
    TheoryCompiler,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_specification import (
    TheorySpecification,
)


@dataclass
class UnparseableSpecificationTextError(DataclassException):
    """
    Raised when a proposal contains no parseable JSON object.
    """

    response_text: str
    """The proposal that could not be parsed."""

    def error_message(self) -> str:
        return "The proposal contains no parseable JSON object"

    def suggest_correction(self) -> str:
        return "Answer with a single JSON object and nothing else"


def extract_json_object(response_text: str) -> Dict[str, Any]:
    """
    Parses the JSON object out of a proposal, tolerating surrounding prose or fences.

    :param response_text: The raw proposal text.
    :return: The parsed object.
    :raises UnparseableSpecificationTextError: if no JSON object can be parsed.
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(response_text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise UnparseableSpecificationTextError(response_text=response_text)


@dataclass
class SynthesizedTheory:
    """
    One successful synthesis: the theory and the artifacts it came from.
    """

    theory: GeneralRDRTheory
    """
    The compiled theory, ready for the chart assembler.
    """

    specification: TheorySpecification
    """
    The validated specification the theory was compiled from.
    """

    response_text: str
    """
    The proposal the specification was parsed from.
    """


@dataclass
class TheorySynthesis:
    """
    Turns instruction sentences into compiled theories through a generator.
    """

    generator: SpecificationGenerator
    """
    Proposes and revises specification text.
    """

    compiler: TheoryCompiler
    """
    Validates and compiles proposals.
    """

    prompt_builder: SynthesisPromptBuilder
    """
    Builds the prompts from the interfaces the compiler accepts.
    """

    def synthesize(
        self, scene: SceneDescription, instruction: str
    ) -> SynthesizedTheory:
        """
        Synthesizes a theory for one instruction, in a fresh conversation.

        :param scene: What the instruction's names refer to.
        :param instruction: The task instruction in natural language.
        :return: The compiled theory and its artifacts.
        """
        response_text = self.generator.propose(
            system_prompt=self.prompt_builder.system_prompt(),
            instruction_prompt=self.prompt_builder.instruction_prompt(
                scene, instruction
            ),
        )
        return self.compile_proposal(response_text)

    def revise(self, feedback: str) -> SynthesizedTheory:
        """
        Asks the generator to revise its last proposal and compiles the revision.

        :param feedback: What was wrong with the last proposal.
        :return: The compiled theory and its artifacts.
        """
        return self.compile_proposal(self.generator.revise(feedback))

    def compile_proposal(self, response_text: str) -> SynthesizedTheory:
        """
        Parses, validates and compiles one proposal.

        :param response_text: The raw proposal text.
        :return: The compiled theory and its artifacts.
        """
        specification = TheorySpecification.from_json(
            extract_json_object(response_text)
        )
        return SynthesizedTheory(
            theory=self.compiler.compile(specification),
            specification=specification,
            response_text=response_text,
        )
