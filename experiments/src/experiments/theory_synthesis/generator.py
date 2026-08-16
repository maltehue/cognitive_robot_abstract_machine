"""
The language-model side of theory synthesis.

The generator's whole contract is text in, text out: it proposes a specification for an
instruction and revises it on feedback, holding whatever conversation state that needs.
Everything that makes the output trustworthy — parsing, validation, compilation —
happens outside it, so the generator is swappable and the tests mock it.
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional

from krrood.exceptions import DataclassException


@dataclass
class SpecificationGenerationError(DataclassException):
    """
    Raised when the generator process fails rather than answers.
    """

    detail: str
    """What went wrong, from the process's output."""

    def error_message(self) -> str:
        return f"The specification generator failed: {self.detail}"

    def suggest_correction(self) -> str:
        return "Check that the claude command line is installed and authenticated"


@dataclass
class SpecificationGenerator(ABC):
    """
    Proposes and revises specification text for an instruction.
    """

    @abstractmethod
    def propose(self, system_prompt: str, instruction_prompt: str) -> str:
        """
        Proposes specification text for an instruction, starting a fresh conversation.

        :param system_prompt: The format rules and vocabulary.
        :param instruction_prompt: The scene and the instruction.
        :return: The raw response text.
        """

    @abstractmethod
    def revise(self, feedback: str) -> str:
        """
        Revises the last proposal given feedback, continuing the conversation.

        :param feedback: What was wrong with the last proposal.
        :return: The raw response text.
        """


@dataclass
class ClaudeCommandLineGenerator(SpecificationGenerator):
    """
    Generates specifications through the locally authenticated claude command line.

    Tools are removed from the model's context, so the generator answers from the prompt
    alone — it cannot read the repository to look up what the validator would accept,
    which keeps the validation results meaningful.
    """

    model: Optional[str] = None
    """
    Model identifier to pin, or ``None`` for the command line's default.
    """

    timeout_seconds: float = 240.0
    """
    How long one generation may take before it counts as failed.
    """

    _session_identifier: Optional[str] = field(default=None, init=False, repr=False)
    """
    The conversation to continue on revision, from the last proposal's envelope.
    """

    def propose(self, system_prompt: str, instruction_prompt: str) -> str:
        command = [
            "claude",
            "-p",
            instruction_prompt,
            "--system-prompt",
            system_prompt,
            "--output-format",
            "json",
            "--tools",
            "",
        ]
        if self.model is not None:
            command.extend(["--model", self.model])
        envelope = self._run(command)
        self._session_identifier = envelope.get("session_id")
        return envelope["result"]

    def revise(self, feedback: str) -> str:
        if self._session_identifier is None:
            raise SpecificationGenerationError(
                detail="revise was called before any proposal"
            )
        command = [
            "claude",
            "--resume",
            self._session_identifier,
            "-p",
            feedback,
            "--output-format",
            "json",
            "--tools",
            "",
        ]
        envelope = self._run(command)
        self._session_identifier = envelope.get("session_id")
        return envelope["result"]

    def _run(self, command: List[str]) -> Dict[str, Any]:
        """
        Runs one command-line generation and parses its JSON envelope.

        :param command: The full command line.
        :return: The parsed envelope.
        :raises SpecificationGenerationError: if the process fails or answers
            unparseably.
        """
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise SpecificationGenerationError(
                detail=f"timed out after {self.timeout_seconds:g} s"
            ) from error
        if completed.returncode != 0:
            raise SpecificationGenerationError(
                detail=completed.stderr.strip() or completed.stdout.strip()
            )
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise SpecificationGenerationError(
                detail=f"unparseable envelope: {completed.stdout[:200]!r}"
            ) from error
