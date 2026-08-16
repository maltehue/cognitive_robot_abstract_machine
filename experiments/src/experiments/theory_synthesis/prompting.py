"""
Builds the prompts that ask a language model for a theory specification.

The prompt is generated from the interfaces it targets — the situation type's facts and
the declaration kinds' parameters are verbalized from their own definitions — so what
the model is offered is exactly what the compiler will accept. The instruction sentence
and the scene are the only free text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields

from typing_extensions import List, Mapping, Tuple, Type

from experiments.theory_synthesis.documentation_extraction import field_documentation
from semantic_digital_twin.reasoning.knowledge_servoing.condition_validation import (
    situation_fact_names,
)
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    ConstraintDeclaration,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import Situation

_DECLARATION_OWN_FIELDS = ("identifier", "gating_decision_type", "parameter_channel")

_FORMAT_RULES = """\
You translate one task instruction into a theory specification: a single JSON object with the
fields "constraints" and "rules", and nothing else.

Each constraint is one requirement on the motion:
- "identifier": a short snake_case name for the constraint.
- "kind": one of the constraint kinds listed below.
- "parameters": exactly the kind's parameters.
- "gated_by" (optional): the name of a decision; the constraint is enforced only while your rules
  conclude that decision. Without it the constraint is enforced for the whole motion.
- "value_from" (optional): the name of a decision whose concluded value sets the constraint's
  runtime target.

Each rule concludes one decision from the current situation:
- "concludes": the decision name. A decision exists only because some constraint names it in
  "gated_by" or "value_from"; do not conclude anything no constraint uses.
- "condition": a boolean expression over the situation facts listed below. Allowed syntax: and,
  or, not, comparisons, + - * /, numbers, True, False, and reads of the form
  case.situation.<fact>. Nothing else — no function calls, no other names.
- "requires_concluded" (optional): decision names that must already be concluded this cycle.
- "defeated_by" (optional): conditions that withdraw the conclusion even when the rule fires.
- "value" (required exactly when the decision is used in "value_from"): a numeric expression,
  either a constant or an expression over case.situation.<fact>.

Express quantities as fill fractions of the receiving container, computed from the capacities the
scene states. Answer with the JSON object only."""

_REFERENCE_EXAMPLE = """\
Write the quantities you computed into the rules themselves — for a target fill fraction of 0.33,
compare case.situation.receiver_fill_level against 0.33 directly. The specification must be
self-contained: it, not the surrounding system, knows what the instruction asked for.

Example. Instruction: "Fill the bowl to about a third from the jug." — with the jug named
example_jug and the bowl named example_bowl, each holding 100 ml, a correct specification is:

{
  "constraints": [
    {"identifier": "aim", "kind": "aimed_transfer",
     "parameters": {"source_name": "example_jug", "receiver_name": "example_bowl"},
     "gated_by": "Align"},
    {"identifier": "rim_clearance", "kind": "rim_clearance",
     "parameters": {"source_name": "example_jug", "receiver_name": "example_bowl"},
     "gated_by": "Align"},
    {"identifier": "quantity", "kind": "transfer_quantity",
     "parameters": {"source_name": "example_jug", "receiver_name": "example_bowl"},
     "gated_by": "Pour", "value_from": "SetGoal"},
    {"identifier": "return_upright", "kind": "return_upright",
     "parameters": {"subject_name": "example_jug"}, "gated_by": "Finish"},
    {"identifier": "abort", "kind": "motion_abort",
     "parameters": {"reason": "the bowl would overflow"}, "gated_by": "Abort"}
  ],
  "rules": [
    {"concludes": "Align",
     "condition": "case.situation.near and case.situation.receiver_fill_level < 0.33"},
    {"concludes": "Pour",
     "condition": "case.situation.opening_within and case.situation.source_above_receiver and case.situation.receiver_fill_level < 0.33",
     "defeated_by": ["case.situation.receiver_overflowing"]},
    {"concludes": "Finish", "condition": "case.situation.receiver_fill_level >= 0.33"},
    {"concludes": "Abort", "condition": "case.situation.receiver_overflowing"},
    {"concludes": "SetGoal", "condition": "True",
     "requires_concluded": ["Pour"], "value": "0.33"}
  ]
}

Always include the aim, rim_clearance, quantity and return_upright constraints for a transfer —
they are what makes the pour physically executable. Add further constraints exactly when the
instruction warrants them. If the instruction asks for something no constraint kind can express,
answer with the JSON object {"constraints": [], "rules": []} and nothing else."""


@dataclass(frozen=True)
class ContainerDescription:
    """
    One container the instruction may refer to.
    """

    annotation_name: str
    """
    The name the specification must use for it.
    """

    description: str
    """
    What it is, in the instruction's terms.
    """

    capacity_milliliters: float
    """
    How much it holds when full, for converting quantities to fill fractions.
    """


@dataclass(frozen=True)
class SceneDescription:
    """
    What the model is told about the scene the instruction happens in.
    """

    containers: Tuple[ContainerDescription, ...]
    """
    The containers, by the names the specification must use.
    """

    other_objects: Mapping[str, str] = field(default_factory=dict)
    """
    Other named objects and what they are.
    """

    def verbalized(self) -> str:
        """
        The scene as prompt text.
        """
        lines: List[str] = ["The scene contains:"]
        for container in self.containers:
            lines.append(
                f"- {container.annotation_name}: {container.description}, holding "
                f"{container.capacity_milliliters:g} ml when full"
            )
        for name, description in self.other_objects.items():
            lines.append(f"- {name}: {description}")
        return "\n".join(lines)


@dataclass
class SynthesisPromptBuilder:
    """
    Builds the system and instruction prompts from the interfaces they target.
    """

    situation_type: Type[Situation]
    """
    The situation type whose facts conditions may read.
    """

    declaration_kinds: Mapping[str, Type[ConstraintDeclaration]]
    """
    The constraint vocabulary the specification may declare from.
    """

    def system_prompt(self) -> str:
        """
        The full system prompt: format rules, fact vocabulary, kinds, and the example.
        """
        return "\n\n".join(
            [
                _FORMAT_RULES,
                self._verbalized_facts(),
                self._verbalized_kinds(),
                _REFERENCE_EXAMPLE,
            ]
        )

    def instruction_prompt(self, scene: SceneDescription, instruction: str) -> str:
        """
        The per-request prompt: the scene and the instruction to translate.

        :param scene: What the instruction's names refer to.
        :param instruction: The task instruction in natural language.
        """
        return f"{scene.verbalized()}\n\nInstruction: {json.dumps(instruction)}"

    def _verbalized_facts(self) -> str:
        """
        The situation facts as prompt text, from the situation type's own documentation.
        """
        documentation = field_documentation(self.situation_type)
        lines = ["Situation facts a condition may read:"]
        for fact_name in sorted(situation_fact_names(self.situation_type)):
            summary = documentation.get(fact_name, "")
            lines.append(f"- case.situation.{fact_name}: {summary}")
        return "\n".join(lines)

    def _verbalized_kinds(self) -> str:
        """
        The constraint kinds as prompt text, from the declarations' own documentation.
        """
        lines = ["Constraint kinds a specification may declare:"]
        for kind_name in sorted(self.declaration_kinds):
            declaration_type = self.declaration_kinds[kind_name]
            summary = (declaration_type.__doc__ or "").strip().splitlines()[0]
            lines.append(f"- {kind_name}: {summary}")
            documentation = field_documentation(declaration_type)
            for declared_field in dataclass_fields(declaration_type):
                if declared_field.name in _DECLARATION_OWN_FIELDS:
                    continue
                lines.append(
                    f"    parameter {declared_field.name}: "
                    f"{documentation.get(declared_field.name, '')}"
                )
        return "\n".join(lines)
