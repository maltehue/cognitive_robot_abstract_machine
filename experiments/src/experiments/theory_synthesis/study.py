"""The sentence study: synthesize, execute, and report — judgment stays with the human.

Nothing here verifies that a synthesized theory does what the sentence meant; knowing the intent is
exactly what the system does not have. Each sentence's outcome is therefore a faithful record of
what happened — what was synthesized, what was rejected and why, what the run did — for the person
who wrote the sentence to judge.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from typing_extensions import Callable, List, Optional, Tuple

from experiments.knowledge_servoing.constraint_factories import (
    DeclaredMotionAborted,
    build_transfer_catalog,
)
from experiments.knowledge_servoing.scenario import (
    TransferScenario,
    build_transfer_scenario,
    pouring_plane_stabilization,
)
from experiments.theory_synthesis.generator import SpecificationGenerator
from experiments.theory_synthesis.prompting import (
    ContainerDescription,
    SceneDescription,
    SynthesisPromptBuilder,
)
from experiments.theory_synthesis.synthesis import SynthesizedTheory, TheorySynthesis
from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.knowledge_servoing.chart_assembler import (
    PluggedTheory,
    TheoryChartAssembler,
)
from giskardpy.motion_statechart.knowledge_servoing.decision_transcript import (
    DecisionTranscript,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.exceptions import DataclassException
from semantic_digital_twin.reasoning.knowledge_servoing.constraint_declarations import (
    GENERIC_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.knowledge_servoing.theory_compiler import (
    TheoryCompiler,
)
from semantic_digital_twin.reasoning.substance_transfer import (
    TransferSituationGrounding,
)
from semantic_digital_twin.reasoning.substance_transfer.declarations import (
    ReturnUprightDeclaration,
    TRANSFER_DECLARATION_KINDS,
)
from semantic_digital_twin.reasoning.substance_transfer.situation import (
    TransferSituation,
)

STUDY_DECLARATION_KINDS = {**GENERIC_DECLARATION_KINDS, **TRANSFER_DECLARATION_KINDS}
"""The constraint vocabulary the study offers."""

STUDY_SCENE = SceneDescription(
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
    other_objects={
        "laptop": "a laptop directly beside the flask",
        "balance": "a laboratory balance on the table, 0.4 m from the flask",
    },
)
"""What the model is told about the demonstration scene."""

MAXIMUM_CONTROL_CYCLES = 3000
"""Cycle budget after which a run counts as not terminating."""


class SentenceExpectation(Enum):
    """What kind of outcome a sentence was put in the set to demonstrate."""

    EXECUTES = auto()
    """The sentence is expressible and the synthesized theory should run."""

    DECLINED = auto()
    """The sentence is outside the vocabulary and must be declined or rejected, not guessed."""


@dataclass(frozen=True)
class StudySentence:
    """One sentence of the study."""

    identifier: str
    """Short name for the sentence's artifacts."""

    instruction: str
    """The instruction as the human would give it."""

    expectation: SentenceExpectation
    """Why the sentence is in the set."""


STUDY_SENTENCES: Tuple[StudySentence, ...] = (
    StudySentence(
        identifier="corrosive_flagship",
        instruction=(
            "Transfer 40 ml of reagent B into the flask; it is corrosive, so keep the "
            "cup well clear of the balance and pour gently."
        ),
        expectation=SentenceExpectation.EXECUTES,
    ),
    StudySentence(
        identifier="quantity_only_control",
        instruction="Pour 60 ml into the flask.",
        expectation=SentenceExpectation.EXECUTES,
    ),
    StudySentence(
        identifier="half_full",
        instruction="Fill the flask about half full.",
        expectation=SentenceExpectation.EXECUTES,
    ),
    StudySentence(
        identifier="tight_tolerance",
        instruction="Transfer 25 ml into the flask; precision matters, stay within 3 ml.",
        expectation=SentenceExpectation.EXECUTES,
    ),
    StudySentence(
        identifier="keep_clear_of_laptop",
        instruction=(
            "Top the flask up to 70 ml, and keep the cup away from the laptop while "
            "you carry it."
        ),
        expectation=SentenceExpectation.EXECUTES,
    ),
    StudySentence(
        identifier="gentle_only",
        instruction="Pour 50 ml into the flask very carefully and slowly.",
        expectation=SentenceExpectation.EXECUTES,
    ),
    StudySentence(
        identifier="out_of_vocabulary",
        instruction="Stir the mixture in the flask three times.",
        expectation=SentenceExpectation.DECLINED,
    ),
)
"""The study's sentences: the flagship, a plain control, variations, and one outside the vocabulary."""


@dataclass
class SentenceOutcome:
    """What actually happened for one sentence, for the human to judge."""

    sentence: StudySentence
    """The sentence this outcome belongs to."""

    response_text: Optional[str] = None
    """The generator's raw proposal, if generation succeeded."""

    rejection: Optional[str] = None
    """The typed rejection, as ``ErrorType: message``, if the proposal did not compile."""

    declined: bool = False
    """Whether the model answered with an empty specification, declining the sentence."""

    constraint_identifiers: Tuple[str, ...] = ()
    """The constraints the synthesized theory declared."""

    constraint_kinds: Tuple[str, ...] = ()
    """The declaration kinds of those constraints, in the same order."""

    executed: bool = False
    """Whether the synthesized theory was run."""

    ended_by_theory: bool = False
    """Whether the run terminated through the theory's own conclusions."""

    aborted_reason: Optional[str] = None
    """Why the theory aborted the motion, if it did."""

    final_fill_level: Optional[float] = None
    """The receiving container's fill level after the run."""

    control_cycles: Optional[float] = None
    """How many control cycles the run took."""

    decision_transcript: str = ""
    """The regime turnovers of the run, in the theory's vocabulary."""


@dataclass
class SynthesisStudy:
    """Runs the sentence set: one fresh conversation and one fresh world per sentence."""

    generator_factory: Callable[[], SpecificationGenerator]
    """Builds a fresh generator per sentence, so sentences cannot leak into each other."""

    visualize: bool = False
    """Whether to publish each run's world to RViz while it executes."""

    def run(self) -> List[SentenceOutcome]:
        """Runs every sentence of the study.

        :return: One outcome per sentence, in study order.
        """
        return [self.run_sentence(sentence) for sentence in STUDY_SENTENCES]

    def run_sentence(self, sentence: StudySentence) -> SentenceOutcome:
        """Synthesizes and, if a theory came out, executes one sentence.

        :param sentence: The sentence to run.
        :return: What happened.
        """
        outcome = SentenceOutcome(sentence=sentence)
        synthesis = TheorySynthesis(
            generator=self.generator_factory(),
            compiler=TheoryCompiler(
                declaration_kinds=STUDY_DECLARATION_KINDS,
                situation_type=TransferSituation,
            ),
            prompt_builder=SynthesisPromptBuilder(
                situation_type=TransferSituation,
                declaration_kinds=STUDY_DECLARATION_KINDS,
            ),
        )
        try:
            synthesized = synthesis.synthesize(STUDY_SCENE, sentence.instruction)
        except DataclassException as error:
            outcome.rejection = f"{type(error).__name__}: {error.error_message()}"
            return outcome
        outcome.response_text = synthesized.response_text
        outcome.constraint_identifiers = tuple(
            declaration.identifier
            for declaration in synthesized.theory.required_constraints
        )
        outcome.constraint_kinds = tuple(
            type(declaration).__name__
            for declaration in synthesized.theory.required_constraints
        )
        if not synthesized.specification.constraints:
            outcome.declined = True
            return outcome
        self._execute(synthesized, outcome)
        return outcome

    def _execute(
        self, synthesized: SynthesizedTheory, outcome: SentenceOutcome
    ) -> None:
        """Runs a synthesized theory on a fresh scenario and records what happened.

        :param synthesized: The compiled theory and its artifacts.
        :param outcome: The record the run is written into.
        """
        scenario = build_transfer_scenario()
        visualization = None
        if self.visualize:
            from experiments.knowledge_servoing.visualization import (
                WorldVisualization,
            )

            visualization = WorldVisualization.attach(scenario.world)
        statechart = MotionStatechart()
        assembler = TheoryChartAssembler(
            catalog=build_transfer_catalog(), world=scenario.world
        )
        [assembled] = assembler.assemble(
            [
                PluggedTheory(
                    name="synthesized",
                    theory=synthesized.theory,
                    grounding=_inert_goal_grounding(scenario),
                )
            ],
            statechart,
        )
        statechart.add_node(pouring_plane_stabilization(scenario))
        for declaration in synthesized.theory.required_constraints:
            if isinstance(declaration, ReturnUprightDeclaration):
                statechart.add_node(
                    EndMotion.when_true(assembled.monitors[declaration.identifier])
                )
                break

        executor = Executor(
            MotionStatechartContext(world=scenario.world),
            pacer=SimulationPacer(real_time_factor=1),
        )
        executor.compile(motion_statechart=statechart)
        transcript = DecisionTranscript()
        outcome.executed = True
        for _ in range(MAXIMUM_CONTROL_CYCLES):
            try:
                executor.tick()
            except DeclaredMotionAborted as aborted:
                outcome.aborted_reason = aborted.reason
                break
            transcript.record(assembled.decision_slot.latest, executor.control_cycles)
            if executor.motion_statechart.is_end_motion():
                outcome.ended_by_theory = True
                break
        outcome.final_fill_level = float(scenario.receiving_cup.fill_level)
        outcome.control_cycles = float(executor.control_cycles)
        outcome.decision_transcript = str(transcript)
        if visualization is not None:
            visualization.close()


def _inert_goal_grounding(scenario: TransferScenario) -> TransferSituationGrounding:
    """A grounding whose own goal facts never fire, since the theory carries its goal itself.

    The synthesized specification is self-contained — its rules compare the fill level against the
    quantities it computed — so the grounding's requested fill level is set where its derived goal
    facts stay out of the way.
    """
    return TransferSituationGrounding(
        source=scenario.source_cup,
        receiver=scenario.receiving_cup,
        requested_fill_level=1.0,
    )
