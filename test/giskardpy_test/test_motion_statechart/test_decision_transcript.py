"""
The decision transcript, which records what the theory concluded and when.
"""

from __future__ import annotations

from dataclasses import dataclass

from giskardpy.motion_statechart.knowledge_servoing.decision_transcript import (
    DecisionTranscript,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    DecisionSet,
    ParameterDecision,
    RegimeDecision,
)


@dataclass(frozen=True)
class Approach(RegimeDecision):
    """
    A regime decision standing in for a theory's first phase.
    """


@dataclass(frozen=True)
class Act(RegimeDecision):
    """
    A regime decision standing in for a theory's second phase.
    """


@dataclass(frozen=True)
class SetRate(ParameterDecision):
    """
    A parameter decision carrying a value the transcript should preserve.
    """

    rate: float
    """
    The value supplied to the controller.
    """


class TestDecisionTranscript:
    """
    Whether a run's regime turnovers are recoverable from the transcript afterwards.
    """

    def test_a_steady_conclusion_records_one_change(self):
        transcript = DecisionTranscript()
        for control_cycle in range(10):
            transcript.record(DecisionSet((Approach(),)), control_cycle)
        assert len(transcript.changes) == 1

    def test_a_turnover_records_what_entered_and_what_was_withdrawn(self):
        transcript = DecisionTranscript()
        transcript.record(DecisionSet((Approach(),)), 0)
        transcript.record(DecisionSet((Act(),)), 7)

        turnover = transcript.changes[-1]
        assert turnover.control_cycle == 7
        assert turnover.entered == (Act,)
        assert turnover.withdrawn == (Approach,)

    def test_nothing_is_recorded_before_the_first_inference(self):
        transcript = DecisionTranscript()
        transcript.record(None, 0)
        assert transcript.changes == []

    def test_the_cycle_a_decision_first_appeared_is_recoverable(self):
        transcript = DecisionTranscript()
        transcript.record(DecisionSet((Approach(),)), 0)
        transcript.record(DecisionSet((Approach(), Act())), 4)

        assert transcript.cycle_of_first(Act) == 4
        assert transcript.cycle_of_first(SetRate) is None

    def test_parameter_values_are_preserved_with_the_change(self):
        transcript = DecisionTranscript()
        transcript.record(DecisionSet((Act(), SetRate(0.25))), 3)

        [set_rate] = [
            decision
            for decision in transcript.changes[-1].decisions
            if isinstance(decision, SetRate)
        ]
        assert set_rate.rate == 0.25
