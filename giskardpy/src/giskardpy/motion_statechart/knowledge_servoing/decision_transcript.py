"""
A record of what the theory concluded, when, and what changed.

The interpretability claim of knowledge-based servoing rests on being able to say
afterwards why the robot did what it did. The controller's own logs answer that in joint
velocities; this answers it in the theory's own vocabulary, timestamped against the
control cycles the conclusions took effect on.

Only changes are recorded. A reasoner that concludes the same thing for two hundred
cycles produces two entries, not two hundred, because what a reader wants from a
transcript is the points where the regime turned over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import List, Optional, Tuple, Type

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    DecisionSet,
)


@dataclass(frozen=True)
class DecisionChange:
    """
    One turnover in what the theory concluded.
    """

    control_cycle: int
    """
    The control cycle the change became visible on.
    """

    entered: Tuple[Type[ControlDecision], ...]
    """
    Decision types concluded now that were not concluded before.
    """

    withdrawn: Tuple[Type[ControlDecision], ...]
    """
    Decision types concluded before that are no longer concluded.
    """

    decisions: Tuple[ControlDecision, ...]
    """
    The full decision set as of this change, including its parameter values.
    """

    def __str__(self) -> str:
        entered = ", ".join(sorted(decision.__name__ for decision in self.entered))
        withdrawn = ", ".join(sorted(decision.__name__ for decision in self.withdrawn))
        parts = [
            part
            for part in (
                f"+{entered}" if entered else "",
                f"-{withdrawn}" if withdrawn else "",
            )
            if part
        ]
        return f"cycle {self.control_cycle}: {' '.join(parts)}"


@dataclass
class DecisionTranscript:
    """
    Accumulates the theory's conclusions as a sequence of changes.

    Recording is driven by whoever ticks the statechart rather than by the theory node,
    so a run can be transcribed without the reasoning path knowing it is being observed.
    """

    changes: List[DecisionChange] = field(default_factory=list)
    """
    The recorded turnovers, oldest first.
    """

    _previous_types: Optional[frozenset] = field(default=None, init=False, repr=False)
    """
    Decision types concluded at the last recorded change.
    """

    def record(self, decisions: Optional[DecisionSet], control_cycle: int) -> None:
        """
        Records a change if the concluded decision types differ from the last recorded
        set.

        :param decisions: The theory's latest decision set, or ``None`` before the first
            inference.
        :param control_cycle: The control cycle this decision set is visible on.
        """
        if decisions is None:
            return
        current_types = frozenset(type(decision) for decision in decisions)
        if current_types == self._previous_types:
            return
        previous_types = self._previous_types or frozenset()
        self.changes.append(
            DecisionChange(
                control_cycle=control_cycle,
                entered=tuple(current_types - previous_types),
                withdrawn=tuple(previous_types - current_types),
                decisions=tuple(decisions),
            )
        )
        self._previous_types = current_types

    def cycle_of_first(self, decision_type: Type[ControlDecision]) -> Optional[int]:
        """
        The control cycle a decision type was first concluded on.

        :param decision_type: The decision type to look for.
        :return: The cycle, or ``None`` if it was never concluded.
        """
        for change in self.changes:
            if decision_type in change.entered:
                return change.control_cycle
        return None

    def __str__(self) -> str:
        return "\n".join(str(change) for change in self.changes)
