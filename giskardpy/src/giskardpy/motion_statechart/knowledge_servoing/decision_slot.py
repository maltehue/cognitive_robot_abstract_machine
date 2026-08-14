"""The hand-off point between the theory node and the monitors that read its decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import DecisionSet


@dataclass
class DecisionSlot:
    """Holds the most recent decision set a theory node published.

    The theory node writes it each tick and the :class:`ConcludedMonitor`s read it; while inference
    runs synchronously on the control tick there is a single writer before any reader, so no lock is
    needed.
    """

    _latest: Optional[DecisionSet] = field(default=None, init=False)
    """The decision set from the last inference, or ``None`` before the first inference."""

    def publish(self, decisions: DecisionSet) -> None:
        """Stores the decision set from the current inference."""
        self._latest = decisions

    @property
    def latest(self) -> Optional[DecisionSet]:
        """The last published decision set, or ``None`` before the first inference."""
        return self._latest
