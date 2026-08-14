"""A monitor that gates statechart nodes on whether a theory concluded a given decision type."""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, Type

from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
)

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot


@dataclass(eq=False, repr=False)
class ConcludedMonitor(MotionStatechartNode):
    """Observes whether the latest decision set contains a decision of a given type (channel 1).

    The observation is TRUE while the latest decision set contains such a decision, FALSE once
    inference has run without one, and UNKNOWN before the first inference. Gating a task's start
    condition on this monitor activates that task exactly while the theory concludes the decision.
    """

    decision_type: Type[ControlDecision] = field(kw_only=True)
    """The decision type whose presence turns the observation TRUE."""

    decision_slot: DecisionSlot = field(kw_only=True)
    """The slot the theory node publishes its decisions to."""

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        latest = self.decision_slot.latest
        if latest is None:
            return ObservationStateValues.UNKNOWN
        if latest.contains_type(self.decision_type):
            return ObservationStateValues.TRUE
        return ObservationStateValues.FALSE
