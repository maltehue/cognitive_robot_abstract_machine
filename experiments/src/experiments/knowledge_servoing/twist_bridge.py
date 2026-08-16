"""The published system's bridge from Boolean primitives to a commanded twist.

This is equation (1) of Huerkamp et al. 2025: each opposed pair of primitives contributes the
difference of its members, scaled by a gain held constant for the whole run. It is the step the
regime vocabulary replaces, and reproducing it faithfully is what lets the comparison attribute a
difference in precision to the bridge rather than to anything else.

It lives in ``experiments`` rather than in the controller because it knows a specific theory's
primitive vocabulary, which the framework deliberately does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, Type

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import MotionStatechartNode, NodeArtifacts
from giskardpy.motion_statechart.knowledge_servoing.decision_slot import DecisionSlot
from giskardpy.motion_statechart.tasks.commanded_velocity import (
    CommandedTiltVelocity,
    CommandedTranslationVelocity,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import (
    ControlDecision,
    DecisionSet,
)
from semantic_digital_twin.reasoning.substance_transfer.motion_primitives import (
    DecreaseTilt,
    IncreaseTilt,
    MoveBack,
    MoveForward,
    MoveLeft,
    MoveRight,
)

LINEAR_GAIN = 0.02
"""The paper's α: linear speed contributed by one active translation primitive, in m/s."""

TILT_GAIN = 0.03
"""The paper's β: tilt rate contributed by an active increase-tilt primitive, in rad/s."""

TILT_BACK_GAIN = 1.0
"""The paper's γ: tilt rate contributed by an active decrease-tilt primitive, in rad/s.

Far larger than :data:`TILT_GAIN`, because the published system tilts back quickly to stop a pour it
has decided to end but cannot otherwise slow.
"""


@dataclass(eq=False, repr=False)
class TwistBridgeNode(MotionStatechartNode):
    """Turns the primitives a theory concluded into commanded velocities, at constant gain."""

    decision_slot: DecisionSlot = field(kw_only=True)
    """The slot the primitive-concluding theory publishes to."""

    translation: CommandedTranslationVelocity = field(kw_only=True)
    """The task whose commanded translational velocity this bridge writes."""

    tilt: CommandedTiltVelocity = field(kw_only=True)
    """The task whose commanded tilt rate this bridge writes."""

    linear_gain: float = field(default=LINEAR_GAIN, kw_only=True)
    """The paper's α."""

    tilt_gain: float = field(default=TILT_GAIN, kw_only=True)
    """The paper's β."""

    tilt_back_gain: float = field(default=TILT_BACK_GAIN, kw_only=True)
    """The paper's γ."""

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        return NodeArtifacts()

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        decisions = self.decision_slot.latest
        if decisions is None:
            return ObservationStateValues.UNKNOWN
        float_variable_data = context.float_variable_data
        forward, left, tilt_rate = self.commanded_velocities(decisions)
        float_variable_data.set_value(self.translation.commanded_velocity[0], forward)
        float_variable_data.set_value(self.translation.commanded_velocity[1], left)
        float_variable_data.set_value(self.translation.commanded_velocity[2], 0.0)
        float_variable_data.set_value(self.tilt.commanded_tilt_rate, tilt_rate)
        return ObservationStateValues.TRUE

    def commanded_velocities(
        self, decisions: DecisionSet
    ) -> tuple[float, float, float]:
        """Evaluates the fixed-gain twist for a set of concluded primitives.

        :param decisions: What the theory concluded this cycle.
        :return: Commanded velocity along world x, along world y, and tilt rate.
        """
        forward = self.linear_gain * self._difference(decisions, MoveForward, MoveBack)
        left = self.linear_gain * self._difference(decisions, MoveLeft, MoveRight)
        tilt_rate = self.tilt_gain * self._active(
            decisions, IncreaseTilt
        ) - self.tilt_back_gain * self._active(decisions, DecreaseTilt)
        return forward, left, tilt_rate

    @staticmethod
    def _active(decisions: DecisionSet, primitive: Type[ControlDecision]) -> float:
        """Whether a primitive is among the concluded decisions, as one or zero."""
        return 1.0 if decisions.contains_type(primitive) else 0.0

    def _difference(
        self,
        decisions: DecisionSet,
        positive: Type[ControlDecision],
        negative: Type[ControlDecision],
    ) -> float:
        """The signed contribution of one opposed primitive pair."""
        return self._active(decisions, positive) - self._active(decisions, negative)
