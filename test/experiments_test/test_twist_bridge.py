"""
The fixed-gain bridge from Boolean primitives to a commanded twist.

The arithmetic is equation (1) of the published system, so these assertions pin the
replication rather than a design choice: opposed primitives cancel, a single primitive
contributes its gain, and tilting back is deliberately far faster than tilting over.
"""

from __future__ import annotations

import pytest

from experiments.knowledge_servoing.twist_bridge import (
    LINEAR_GAIN,
    TILT_BACK_GAIN,
    TILT_GAIN,
    TwistBridgeNode,
)
from semantic_digital_twin.reasoning.knowledge_servoing.interfaces import DecisionSet
from semantic_digital_twin.reasoning.substance_transfer.motion_primitives import (
    DecreaseTilt,
    IncreaseTilt,
    MoveBack,
    MoveForward,
    MoveLeft,
    MoveRight,
)


@pytest.fixture
def bridge():
    """
    A bridge with no controller attached; only its arithmetic is under test.
    """
    return TwistBridgeNode(
        name="bridge", decision_slot=None, translation=None, tilt=None
    )


class TestFixedGainTwist:
    """
    What the bridge commands for a given set of concluded primitives.
    """

    def test_no_primitives_command_no_motion(self, bridge):
        assert bridge.commanded_velocities(DecisionSet(())) == (0.0, 0.0, 0.0)

    def test_one_primitive_contributes_its_gain(self, bridge):
        forward, left, tilt_rate = bridge.commanded_velocities(
            DecisionSet((MoveForward(),))
        )
        assert forward == pytest.approx(LINEAR_GAIN)
        assert left == 0.0
        assert tilt_rate == 0.0

    def test_opposed_primitives_cancel(self, bridge):
        forward, _left, _tilt = bridge.commanded_velocities(
            DecisionSet((MoveForward(), MoveBack()))
        )
        assert forward == 0.0

    def test_the_opposite_primitive_reverses_the_sign(self, bridge):
        _forward, left, _tilt = bridge.commanded_velocities(DecisionSet((MoveRight(),)))
        assert left == pytest.approx(-LINEAR_GAIN)

    def test_axes_are_independent(self, bridge):
        forward, left, _tilt = bridge.commanded_velocities(
            DecisionSet((MoveForward(), MoveLeft()))
        )
        assert (forward, left) == pytest.approx((LINEAR_GAIN, LINEAR_GAIN))

    def test_increasing_tilt_uses_the_slow_gain(self, bridge):
        _forward, _left, tilt_rate = bridge.commanded_velocities(
            DecisionSet((IncreaseTilt(),))
        )
        assert tilt_rate == pytest.approx(TILT_GAIN)

    def test_decreasing_tilt_uses_the_fast_gain_and_the_opposite_sign(self, bridge):
        _forward, _left, tilt_rate = bridge.commanded_velocities(
            DecisionSet((DecreaseTilt(),))
        )
        assert tilt_rate == pytest.approx(-TILT_BACK_GAIN)

    def test_the_gains_are_asymmetric_by_design(self):
        """
        The published system tilts back far faster than it tilts over, because a fixed-
        gain bridge has no other way to stop a pour it has decided to end.
        """
        assert TILT_BACK_GAIN > TILT_GAIN
