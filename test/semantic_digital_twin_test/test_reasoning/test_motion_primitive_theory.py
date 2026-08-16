"""
The replication arm's rule set, which concludes directions instead of constraint
regimes.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.reasoning.substance_transfer.motion_primitives import (
    DecreaseTilt,
    IncreaseTilt,
    MoveBack,
    MoveForward,
    MoveLeft,
    MoveRight,
)
from semantic_digital_twin.reasoning.substance_transfer.primitive_theory import (
    build_motion_primitive_theory,
)

from .test_substance_transfer_theory import situation


@pytest.fixture
def theory():
    return build_motion_primitive_theory()


class TestAlignmentPrimitives:
    """
    Which direction the theory picks while the pour is not yet aimed.
    """

    def test_a_receiver_ahead_of_the_source_moves_forward(self, theory):
        decisions = theory.infer(
            [situation(opening_within=False, receiver_offset_forward=0.05)]
        )
        assert decisions.contains_type(MoveForward)
        assert not decisions.contains_type(MoveBack)

    def test_a_receiver_behind_the_source_moves_back(self, theory):
        decisions = theory.infer(
            [situation(opening_within=False, receiver_offset_forward=-0.05)]
        )
        assert decisions.contains_type(MoveBack)

    def test_offsets_on_both_axes_conclude_both_directions(self, theory):
        decisions = theory.infer(
            [
                situation(
                    opening_within=False,
                    receiver_offset_forward=0.05,
                    receiver_offset_left=0.05,
                )
            ]
        )
        assert decisions.contains_type(MoveForward)
        assert decisions.contains_type(MoveLeft)

    def test_an_aligned_source_concludes_no_direction(self, theory):
        decisions = theory.infer(
            [situation(opening_within=False, receiver_offset_left=0.0)]
        )
        assert not decisions.contains_type(MoveLeft)
        assert not decisions.contains_type(MoveRight)


class TestTiltPrimitives:
    """
    When the theory tilts the source over and when it tilts it back.
    """

    def test_an_aimed_pour_short_of_the_goal_increases_tilt(self, theory):
        decisions = theory.infer([situation()])
        assert decisions.contains_type(IncreaseTilt)
        assert not decisions.contains_type(DecreaseTilt)

    def test_reaching_the_goal_decreases_tilt(self, theory):
        decisions = theory.infer(
            [situation(goal_reached=True, almost_goal_reached=True)]
        )
        assert decisions.contains_type(DecreaseTilt)


class TestNearGoalSuperiority:
    """
    The paper's superiority pair: near-goal decrease-tilt overrides increase-tilt.
    """

    def test_being_near_the_goal_defeats_increasing_tilt(self, theory):
        decisions = theory.infer([situation(almost_goal_reached=True)])
        assert not decisions.contains_type(IncreaseTilt)

    def test_being_near_the_goal_decreases_tilt_instead(self, theory):
        decisions = theory.infer([situation(almost_goal_reached=True)])
        assert decisions.contains_type(DecreaseTilt)
