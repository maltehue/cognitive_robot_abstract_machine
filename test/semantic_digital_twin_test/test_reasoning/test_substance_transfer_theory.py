"""
The substance-transfer theory's rules, exercised on hand-built situations.

Each test fixes the qualitative facts directly rather than going through a world, so a
rule's behaviour is pinned independently of whether grounding computes the fact
correctly; grounding has its own tests.
"""

from __future__ import annotations

import pytest

from semantic_digital_twin.reasoning.substance_transfer import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
    RetargetFillLevel,
    TransferDefeat,
    TransferSituation,
    build_substance_transfer_theory,
)

REQUESTED_FILL_LEVEL = 0.7
"""
Fill level the transfer is asked to reach in these tests.
"""


def situation(**overrides) -> TransferSituation:
    """
    Builds a situation whose facts default to an aligned, tilted, actively pouring
    transfer.
    """
    facts = {
        "source": None,
        "receiver": None,
        "requested_fill_level": REQUESTED_FILL_LEVEL,
        "receiver_fill_level": 0.2,
        "near": True,
        "source_above_receiver": True,
        "opening_within": True,
        "is_tilted": True,
        "pours_to": True,
        "goal_reached": False,
        "almost_goal_reached": False,
        "receiver_offset_forward": 0.0,
        "receiver_offset_left": 0.0,
        "receiver_overflowing": False,
    }
    facts.update(overrides)
    return TransferSituation(**facts)


@pytest.fixture
def theory():
    return build_substance_transfer_theory()


class TestTransferRegimeRules:
    """
    Which constraint regime the theory concludes for a given qualitative state.
    """

    def test_an_aimed_pour_concludes_align_and_pour(self, theory):
        decisions = theory.infer([situation()])
        assert decisions.contains_type(AlignSourceOverReceiver)
        assert decisions.contains_type(PourIntoReceiver)

    def test_being_near_but_not_aimed_concludes_align_without_pour(self, theory):
        decisions = theory.infer([situation(opening_within=False)])
        assert decisions.contains_type(AlignSourceOverReceiver)
        assert not decisions.contains_type(PourIntoReceiver)

    def test_a_source_below_the_rim_does_not_pour(self, theory):
        decisions = theory.infer([situation(source_above_receiver=False)])
        assert not decisions.contains_type(PourIntoReceiver)

    def test_reaching_the_goal_concludes_the_transfer_and_stops_pouring(self, theory):
        decisions = theory.infer(
            [situation(receiver_fill_level=REQUESTED_FILL_LEVEL, goal_reached=True)]
        )
        assert decisions.contains_type(ConcludeTransfer)
        assert not decisions.contains_type(PourIntoReceiver)
        assert not decisions.contains_type(AlignSourceOverReceiver)


class TestOverflowDefeatsPouring:
    """
    Whether the overflow stop rule withdraws the pour regime it refines.
    """

    def test_an_overflowing_receiver_defeats_pouring(self, theory):
        decisions = theory.infer(
            [situation(receiver_fill_level=1.0, receiver_overflowing=True)]
        )
        assert not decisions.contains_type(PourIntoReceiver)

    def test_an_overflowing_receiver_abandons_the_transfer(self, theory):
        decisions = theory.infer(
            [situation(receiver_fill_level=1.0, receiver_overflowing=True)]
        )
        [abandon] = decisions.of_type(AbandonTransfer)
        assert abandon.defeat is TransferDefeat.RECEIVER_WOULD_OVERFLOW


class TestFillGoalParameterization:
    """
    Whether the parameter family supplies the numeric goal by chaining on the regime
    family.
    """

    def test_pouring_retargets_the_fill_goal_to_the_requested_level(self, theory):
        [retarget] = theory.infer([situation()]).of_type(RetargetFillLevel)
        assert retarget.goal_fill_level == REQUESTED_FILL_LEVEL

    def test_no_goal_is_supplied_when_pouring_is_not_concluded(self, theory):
        decisions = theory.infer([situation(opening_within=False)])
        assert not decisions.contains_type(RetargetFillLevel)


class TestSpillRisk:
    """
    The predictive spill fact, which the analytic world cannot observe directly.
    """

    def test_tilting_while_unaimed_is_a_spill_risk(self):
        assert situation(opening_within=False).spill_risk

    def test_tilting_while_aimed_is_not_a_spill_risk(self):
        assert not situation().spill_risk
