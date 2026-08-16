"""
The knowledge-servoing demonstration, run end to end as it is shown.

These assertions are what the demonstration's figures claim, checked against a real run,
so the scenario shown in the thesis and the scenario verified here cannot diverge.
"""

from __future__ import annotations

import pytest

from experiments.knowledge_servoing.demonstration import (
    FILL_LEVEL_TOLERANCE,
    REQUESTED_FILL_LEVEL,
    build_primitive_arm_demonstration,
    build_transfer_demonstration,
)
from experiments.knowledge_servoing.scenario import build_transfer_scenario
from semantic_digital_twin.reasoning.contextual_safety import EnforceCaution
from semantic_digital_twin.reasoning.substance_transfer import (
    AbandonTransfer,
    AlignSourceOverReceiver,
    ConcludeTransfer,
    PourIntoReceiver,
)
from semantic_digital_twin.utils import tracy_installed


@pytest.fixture(scope="module")
def completed_demonstration():
    """
    Runs the demonstration once and shares the finished run across the assertions.
    """
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    demonstration = build_transfer_demonstration(build_transfer_scenario())
    demonstration.run()
    return demonstration


class TestTransferDemonstration:
    """
    What the demonstration's figures assert about the run behind them.
    """

    def test_the_transfer_reaches_the_requested_fill_level(
        self, completed_demonstration
    ):
        assert (
            completed_demonstration.scenario.receiving_cup.fill_level
            == pytest.approx(REQUESTED_FILL_LEVEL, abs=FILL_LEVEL_TOLERANCE)
        )

    def test_the_transfer_theory_turns_over_align_then_pour_then_conclude(
        self, completed_demonstration
    ):
        transcript = completed_demonstration.transfer_transcript
        assert transcript.cycle_of_first(
            AlignSourceOverReceiver
        ) < transcript.cycle_of_first(PourIntoReceiver)
        assert transcript.cycle_of_first(PourIntoReceiver) < transcript.cycle_of_first(
            ConcludeTransfer
        )
        assert transcript.cycle_of_first(AbandonTransfer) is None

    def test_the_safety_theory_restricts_the_motion_independently(
        self, completed_demonstration
    ):
        """
        The safety theory reaches its own conclusion about the same run without the
        transfer theory declaring, or knowing about, the decision type it concludes.
        """
        assert (
            completed_demonstration.safety_transcript.cycle_of_first(EnforceCaution)
            is not None
        )

    def test_the_gantt_chart_renders(self, completed_demonstration, tmp_path):
        chart_path = tmp_path / "gantt.pdf"
        completed_demonstration.plot_gantt_chart(str(chart_path))
        assert chart_path.stat().st_size > 0


@pytest.fixture(scope="module")
def completed_primitive_arm():
    """
    Runs the replication arm once, driven through the fixed-gain twist bridge.
    """
    if not tracy_installed():
        pytest.skip("Tracy not installed")
    demonstration = build_primitive_arm_demonstration(build_transfer_scenario())
    demonstration.run(maximum_control_cycles=4000)
    return demonstration


class TestPrimitiveArm:
    """
    What the replication arm does with the same facts and a fixed-gain bridge.
    """

    def test_the_bridge_drives_a_real_transfer(self, completed_primitive_arm):
        """
        The commanded twist reaches the QP and moves substance, which is what makes the
        arm a comparison rather than a straw man.
        """
        assert completed_primitive_arm.scenario.receiving_cup.fill_level > 0.1
        assert completed_primitive_arm.scenario.source_cup.fill_level < 1.0

    def test_the_fixed_gain_bridge_churns_far_more_than_the_regime_arm(
        self, completed_primitive_arm, completed_demonstration
    ):
        """
        Bang-bang primitives re-decide constantly around their thresholds where regime
        decisions turn over a handful of times, which is the qualitative difference the
        comparison measures.
        """
        primitive_changes = len(completed_primitive_arm.transfer_transcript.changes)
        regime_changes = len(completed_demonstration.transfer_transcript.changes)
        assert primitive_changes > 10 * regime_changes
