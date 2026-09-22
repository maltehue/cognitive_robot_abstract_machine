"""
Live recordings retain the executed hierarchy for offline presentation.
"""

from __future__ import annotations

from pathlib import Path

from cramera.knowledge.enums import PlanNodeGroup
from cramera.live.bridge import PlanNodeEntry, PlanSnapshot
from cramera.live.recording_bundle import write_recording_bundle

from .test_live_bundle import attached_bridge
from .test_recording_bundle import frame_with_milk


# %% recorded hierarchy
class TestRecordedPlanHierarchy:
    """
    The recorded tree preserves the live snapshot's nesting and metadata.
    """

    @staticmethod
    def snapshot() -> PlanSnapshot:
        """
        Create a root and a child whose action metadata must survive saving.
        """
        return PlanSnapshot(
            nodes=[
                PlanNodeEntry(
                    id="root",
                    parent=None,
                    kind="SequentialNode",
                    label="sequence",
                    group=PlanNodeGroup.of_plan_node_kind("SequentialNode"),
                    status="SUCCEEDED",
                    derived=False,
                ),
                PlanNodeEntry(
                    id="move",
                    parent="root",
                    kind="MotionNode",
                    label="NavigateAction",
                    group=PlanNodeGroup.of_plan_node_kind("MotionNode"),
                    status="SUCCEEDED",
                    derived=False,
                    arm="LEFT",
                    target="countertop",
                ),
            ]
        )

    def test_finalized_bundle_keeps_the_plan(self, tmp_path: Path) -> None:
        """
        A saved live run contains the hierarchy used by the recorded Plan tab.
        """
        bridge = attached_bridge()
        bridge.plan_state = self.snapshot()
        scene = write_recording_bundle(
            bridge, [frame_with_milk()], 20.0, tmp_path / "episode", "episode"
        )
        tree = scene["planTrees"][0]
        child = tree["children"][0]
        expected = bridge.plan_state.nodes[1]
        assert (child["label"], child["arm"], child["target"]) == (
            expected.label,
            expected.arm,
            expected.target,
        )
        assert tree["label"] == bridge.plan_state.nodes[0].label
        assert child["children"] == []

    def test_a_recording_without_a_plan_has_no_tree(self) -> None:
        """
        A teleoperation-only recording remains a valid empty hierarchy.
        """
        assert PlanSnapshot().recorded_trees() == []
