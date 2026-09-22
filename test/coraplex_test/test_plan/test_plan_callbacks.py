"""
Tests for plan node callbacks: registered ``PlanCallback``s are notified when the
performed root starts and ends, and when each executed motion's giskard task transitions
through its life cycle during simulated execution.
"""

from dataclasses import dataclass, field

from typing_extensions import List, Tuple

from coraplex.execution_environment import simulated_robot
from coraplex.plans.attachment_nodes import ReAttachNode
from coraplex.plans.executables import MotionPlanHistory
from coraplex.plans.factories import sequential
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from giskardpy.motion_statechart.graph_node import LifeCycleValues
from semantic_digital_twin.datastructures.definitions import TorsoState
from .test_execution_observers import RecordedMotion

# %% recording callback


@dataclass
class StartEndRecorder(PlanCallback):
    """
    Records every start and end notification it receives, in order.
    """

    events: List[Tuple[str, PlanNode]] = field(default_factory=list)
    """
    The recorded (event, node) pairs in notification order.
    """

    def on_start(self, node: PlanNode):
        self.events.append(("start", node))

    def on_end(self, node: PlanNode):
        self.events.append(("end", node))


# %% perform notifies callbacks


def test_perform_notifies_root_and_motion_nodes(immutable_model_world):
    """
    Performing a plan notifies registered callbacks of the performed root's start and
    end, and of every executed motion node's start and end, each exactly once and in
    execution order.
    """
    world, robot_view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    recorder = StartEndRecorder()
    plan.node_callbacks.append(recorder)

    with simulated_robot:
        plan.perform()

    assert recorder.events[0] == ("start", plan.root)
    assert recorder.events[-1] == ("end", plan.root)

    motion_nodes = [node for node in plan.all_nodes if isinstance(node, MotionNode)]
    assert len(motion_nodes) > 0
    started_motions = [
        node
        for event, node in recorder.events
        if event == "start" and isinstance(node, MotionNode)
    ]
    ended_motions = [
        node
        for event, node in recorder.events
        if event == "end" and isinstance(node, MotionNode)
    ]
    assert started_motions == motion_nodes
    assert ended_motions == motion_nodes
    for motion_node in motion_nodes:
        start_index = recorder.events.index(("start", motion_node))
        end_index = recorder.events.index(("end", motion_node))
        assert start_index < end_index


# %% simulated execution notifies motion ticks


@dataclass
class MotionTickRecorder(PlanCallback):
    """
    Records every statechart the motion executor reports a tick for.
    """

    statecharts: List[object] = field(default_factory=list)
    """
    The reported statecharts in notification order.
    """

    def on_motion_tick(self, statechart):
        self.statecharts.append(statechart)


def test_simulated_execution_notifies_every_motion_tick(immutable_model_world):
    """
    While the simulated executor ticks a plan's motions, every tick is reported with the
    statechart being executed.
    """
    world, robot_view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan
    recorder = MotionTickRecorder()
    plan.node_callbacks.append(recorder)

    with simulated_robot:
        plan.perform()

    assert len(recorder.statecharts) > 0
    assert all(
        statechart is recorder.statecharts[0] for statechart in recorder.statecharts
    )


# %% statuses follow the tasks realizing the motions


@dataclass
class NotifiedPlan:
    """
    The plan a tracked node notifies, recording the status each notification carried.
    """

    events: List[Tuple[str, LifeCycleValues]] = field(default_factory=list)
    """
    The recorded (event, status) pairs in notification order.
    """

    def notify_motion_tick(self, statechart) -> None:
        """Accept native snapshots while this test records boundaries only."""

    def notify_node_started(self, node) -> None:
        self.events.append(("start", node.status))

    def notify_node_ended(self, node) -> None:
        self.events.append(("end", node.status))


@dataclass(eq=False)
class TrackedMotionNode:
    """
    A motion node as the tracker touches it: a status to set and a plan to notify.

    Hashable by identity, since the tracker keys its mappings on the node itself.
    """

    plan: NotifiedPlan = field(default_factory=NotifiedPlan)
    """
    The plan told about this node's transitions.
    """

    status: LifeCycleValues = LifeCycleValues.NOT_STARTED
    """
    The status the tracker keeps on the node.
    """


def test_a_motion_node_runs_and_succeeds_with_its_task():
    """
    Nothing else sets a motion node's status -- it is realized by a statechart task
    rather than performed -- so the tracker has to, or a finished plan reads as untouched.
    """
    motion = RecordedMotion()
    node = TrackedMotionNode()
    tracker = MotionPlanHistory(
        statechart=motion.chart, motion_mappings={node: motion.task}
    )

    motion.record(LifeCycleValues.RUNNING)
    running = node.status

    motion.record(LifeCycleValues.SUCCEEDED)

    assert running is LifeCycleValues.RUNNING
    assert node.status is LifeCycleValues.SUCCEEDED
    assert node.plan.events == [
        ("start", LifeCycleValues.RUNNING),
        ("end", LifeCycleValues.SUCCEEDED),
    ]


def test_a_motion_node_whose_task_failed_ends_failed():
    """
    A failed motion must not read as a successful one, however the plan carried on.
    """
    motion = RecordedMotion()
    node = TrackedMotionNode()
    tracker = MotionPlanHistory(
        statechart=motion.chart, motion_mappings={node: motion.task}
    )

    motion.record(LifeCycleValues.FAILED)

    assert node.status is LifeCycleValues.FAILED
    assert node.plan.events == [
        ("start", LifeCycleValues.RUNNING),
        ("end", LifeCycleValues.FAILED),
    ]


def test_every_executed_motion_of_a_performed_plan_reports_it_succeeded(
    immutable_model_world,
):
    """
    What the viewer and a recording read off the plan tree afterwards: the motions that
    ran say so, while the conditions that never execute stay as created.
    """
    world, robot_view, context = immutable_model_world
    plan = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context).plan

    with simulated_robot:
        plan.perform()

    motion_nodes = [node for node in plan.all_nodes if isinstance(node, MotionNode)]
    assert motion_nodes
    assert {node.status for node in motion_nodes} == {LifeCycleValues.SUCCEEDED}


def test_a_performed_model_change_reports_it_succeeded(mutable_model_world):
    """
    Attaching and detaching is not a motion and is never performed on its own either, so
    without saying so a finished pick reads as still running: its motions are done while
    the attach that ended it is not.
    """
    world, robot_view, context = mutable_model_world
    attach = ReAttachNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )
    plan = sequential([attach], context=context).plan

    with simulated_robot:
        plan.perform()

    assert attach.status is LifeCycleValues.SUCCEEDED
