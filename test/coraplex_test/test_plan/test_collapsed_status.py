"""
Collapsed motion charts retain the lifecycle of their native plan hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.execution_environment import simulated_robot
from coraplex.plans.condition_nodes import ConditionNode
from coraplex.plans.executables import GiskardExecutable, MotionLifeCycleTracker
from coraplex.plans.factories import (
    code,
    execute_single,
    sequential,
    try_all,
    try_in_order,
)
from coraplex.plans.failures import EmptyUnderspecified, PlanFailure
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import MotionNode, PlanNode, UnderspecifiedNode
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction
from giskardpy.motion_statechart.data_types import LifeCycleValues
from krrood.entity_query_language.factories import a
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

RobotContext = tuple[World, AbstractRobot, Context]


# %% lifecycle evidence
@dataclass
class NodeEvents(PlanCallback):
    """
    Record the status delivered with each lifecycle event.
    """

    events: list[tuple[str, PlanNode, TaskStatus]] = field(default_factory=list)
    """
    Events in the order observers received them.
    """

    def on_start(self, node: PlanNode) -> None:
        """:param node: Node whose execution started."""
        self.events.append(("start", node, node.status))

    def on_end(self, node: PlanNode) -> None:
        """:param node: Node whose execution ended."""
        self.events.append(("end", node, node.status))


@dataclass
class TaskState:
    """
    The lifecycle state observed by the existing motion tracker.
    """

    life_cycle_state: LifeCycleValues
    """
    Current native controller state.
    """


def test_collapsed_and_grounded_actions_report_success(
    immutable_model_world: RobotContext,
) -> None:
    """
    Native actions finish even when unevaluated condition nodes remain untouched.
    """
    _, _, context = immutable_model_world
    context.evaluate_conditions = True
    root = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            a(MoveTorsoAction)(torso_state=TorsoState.LOW),
        ],
        context=context,
    )
    recorder = NodeEvents()
    root.plan.node_callbacks.append(recorder)

    with simulated_robot:
        root.plan.perform()

    actions = root.plan.get_nodes_by_designator_type(MoveTorsoAction)
    assert len(actions) == 2
    assert {node.status for node in actions} == {TaskStatus.SUCCEEDED}
    grounded = [
        node for node in root.descendants if isinstance(node, UnderspecifiedNode)
    ]
    assert len(grounded) == 1
    assert grounded[0].status is TaskStatus.SUCCEEDED
    for node in [root, *actions, *grounded]:
        events = [
            (event, status)
            for event, source, status in recorder.events
            if source is node
        ]
        assert events == [("start", TaskStatus.RUNNING), ("end", TaskStatus.SUCCEEDED)]
        assert node.end_time is not None
    conditions = [node for node in root.descendants if isinstance(node, ConditionNode)]
    assert conditions
    assert {node.status for node in conditions} == {TaskStatus.CREATED}


def test_unstarted_siblings_are_not_reported_as_success(
    immutable_model_world: RobotContext,
) -> None:
    """
    Completing one collapsed action leaves the remaining action untouched.
    """
    _, _, context = immutable_model_world
    root = sequential(
        [MoveTorsoAction(TorsoState.HIGH), MoveTorsoAction(TorsoState.LOW)],
        context=context,
    )
    root.notify()
    first, second = root.children
    motion = next(node for node in first.descendants if isinstance(node, MotionNode))
    task = TaskState(LifeCycleValues.RUNNING)
    tracker = MotionLifeCycleTracker({motion: task})
    tracker.emit_transitions()
    assert first.status is TaskStatus.RUNNING
    task.life_cycle_state = LifeCycleValues.DONE
    tracker.emit_transitions()
    assert first.status is TaskStatus.SUCCEEDED
    assert second.status is TaskStatus.CREATED
    assert root.status is TaskStatus.RUNNING


@pytest.mark.parametrize("terminal", [TaskStatus.FAILED, TaskStatus.INTERRUPTED])
def test_terminal_parent_status_is_preserved(
    immutable_model_world: RobotContext, terminal: TaskStatus
) -> None:
    """
    A late controller completion cannot undo failure or cancellation.
    """
    _, _, context = immutable_model_world
    root = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context)
    root.notify()
    action = root.children[0]
    action.status = terminal
    motion = next(node for node in action.descendants if isinstance(node, MotionNode))
    MotionLifeCycleTracker({motion: TaskState(LifeCycleValues.DONE)}).emit_transitions()
    assert action.status is terminal


def test_failed_motion_marks_its_collapsed_action_failed(
    immutable_model_world: RobotContext,
) -> None:
    """
    A controller failure remains visible in the enclosing native action.
    """
    _, _, context = immutable_model_world
    root = sequential([MoveTorsoAction(TorsoState.HIGH)], context=context)
    root.notify()
    action = root.children[0]
    motion = next(node for node in action.descendants if isinstance(node, MotionNode))
    MotionLifeCycleTracker(
        {motion: TaskState(LifeCycleValues.FAILED)}
    ).emit_transitions()
    assert action.status is TaskStatus.FAILED
    assert root.status is TaskStatus.FAILED


def test_grounding_retry_preserves_failed_candidate(
    immutable_model_world: RobotContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Retry success belongs to the accepted candidate, not its failed predecessor.
    """
    _, _, context = immutable_model_world
    root = sequential(
        [a(MoveTorsoAction)(torso_state=TorsoState.HIGH)], context=context
    )
    node = root.children[0]
    node._action_iterator = (
        MoveTorsoAction(state) for state in [TorsoState.HIGH, TorsoState.LOW]
    )
    execute = GiskardExecutable.execute
    attempts = 0

    def fail_first(executable: GiskardExecutable) -> None:
        """:param executable: Real motion executable, failing its first attempt."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PlanFailure()
        execute(executable)

    monkeypatch.setattr(GiskardExecutable, "execute", fail_first)
    with simulated_robot:
        root.plan.perform()

    assert attempts == 2
    assert [candidate.status for candidate in node.children] == [
        TaskStatus.FAILED,
        TaskStatus.SUCCEEDED,
    ]
    assert node.status is TaskStatus.SUCCEEDED
    assert root.status is TaskStatus.SUCCEEDED
    assert node.execution_children == [node.current_candidate]


@pytest.mark.parametrize("factory", [try_in_order, try_all])
@pytest.mark.parametrize("nested", [False, True])
def test_successful_alternative_preserves_failed_sibling(
    immutable_model_world: RobotContext, factory: Callable[..., PlanNode], nested: bool
) -> None:
    """
    One failed alternative cannot turn a successful retry combinator into failure.
    """
    _, _, context = immutable_model_world

    def fail() -> None:
        """
        Fail the first alternative before the next candidate succeeds.
        """
        raise PlanFailure()

    alternatives = factory([code(fail), code(lambda: None)], context=context)
    root = sequential([alternatives], context=context) if nested else alternatives
    with simulated_robot:
        root.plan.perform()

    assert alternatives.status is TaskStatus.SUCCEEDED
    assert [child.status for child in alternatives.children] == [
        TaskStatus.FAILED,
        TaskStatus.SUCCEEDED,
    ]
    assert root.status is TaskStatus.SUCCEEDED


def test_grounded_root_emits_one_lifecycle(immutable_model_world: RobotContext) -> None:
    """
    Perform and deferred grounding share one execution scope for the same node.
    """
    _, _, context = immutable_model_world
    root = execute_single(
        a(MoveTorsoAction)(torso_state=TorsoState.HIGH), context=context
    )
    recorder = NodeEvents()
    root.plan.node_callbacks.append(recorder)
    with simulated_robot:
        root.plan.perform()
    assert [event for event, node, _ in recorder.events if node is root] == [
        "start",
        "end",
    ]
    assert root.status is TaskStatus.SUCCEEDED


def test_exhausted_grounding_reports_failure(
    immutable_model_world: RobotContext,
) -> None:
    """
    An exhausted query terminates its own node and the enclosing sequence.
    """
    _, _, context = immutable_model_world
    root = sequential(
        [a(MoveTorsoAction)(torso_state=TorsoState.HIGH)], context=context
    )
    node = root.children[0]
    node._action_iterator = (MoveTorsoAction(state) for state in [])
    with simulated_robot, pytest.raises(EmptyUnderspecified):
        root.plan.perform()
    assert node.status is root.status is TaskStatus.FAILED
    assert isinstance(node.reason, EmptyUnderspecified)
    assert node.execution_children == []


def test_perform_preserves_an_interruption(immutable_model_world: RobotContext) -> None:
    """
    Returning from a stopped executable does not turn cancellation into success.
    """
    _, _, context = immutable_model_world
    root = code(lambda: root.interrupt(), context=context)
    with simulated_robot:
        root.plan.perform()
    assert root.status is TaskStatus.INTERRUPTED
