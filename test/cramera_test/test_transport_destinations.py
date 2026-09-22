"""
Retry placement alternatives while retaining a transport's acquired object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms, TaskStatus
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.locations.factories import _get_object_in_hand
from coraplex.plans.attachment_nodes import AttachNode, DetachNode
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.plans.plan_node import ActionNode, PlanNode
from coraplex.robot_plans.actions.composite import transporting
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.composite.transporting import (
    MoveAndPlaceAction,
    TransportAction,
)
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body


# %% execution observations
@dataclass
class DestinationAttempts(PlanCallback):
    """
    Observe native grounding and attachment without running arm controllers.
    """

    context: Context
    """Existing robot world used by the native plan and model-change nodes."""
    body: Body
    """
    Object acquired once and retained through failed destination grounding.
    """

    targets: list[Pose]
    """First unreachable and second reachable destination."""
    checked: list[Pose] = field(default_factory=list)
    """
    Destination poses requested by deferred reachability search.
    """

    held: list[Body] = field(default_factory=list)
    """
    Actual object held at each destination search.
    """

    navigation_targets: list[Pose] = field(default_factory=list)
    """
    Base destinations actually executed by the plan.
    """

    def locations(
        self,
        target: Body | Pose,
        context: Context,
        arm: Arms,
        grasp_description: GraspDescription | None,
    ) -> list[Pose]:
        """
        Reject the first destination after observing the acquired object.

        :param target: Pick-up object or current placement candidate.
        :param context: Context observed at grounding time.
        :param arm: Selected carrying arm.
        :param grasp_description: Grasp used to validate each approach.
        """
        if isinstance(target, Body):
            return [context.robot.root.global_pose]
        self.checked.append(target)
        self.held.append(_get_object_in_hand(context.robot, context.world, arm))
        if target is self.targets[0]:
            return []
        assert target is self.targets[1]
        return [context.robot.root.global_pose]

    def pickup(self, action: PickUpAction) -> PlanNode:
        """
        Attach with the same native model-change node as a real pick-up.

        :param action: Pick-up whose selected arm determines the attachment.
        """
        tool = ViewManager.get_end_effector_view(action.arm, self.context.robot)
        return sequential(
            [
                MoveGripperMotion(GripperState.CLOSE, action.arm),
                AttachNode(body=self.body, new_parent=tool.tool_frame),
            ]
        )

    def place(self, action: PlaceAction) -> PlanNode:
        """
        Release the acquired object at the successful destination.

        :param action: Placement selected by the destination query.
        """
        return sequential(
            [
                MoveGripperMotion(GripperState.OPEN, action.arm),
                DetachNode(body=self.body),
            ]
        )

    def on_start(self, node: PlanNode) -> None:
        """
        Observe placement only when its native lifecycle starts.

        :param node: Plan node whose execution is starting.
        """
        if isinstance(node, ActionNode) and isinstance(node.designator, PlaceAction):
            self.record_release(node.designator)

    def record_release(self, action: PlaceAction) -> None:
        """
        Check the held object when placement actually executes.

        :param action: Placement selected after successful base grounding.
        """
        assert action.target_location is self.targets[1]
        assert (
            _get_object_in_hand(self.context.robot, self.context.world, action.arm)
            is self.body
        )

    def navigate(self, action: NavigateAction) -> PlanNode:
        """
        Record executed base approaches.

        :param action: Navigation selected by the deferred approach query.
        """
        self.navigation_targets.append(action.target_location)
        return sequential([MoveGripperMotion(GripperState.CLOSE, Arms.LEFT)])


# %% destination retry
def test_transport_retries_destinations_after_one_pickup(
    pr2_world_copy: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An unreachable placement never repeats pickup or source navigation.

    :param pr2_world_copy: Existing robot world fixture.
    :param monkeypatch: Replace controller plans while retaining native grounding.
    """
    world = pr2_world_copy
    context = Context(
        world, world.get_semantic_annotations_by_type(PR2)[0], evaluate_conditions=False
    )
    with world.modify_world():
        body = Milk.create_with_new_body_in_world(
            name="transported_milk", world=world, scale=Scale(0.06, 0.06, 0.14)
        ).root
    targets = [
        Pose.from_xyz_rpy(x=x, y=1, z=1, reference_frame=world.root) for x in (1, 2)
    ]
    observations = DestinationAttempts(context, body, targets)
    monkeypatch.setattr(transporting, "reachability_location", observations.locations)
    monkeypatch.setattr(TransportAction, "inside_container", lambda action: [])
    monkeypatch.setattr(PickUpAction, "_action_plan", property(observations.pickup))
    monkeypatch.setattr(PlaceAction, "_action_plan", property(observations.place))
    monkeypatch.setattr(NavigateAction, "_action_plan", property(observations.navigate))
    for preparation in (ParkArmsAction, MoveTorsoAction):
        monkeypatch.setattr(
            preparation,
            "_action_plan",
            property(
                lambda action: sequential(
                    [MoveGripperMotion(GripperState.CLOSE, Arms.LEFT)]
                )
            ),
        )
    action = TransportAction(body, targets, Arms.LEFT)
    root = sequential([action], context=context)
    root.plan.node_callbacks.append(observations)

    with simulated_robot:
        root.plan.perform()

    assert observations.checked == targets
    assert observations.held == [body, body]
    assert len(root.plan.get_nodes_by_designator_type(TransportAction)) == 1
    assert len(root.plan.get_nodes_by_designator_type(PickUpAction)) == 1
    assert len(observations.navigation_targets) == 2
    attempts = root.plan.get_nodes_by_designator_type(MoveAndPlaceAction)
    assert [node.status for node in attempts] == [
        TaskStatus.FAILED,
        TaskStatus.SUCCEEDED,
    ]
    assert root.status is TaskStatus.SUCCEEDED
    assert body.parent_connection.parent is world.root


# %% explicit and automatic approach contracts
@pytest.mark.parametrize("robot_fixture", ["pr2_world_copy", "tracy_world"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("look", [False, True])
def test_placement_approach_preserves_explicit_facing_and_robot_capabilities(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    robot_fixture: str,
    explicit: bool,
    look: bool,
) -> None:
    """
    Automatic approaches never add a second base turn or require a drive.

    :param request: Resolve an existing mobile or stationary robot fixture.
    :param monkeypatch: Observe deferred location construction.
    :param robot_fixture: Existing annotated robot world.
    :param explicit: Whether the caller supplied a legacy standing position.
    :param look: Whether an automatic approach should aim the camera.
    """
    world = request.getfixturevalue(robot_fixture)
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    context = Context(world, robot)
    target = Pose.from_xyz_rpy(x=1, z=1, reference_frame=world.root)
    standing = robot.root.global_pose
    action = MoveAndPlaceAction(
        standing if explicit else None,
        robot.root,
        target,
        Arms.LEFT,
        keep_joint_states=True,
        look_at_operation_site=look,
    )
    execute_single(action, context=context)
    search = Mock(return_value=[standing])
    monkeypatch.setattr(transporting, "reachability_location", search)

    approaches = action._make_approach_actions()

    search.assert_not_called()
    if explicit:
        assert [type(item) for item in approaches] == [NavigateAction, FaceAtAction]
        assert approaches[0].target_location is standing
        assert approaches[1].pose is target
        assert all(item.keep_joint_states for item in approaches)
        return
    assert len(approaches) == int(robot.drive is not None) + int(look)
    if robot.drive is not None:
        navigation = next(context.query_backend.evaluate(approaches[0]))
        assert isinstance(navigation, NavigateAction)
        assert navigation.target_location is standing
        assert navigation.keep_joint_states is True
        search.assert_called_once_with(
            target, context, action.arm, action.grasp_description
        )
    if look:
        assert isinstance(approaches[-1], LookAtAction)
        assert approaches[-1].target is target
