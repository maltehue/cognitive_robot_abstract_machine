"""
Transport composes only the preparation actions its robot can perform.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from unittest.mock import Mock
from typing_extensions import Optional

from coraplex.datastructures.grasp import GraspDescription
from semantic_digital_twin.world_description.world_entity import Body

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.locations.factories import _get_object_in_hand
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import UnderspecifiedNode
from coraplex.robot_plans.actions.composite import transporting
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world_description.geometry import Scale

# %% transport composition


@dataclass
class ContainerFreeTransport(TransportAction):
    """
    Transport of an object placed outside all containers.
    """

    def inside_container(self) -> list[Body]:
        """
        Return the empty container set of this test scenario.
        """
        return []


@pytest.fixture(params=["tracy_world", "pr2_world_copy", "_hsr_world_setup"])
def transport_action(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> TransportAction:
    """
    Bind the transport to the shared stationary or mobile robot fixture.

    :param request: Fixture request selecting the existing annotated robot world.
    :param monkeypatch: Fixture substituting base-location search during composition.
    """
    world = deepcopy(request.getfixturevalue(request.param))
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    with world.modify_world():
        object = Milk.create_with_new_body_in_world(
            name="transported_milk",
            world=world,
            scale=Scale(0.08, 0.08, 0.2),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=3, y=3, z=0.5
            ),
        )
    action = ContainerFreeTransport(
        object.root,
        Pose.from_xyz_rpy(3, 3.2, 0.5, reference_frame=world.root),
        Arms.LEFT,
    )
    execute_single(action, context=Context(world=world, robot=robot))
    monkeypatch.setattr(transporting, "reachability_location", Mock(return_value=[]))
    return action


def test_transport_omits_unavailable_preparations(
    transport_action: TransportAction,
) -> None:
    """
    Stationary robots retain pick/place while mobile robots retain both approaches.

    :param transport_action: Transport bound to an existing annotated robot.
    """
    root = transport_action._action_plan
    types = [
        (
            node.designator_type
            if isinstance(node, UnderspecifiedNode)
            else type(node.designator)
        )
        for node in root.children
    ]
    expected = [ParkArmsAction]
    if transport_action.robot.drive is not None:
        expected.append(NavigateAction)
    expected += [PickUpAction, ParkArmsAction]
    if transport_action.robot.get_torso_if_specified() is not None:
        expected.append(MoveTorsoAction)
    if transport_action.robot.drive is not None:
        expected.append(NavigateAction)
    expected += [PlaceAction, ParkArmsAction]
    assert types == expected


def test_navigation_search_is_skipped_without_a_drive(
    transport_action: TransportAction, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A stationary transport never constructs a mobile-base reachability search.

    :param transport_action: Transport bound to an existing annotated robot.
    :param monkeypatch: Fixture replacing the expensive location generator.
    """
    search = Mock(return_value=[])
    monkeypatch.setattr(transporting, "reachability_location", search)
    pickup = transport_action._make_navigation_actions(
        transport_action.object_designator
    )
    search.assert_not_called()
    placing = transport_action._make_navigation_actions(
        transport_action.target_location
    )
    if transport_action.robot.drive is None:
        assert pickup == placing == []
        search.assert_not_called()
    else:
        assert len(pickup) == len(placing) == 1
        search.assert_not_called()


@pytest.mark.parametrize("transport_action", ["pr2_world_copy"], indirect=True)
@pytest.mark.parametrize("target_type", [Body, Pose])
def test_navigation_search_observes_the_attachment_and_base_at_grounding(
    transport_action: TransportAction,
    target_type: type[Body] | type[Pose],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Build manipulation approaches from the state after earlier transport steps.

    :param transport_action: Mobile transport bound to the existing PR2 fixture.
    :param target_type: Whether the navigation approaches an object or placement pose.
    :param monkeypatch: Fixture replacing sampling with a current-state observer.
    """
    action = transport_action
    target = action.object_designator if target_type is Body else action.target_location
    end_effector = ViewManager.get_end_effector_view(action.arm, action.robot)

    def current_base_location(
        target: Body | Pose,
        context: Context,
        arm: Arms,
        grasp_description: GraspDescription | None,
    ) -> list[Pose]:
        """
        Observe the held object and return the current base pose.

        :param target: Manipulation target passed to the location factory.
        :param context: State from which the location is built.
        :param arm: Arm carrying the object.
        :param grasp_description: Grasp used by the transport.
        """
        assert (
            _get_object_in_hand(context.robot, context.world, arm)
            is action.object_designator
        )
        return [context.robot.root.global_pose]

    search = Mock(side_effect=current_base_location)
    monkeypatch.setattr(transporting, "reachability_location", search)
    navigation = action._make_navigation_actions(target)[0]
    search.assert_not_called()

    with action.world.modify_world():
        action.world.move_branch_with_fixed_connection(
            branch_root=action.object_designator,
            new_parent=end_effector.tool_frame,
        )
    current_base = Pose.from_xyz_rpy(
        x=2, y=1, yaw=0.5, reference_frame=action.world.root
    )
    action.robot.set_root_pose(current_base)

    grounded = next(action.context.query_backend.evaluate(navigation))

    np.testing.assert_array_equal(
        grounded.target_location.to_np(), action.robot.root.global_pose.to_np()
    )
    search.assert_called_once_with(target, action.context, action.arm, None)


def test_container_opening_approaches_only_with_a_drive(
    transport_action: TransportAction, apartment_world_copy
) -> None:
    """
    Opening an annotated drawer preserves the stationary robot's current base.

    :param transport_action: Transport bound to an existing annotated robot.
    :param apartment_world_copy: Existing apartment with a drawer and handle annotation.
    """
    from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
    from coraplex.robot_plans.actions.core.container import OpenAction

    transport_action.world.merge_world_at_pose(
        apartment_world_copy, HomogeneousTransformationMatrix()
    )
    drawer = transport_action.world.get_semantic_annotations_by_type(Drawer)[0]
    actions = transport_action._make_open_container_actions(drawer.root)
    assert isinstance(actions[-1], OpenAction)
    assert actions[-1].arm == transport_action.arm
    assert len(actions) == (2 if transport_action.robot.drive is not None else 1)
    assert (
        transport_action._make_open_container_actions(
            transport_action.object_designator
        )
        == []
    )


@pytest.mark.parametrize("arm", [Arms.LEFT, Arms.RIGHT])
def test_pick_and_place_keep_the_selected_stationary_or_mobile_arm(
    transport_action: TransportAction, arm: Arms
) -> None:
    """
    Both manipulations select the requested semantic arm without requiring a base.

    :param transport_action: Transport bound to an existing annotated robot.
    :param arm: Requested semantic arm.
    """
    from coraplex.plans.attachment_nodes import AttachNode, DetachNode
    from coraplex.view_manager import ViewManager

    transport_action.arm = arm
    transport_action._action_plan
    end_effector = ViewManager.get_end_effector_view(arm, transport_action.robot)
    pick = PickUpAction(
        transport_action.object_designator, arm, transport_action.grasp_description
    )
    execute_single(pick, context=transport_action.context)
    attach = next(
        node for node in pick._action_plan.descendants if isinstance(node, AttachNode)
    )
    assert isinstance(attach, AttachNode)
    assert attach.new_parent is end_effector.tool_frame
    place = PlaceAction(
        transport_action.object_designator, transport_action.target_location, arm
    )
    execute_single(place, context=transport_action.context)
    release = next(
        node for node in place._action_plan.descendants if isinstance(node, DetachNode)
    )
    assert isinstance(release, DetachNode)
    assert release.new_parent is transport_action.world.root
