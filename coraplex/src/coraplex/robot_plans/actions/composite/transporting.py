from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import List

from typing_extensions import Optional, Any

from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.factories import (
    a,
    an,
    entity,
    variable,
)
from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.base import DeferredLocation
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import sequential
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.composite.facing import FaceAtAction
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.predicates import InsideOf
from semantic_digital_twin.semantic_annotations.semantic_annotations import Drawer
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class TransportAction(ActionDescription):
    """
    Transports an object to a position using an arm.
    """

    object_designator: Body = field(repr=False)
    """
    Object designator_description describing the object that should be
    transported.
    """

    target_location: Pose | Iterable[Pose]
    """
    Destination pose or lazy alternatives tried after acquiring the object.
    """

    arm: Arms
    """
    Arm that should be used.
    """

    grasp_description: Optional[GraspDescription] = None
    """
    Grasp Description that should be used for picking up the object.
    """

    look_at_operation_site: bool = ActionConfig.transport_look_at_operation_site
    """
    Whether the robot looks at the object before picking it up and at the target
    location before placing it, each time from where it has already navigated to.
    """

    def inside_container(self) -> List[Body]:
        bodies = []
        for body in self.world.bodies:
            if body == self.object_designator:
                continue
            if InsideOf(self.object_designator, body).compute_containment_ratio() > 0.9:
                bodies.append(body)
        return bodies

    def _make_look_at_actions(self, pose: Pose) -> List[LookAtAction]:
        """
        :param pose: The place the robot is about to work at.
        :return: The action aiming the robot's camera there, empty if this transport
            does not look at what it operates on.
        """
        if not self.look_at_operation_site:
            return []
        return [LookAtAction(pose)]

    def _make_open_container_actions(
        self, container: Body
    ) -> list[Match[NavigateAction] | OpenAction]:
        """
        :param container: The container body in which the object is located.
        :return: The actions needed to open the given container, empty if the container is not a known drawer.
        """
        drawer_annotation = an(
            entity(
                drawer := variable(Drawer, domain=self.world.semantic_annotations)
            ).where(drawer.root == container)
        )
        drawer_annotation = list(drawer_annotation.evaluate())
        if len(drawer_annotation) == 0:
            return []
        handle = drawer_annotation[0].handle.root

        return [
            *self._make_navigation_actions(handle.global_pose),
            OpenAction(handle, self.arm),
        ]

    def _make_navigation_actions(
        self, target: Body | Pose, grasp_description: Optional[GraspDescription] = None
    ) -> list[Match[NavigateAction]]:
        """Approach a manipulation target when the robot has a drive.

        :param target: Object or pose approached using the state at execution time.
        :param grasp_description: Grasp used to validate reachable base positions.
        :return: A navigation query, or an empty list for a stationary robot.
        """
        if self.robot.drive is None:
            return []
        location = DeferredLocation(
            lambda: reachability_location(
                target, self.context, self.arm, grasp_description
            )
        )
        return [
            a(NavigateAction)(
                target_location=variable(Pose, domain=location),
                keep_joint_states=True,
            )
        ]

    def _make_torso_actions(self) -> list[MoveTorsoAction]:
        """Raise the torso when one is specified by the robot annotation.

        :return: The torso preparation action, or an empty list without a torso.
        """
        if self.robot.get_torso_if_specified() is None:
            return []
        return [MoveTorsoAction(TorsoState.HIGH)]

    def _make_placement_actions(self) -> list[Match | LookAtAction]:
        """Ground alternative destinations within the carrying phase.

        :return: Navigation and placement for a pose, or a destination query.
        """
        if isinstance(self.target_location, Pose):
            return [
                *self._make_navigation_actions(
                    self.target_location, self.grasp_description
                ),
                *self._make_look_at_actions(self.target_location),
                a(PlaceAction)(
                    object_designator=self.object_designator,
                    target_location=self.target_location,
                    arm=self.arm,
                ),
            ]
        return [
            a(MoveAndPlaceAction)(
                standing_position=None,
                object_designator=self.object_designator,
                target_location=variable(Pose, domain=self.target_location),
                arm=self.arm,
                keep_joint_states=True,
                grasp_description=self.grasp_description,
                look_at_operation_site=self.look_at_operation_site,
            )
        ]

    @property
    def _action_plan(self) -> PlanNode:
        # the side to approach from follows the robot's reach: its own front for a robot
        # that drives to the object, the facing side for one that reaches from its stand
        self.grasp_description = (
            self.grasp_description
            or GraspDescription.robot_relative_default(
                ViewManager.get_end_effector_view(self.arm, self.robot),
                self.object_designator.global_pose,
                self.object_designator,
            )
        )

        children = []
        for container in self.inside_container():
            children.extend(self._make_open_container_actions(container))

        children.extend(
            [
                ParkArmsAction(Arms.BOTH),
                *self._make_navigation_actions(
                    self.object_designator, self.grasp_description
                ),
                *self._make_look_at_actions(self.object_designator.global_pose),
                a(PickUpAction)(
                    object_designator=self.object_designator,
                    arm=self.arm,
                    grasp_description=self.grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                *self._make_torso_actions(),
                *self._make_placement_actions(),
                ParkArmsAction(Arms.BOTH),
            ]
        )

        return sequential(children)


@dataclass
class PickAndPlaceAction(ActionDescription):
    """
    Transports an object to a position using an arm without moving the base of
    the robot.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be
    transported.
    """

    target_location: Pose
    """
    Target Location to which the object should be transported.
    """

    arm: Arms
    """
    Arm that should be used.
    """
    grasp_description: GraspDescription
    """
    Description of the grasp to pick up the target.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                ParkArmsAction(Arms.BOTH),
                PickUpAction(
                    self.object_designator,
                    self.arm,
                    grasp_description=self.grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                PlaceAction(self.object_designator, self.target_location, self.arm),
                ParkArmsAction(Arms.BOTH),
            ]
        )


@dataclass
class MoveAndPlaceAction(ActionDescription):
    """
    Approach a placement from an explicit or reachable standing position.
    """

    standing_position: Pose | None
    """
    Explicit base pose, or None to ground an approach with the current held object.
    """
    object_designator: Body
    """
    The object to place.
    """
    target_location: Pose
    """
    The location to place the object.
    """
    arm: Arms
    """
    The arm to use.
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    grasp_description: GraspDescription | None = field(default=None, kw_only=True)
    """Grasp used to validate automatically sampled standing positions."""

    look_at_operation_site: bool = field(default=False, kw_only=True)
    """Aim the camera at the target after an automatically grounded approach."""

    def _make_approach_actions(
        self,
    ) -> list[NavigateAction | FaceAtAction | Match | LookAtAction]:
        """Retain explicit facing or ground a reachable base pose at execution.

        :return: Base and optional camera actions for this placement attempt.
        """
        if self.standing_position is not None:
            return [
                NavigateAction(self.standing_position, self.keep_joint_states),
                FaceAtAction(self.target_location, self.keep_joint_states),
            ]
        actions = []
        if self.robot.drive is not None:
            location = DeferredLocation(
                lambda: reachability_location(
                    self.target_location, self.context, self.arm, self.grasp_description
                )
            )
            actions.append(
                a(NavigateAction)(
                    target_location=variable(Pose, domain=location),
                    keep_joint_states=self.keep_joint_states,
                )
            )
        if self.look_at_operation_site:
            actions.append(LookAtAction(self.target_location))
        return actions

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                *self._make_approach_actions(),
                PlaceAction(self.object_designator, self.target_location, self.arm),
            ]
        )


@dataclass
class MoveAndPickUpAction(ActionDescription):
    """
    Navigate to `standing_position`, then turn towards the object and pick it
    up.
    """

    standing_position: Pose
    """
    The pose to stand before trying to pick up the object.
    """
    object_designator: Body
    """
    The object to pick up.
    """
    arm: Arms
    """
    The arm to use.
    """
    grasp_description: GraspDescription
    """
    The grasp to use.
    """

    keep_joint_states: bool = ActionConfig.navigate_keep_joint_states
    """
    Keep the joint states of the robot the same during the navigation.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                NavigateAction(self.standing_position, self.keep_joint_states),
                FaceAtAction(
                    self.object_designator.global_pose, self.keep_joint_states
                ),
                PickUpAction(self.object_designator, self.arm, self.grasp_description),
            ]
        )
