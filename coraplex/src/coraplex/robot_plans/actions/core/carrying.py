from __future__ import annotations

from dataclasses import dataclass

from coraplex.exceptions import ObjectIsNotHeld
from coraplex.plans.factories import execute_single
from coraplex.plans.plan_node import PlanNode
from coraplex.querying.predicates import GripperHolds
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.mixins import (
    CartesianVelocityLimitParameters,
    HasTcpGoalThresholds,
)
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.grasping.grasp_candidates import (
    GraspCandidate,
    HasGraspCandidates,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% positioning a held object


@dataclass
class CarryAction(
    ActionDescription, CartesianVelocityLimitParameters, HasTcpGoalThresholds
):
    """
    Move an attached object to a carrying pose without changing its grasp.
    """

    object_designator: HasGraspCandidates
    """The object already attached beneath an end effector."""

    target: Pose
    """
    Desired object pose, usually expressed relative to the robot root.
    """

    @property
    def _action_plan(self) -> PlanNode:
        """
        Move the holding arm using the current object-to-tool transform.
        """
        object_body = self.object_designator.root
        arm = next(
            (
                arm
                for arm in self.robot.all_arms
                if GripperHolds(arm.end_effector, object_body)()
            ),
            None,
        )
        if arm is None:
            raise ObjectIsNotHeld(self.object_designator)
        gripper = arm.end_effector
        grasp = GraspCandidate(self.object_designator, gripper.held_body_T_grasp)
        target = gripper.tool_frame_goal(grasp.moved_to(self.target))
        return execute_single(
            MoveToolCenterPointMotion(
                target=target,
                arm=arm,
                allowed_contacts=[
                    AllowCollisionBetweenGroups(
                        body_group_a=gripper.bodies_with_collision,
                        body_group_b=[object_body],
                    )
                ],
                max_linear_velocity=self.max_linear_velocity,
                max_angular_velocity=self.max_angular_velocity,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            )
        )
