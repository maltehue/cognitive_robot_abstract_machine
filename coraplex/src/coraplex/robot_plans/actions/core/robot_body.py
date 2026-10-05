from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Tuple, List

from typing_extensions import Optional, Dict, Any

import numpy as np

from coraplex.plans.plan_node import PlanNode
from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from coraplex.datastructures.dataclasses import Context
from coraplex.robot_plans import MoveManipulatorMotion
from krrood.entity_query_language.factories import variable_from
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.connections import Connection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    Vector3,
)
from coraplex.datastructures.enums import AxisIdentifier, Arms, MovementType

from coraplex.datastructures.trajectory import PoseTrajectory
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans.actions.base import ActionDescription, DescriptionType
from coraplex.robot_plans.mixins import HasMaxJointVelocity, HasTcpGoalThresholds
from coraplex.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveTCPWaypointsMotion,
    MoveToolCenterPointKeepingAxisMotion,
    MoveToolCenterPointMotion,
)
from coraplex.robot_plans.motions.base import BaseMotion
from coraplex.robot_plans.motions.robot_body import MoveJointsMotion
from coraplex.validation.goal_validator import create_multiple_joint_goal_validator
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import (
    TorsoState,
    GripperState,
    StaticJointState,
)


@dataclass
class MoveTorsoAction(ActionDescription):
    """
    Move the torso of the robot up and down.
    """

    torso_state: TorsoState
    """
    The state of the torso that should be set
    """

    @property
    def _action_plan(self) -> PlanNode:
        joint_state = self.robot.get_torso().get_joint_state_by_type(self.torso_state)
        return execute_single(
            MoveJointsMotion(
                [c.name.name for c in joint_state.connections],
                joint_state.target_values,
            ),
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression | bool:
        """
        The target joint state for the torso needs to be achieved.
        """
        joint_state = context.robot.get_torso().get_joint_state_by_type(
            kwargs["torso_state"]
        )
        return variable_from(joint_state).is_achieved()


@dataclass
class SetGripperAction(ActionDescription):
    """
    Set the gripper state of the robot.
    """

    gripper: Arms
    """
    The gripper that should be set.
    """

    motion: GripperState
    """
    The motion that should be set on the gripper.
    """

    @property
    def _action_plan(self) -> PlanNode:
        arms = [Arms.LEFT, Arms.RIGHT] if self.gripper == Arms.BOTH else [self.gripper]
        return sequential(
            [MoveGripperMotion(gripper=arm, motion=self.motion) for arm in arms]
        )


@dataclass
class ParkArmsAction(ActionDescription, HasMaxJointVelocity):
    """
    Park the arms of the robot, and its torso and neck too where they declare a parked
    state: a humanoid that bent to reach and looked at what it reached for parks
    standing upright and facing forward.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    carry_lift: float = field(default=0.6, kw_only=True)
    """
    How high above its parked place a hand holding a body is brought, in towards the
    chest, before it comes down to that place, in meters.

    Folding the arm is a motion of its joints, which takes the hand on whatever arc
    the joints describe, and with a body hanging from the hand that arc swung the
    body forward and down through the table it had just been lifted from, from
    whatever height the fold started. So a hand that holds a body does not fold: it
    is carried up and in to above its parked place, then down to it, both as straight
    lines that keep the body hanging as it hangs, and the joints only settle the
    parked pose at the end.
    """

    carry_retreat_keeps_hold: bool = field(default=True, kw_only=True)
    """
    Whether the hand keeps its approach axis pointing the way it points while it is
    carried in, so that the body it holds stays hanging as it hangs, the hand free to
    turn about that axis; else the orientation is left to the arm altogether. Holding
    the whole orientation was tried and asked the waist to lean back, or the wrist
    to go past its range.
    """

    @property
    def _action_plan(self) -> PlanNode:
        joint_names, joint_poses = self.get_joint_poses()
        park = MoveJointsMotion(
            names=joint_names,
            positions=joint_poses,
            max_joint_velocity=self.max_joint_velocity,
        )
        retreats = [
            retreat for arm in self.arms for retreat in self.retreat_carrying(arm)
        ]
        if not retreats:
            return execute_single(park)
        return sequential([*retreats, park])

    @property
    def arms(self) -> List[Arms]:
        """
        :return: The arms parked, one by one.
        """
        if self.arm == Arms.BOTH:
            return [Arms.LEFT, Arms.RIGHT]
        return [self.arm]

    def retreat_carrying(self, arm: Arms) -> List[BaseMotion]:
        """
        :param arm: An arm about to be parked.
        :return: The motions carrying its hand, when it holds a body, up and in to
            :attr:`carry_lift` above where the hand will be parked and then down to
            there, keeping the body hanging as it hangs; the torso and the neck park
            alongside, so that the controller does not lean the torso to bring the
            hand closer. Nothing when the hand is empty.
        """
        views = ViewManager().get_all_arm_views(arm, self.robot)
        if not views or views[0] is None:
            return []
        end_effector = views[0].end_effector
        if not end_effector.held_bodies:
            return []
        root = self.robot.root
        root_T_tool = self.world.compute_forward_kinematics_np(
            root, end_effector.tool_frame
        )
        parked = self.parked_pose_of(end_effector.tool_frame)
        above = root_T_tool.copy()
        above[:3, 3] = parked[:3, 3]
        above[2, 3] += self.carry_lift
        down = above.copy()
        down[2, 3] = parked[2, 3]
        trunk_names, trunk_poses = self.get_joint_poses(arms=False)
        alongside = dict(zip(trunk_names, trunk_poses))
        approach = end_effector.front_facing_axis.to_np()[:3]
        motions = []
        # Up and in, the hand's approach keeps pointing the way it does now; coming
        # down, the way it will point once parked, so that the descent ends in the
        # parked pose rather than fighting it.
        for waypoint, pointing in (
            (above, root_T_tool[:3, :3] @ approach),
            (down, parked[:3, :3] @ approach),
        ):
            target = HomogeneousTransformationMatrix(
                waypoint, reference_frame=root
            ).to_pose()
            if self.carry_retreat_keeps_hold:
                motions.append(
                    MoveToolCenterPointKeepingAxisMotion(
                        target,
                        arm,
                        allow_gripper_collision=False,
                        kept_axis=Vector3(*approach),
                        kept_direction=Vector3(*pointing),
                        joint_goals=alongside,
                    )
                )
            else:
                motions.append(
                    MoveToolCenterPointMotion(
                        target,
                        arm,
                        allow_gripper_collision=False,
                        movement_type=MovementType.TRANSLATION,
                    )
                )
        return motions

    def parked_pose_of(self, tool_frame: Body) -> np.ndarray:
        """
        :param tool_frame: A hand's tool frame.
        :return: Where it will be once everything this action parks is parked, as a
            4 by 4 transformation matrix in the frame of the robot's root, read off the
            forward kinematics with the parked joints at their parked positions and
            every other joint as it is now; nothing is moved.
        """
        expression = self.world.compose_forward_kinematics_expression(
            self.robot.root, tool_frame
        )
        targets = self.parked_joint_targets()
        variables = [connection.dof.variables.position for connection in targets]
        return expression.substitute(variables, list(targets.values())).evaluate()

    def parked_joint_targets(
        self, arms: bool = True, trunk: bool = True
    ) -> Dict[Connection, float]:
        """
        :param arms: Whether the arms' parked joints are included.
        :param trunk: Whether the torso's and the neck's parked joints are included,
            where they declare a parked state.
        :return: The parked position of every joint this action parks, by joint.
        """
        parked = (
            list(ViewManager().get_all_arm_views(self.arm, self.robot)) if arms else []
        )
        trunk_parts = (
            (self.robot.get_torso_if_specified(), self.robot.get_neck_if_specified())
            if trunk
            else ()
        )
        for part in trunk_parts:
            if part is not None and part.has_joint_state_of_type(StaticJointState.PARK):
                parked.append(part)
        targets: Dict[Connection, float] = {}
        for part in parked:
            targets.update(part.get_joint_state_by_type(StaticJointState.PARK).items())
        return targets

    def get_joint_poses(
        self, arms: bool = True, trunk: bool = True
    ) -> Tuple[List[str], List[float]]:
        """
        :param arms: Whether the arms' parked joints are included.
        :param trunk: Whether the torso's and the neck's parked joints are included,
            where they declare a parked state.
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        targets = self.parked_joint_targets(arms=arms, trunk=trunk)
        return [c.name.name for c in targets], list(targets.values())


@dataclass
class CarryAction(ActionDescription):
    """
    Parks the robot's arms.

    And align the arm with the given Axis of a frame.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    align: Optional[bool] = False
    """
    If True, aligns the end-effector with a specified axis.
    """

    tip_link: Optional[str] = None
    """
    Name of the tip link to align with, e.g the object.
    """

    tip_axis: Optional[AxisIdentifier] = None
    """
    Tip axis of the tip link, that should be aligned.
    """

    root_link: Optional[str] = None
    """
    Base link of the robot; typically set to the torso.
    """

    root_axis: Optional[AxisIdentifier] = None
    """
    Goal axis of the root link, that should be used to align with.
    """

    def execute(self) -> None:
        joint_poses = self.get_joint_poses()
        tip_normal = self.axis_to_vector3_stamped(self.tip_axis, link=self.tip_link)
        root_normal = self.axis_to_vector3_stamped(self.root_axis, link=self.root_link)

        self.add_subplan(
            execute_single(
                MoveJointsMotion(
                    names=list(joint_poses.keys()),
                    positions=list(joint_poses.values()),
                    align=self.align,
                    tip_link=self.tip_link,
                    tip_normal=tip_normal,
                    root_link=self.root_link,
                    root_normal=root_normal,
                )
            )
        ).perform()

    def get_joint_poses(self) -> Dict[str, float]:
        """
        :return: The joint positions that should be set for the arm to be in the park position.
        """
        joint_poses = {}
        arm_chains = RobotDescription.current_robot_description.get_arm_chain(self.arm)
        if type(arm_chains) is not list:
            joint_poses = arm_chains.get_static_joint_states(StaticJointState.Park)
        else:
            for arm_chain in RobotDescription.current_robot_description.get_arm_chain(
                self.arm
            ):
                joint_poses.update(
                    arm_chain.get_static_joint_states(StaticJointState.Park)
                )
        return joint_poses

    def axis_to_vector3_stamped(
        self, axis: AxisIdentifier, link: str = "base_link"
    ) -> Vector3:
        v = {
            AxisIdentifier.X: Vector3(x=1.0, y=0.0, z=0.0),
            AxisIdentifier.Y: Vector3(x=0.0, y=1.0, z=0.0),
            AxisIdentifier.Z: Vector3(x=0.0, y=0.0, z=1.0),
        }[axis]
        v.frame_id = link
        return v


@dataclass
class FollowToolCenterPointPathAction(ActionDescription, HasTcpGoalThresholds):
    """
    Represents an action to move a robotic arm's TCP (Tool Center Point) along a path of
    poses.
    """

    target_locations: PoseTrajectory
    """
    Path poses for the TCP motion.
    """

    arm: Arms
    """
    Entry from the enum for which arm should be parked.
    """

    @property
    def _action_plan(self) -> PlanNode:
        target_locations = list(self.target_locations.poses)

        motion = MoveTCPWaypointsMotion(
            target_locations,
            self.arm,
            allow_gripper_collision=True,
            position_threshold=self.position_threshold,
            orientation_threshold=self.orientation_threshold,
        )

        return execute_single(motion)

    def validate(
        self,
        result: Optional[Any] = None,
        max_wait_time: timedelta = timedelta(seconds=2),
    ):
        pass


@dataclass
class MoveManipulatorAction(ActionDescription, HasTcpGoalThresholds):
    """
    Move the end_effector to a specific pose.
    """

    target_pose: Pose
    """
    The pose where the end_effector should be moved to.
    """

    end_effector: EndEffector
    """
    The end_effector that should be moved.
    """

    allow_gripper_collision: bool
    """
    If the gripper can collide with something.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(
            MoveManipulatorMotion(
                self.target_pose,
                self.end_effector,
                self.allow_gripper_collision,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            )
        )

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        end_effector = variables["end_effector"]
        target_pose = variables["target_pose"]
        return allclose(
            end_effector.tool_frame.global_pose.to_np(),
            target_pose.to_np(),
            atol=0.1,
        )
