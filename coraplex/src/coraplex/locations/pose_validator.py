from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

from typing_extensions import List

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.exceptions import (
    CollisionViolatedError,
    NoProgressError,
)
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.qp.exceptions import InfeasibleException
from coraplex.plans.plan_node import MotionNode
from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.manipulation_contacts import (
    ManipulationContactPolicy,
    TemporaryCollisionScope,
)
from coraplex.exceptions import TipLinkDoesNotMatchAnyArm
from coraplex.locations.base import PoseValidator
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.plan import Plan
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.robot_plans.mixins import HasTcpGoalThresholds
from coraplex.view_manager import ViewManager
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionForEndEffector,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import (
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
    KinematicStructureEntity,
)

logger = logging.getLogger("coraplex")


@dataclass
class IsVisibleBy(PoseValidator):
    """
    Validator for checking if either the given pose or body is visible for the robot.

    One has to be given, if both are provided the body is prefered
    """

    target_pose: Pose = field(default=None)
    """
    Pose for which visibility should be checked.
    """

    target_body: Body = field(default=None)
    """
    Body for which visibility should be checked.
    """

    def __call__(self, *args, **kwargs) -> bool:
        if not (self.target_pose or self.target_body):
            raise AttributeError("Either a pose or a body have to be given")
        return self.validate_body() if self.target_body else self.validate_pose()

    def validate_pose(self) -> bool:
        """
        Validates if the target_pose is visible for the robot by creating a temporary
        body at the pose and performing a ray test to see if there is a viewing axis
        between the robot and the target pose.

        :return: True if the target pose is visible for the robot, False otherwise
        """
        gen_body = Body(
            name=PrefixedName("visibility_test_obj", "coraplex"),
            collision=ShapeCollection([Box(scale=Scale(0.1, 0.1, 0.1))]),
        )
        with self.world.modify_world():
            self.world.add_connection(
                FixedConnection(
                    parent=self.world.root,
                    child=gen_body,
                    parent_T_connection_expression=self.target_pose.to_homogeneous_matrix(),
                )
            )

        result = self._ray_test(gen_body)

        if isinstance(self.target_pose, Pose):
            with self.world.modify_world():
                self.world.remove_connection(gen_body.parent_connection)
                self.world.remove_kinematic_structure_entity(gen_body)

        return result

    def validate_body(self) -> bool:
        return self._ray_test(self.target_body)

    def _ray_test(self, target_body: Body) -> bool:
        """
        Performs a ray test from the robot to check if the given body is visible, the
        check filters out bodies of the ' robot form the hit list of the ray test.

        :param target_body: The body for which the ray test is to be performed
        :return: True if the target body is visible for the robot, False otherwise
        """
        ray_tracer = self.world.ray_tracer
        camera = self.robot.get_default_camera()
        ray = ray_tracer.ray_test(
            camera.bodies[0].global_transform.to_position()[:3].to_np(),
            target_body.global_transform.to_position()[:3].to_np(),
            multiple_hits=True,
        )

        hit_bodies = [body for body in ray[2] if body not in self.robot.bodies]

        return hit_bodies[0] == target_body if len(hit_bodies) > 0 else False


@dataclass
class IsReachableBy(PoseValidator):
    """
    Validator that checks if a single pose is reachable with a link of the robot.
    """

    pose: Pose
    """
    Pose that should be reached with the tip_link.
    """

    tip_link: KinematicStructureEntity
    """
    Link that should be moved to the given pose.
    """

    grasp_description: GraspDescription = field(default=None)
    """
    The grasp description that should be used for validation.
    """

    contact_policy: ManipulationContactPolicy | None = field(default=None, kw_only=True)
    """
    Intended contacts preserved while validating the target pose.
    """

    allow_gripper_collision: bool = field(default=False, kw_only=True)
    """
    Explicit allowance matching a motion configured to ignore gripper contacts.
    """

    def __call__(self) -> bool:
        return AreReachableBy(
            pose_sequence=[self.pose],
            tip_link=self.tip_link,
            context=self.context,
            grasp_description=self.grasp_description,
            contact_policy=self.contact_policy,
            allow_gripper_collision=self.allow_gripper_collision,
        ).__call__()


@dataclass
class AreReachableBy(PoseValidator, HasTcpGoalThresholds):
    """
    Validator that checks if a sequence of poses is reachable with the given robot link.

    Poses are addressed in the order they are given. The active execution environment
    determines collision avoidance, and goal tolerances are resolved through the same
    configuration as native TCP motions.
    """

    pose_sequence: List[Pose]
    """
    Sequence of poses that should be reached.
    """

    tip_link: KinematicStructureEntity
    """
    Link of the robot which should be used for reachability checking.
    """

    grasp_description: GraspDescription = field(default=None)
    """
    The grasp description that should be used for validation.
    """

    contact_policy: ManipulationContactPolicy | None = field(default=None, kw_only=True)
    """
    Intended object contacts preserved during candidate validation.
    """

    allow_gripper_collision: bool = field(default=False, kw_only=True)
    """
    Explicit allowance matching a motion configured to ignore gripper contacts.
    """

    @property
    def gripper_collision_rules(self) -> list[CollisionRule]:
        """
        :return: The explicitly requested gripper allowance, if this is a tool frame.
        """
        if not self.allow_gripper_collision:
            return []
        arm = ViewManager.get_arm_by_tool_frame(self.tip_link, self.robot)
        if arm is None:
            return []
        return [
            AllowCollisionForEndEffector(
                end_effector=ViewManager.get_end_effector_view(arm, self.robot)
            )
        ]

    def create_msc(self) -> MotionStatechart:
        """
        Creates the Motion state chart to reach the given pose sequence with the given
        tip link.

        Also takes into account if there are alternative motion mappings for moving the
        end effector to the given pose.
        """
        alternative_motion = AlternativeMotion.check_for_alternative(
            self.alternative_motion_mappings, self.robot, MoveToolCenterPointMotion
        )
        if alternative_motion:
            correct_arm = ViewManager.get_arm_by_tool_frame(self.tip_link, self.robot)
            if correct_arm is None:
                raise TipLinkDoesNotMatchAnyArm(self.tip_link, self.robot)
            sequence = []
            for pose in self.pose_sequence:

                if self.grasp_description:
                    pose = self.grasp_description.pose_sequence(pose)[1]

                motion = alternative_motion(
                    pose,
                    correct_arm,
                    False,
                    position_threshold=self.resolved_position_threshold(),
                    orientation_threshold=self.resolved_orientation_threshold(),
                )
                node = MotionNode(designator=motion)
                # Imagine a plan for the motion node
                plan = Plan(replace(self.context, plan=None))
                plan.add_node(node)
                motion.plan_node = node
                sequence.append(motion._motion_chart)

        else:
            root = (
                self.robot.root
                if not (
                    self.robot.mobile_base.full_body_controlled
                    if isinstance(self.robot, HasMobileBase)
                    else False
                )
                else self.world.root
            )

            sequence = (
                [
                    self.grasp_description.pose_sequence(pose)[1]
                    for pose in self.pose_sequence
                ]
                if self.grasp_description
                else self.pose_sequence
            )

            sequence = [
                CartesianPose(
                    root_link=root,
                    tip_link=self.tip_link,
                    goal_pose=pose,
                    translation_threshold=self.resolved_position_threshold(),
                    orientation_threshold=self.resolved_orientation_threshold(),
                )
                for pose in sequence
            ]

        msc = MotionStatechart()
        msc.add_node(sequence_node := Sequence(sequence))
        if GiskardExecutable.collision_avoidance:
            msc.add_node(ExternalCollisionAvoidance(robot=self.robot))
            msc.add_node(SelfCollisionAvoidance(robot=self.robot))
            if rules := self.gripper_collision_rules:
                msc.add_node(UpdateTemporaryCollisionRules(temporary_rules=rules))
        msc.add_node(EndMotion.when_true(sequence_node))
        msc.add_node(
            still_progressing := StillProgressing(monitored_node=sequence_node)
        )
        msc.add_node(still_progressing.cancel_motion())

        return msc

    def create_executor(self, msc: MotionStatechart) -> Executor:
        """
        Creates the executor that runs a probe of this validator.

        :param msc: The motion statechart the executor is compiled against.
        """
        executor = Executor(
            context=MotionStatechartContext(
                world=self.world,
                qp_controller_config=QPControllerConfig(
                    target_frequency=50, prediction_horizon=4, verbose=False
                ),
            ),
        )
        try:
            executor.compile(msc)
        except BaseException:
            executor.context.cleanup()
            raise
        return executor

    def __call__(self, *args, **kwargs) -> bool:
        logger.debug(
            f"Hash of input for pose_sequence_reachability_validator: {hash((*self.pose_sequence, self.tip_link, self.robot))}"
        )

        scope = (
            self.contact_policy.scope(self.world)
            if self.contact_policy is not None
            else TemporaryCollisionScope(self.world)
        )
        scope.rules.extend(self.gripper_collision_rules)
        with self.world.reset_state_context(), scope.activate():
            executor = None
            try:
                executor = self.create_executor(self.create_msc())
                # These failures mean this candidate cannot execute the requested
                # sequence under the configured collision constraints.
                executor.tick_until_end(timeout=1500)
            except (
                TimeoutError,
                CollisionViolatedError,
                InfeasibleException,
                NoProgressError,
            ):
                logger.debug(f"Infeasible pose sequence: {self.pose_sequence}")
                return False
            finally:
                # tick_until_end cleans up its execution; compilation can fail
                # before entering it and must release collision consumers as well.
                if executor is not None:
                    executor.context.cleanup()
            return True


@dataclass
class IsObjectReachableBy(PoseValidator):
    """
    Reachability check that is evaluated against a *fresh* copy of the world.

    Both the world copy and the grasp pose sequence are produced inside
    :meth:`__call__`, i.e. when the surrounding condition/monitor is evaluated,
    so the result reflects the current world state instead of the state at the
    time the plan was parsed. The actual reachability simulation is delegated to
    :class:`AreReachableBy` / :class:`IsReachableBy`, which run on the throwaway
    copy so the live world is left untouched.
    """

    arm: Arms
    """
    The arm whose end effector should reach the object.
    """

    object_designator: Body
    """
    The object that should be reachable.
    """

    grasp_description: GraspDescription = field(default=None)
    """
    Grasp description used to build the pose sequence.

    Required unless
    ``as_single_grasp`` is set.
    """

    target_pose: Pose = field(default=None)
    """
    Optional explicit target pose.

    If omitted, the object's own frame is used as the grasp target (as in
    :meth:`GraspDescription.grasp_pose_sequence`).
    """

    reverse: bool = field(default=False)
    """
    Whether the grasp pose sequence should be reversed.
    """

    as_single_grasp: bool = field(default=False)
    """
    If set, check reachability of a single grasp pose at the object (used for grasping
    handles of containers) instead of a full pick pose sequence.
    """

    def __call__(self, *args, **kwargs) -> bool:
        context = self.copy_context_for_validation(self.context)
        end_effector = ViewManager.get_end_effector_view(self.arm, context.robot)

        if self.as_single_grasp:
            return IsReachableBy(
                context=context,
                pose=self.object_designator.global_pose,
                tip_link=end_effector.tool_frame,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    end_effector,
                ),
                contact_policy=ManipulationContactPolicy(
                    self.object_designator,
                    end_effector.bodies_with_collision,
                    self.object_designator.global_pose,
                ),
            ).__call__()

        if self.target_pose is not None:
            pose_sequence = self.grasp_description.pose_sequence(
                self.target_pose, self.object_designator, reverse=self.reverse
            )
        else:
            pose_sequence = self.grasp_description.grasp_pose_sequence(
                self.object_designator
            )

        return AreReachableBy(
            context=context,
            pose_sequence=pose_sequence,
            tip_link=end_effector.tool_frame,
            contact_policy=ManipulationContactPolicy(
                self.object_designator,
                end_effector.bodies_with_collision,
                (
                    self.target_pose
                    if self.target_pose is not None
                    else self.object_designator.global_pose
                ),
            ),
        ).__call__()
