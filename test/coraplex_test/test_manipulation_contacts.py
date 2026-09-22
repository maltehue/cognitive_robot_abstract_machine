"""
Native manipulation retains avoidance while permitting intended contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.datastructures.grasp import GraspDescription
from coraplex.datastructures.manipulation_contacts import (
    ManipulationContactPolicy,
    TemporaryCollisionScope,
)
from coraplex.execution_environment import simulated_robot_advanced, real_robot
from coraplex.locations.base import PoseValidator
from coraplex.locations.factories import reachability_location
from coraplex.locations.pose_validator import AreReachableBy
from coraplex.plans.factories import execute_single, sequential
from coraplex.plans.attachment_nodes import AttachNode
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from semantic_digital_twin.api import BodySpecification, Connection6DoFSpecification
from semantic_digital_twin.collision_checking.collision_rules import (
    AvoidCollisionBetweenGroups,
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.collision_checking.collision_matrix import CollisionCheck
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import FixedConnection
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from coraplex.robot_plans.motions.placement import (
    PlacementPoseSequence,
    PlacementStage,
    MovePlacementMotion,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from krrood.adapters.json_serializer import to_json, from_json
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)


# %% supported manipulation fixture
class ManipulationBody(StrEnum):
    """
    Bodies participating in the contact regression.
    """

    TABLE = "manipulation_table"
    """
    Support for the manipulated object.
    """

    OBJECT = "manipulation_object"
    """
    Object selected by the actions.
    """

    REFERENCE = "rotated_reference"
    """
    Independent frame for a support-pose regression.
    """


# %% scene observations
@dataclass
class SupportedManipulation:
    """
    A supported object at an independently reachable PR2 stance.
    """

    context: Context
    """
    Context containing the annotated robot and table.
    """

    body: Body
    """
    Object which may touch its gripper and support.
    """
    table: Body
    """
    Body supporting the initial and final object pose.
    """
    grasp: GraspDescription
    """
    Native grasp for the selected end effector.
    """


@dataclass
class ContactStage:
    """
    Observed policy at a control tick.
    """

    attached: bool
    """
    Whether the selected object is attached to the tool.
    """

    contact_active: bool
    """
    Whether selected gripper-to-object contact is temporarily allowed.
    """

    support_checked: bool
    """
    Whether gripper-to-support avoidance remains in the collision matrix.
    """


@dataclass
class ContactScopeTrace(PlanCallback):
    """
    Observe contact-rule transitions during actual native execution.
    """

    scene: SupportedManipulation
    """
    Scene whose attachment and collision pairs are observed.
    """

    stages: list[ContactStage] = field(default_factory=list)
    """
    Recorded policy states in controller execution order.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """:param statechart: Native statechart that completed one control tick."""
        gripper = self.scene.context.robot.left_arm.end_effector
        checks = (
            self.scene.context.world.collision_manager.collision_matrix.collision_checks
        )
        temporary = self.scene.context.world.collision_manager.temporary_rules
        self.stages.append(
            ContactStage(
                self.scene.body.parent_connection.parent is gripper.tool_frame,
                any(
                    isinstance(rule, AllowCollisionBetweenGroups)
                    and gripper.root in rule.body_group_a
                    and self.scene.body in rule.body_group_b
                    for rule in temporary
                ),
                CollisionCheck.create_and_validate(gripper.root, self.scene.table)
                in checks,
            )
        )


@pytest.fixture
def supported_manipulation(pr2_world_copy: World) -> SupportedManipulation:
    """:param pr2_world_copy: Existing isolated native PR2 world."""
    world = pr2_world_copy
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    robot.mobile_base.full_body_controlled = False
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    robot.get_torso().get_joint_state_by_type(TorsoState.HIGH).apply_to(world)
    robot.set_root_pose(Pose.from_xyz_rpy(0.82, 0, 0, reference_frame=world.root))
    BodySpecification.box(
        ManipulationBody.TABLE,
        Scale(0.5, 0.5, 0.04),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 0, 0.8),
    ).spawn(world)
    BodySpecification.box(
        ManipulationBody.OBJECT,
        Scale(0.06, 0.06, 0.14),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 0, 0.89),
        connection_specification=Connection6DoFSpecification(),
    ).spawn(world)
    table = world.get_body_by_name(ManipulationBody.TABLE)
    with world.modify_world():
        world.add_semantic_annotation(Table(root=table))
    body = world.get_body_by_name(ManipulationBody.OBJECT)
    return SupportedManipulation(
        Context(world=world, robot=robot),
        body,
        table,
        GraspDescription.robot_relative_default(
            robot.left_arm.end_effector, body.global_pose, body
        ),
    )


# %% scoped pair policy
@pytest.mark.parametrize("fail", [False, True])
def test_contact_scope_restores_caller_rules(
    supported_manipulation: SupportedManipulation,
    fail: bool,
) -> None:
    """
    Successful and failed contact scopes preserve the caller's complete policy.
    """
    scene = supported_manipulation
    world = scene.context.world
    manager = world.collision_manager
    policy = ManipulationContactPolicy(
        scene.body,
        scene.context.robot.left_arm.end_effector.bodies_with_collision,
        scene.body.global_pose,
    )
    previous = AvoidCollisionBetweenGroups(
        body_group_a=[scene.context.robot.right_arm.end_effector.root],
        body_group_b=[scene.table],
        buffer_zone_distance=0.23,
    )
    manager.add_temporary_rule(previous)
    manager.update_collision_matrix()
    defaults = list(manager.default_rules)
    ignored = list(manager.ignore_collision_rules)
    checks = set(manager.collision_matrix.collision_checks)
    with (
        pytest.raises(ValueError) if fail else TemporaryCollisionScope(world).activate()
    ):
        with policy.scope(world).activate():
            assert manager.temporary_rules[0] is previous
            assert (
                manager.get_buffer_zone_distance(
                    scene.context.robot.right_arm.end_effector.root, scene.table
                )
                == previous.buffer_zone_distance
            )
            if fail:
                raise ValueError()
    assert manager.temporary_rules == [previous]
    assert manager.default_rules == defaults
    assert manager.ignore_collision_rules == ignored
    assert manager.collision_matrix.collision_checks == checks


def test_contact_policy_retains_unrelated_gripper_avoidance(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    Only the selected gripper-to-object pair is removed from collision checks.
    """
    scene = supported_manipulation
    world = scene.context.world
    gripper = scene.context.robot.left_arm.end_effector
    other_gripper = scene.context.robot.right_arm.end_effector
    policy = ManipulationContactPolicy(
        scene.body, gripper.bodies_with_collision, scene.body.global_pose
    )
    with policy.scope(world).activate():
        checks = world.collision_manager.collision_matrix.collision_checks
        assert (
            CollisionCheck.create_and_validate(gripper.root, scene.body) not in checks
        )
        assert CollisionCheck.create_and_validate(gripper.root, scene.table) in checks
        assert (
            CollisionCheck.create_and_validate(other_gripper.root, scene.body) in checks
        )
        assert (
            world.collision_manager.get_buffer_zone_distance(gripper.root, scene.table)
            == policy.gripper_clearance
        )


def test_support_lookup_respects_translated_rotated_reference_frames(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    A frame-relative target identifies the same physical support as its world pose.
    """
    scene = supported_manipulation
    world = scene.context.world
    BodySpecification.box(
        ManipulationBody.REFERENCE,
        Scale(0.01, 0.01, 0.01),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(4, 5, 0, yaw=1.1),
    ).spawn(world)
    reference = world.get_body_by_name(ManipulationBody.REFERENCE)
    target = world.transform(scene.body.global_pose, reference)
    policy = ManipulationContactPolicy(
        scene.body,
        scene.context.robot.left_arm.end_effector.bodies_with_collision,
        target,
    )
    assert policy.supporting_bodies(world) == [scene.table]


def test_contact_policy_rebinds_validation_world_entities(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    Candidate validation applies policy only to bodies in its isolated world.
    """
    scene = supported_manipulation
    location = reachability_location(scene.body, scene.context, Arms.LEFT, scene.grasp)
    validator = location.validators[0]
    assert isinstance(validator, AreReachableBy)
    context = PoseValidator.copy_context_for_validation(scene.context)
    scope = validator.contact_policy.scope(context.world)
    for rule in scope.rules:
        assert all(
            part._world is context.world
            for part in rule.body_group_a + rule.body_group_b
        )
    assert scene.context.world.collision_manager.temporary_rules == []


# %% real native controller regressions
def test_remote_placement_serializes_only_native_controller_goals(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    The remote controller receives a native goal independent of client plan state.
    """
    scene = supported_manipulation
    action = execute_single(
        PlaceAction(scene.body, scene.body.global_pose, Arms.LEFT),
        context=scene.context,
    )
    with real_robot:
        action.notify()
        [approach, _, _] = action.plan.get_nodes_by_designator_type(MovePlacementMotion)
        goal = approach.motion.motion_chart
    assert type(goal) is CartesianPose
    tracker = WorldEntityWithIDKwargsTracker.from_world(scene.context.world)
    restored = from_json(to_json(goal), **tracker.create_kwargs())
    assert type(restored) is CartesianPose


def test_held_object_keeps_support_contact_distinct_from_gripper_clearance(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    Attachment must not reclassify payload geometry as a gripper link.
    """
    scene = supported_manipulation
    end_effector = scene.context.robot.left_arm.end_effector
    execute_single(
        AttachNode(body=scene.body, new_parent=end_effector.tool_frame),
        context=scene.context,
    ).perform()
    policy = ManipulationContactPolicy(
        scene.body, end_effector.bodies_with_collision, scene.body.global_pose
    )
    with policy.scope(scene.context.world).activate():
        manager = scene.context.world.collision_manager
        assert manager.get_buffer_zone_distance(scene.body, scene.table) == 0.0
        assert (
            manager.get_violated_distance(scene.body, scene.table)
            == -policy.contact_tolerance
        )


def test_pickup_and_park_preserve_intended_contact(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    A normal pickup lifts and parks the object with collision avoidance enabled.
    """
    scene = supported_manipulation
    initial_height = float(scene.body.global_pose.z)
    plan = sequential(
        [PickUpAction(scene.body, Arms.LEFT, scene.grasp), ParkArmsAction(Arms.BOTH)],
        context=scene.context,
    ).plan
    trace = ContactScopeTrace(scene)
    plan.node_callbacks.append(trace)
    with simulated_robot_advanced:
        plan.perform()
    assert (
        scene.body.parent_connection.parent
        is scene.context.robot.left_arm.end_effector.tool_frame
    )
    assert float(scene.body.global_pose.z) > initial_height
    assert any(stage.attached and stage.contact_active for stage in trace.stages)
    assert trace.stages[-1] == ContactStage(True, False, True)
    assert scene.context.world.collision_manager.temporary_rules == []


def test_place_reaches_supported_pose_and_retracts(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    A normal place releases its object at the requested support pose.
    """
    scene = supported_manipulation
    source = scene.body.global_pose
    target = Pose.from_xyz_rpy(
        source.x, source.y + 0.08, source.z, reference_frame=scene.context.world.root
    )
    plan = sequential(
        [
            PickUpAction(scene.body, Arms.LEFT, scene.grasp),
            PlaceAction(scene.body, target, Arms.LEFT),
            ParkArmsAction(Arms.BOTH),
        ],
        context=scene.context,
    ).plan
    trace = ContactScopeTrace(scene)
    plan.node_callbacks.append(trace)
    with simulated_robot_advanced:
        plan.perform()
    assert scene.body.parent_connection.parent is scene.context.world.root
    np.testing.assert_allclose(
        scene.body.global_pose.to_np()[:3, 3],
        target.to_np()[:3, 3],
        atol=scene.context.motion_tolerances.default_tcp_position_threshold,
    )
    assert all(stage.support_checked for stage in trace.stages)
    assert any(stage.attached and stage.contact_active for stage in trace.stages)
    assert any(not stage.attached and stage.contact_active for stage in trace.stages)
    assert trace.stages[-1] == ContactStage(False, False, True)


def test_repeated_placement_refreshes_the_measured_attachment(
    supported_manipulation: SupportedManipulation,
) -> None:
    """
    A new approach refreshes cached tool goals after the attachment changes.
    """
    scene = supported_manipulation
    target = scene.body.global_pose
    sequence = PlacementPoseSequence(scene.grasp, scene.body, target)
    sequence.resolve(PlacementStage.APPROACH)
    initial = sequence.resolve(PlacementStage.RELEASE).to_np()
    end_effector = scene.context.robot.left_arm.end_effector
    execute_single(
        AttachNode(body=scene.body, new_parent=end_effector.tool_frame),
        context=scene.context,
    ).perform()
    with scene.context.world.modify_world():
        scene.context.world.remove_connection(scene.body.parent_connection)
        scene.context.world.add_connection(
            FixedConnection(
                parent=end_effector.tool_frame,
                child=scene.body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.03, 0.01, 0.02, yaw=0.2, reference_frame=end_effector.tool_frame
                ),
            )
        )
    np.testing.assert_array_equal(
        sequence.resolve(PlacementStage.RELEASE).to_np(), initial
    )
    sequence.resolve(PlacementStage.APPROACH)
    expected = scene.grasp.place_pose_sequence(target, scene.body)[
        PlacementStage.RELEASE
    ]
    np.testing.assert_allclose(
        sequence.resolve(PlacementStage.RELEASE).to_np(), expected.to_np(), atol=1e-12
    )
    assert not np.allclose(sequence.resolve(PlacementStage.RELEASE).to_np(), initial)
