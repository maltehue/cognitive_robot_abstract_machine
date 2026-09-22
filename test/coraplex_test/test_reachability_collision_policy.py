"""
Manipulation candidates obey the same collision policy as their execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot, simulated_robot_advanced
from coraplex.locations.base import Location, PoseValidator
from coraplex.locations.factories import reachability_location
from coraplex.locations.pose_validator import AreReachableBy, IsObjectReachableBy
from giskardpy.executor import Executor
from giskardpy.motion_statechart.exceptions import CollisionViolatedError
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.qp.exceptions import InfeasibleException
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.collision_checking.collision_manager import CollisionManager
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
    AvoidExternalCollisions,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale

from .test_locations import FixedPoseGenerator
from .test_pose_validator import _MoveTcpAlternativeForPr2


# %% preserved validation context
@dataclass
class CollisionPolicyRecorder(PoseValidator):
    """
    Observe the collision policy seen by a candidate validator.
    """

    observed_rules: list[list[CollisionRule]] = field(default_factory=list)
    """
    Temporary rules at each invocation, after static screening has finished.
    """

    def __call__(self, pose_candidate: Pose) -> bool:
        """:param pose_candidate: Candidate accepted after recording its policy."""
        self.observed_rules.append(list(self.world.collision_manager.temporary_rules))
        return True


@pytest.fixture
def validation_context(pr2_world_copy: World) -> Context:
    """:param pr2_world_copy: Existing isolated PR2 fixture."""
    return Context(
        world=pr2_world_copy,
        robot=pr2_world_copy.get_semantic_annotations_by_type(PR2)[0],
        motion_tolerances=MotionToleranceConfig(0.003, 0.04),
    )


@pytest.mark.parametrize("avoidance", [False, True])
@pytest.mark.parametrize("alternative", [False, True])
def test_reachability_chart_matches_the_active_motion_policy(
    validation_context: Context, avoidance: bool, alternative: bool
) -> None:
    """
    Collision avoidance runs beside the complete sequence at execution precision.
    """
    tip = validation_context.robot.left_arm.end_effector.tool_frame
    if alternative:
        validation_context.alternative_motion_mappings = [_MoveTcpAlternativeForPr2]
    validator = AreReachableBy([tip.global_pose], tip, context=validation_context)
    environment = simulated_robot_advanced if avoidance else simulated_robot
    with environment:
        chart = validator.create_msc()
    assert len(chart.get_nodes_by_type(ExternalCollisionAvoidance)) == int(avoidance)
    sequence = chart.get_nodes_by_type(Sequence)[0]
    assert all(isinstance(task, CartesianPose) for task in sequence.nodes)
    task = sequence.nodes[0]
    assert task.translation_threshold == 0.003
    assert task.orientation_threshold == 0.04


def test_location_restores_rules_before_validators_and_preserves_tolerances(
    validation_context: Context,
) -> None:
    """
    Static base clearance must not replace the manipulation contact policy.
    """
    context = validation_context
    rule = AvoidExternalCollisions(
        robot=context.robot, buffer_zone_distance=0.023, violated_distance=0.012
    )
    context.world.collision_manager.add_temporary_rule(rule)
    recorder = CollisionPolicyRecorder()
    candidate = context.robot.root.global_pose
    location = Location(context, candidate, FixedPoseGenerator([candidate]), [recorder])
    assert len(list(location)) == 1
    observed = recorder.observed_rules[0]
    assert len(observed) == 1
    assert type(observed[0]) is AvoidExternalCollisions
    assert observed[0].violated_distance == 0.012
    assert observed[0].buffer_zone_distance == 0.023
    assert observed[0].robot is recorder.robot
    assert recorder.context.motion_tolerances == context.motion_tolerances
    assert context.world.collision_manager.temporary_rules == [rule]


def test_factory_preserves_motion_tolerances(validation_context: Context) -> None:
    """
    The factory must pass customized native task accuracy into its validator.
    """
    tip = validation_context.robot.left_arm.end_effector.tool_frame
    location = reachability_location(tip.global_pose, validation_context, Arms.LEFT)
    assert (
        location.validators[0].context.motion_tolerances
        == validation_context.motion_tolerances
    )


def test_location_preserves_live_default_and_ignore_rules(
    validation_context: Context,
) -> None:
    """
    The copied world must retain clearances lost by model-history serialization.
    """
    context = validation_context
    source_manager = context.world.collision_manager
    with context.world.modify_world():
        source_manager.add_default_rule(
            AvoidExternalCollisions(
                robot=context.robot, buffer_zone_distance=0.17, violated_distance=0.03
            )
        )
        source_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(
                body_group_a=list(
                    context.robot.left_arm.end_effector.bodies_with_collision
                ),
                body_group_b=list(
                    context.robot.right_arm.end_effector.bodies_with_collision
                ),
            )
        )
    recorder = CollisionPolicyRecorder()
    candidate = context.robot.root.global_pose
    location = Location(context, candidate, FixedPoseGenerator([candidate]), [recorder])
    assert len(list(location)) == 1
    manager = recorder.world.collision_manager
    expected = [
        (rule.buffer_zone_distance, rule.violated_distance)
        for rule in source_manager.default_rules
        if isinstance(rule, AvoidExternalCollisions)
    ]
    actual = [
        (rule.buffer_zone_distance, rule.violated_distance)
        for rule in manager.default_rules
        if isinstance(rule, AvoidExternalCollisions)
    ]
    assert actual == expected
    assert actual[-1] == (0.17, 0.03)
    ignored = manager.ignore_collision_rules[-1]
    original = source_manager.ignore_collision_rules[-1]
    assert type(ignored) is AllowCollisionBetweenGroups
    assert [body.id for body in ignored.body_group_a] == [
        body.id for body in original.body_group_a
    ]
    assert all(body._world is recorder.world for body in ignored.body_group_a)
    assert all(body._world is context.world for body in original.body_group_a)


@pytest.mark.parametrize("single_grasp", [False, True])
def test_object_reachability_preserves_the_active_validation_context(
    validation_context: Context, monkeypatch: pytest.MonkeyPatch, single_grasp: bool
) -> None:
    """
    Object predicates use the same isolated policy and precision as locations.
    """
    context = validation_context
    BodySpecification.box(
        "reachability_object",
        Scale(0.06, 0.06, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(1, 0, 0.8),
    ).spawn(context.world)
    body = context.world.get_body_by_name("reachability_object")
    with context.world.modify_world():
        context.world.collision_manager.add_default_rule(
            AvoidExternalCollisions(robot=context.robot, buffer_zone_distance=0.17)
        )
    context.world.collision_manager.add_temporary_rule(
        AvoidExternalCollisions(robot=context.robot, violated_distance=0.012)
    )
    captured: list[Context] = []

    def capture(validator: AreReachableBy) -> bool:
        """:param validator: Final native validator built by either predicate branch."""
        captured.append(validator.context)
        return True

    monkeypatch.setattr(AreReachableBy, "__call__", capture)
    predicate = IsObjectReachableBy(
        arm=Arms.LEFT,
        object_designator=body,
        context=context,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            context.robot.left_arm.end_effector,
        ),
        as_single_grasp=single_grasp,
    )
    assert predicate()
    assert len(captured) == 1
    copied = captured[0]
    assert copied.world is not context.world
    assert copied.robot._world is copied.world
    assert copied.motion_tolerances == context.motion_tolerances
    assert copied.world.collision_manager.default_rules[-1].buffer_zone_distance == 0.17
    assert len(copied.world.collision_manager.temporary_rules) == 1
    assert copied.world.collision_manager.temporary_rules[0].violated_distance == 0.012


def test_static_screening_failure_restores_rules(
    validation_context: Context, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    An unexpected collision-detector error cannot leave screening rules installed.
    """
    context = validation_context
    context.world.collision_manager.add_temporary_rule(
        AvoidExternalCollisions(robot=context.robot, violated_distance=0.012)
    )
    recorder = CollisionPolicyRecorder()
    candidate = context.robot.root.global_pose
    location = Location(context, candidate, FixedPoseGenerator([candidate]), [recorder])

    def fail(manager: CollisionManager) -> None:
        """:param manager: Test-world manager whose static query fails."""
        raise ValueError("collision detector unavailable")

    monkeypatch.setattr(CollisionManager, "compute_collisions", fail)
    with pytest.raises(ValueError, match="collision detector unavailable"):
        list(location)
    restored = recorder.world.collision_manager.temporary_rules
    assert len(restored) == 1
    assert restored[0].violated_distance == 0.012


def test_static_collision_rejection_restores_rules(validation_context: Context) -> None:
    """
    Rejecting an occupied base pose restores its policy without running validators.
    """
    context = validation_context
    base = context.robot.root
    BodySpecification.box(
        "base_obstruction",
        Scale(1, 1, 0.5),
        parent_T_self=(
            base.global_transform
            @ HomogeneousTransformationMatrix.from_xyz_rpy(z=0.2, reference_frame=base)
        ),
    ).spawn(context.world)
    context.world.collision_manager.add_temporary_rule(
        AvoidExternalCollisions(robot=context.robot, violated_distance=0.012)
    )
    recorder = CollisionPolicyRecorder()
    location = Location(
        context, base.global_pose, FixedPoseGenerator([base.global_pose]), [recorder]
    )
    assert list(location) == []
    assert recorder.observed_rules == []
    restored = recorder.world.collision_manager.temporary_rules
    assert len(restored) == 1
    assert restored[0].violated_distance == 0.012


# %% native execution and resource lifetime
def test_reachable_pose_is_rejected_when_its_gripper_collides(
    validation_context: Context,
) -> None:
    """
    A geometric IK success is insufficient, but permitted contact remains usable.
    """
    context = validation_context
    tip = context.robot.left_arm.end_effector.tool_frame
    target = (
        tip.global_transform
        @ HomogeneousTransformationMatrix.from_xyz_rpy(x=-0.05, reference_frame=tip)
    ).to_pose()
    BodySpecification.box(
        "reachability_obstacle",
        Scale(0.1, 0.1, 0.1),
        parent_T_self=tip.global_transform,
    ).spawn(context.world)
    obstacle = context.world.get_body_by_name("reachability_obstacle")
    validator = AreReachableBy([target], tip, context=context)
    initial_state = context.world.state._data.copy()
    consumer_count = len(context.world.collision_manager.collision_consumers)

    with simulated_robot:
        assert validator()
    with simulated_robot_advanced:
        assert not validator()
    assert len(context.world.collision_manager.collision_consumers) == consumer_count
    np.testing.assert_array_equal(context.world.state._data, initial_state)

    with context.world.modify_world():
        context.world.collision_manager.add_ignore_collision_rule(
            AllowCollisionBetweenGroups(
                body_group_a=list(context.robot.bodies_with_collision),
                body_group_b=[obstacle],
            )
        )
    with simulated_robot_advanced:
        assert validator()
    assert len(context.world.collision_manager.collision_consumers) == consumer_count


@pytest.mark.parametrize("during_compile", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError(),
        CollisionViolatedError([], []),
        InfeasibleException(),
        ValueError("invalid model"),
    ],
)
def test_candidate_failure_releases_collision_consumers(
    validation_context: Context,
    monkeypatch: pytest.MonkeyPatch,
    during_compile: bool,
    failure: Exception,
) -> None:
    """
    Expected infeasibility rejects candidates; programming errors remain visible.
    """
    tip = validation_context.robot.left_arm.end_effector.tool_frame
    validator = AreReachableBy([tip.global_pose], tip, context=validation_context)
    consumers = validation_context.world.collision_manager.collision_consumers
    initial_count = len(consumers)

    def fail_compile(executor: Executor, chart: MotionStatechart) -> None:
        """:param executor: Compiler registering a consumer before its failure.
        :param chart: Chart whose setup cannot complete.
        """
        executor.context.external_collision_manager
        raise failure

    def fail_tick(executor: Executor, timeout: int) -> None:
        """:param executor: Compiled executor whose integration fails.
        :param timeout: Existing finite controller tick budget.
        """
        raise failure

    if during_compile:
        monkeypatch.setattr(Executor, "compile", fail_compile)
    else:
        monkeypatch.setattr(Executor, "tick_until_end", fail_tick)
    with simulated_robot_advanced:
        if isinstance(failure, ValueError):
            with pytest.raises(ValueError, match="invalid model"):
                validator()
        else:
            assert not validator()
    assert len(consumers) == initial_count
