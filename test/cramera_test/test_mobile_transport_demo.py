"""
Acceptance of semantic transport with a mobile manipulator.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from coraplex.datastructures.enums import Arms, TaskStatus
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.factories import execute_single
from coraplex.execution_environment import simulated_robot_advanced
from coraplex.plans.plan_callbacks import PlanCallback
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.motions.navigation import MoveMotion
from cramera.mobile_transport_demo import MobileTransportDemo, TransportSceneBody
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.datastructures.definitions import (
    StaticJointState,
    TorsoState,
)
from semantic_digital_twin.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import Body


# %% authoritative execution observations
@dataclass
class CarryTrajectory(PlanCallback):
    """
    Observe attachment and base motion during executed controller ticks.
    """

    robot: PR2
    """Robot performing the transport."""
    body: Body
    """
    Object whose attachment is observed.
    """

    positions: list[np.ndarray] = field(default_factory=list)
    """
    Base positions during all executed motions.
    """

    carried_positions: list[np.ndarray] = field(default_factory=list)
    """
    Base positions while the object belongs to the gripper.
    """

    avoidance_counts: set[int] = field(default_factory=set)
    """
    Avoidance goals present in the executed motion charts.
    """

    def on_motion_tick(self, statechart: MotionStatechart) -> None:
        """
        Capture actual controller states without observing hypothetical poses.

        :param statechart: Motion chart that advanced the robot.
        """
        position = self.robot.root.global_pose.to_np()[:3, 3].copy()
        self.positions.append(position)
        if (
            self.body.parent_connection.parent
            is self.robot.left_arm.end_effector.tool_frame
        ):
            self.carried_positions.append(position)
        self.avoidance_counts.add(
            len(statechart.get_nodes_by_type(ExternalCollisionAvoidance))
        )


# %% complete simulated transport
def test_mobile_demo_grasps_from_a_reachable_stance() -> None:
    """
    The native pickup controller lifts the supported object into its gripper.
    """
    demonstration = MobileTransportDemo(used_robot=PR2)
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    context = demonstration.build_context(world)
    robot = context.robot
    for arm in (robot.left_arm, robot.right_arm):
        arm.get_joint_state_by_type(StaticJointState.PARK).apply_to(world)
    robot.get_torso().get_joint_state_by_type(TorsoState.HIGH).apply_to(world)
    robot.set_root_pose(Pose.from_xyz_rpy(0.82, 0, 0, reference_frame=world.root))
    body = world.get_body_by_name(TransportSceneBody.OBJECT)
    initial_height = body.global_pose.z
    grasp = GraspDescription.robot_relative_default(
        robot.left_arm.end_effector, body.global_pose, body
    )
    plan = execute_single(PickUpAction(body, Arms.LEFT, grasp), context=context).plan

    with simulated_robot_advanced:
        plan.perform()

    assert body.parent_connection.parent is robot.left_arm.end_effector.tool_frame
    assert float(body.global_pose.z) == pytest.approx(
        float(initial_height) + grasp.manipulation_offset, abs=0.005
    )


def test_mobile_demo_carries_and_places_on_the_semantic_table() -> None:
    """
    PR2 drives with an attached object and releases it on the named surface.
    """
    demonstration = MobileTransportDemo(used_robot=PR2)
    assert demonstration.collision_avoidance is True
    world = demonstration.build_simulated_world()
    demonstration.populate_scene(world)
    context = demonstration.build_context(world)
    body = world.get_body_by_name(TransportSceneBody.OBJECT)
    trajectory = CarryTrajectory(robot=context.robot, body=body)
    plan = demonstration.build_plan(context)
    plan.node_callbacks.append(trajectory)

    with simulated_robot_advanced:
        plan.perform()

    [transport] = plan.get_nodes_by_designator_type(TransportAction)
    assert transport.status is TaskStatus.SUCCEEDED
    navigations = plan.get_nodes_by_designator_type(MoveMotion)
    assert len(navigations) == 2
    assert all(node.status is TaskStatus.SUCCEEDED for node in navigations)
    positions = np.asarray(trajectory.positions)
    carried = np.asarray(trajectory.carried_positions)
    assert np.linalg.norm(carried[-1] - carried[0]) > demonstration.table_distance / 2
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) < 0.02
    assert min(trajectory.avoidance_counts) > 0
    assert body.parent_connection.parent is world.root
    table = world.get_body_by_name(TransportSceneBody.DESTINATION)
    final_position = body.global_pose.to_np()[:3, 3]
    table_position = table.global_pose.to_np()[:3, 3]
    assert (
        abs(final_position[0] - table_position[0])
        <= (demonstration.table_size.x - demonstration.object_size.x) / 2
    )
    assert (
        abs(final_position[1] - table_position[1])
        <= (demonstration.table_size.y - demonstration.object_size.y) / 2
    )
    expected_height = (
        table_position[2]
        + (demonstration.table_size.z + demonstration.object_size.z) / 2
    )
    assert final_position[2] == pytest.approx(expected_height, abs=0.005)
    table_T_object = world.transform(body.global_pose, table).to_np()
    orientation_error = np.arccos(
        np.clip((np.trace(table_T_object[:3, :3]) - 1) / 2, -1, 1)
    )
    assert orientation_error <= context.motion_tolerances.tool_orientation_threshold
    bounds = body.collision.as_bounding_box_collection_in_frame(table).bounding_box()
    assert bounds.min_x >= -demonstration.table_size.x / 2
    assert bounds.max_x <= demonstration.table_size.x / 2
    assert bounds.min_y >= -demonstration.table_size.y / 2
    assert bounds.max_y <= demonstration.table_size.y / 2
