"""
Placement goals honor the observed object pose inside the selected gripper.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.datastructures.enums import ApproachDirection, Arms, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.locations.factories import reachability_location
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Scale


# %% object poses retained during carrying
@pytest.mark.parametrize("carry_yaw", [0.0, 1.2])
@pytest.mark.parametrize(
    "tool_T_object",
    [
        pytest.param(
            HomogeneousTransformationMatrix.from_xyz_rpy(0.03, -0.02, 0.01),
            id="translated-grasp",
        ),
        pytest.param(
            HomogeneousTransformationMatrix.from_xyz_rpy(yaw=0.4),
            id="rotated-grasp",
        ),
    ],
)
def test_place_reaches_the_object_goal_with_its_actual_grasp(
    mutable_model_world,
    carry_yaw: float,
    tool_T_object: HomogeneousTransformationMatrix,
) -> None:
    """
    The commanded tool pose places the attached body at the requested object pose.

    :param mutable_model_world: Existing apartment fixture with PR2 and movable milk.
    :param carry_yaw: Base rotation acquired while carrying the object.
    :param tool_T_object: Observed attachment transform after the grasp.
    """
    world, robot, context = mutable_model_world
    body = world.get_body_by_name("milk.stl")
    end_effector = robot.left_arm.end_effector
    with world.modify_world():
        world.remove_connection(body.parent_connection)
        world.add_connection(
            FixedConnection(
                parent=end_effector.tool_frame,
                child=body,
                parent_T_connection_expression=tool_T_object,
            )
        )
    robot.set_root_pose(
        Pose.from_xyz_rpy(0.5, 0.2, yaw=carry_yaw, reference_frame=world.root)
    )
    grasp = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
    )
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, yaw=-0.7, reference_frame=world.root)
    place = PlaceAction(body, target, Arms.LEFT)
    sequential([PickUpAction(body, Arms.LEFT, grasp), place], context=context)

    placing_motion = place._action_plan.children[1].designator
    world_T_tool_goal = world.transform(
        placing_motion.target, world.root
    ).to_homogeneous_matrix()
    measured_tool_T_object = world.compute_forward_kinematics(
        end_effector.tool_frame, body
    )

    np.testing.assert_allclose(
        (world_T_tool_goal @ measured_tool_T_object).to_np(),
        target.to_np(),
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("held", [False, True])
def test_reachability_validates_the_same_pick_or_place_sequence(
    mutable_model_world, held: bool
) -> None:
    """
    Reachability uses the observed attachment for placement and nominal unheld reach.

    :param mutable_model_world: Existing PR2 apartment fixture with movable milk.
    :param held: Whether the selected arm already holds the object.
    """
    world, robot, context = mutable_model_world
    body = world.get_body_by_name("milk.stl")
    end_effector = robot.left_arm.end_effector
    grasp = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
    )
    if held:
        with world.modify_world():
            world.remove_connection(body.parent_connection)
            world.add_connection(
                FixedConnection(
                    parent=end_effector.tool_frame,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                        0.03, -0.02, 0.01, yaw=0.4
                    ),
                )
            )
    target = Pose.from_xyz_rpy(1.2, 0.4, 0.9, yaw=-0.7, reference_frame=world.root)

    location = reachability_location(target, context, Arms.LEFT, grasp)
    expected = (
        grasp.place_pose_sequence(target, body) if held else grasp.pose_sequence(target)
    )

    np.testing.assert_allclose(
        [pose.to_np() for pose in location.validators[0].pose_sequence],
        [pose.to_np() for pose in expected],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize("body_count", [0, 2])
def test_placement_requires_a_unique_implicit_object(
    mutable_model_world, body_count: int
) -> None:
    """
    An omitted object cannot silently select one of several tool attachments.

    :param mutable_model_world: Existing PR2 apartment fixture.
    :param body_count: Number of bodies attached below the selected tool frame.
    """
    world, robot, _ = mutable_model_world
    end_effector = robot.left_arm.end_effector
    for index in range(body_count):
        body = BodySpecification.box(
            f"held_object_{index}", Scale(0.03, 0.03, 0.03)
        ).spawn(world)
        with world.modify_world():
            world.move_branch_with_fixed_connection(body, end_effector.tool_frame)
    grasp = GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, end_effector
    )

    with pytest.raises(ValueError):
        grasp.place_pose_sequence(Pose(reference_frame=world.root))


# %% semantic targets expressed in surface frames
def test_reachability_samples_world_positions_for_a_transformed_target_frame(
    mutable_model_world,
) -> None:
    """
    A surface-relative target samples the same base poses as its world equivalent.

    :param mutable_model_world: Existing PR2 world used for both equivalent searches.
    """
    world, robot, context = mutable_model_world
    reference = BodySpecification.box(
        "rotated_placement_reference",
        Scale(0.2, 0.2, 0.04),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
            3.5, -1.0, 0.8, yaw=0.7
        ),
    ).spawn(world)
    target = Pose.from_xyz_rpy(0.12, -0.04, 0.09, yaw=-0.3, reference_frame=reference)
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.end_effector,
    )
    world_target = world.transform(target, world.root)
    expected = reachability_location(world_target, context, Arms.LEFT, grasp)

    actual = reachability_location(target, context, Arms.LEFT, grasp)

    np.testing.assert_allclose(
        [pose.to_np() for pose in actual.generator],
        [pose.to_np() for pose in expected.generator],
        atol=1e-12,
        rtol=0,
    )


def test_reachability_preserves_the_manipulation_target_frame(
    mutable_model_world,
) -> None:
    """
    Normalizing the base map keeps manipulation goals in their supplied frame.

    :param mutable_model_world: Existing PR2 apartment with a movable reference body.
    """
    world, robot, context = mutable_model_world
    body = world.get_body_by_name("milk.stl")
    target = Pose.from_xyz_rpy(0.1, 0.2, 0.3, reference_frame=body)
    grasp = GraspDescription(
        ApproachDirection.FRONT,
        VerticalAlignment.NoAlignment,
        robot.left_arm.end_effector,
    )

    location = reachability_location(target, context, Arms.LEFT, grasp)

    assert all(
        pose.reference_frame is body for pose in location.validators[0].pose_sequence
    )
