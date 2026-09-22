"""
Hypothetical base footprints retain the original collision-shape bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from coraplex.locations.navigation import NavigationPath
from semantic_digital_twin.api import BodySpecification
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Scale


# %% footprint projection
@pytest.mark.parametrize("heading", [0.4, -0.7])
def test_projected_footprint_matches_actual_rotated_collision_geometry(
    cylinder_bot_world: World, heading: float
) -> None:
    """
    Projecting a rotated payload does not inflate an already axis-aligned box.

    :param cylinder_bot_world: Existing mobile-robot fixture.
    :param heading: Hypothetical base heading used for comparison.
    """
    world = cylinder_bot_world
    robot = world.get_semantic_annotations_by_type(AbstractRobot)[0]
    payload = BodySpecification.box(
        "angled_payload",
        Scale(0.2, 1.0, 0.1),
        parent_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=0.8, yaw=0.6),
    ).spawn(world)
    with world.modify_world():
        world.move_branch_with_fixed_connection(
            branch_root=payload, new_parent=robot.root
        )
    target = Pose.from_xyz_rpy(1, -0.5, yaw=heading, reference_frame=world.root)
    initial = robot.root.global_pose.to_np().copy()
    predicted = NavigationPath(world, robot, target).bounds_at_pose(target)
    np.testing.assert_array_equal(robot.root.global_pose.to_np(), initial)

    robot.set_root_pose(target)
    actual = robot.as_bounding_box_collection_in_frame(world.root)
    assert len(predicted) == len(actual)
    for predicted_box, actual_box in zip(predicted, actual):
        np.testing.assert_allclose(
            predicted_box.to_array_bounds().lower,
            actual_box.to_array_bounds().lower,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            predicted_box.to_array_bounds().upper,
            actual_box.to_array_bounds().upper,
            atol=1e-12,
        )
