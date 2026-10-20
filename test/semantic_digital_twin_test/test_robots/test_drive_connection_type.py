from enum import StrEnum

import numpy as np
import pytest

from semantic_digital_twin.robots import exceptions as robot_errors
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.connections import (
    PrismaticConnectionSpecification,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Vector3
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.robots.pr2 import PR2, PR2MobileBase
from semantic_digital_twin.robots.stretch import Stretch, StretchMobileBase
from semantic_digital_twin.robots.tiago import Tiago, TiagoMujoco
from semantic_digital_twin.world_description.connections import (
    DifferentialDrive,
    FixedConnection,
    OmniDrive,
)

# %% drive type resolved from the robot's mobile base


def test_omni_drive_robot_reports_omni_drive():
    assert PR2.get_drive_connection_type() is OmniDrive


def test_differential_drive_robots_report_differential_drive():
    assert Tiago.get_drive_connection_type() is DifferentialDrive
    assert TiagoMujoco.get_drive_connection_type() is DifferentialDrive
    assert Stretch.get_drive_connection_type() is DifferentialDrive


# %% robot without a mobile base


def test_robot_without_mobile_base_uses_fixed_connection():
    assert MinimalRobot.get_drive_connection_type() is FixedConnection


# %% mobile base exposes its bound drive


def test_mobile_base_reports_its_bound_drive():
    assert PR2MobileBase.get_drive_connection_type() is OmniDrive
    assert StretchMobileBase.get_drive_connection_type() is DifferentialDrive


# %% collision geometry below a frame-only base root


class BaseBody(StrEnum):
    """
    Bodies distinguishing the rigid platform from its articulated payload.
    """

    ARTICULATED = "articulated_payload"


@pytest.fixture
def frame_only_base(world_with_two_bodies):
    """
    A geometryless base frame carries an offset platform and a movable part.
    """
    world, root, platform = world_with_two_bodies
    platform.collision = Box(
        scale=Scale(0.5, 0.75, 0.25),
        origin=HomogeneousTransformationMatrix(reference_frame=platform),
    ).as_shape_collection()
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=platform,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.125
                ),
            )
        )
    BodySpecification.box(
        BaseBody.ARTICULATED,
        Scale(4, 4, 4),
        connection_specification=PrismaticConnectionSpecification(axis=Vector3.Z()),
    ).spawn(world, parent=platform)
    return PR2MobileBase(root=root, _world=world), platform


def test_frame_only_base_uses_rigid_collision_geometry(frame_only_base):
    """
    The platform determines bounds while its articulated part is excluded.
    """
    base, platform = frame_only_base
    expected = platform.collision.as_bounding_box_collection_in_frame(
        platform._world.root
    ).bounding_box()

    np.testing.assert_allclose(
        base.bounding_box.to_array_bounds().lower, expected.to_array_bounds().lower
    )
    np.testing.assert_allclose(
        base.bounding_box.to_array_bounds().upper, expected.to_array_bounds().upper
    )
    assert base.base_radius == (expected.depth / 2 + expected.width / 2) / 2


def test_base_with_own_geometry_retains_its_existing_bounds(frame_only_base):
    """
    An existing geometric root keeps the established base-radius definition.
    """
    base, platform = frame_only_base
    base.root.collision = Box(
        scale=Scale(0.25, 0.25, 0.25),
        origin=HomogeneousTransformationMatrix(reference_frame=base.root),
    ).as_shape_collection()
    expected = base.root.collision.as_bounding_box_collection_in_frame(
        platform._world.root
    ).bounding_box()

    np.testing.assert_allclose(
        base.bounding_box.to_array_bounds().lower, expected.to_array_bounds().lower
    )
    np.testing.assert_allclose(
        base.bounding_box.to_array_bounds().upper, expected.to_array_bounds().upper
    )


def test_base_without_any_rigid_collision_geometry_reports_the_missing_geometry(
    world_with_two_bodies,
):
    """
    A frame-only base cannot silently produce invalid clearance bounds.
    """
    world, root, child = world_with_two_bodies
    with world.modify_world():
        world.add_connection(FixedConnection(parent=root, child=child))
    base = PR2MobileBase(root=root, _world=world)

    with pytest.raises(robot_errors.MissingBaseCollisionGeometry) as raised:
        base.bounding_box

    assert raised.value.root is root
