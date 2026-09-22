"""
Explicit robot namespaces stay distinct through parsing and native world merging.
"""

from __future__ import annotations

import numpy as np

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.api import RobotSpecification, WorldSpecification
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


# %% explicit parser and instance namespaces
def test_parser_prefix_applies_to_connections_and_degrees_of_freedom(
    pr2_world_copy: World,
) -> None:
    """
    A supplied namespace covers commandable joints as well as visible bodies.

    :param pr2_world_copy: Existing fixture verifying PR2 model availability.
    """
    prefix = f"second_{pr2_world_copy.name}"
    world = URDFParser.from_file(PR2.get_ros_file_path(), prefix=prefix).parse()
    assert {body.name.prefix for body in world.bodies} == {prefix}
    assert {connection.name.prefix for connection in world.connections} == {prefix}
    assert {degree.name.prefix for degree in world.degrees_of_freedom} == {prefix}


def test_duplicate_robot_models_keep_distinct_names_and_poses(
    pr2_world_copy: World,
) -> None:
    """
    Native specifications retain separate names and localization for two PR2s.

    :param pr2_world_copy: Existing fixture verifying PR2 model availability.
    """
    specifications = [
        RobotSpecification(
            semantic_annotation_type=PR2,
            prefix=f"{pr2_world_copy.name}_{index}",
            world_T_odom=HomogeneousTransformationMatrix.from_xyz_rpy(x=index * 3.0),
        )
        for index in range(2)
    ]
    world = WorldSpecification(robots=specifications).to_domain_object()
    robots = world.get_semantic_annotations_by_type(PR2)
    assert len(robots) == len(specifications)
    for robot, specification in zip(robots, specifications):
        assert robot.root.name.prefix == specification.prefix
        assert robot.root.parent_connection.parent.name.prefix == specification.prefix
        assert {body.name.prefix for body in robot.bodies} == {specification.prefix}
        assert {connection.name.prefix for connection in robot.connections} == {
            specification.prefix
        }
        np.testing.assert_allclose(
            robot.root.global_transform.to_np(), specification.world_T_odom.to_np()
        )
    names = [str(degree.name) for degree in world.degrees_of_freedom]
    assert len(names) == len(set(names))
