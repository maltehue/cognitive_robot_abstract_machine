import os

import pytest

import numpy as np
import trimesh

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.pipeline.mesh_decomposition.bounding_box import (
    BoundingBoxDecomposer,
)
from semantic_digital_twin.pipeline.mesh_decomposition.box_decomposer import (
    BoxDecomposer,
)
from semantic_digital_twin.pipeline.mesh_decomposition.coacd import COACDMeshDecomposer
from semantic_digital_twin.pipeline.mesh_decomposition.vhacd import VHACDMeshDecomposer
from semantic_digital_twin.pipeline.pipeline import Pipeline
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Box, Mesh
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture(scope="function")
def jeroen_cup_world_fixture():
    stl_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "semantic_digital_twin",
        "resources",
        "stl",
    )
    world = STLParser(os.path.join(stl_dir, "jeroen_cup.stl")).parse()
    world.root.name = PrefixedName("root")
    return world


def test_coacd(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    old_collision_length = len(cup.collision.shapes)

    pipeline = Pipeline([COACDMeshDecomposer(threshold=0.2)])
    pipeline.apply(jeroen_cup_world_fixture)

    assert len(cup.collision.shapes) > old_collision_length


def test_vhacd(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    old_collision_length = len(cup.collision.shapes)

    pipeline = Pipeline([VHACDMeshDecomposer(max_convex_hulls=10)])
    pipeline.apply(jeroen_cup_world_fixture)

    assert len(cup.collision.shapes) > old_collision_length


def test_box_decomposer(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    old_collision_length = len(cup.collision.shapes)

    pipeline = Pipeline([BoxDecomposer()])

    pipeline.apply(jeroen_cup_world_fixture)

    assert len(cup.collision.shapes) > old_collision_length
    assert all([isinstance(shape, Box) for shape in cup.collision.shapes])


# %% enclosing a mesh in its bounding box


def test_bounding_box_decomposer_encloses_the_mesh_in_one_box(jeroen_cup_world_fixture):
    [cup] = jeroen_cup_world_fixture.bodies
    [visual] = cup.visual.shapes
    low, high = visual.bounds

    Pipeline([BoundingBoxDecomposer()]).apply(jeroen_cup_world_fixture)

    [enclosing_box] = cup.collision.shapes
    assert isinstance(enclosing_box, Box)
    np.testing.assert_allclose(enclosing_box.scale.to_np(), high - low)
    np.testing.assert_allclose(enclosing_box.origin.to_np()[:3, 3], (low + high) / 2)


def test_bounding_box_decomposer_keeps_a_shape_that_is_not_a_mesh(
    jeroen_cup_world_fixture,
):
    [cup] = jeroen_cup_world_fixture.bodies
    already_a_box = Box()
    cup.visual.append(already_a_box)

    Pipeline([BoundingBoxDecomposer()]).apply(jeroen_cup_world_fixture)

    assert already_a_box in cup.collision.shapes


def test_bounding_box_decomposer_does_not_load_the_geometry_of_a_glb(tmp_path):
    # Enclosing a scanned scene is the reason this step exists, and reading every
    # surface back to do it is what makes that scene unloadable.
    path = tmp_path / "surface.glb"
    trimesh.creation.box().export(path, file_type="glb")
    mesh = Mesh(filename=str(path))
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("surface"))
        world.add_kinematic_structure_entity(body)
        body.visual.append(mesh)

    Pipeline([BoundingBoxDecomposer()]).apply(world)

    assert "mesh" not in mesh.__dict__
    assert "unscaled_mesh" not in mesh.__dict__


def test_bounding_box_decomposer_places_the_box_where_the_mesh_sits(tmp_path):
    # A mesh shape placed away from its body's origin has to be enclosed where it sits,
    # not where the vertices in its file happen to be stored.
    path = tmp_path / "offset.glb"
    trimesh.creation.box().export(path, file_type="glb")
    body_T_mesh = HomogeneousTransformationMatrix.from_xyz_rpy(x=5.0, y=-2.0)
    mesh = Mesh(filename=str(path), origin=body_T_mesh)
    low, high = mesh.bounds

    [enclosing_box] = BoundingBoxDecomposer().apply_to_mesh(mesh)

    np.testing.assert_allclose(
        enclosing_box.origin.to_np()[:3, 3],
        (low + high) / 2 + body_T_mesh.to_np()[:3, 3],
    )
