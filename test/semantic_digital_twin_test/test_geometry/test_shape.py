import math
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import (
    Box,
    Color,
    Cylinder,
    Mesh,
    Scale,
    Sphere,
    Texture,
)
from semantic_digital_twin.world_description.mesh_file_storage import MeshFileStorage
from semantic_digital_twin.world_description.world_entity import Body


def test_recenter_origin_centers_bounding_box():
    # A non-planar point cloud whose bounding box is offset from the origin.
    mesh = Mesh.from_3d_points(
        points_3d=[
            Point3(0, 0, 0),
            Point3(2, 0, 0),
            Point3(0, 4, 0),
            Point3(0, 0, 6),
        ]
    )
    bounding_box = mesh.local_frame_bounding_box
    expected_center = np.array(
        [
            (bounding_box.min_x + bounding_box.max_x) / 2,
            (bounding_box.min_y + bounding_box.max_y) / 2,
            (bounding_box.min_z + bounding_box.max_z) / 2,
        ]
    )

    mesh.recenter_origin()

    np.testing.assert_allclose(mesh.origin.to_position().to_np()[:3], -expected_center)


def test_recenter_origin_preserves_existing_rotation():
    # Recentering only moves the origin's translation; a pre-existing rotation must
    # survive so the shape is not silently re-oriented.
    mesh = Mesh.from_3d_points(
        points_3d=[
            Point3(0, 0, 0),
            Point3(2, 0, 0),
            Point3(0, 4, 0),
            Point3(0, 0, 6),
        ]
    )
    mesh.origin = HomogeneousTransformationMatrix.from_xyz_rpy(0, 0, 0, 0, 0, np.pi / 2)
    expected_rotation = mesh.origin.to_rotation_matrix().to_np()

    mesh.recenter_origin()

    np.testing.assert_allclose(
        mesh.origin.to_rotation_matrix().to_np(), expected_rotation, atol=1e-12
    )


def test_shape():
    mesh = Mesh.from_ply_file(
        ply_file_path=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "ply",
            "chair.ply",
        ),
        texture_file_path=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "ply",
            "chair_texture.png",
        ),
    )
    assert mesh.filename.startswith("/tmp/")
    assert mesh.filename.endswith(".obj")
    assert len(mesh.mesh.visual.uv) == 8527


def test_mesh_color_survives_serialization(tmp_path):
    """
    Per-vertex mesh color survives the to_json/from_json round-trip.

    Color travels inside the serialized geometry (re-exported as OBJ, which the
    collision loader and visualizer can read), so a receiver renders it without needing
    the original mesh file.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(source.vertices), 1))

    mesh = Mesh.from_trimesh(mesh=source, directory=tmp_path, file_type="ply")
    restored = Mesh.from_json(mesh.to_json())

    assert restored.filename.endswith(".obj")
    assert (restored.mesh.visual.vertex_colors[:, :3] == [200, 50, 50]).all()


def test_mesh_color_is_lost_without_color_preserving_format(tmp_path):
    """
    A format that cannot store per-vertex color (STL) drops it on export.

    The contrast to the PLY round-trip: without a color-preserving format the color
    is lost.
    """
    source = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(source.vertices), 1))

    mesh = Mesh.from_trimesh(mesh=source, directory=tmp_path, file_type="stl")

    assert not (mesh.mesh.visual.vertex_colors[:, :3] == [200, 50, 50]).all()


# %% where an exported mesh file is written


def test_exported_mesh_gets_a_directory_of_its_own():
    """
    An export writes into a directory holding nothing else, so the material file trimesh
    writes beside the mesh belongs to that mesh alone and a consumer resolving the
    material relative to the mesh finds the right one.
    """
    mesh = Mesh.from_ply_file(
        ply_file_path=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "ply",
            "chair.ply",
        ),
        texture_file_path=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "ply",
            "chair_texture.png",
        ),
    )

    mesh_directory = Path(mesh.filename).parent
    assert mesh_directory.parent.name.startswith(MeshFileStorage.root_prefix)
    assert {path.suffix for path in mesh_directory.iterdir()} == {
        ".obj",
        ".mtl",
        ".png",
    }


def test_exported_mesh_basenames_are_unique(tmp_path):
    """
    Two exports never share a file name, because a consumer identifies a mesh by that
    name and would otherwise treat the second mesh as the first.
    """
    first = Mesh.from_trimesh(
        mesh=trimesh.creation.box(extents=(1.0, 1.0, 1.0)), directory=tmp_path
    )
    second = Mesh.from_trimesh(
        mesh=trimesh.creation.box(extents=(2.0, 2.0, 2.0)), directory=tmp_path
    )

    assert Path(first.filename).stem != Path(second.filename).stem


def test_explicit_directory_overrides_session_root(tmp_path):
    mesh = Mesh.from_trimesh(
        mesh=trimesh.creation.box(extents=(1.0, 1.0, 1.0)), directory=tmp_path
    )

    assert Path(mesh.filename).parent.parent == tmp_path
    assert MeshFileStorage.root_prefix not in mesh.filename


def coplanar_triangles_with_shared_positions() -> trimesh.Trimesh:
    """
    Two triangles that repeat two vertex positions, each triangle in its own color.

    Welding the duplicate positions would merge vertices carrying different colors, so
    this geometry distinguishes a faithful reload from a processed one.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.array([[0, 1, 2], [3, 4, 5]]), process=False
    )
    mesh.visual.vertex_colors = np.array(
        [[200, 50, 50, 255]] * 3 + [[50, 50, 200, 255]] * 3
    )
    return mesh


def test_vertices_sharing_a_position_keep_their_own_colors(tmp_path):
    """
    Reloading a mesh must not weld vertices that share a position, because welding
    collapses their differing colors into one.
    """
    source = coplanar_triangles_with_shared_positions()

    mesh = Mesh.from_trimesh(mesh=source, directory=tmp_path, file_type="obj")

    np.testing.assert_array_equal(
        np.asarray(mesh.mesh.visual.vertex_colors),
        np.asarray(source.visual.vertex_colors),
    )


def test_per_vertex_colors_survive_serialization_unwelded(tmp_path):
    """
    Every per-vertex color survives ``to_json``/``from_json``, not just the subset that
    happens to remain after duplicate positions are merged.
    """
    source = coplanar_triangles_with_shared_positions()
    mesh = Mesh.from_trimesh(mesh=source, directory=tmp_path, file_type="obj")

    restored = Mesh.from_json(mesh.to_json())

    np.testing.assert_array_equal(
        np.asarray(restored.mesh.visual.vertex_colors),
        np.asarray(source.visual.vertex_colors),
    )


def test_serialization_reproduces_the_same_mesh():
    """
    A deserialized mesh is the mesh that was serialized, not a differently tessellated
    reading of the same file: same vertices, same faces, and the same answer to whether
    it bounds a volume.
    """
    original = Mesh(
        filename=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "stl",
            "milk.stl",
        )
    )

    restored = Mesh.from_json(original.to_json())

    np.testing.assert_allclose(
        np.asarray(restored.mesh.vertices),
        np.asarray(original.mesh.vertices),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(restored.mesh.faces), np.asarray(original.mesh.faces)
    )
    assert restored.mesh.is_volume == original.mesh.is_volume


def test_serialization_preserves_watertightness_of_a_closed_mesh():
    """
    Volume and boolean operations need a watertight mesh, so a closed mesh must still
    bound a volume after a round-trip.
    """
    original = Mesh(
        filename=os.path.join(
            Path(files("semantic_digital_twin")).parent.parent,
            "resources",
            "stl",
            "milk.stl",
        )
    )
    assert original.mesh.is_volume

    restored = Mesh.from_json(original.to_json())

    assert restored.mesh.is_volume
    np.testing.assert_allclose(restored.mesh.volume, original.mesh.volume, rtol=1e-6)


def test_texture_defaults():
    texture = Texture(file_path="/textures/wood.png")

    assert texture.repeat == (1.0, 1.0)
    assert texture.uniform is False


def test_texture_survives_serialization():
    """
    A texture's fields survive the to_json/from_json round-trip, so a receiver renders
    the same tiling as the sender without needing the original scene.
    """
    texture = Texture(file_path="/textures/wood.png", repeat=(2.0, 3.0), uniform=True)

    restored = from_json(to_json(texture))

    assert restored == texture


def test_textured_primitive_survives_serialization():
    """
    A primitive shape carrying a texture round-trips through serialization with the
    texture intact, rather than silently collapsing to its flat color.
    """
    box = Box(
        scale=Scale(1.0, 1.0, 1.0), texture=Texture(file_path="/textures/marble.png")
    )

    restored = Box.from_json(box.to_json())

    assert restored.texture == box.texture
    assert restored == box


# %% the volume a shape encloses


def test_box_volume():
    assert Box(scale=Scale(0.5, 2.0, 3.0)).volume == pytest.approx(3.0)


def test_sphere_volume():
    assert Sphere(radius=2.0).volume == pytest.approx(4.0 / 3.0 * math.pi * 8.0)


def test_cylinder_volume():
    """
    A cylinder's volume follows from the circle its width spans, not from the polygon
    its mesh approximates that circle with.
    """
    cylinder = Cylinder(width=2.0, height=3.0)

    assert cylinder.volume == pytest.approx(math.pi * 3.0)
    assert cylinder.volume > cylinder.mesh.volume


def test_mesh_volume(tmp_path):
    source = trimesh.creation.box(extents=(1.0, 2.0, 4.0))

    mesh = Mesh.from_trimesh(mesh=source, directory=tmp_path, file_type="stl")

    assert mesh.volume == pytest.approx(8.0)


# %% the units a mesh file declares


def collada_fixture_path(file_name: str) -> str:
    """
    :param file_name: The name of the COLLADA file inside the collada resources.
    :return: The absolute path of that file.
    """
    return os.path.join(
        Path(files("semantic_digital_twin")).parent.parent,
        "resources",
        "collada",
        file_name,
    )


def test_mesh_declaring_centimeters_loads_in_meters():
    """
    A mesh file stating that its coordinates are centimeters loads at its real size.

    The world is in meters throughout, so a file measuring its cube as 100 across in
    centimeters must arrive as a cube of one meter.
    """
    mesh = Mesh(filename=collada_fixture_path("centimeter_cube.dae"))

    assert mesh.mesh.extents == pytest.approx([1.0, 1.0, 1.0])


def test_mesh_declaring_no_units_is_taken_as_meters():
    """
    A mesh file that states no units is read as it is written.

    Without a declaration there is nothing to convert from, so the coordinates are
    already the meters the world expects.
    """
    mesh = Mesh(filename=collada_fixture_path("unitless_cube.dae"))

    assert mesh.mesh.extents == pytest.approx([100.0, 100.0, 100.0])


def test_declared_units_compose_with_the_meshs_own_scale():
    """
    A mesh keeps scaling by its own :attr:`~Mesh.scale` on top of the file's units.

    The two are independent: the file says what its numbers mean, the shape says how
    much to resize the result.
    """
    mesh = Mesh(
        filename=collada_fixture_path("centimeter_cube.dae"), scale=Scale(2, 2, 2)
    )

    assert mesh.mesh.extents == pytest.approx([2.0, 2.0, 2.0])


def test_stl_without_unit_metadata_loads_unchanged(tmp_path):
    """
    An STL carries no unit metadata at all, which must not be mistaken for a conversion
    request.
    """
    source = trimesh.creation.box(extents=(1.0, 2.0, 4.0))

    mesh = Mesh.from_trimesh(mesh=source, directory=tmp_path, file_type="stl")

    assert mesh.mesh.extents == pytest.approx([1.0, 2.0, 4.0])


# %% expressing a shape's mesh in another frame


def test_mesh_in_frame_applies_the_owning_bodys_world_transform():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("obstacle"))
        world.add_connection(
            FixedConnection(
                root,
                child=obstacle,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=2.0, y=0.0, z=0.0, reference_frame=root
                ),
            )
        )
        shape = Box(scale=Scale(1.0, 1.0, 1.0))
        obstacle.collision.append(shape)

    world_mesh = shape.mesh_in_frame(root)

    np.testing.assert_allclose(
        world_mesh.bounds, shape.mesh.bounds + np.array([2.0, 0.0, 0.0])
    )


def test_mesh_in_frame_in_the_shapes_own_frame_matches_its_local_mesh():
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root)
        obstacle = Body(name=PrefixedName("obstacle"))
        world.add_connection(FixedConnection.create_with_dofs(world, root, obstacle))
        shape = Box(scale=Scale(1.0, 1.0, 1.0))
        obstacle.collision.append(shape)

    world_mesh = shape.mesh_in_frame(obstacle)

    np.testing.assert_allclose(world_mesh.bounds, shape.mesh.bounds)


# %% json round trips


@pytest.mark.parametrize(
    "shape",
    [
        Sphere(radius=0.3),
        Cylinder(width=0.2, height=0.4),
        Box(scale=Scale(1.0, 2.0, 3.0)),
        Box(
            scale=Scale(1.0, 1.0, 1.0),
            color=Color(0.1, 0.2, 0.3, 0.4),
            texture=Texture(file_path="/textures/wood.png"),
        ),
    ],
)
def test_a_shape_survives_a_json_round_trip(shape):
    """
    Shapes are read back by the same code that writes them, so what one half spells and
    the other half looks for can drift apart with nothing else noticing.
    """
    payload = shape.to_json()

    restored = from_json(payload)

    assert restored == shape
    assert restored.to_json() == payload


# %% textures


def test_add_texture_shows_the_texture_at_full_brightness(tmp_path):
    # trimesh's SimpleMaterial defaults to a 40% grey diffuse, which every renderer
    # multiplies the texture by - leaving a textured mesh at 40% brightness.
    texture_file = tmp_path / "wood.png"
    Image.new("RGB", (2, 2), color=(200, 100, 50)).save(texture_file)
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)))

    textured = Mesh.add_texture(mesh=mesh, texture_file_path=str(texture_file))

    np.testing.assert_array_equal(
        textured.visual.material.diffuse, [255, 255, 255, 255]
    )


# %% textured meshes


def textured_glb(tmp_path) -> str:
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.new("RGB", (2, 2), color=(200, 100, 50))
        ),
    )
    path = tmp_path / "textured.glb"
    mesh.export(path, file_type="glb")
    return str(path)


def untextured_glb(tmp_path) -> str:
    path = tmp_path / "plain.glb"
    trimesh.creation.box().export(path, file_type="glb")
    return str(path)


def test_is_textured_finds_the_texture_of_a_glb(tmp_path):
    assert Mesh(filename=textured_glb(tmp_path)).is_textured


def test_is_textured_is_false_for_a_glb_without_one(tmp_path):
    assert not Mesh(filename=untextured_glb(tmp_path)).is_textured


def test_is_textured_does_not_load_the_geometry_of_a_glb(tmp_path):
    # Answering this by loading the mesh costs gigabytes on a scanned scene, for a
    # question the file answers in its first few kilobytes.
    mesh = Mesh(filename=textured_glb(tmp_path))

    mesh.is_textured

    assert "mesh" not in mesh.__dict__
    assert "unscaled_mesh" not in mesh.__dict__


def test_is_textured_agrees_with_the_loaded_mesh_for_other_formats(tmp_path):
    mesh = trimesh.creation.box()
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.new("RGB", (2, 2), color=(200, 100, 50))
        ),
    )
    path = tmp_path / "textured.obj"
    mesh.export(path, file_type="obj")

    assert Mesh(filename=str(path)).is_textured


# %% the bounds a mesh spans


def glb_placed_by_a_node_transform(tmp_path, translation) -> str:
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(),
        node_name="placed",
        transform=trimesh.transformations.translation_matrix(translation),
    )
    path = tmp_path / "placed.glb"
    scene.export(path, file_type="glb")
    return str(path)


def test_bounds_of_a_glb_matches_the_loaded_mesh(tmp_path):
    mesh = Mesh(filename=untextured_glb(tmp_path))

    np.testing.assert_allclose(mesh.bounds, mesh.mesh.bounds)


def test_bounds_does_not_load_the_geometry_of_a_glb(tmp_path):
    # Enclosing a scanned scene in boxes costs gigabytes if every surface has to be
    # read back, for a question the file answers in its first few kilobytes.
    mesh = Mesh(filename=untextured_glb(tmp_path))

    mesh.bounds

    assert "mesh" not in mesh.__dict__
    assert "unscaled_mesh" not in mesh.__dict__


def test_bounds_applies_the_shapes_scale(tmp_path):
    mesh = Mesh(filename=untextured_glb(tmp_path), scale=Scale(2.0, 3.0, 4.0))

    np.testing.assert_allclose(mesh.bounds, mesh.mesh.bounds)


def test_bounds_of_a_glb_placed_by_a_node_transform_matches_the_loaded_mesh(tmp_path):
    # The bounds a glTF accessor states are the ones its own buffer holds, which a node
    # placing that mesh elsewhere moves away from.
    mesh = Mesh(filename=glb_placed_by_a_node_transform(tmp_path, [10.0, 20.0, 30.0]))

    np.testing.assert_allclose(mesh.bounds, mesh.mesh.bounds)


def test_bounds_agrees_with_the_loaded_mesh_for_other_formats(tmp_path):
    path = tmp_path / "plain.obj"
    trimesh.creation.box().export(path, file_type="obj")
    mesh = Mesh(filename=str(path))

    np.testing.assert_allclose(mesh.bounds, mesh.mesh.bounds)
