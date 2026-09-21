from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_digital_twin.adapters.usd.asset_library import (
    CollisionProxy,
    USDAssetLibrary,
    VertexSharing,
)
from semantic_digital_twin.adapters.usd.scene_parser import USDSceneParser
from semantic_digital_twin.adapters.usd.stage_parser import (
    RootPlacement,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)

from .usd_stages import (
    PXR_AVAILABLE,
    build_scene_stage_with_a_triangle_soup,
    build_scene_stage_with_textured_objects,
)

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)

OBJECT_NAMES = ("wall_a", "wall_b", "floor_a")


def texture_file(tmp_path, size=(64, 64)) -> str:
    path = tmp_path / "surface.png"
    Image.new("RGB", size, color=(200, 100, 50)).save(path)
    return str(path)


def written_library(tmp_path, **arguments):
    """
    Split the scanned-scene fixture into an asset library and open it again.

    :return: The source stage, the path of the world layer, and the composed library.
    """
    source = build_scene_stage_with_textured_objects(texture_file(tmp_path))
    world_layer = USDAssetLibrary(stage=source, **arguments).write(tmp_path / "library")
    return source, world_layer, Usd.Stage.Open(str(world_layer))


def world_transform(prim) -> np.ndarray:
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return np.array(matrix, dtype=float)


def placed_bodies(world) -> dict:
    """
    The world pose of every body but the root, by name - the root prim is named after
    the layer it was written into, which says nothing about the scene.
    """
    return {
        body.name.name: np.round(body.global_transform.to_np(), 9).tobytes()
        for body in world.bodies
        if body is not world.root
    }


def asset_layer(world_layer, object_name: str, layer_name: str):
    """
    :return: One layer of a written asset, read without composing it.
    """
    asset = world_layer.parent / "Assets" / object_name
    return Sdf.Layer.FindOrOpen(str(asset / f"{object_name}_{layer_name}"))


def physics_in(layer) -> list:
    """
    :return: The path of every prim spec of a layer that says anything about physics.
    """
    said = []

    def visit(path):
        spec = layer.GetObjectAtPath(path)
        if not isinstance(spec, Sdf.PrimSpec):
            return
        applied = (
            spec.GetInfo("apiSchemas").GetAddedOrExplicitItems()
            if spec.HasInfo("apiSchemas")
            else []
        )
        if any(schema.startswith("Physics") for schema in applied):
            said.append(path.pathString)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return said


def mesh_under(stage, object_name: str):
    [mesh] = [
        prim
        for prim in stage.TraverseAll()
        if prim.IsA(UsdGeom.Mesh) and object_name in prim.GetPath().pathString
    ]
    return mesh


# %% the files a library is written as


def test_write_gives_every_object_an_asset_of_its_own(tmp_path):
    _, world_layer, _ = written_library(tmp_path)

    assets = world_layer.parent / "Assets"
    assert sorted(path.name for path in assets.iterdir()) == sorted(OBJECT_NAMES)


def test_an_asset_keeps_its_geometry_and_its_material_in_separate_layers(tmp_path):
    _, world_layer, _ = written_library(tmp_path)

    asset = world_layer.parent / "Assets" / "wall_a"
    assert sorted(path.name for path in asset.iterdir() if path.is_file()) == [
        "wall_a.usda",
        "wall_a_geo.usd",
        "wall_a_look.usda",
        "wall_a_payload.usda",
        "wall_a_physics.usda",
    ]


def test_the_library_keeps_the_name_the_stage_gave_its_root(tmp_path):
    # The library stands in for the stage it was written from, so anything addressing
    # the scene by the name of its root goes on working.
    source, _, library = written_library(tmp_path)

    assert library.GetDefaultPrim().GetName() == source.GetDefaultPrim().GetName()


def test_the_world_layer_carries_no_geometry_of_its_own(tmp_path):
    # The point of the split: the file a person edits stays small enough to read.
    _, world_layer, _ = written_library(tmp_path)

    text = world_layer.read_text()
    assert "points" not in text
    assert text.count("prepend references") == len(OBJECT_NAMES)


# %% what the library composes back to


def test_the_library_places_every_object_where_the_stage_had_it(tmp_path):
    source, _, library = written_library(tmp_path)

    for name in OBJECT_NAMES:
        np.testing.assert_allclose(
            world_transform(mesh_under(library, name)),
            world_transform(mesh_under(source, name)),
            atol=1e-9,
        )


def test_the_library_carries_the_geometry_the_stage_held(tmp_path):
    source, _, library = written_library(tmp_path)

    for name in OBJECT_NAMES:
        np.testing.assert_allclose(
            UsdGeom.Mesh(mesh_under(library, name)).GetPointsAttr().Get(),
            UsdGeom.Mesh(mesh_under(source, name)).GetPointsAttr().Get(),
        )


def test_every_asset_is_a_component_deferred_behind_a_payload(tmp_path):
    _, world_layer, _ = written_library(tmp_path)

    unloaded = Usd.Stage.Open(str(world_layer), Usd.Stage.LoadNone)
    assert [prim for prim in unloaded.TraverseAll() if prim.IsA(UsdGeom.Mesh)] == []
    assert [
        Usd.ModelAPI(prim).GetKind()
        for prim in unloaded.TraverseAll()
        if Usd.ModelAPI(prim).GetKind()
    ] == ["component"] * len(OBJECT_NAMES)


# %% the textures the library holds


def test_a_texture_is_copied_beside_the_asset_that_reads_it(tmp_path):
    _, world_layer, library = written_library(tmp_path)

    copied = world_layer.parent / "Assets" / "wall_a" / "Materials" / "Textures"
    assert [path.name for path in copied.iterdir()] == ["surface.png"]

    [reader] = [
        prim
        for prim in library.TraverseAll()
        if prim.IsA(UsdShade.Shader)
        and UsdShade.Shader(prim).GetShaderId() == "UsdUVTexture"
        and "wall_a" in prim.GetPath().pathString
    ]
    asset_path = UsdShade.Shader(reader).GetInput("file").Get()
    assert asset_path.path.startswith("./")
    assert Path(asset_path.resolvedPath).samefile(copied / "surface.png")


def test_a_texture_is_downscaled_to_the_given_cap(tmp_path):
    _, world_layer, _ = written_library(tmp_path, maximum_texture_size=16)

    copied = world_layer.parent / "Assets" / "wall_a" / "Materials" / "Textures"
    with Image.open(copied / "surface.png") as image:
        assert max(image.size) == 16


def test_a_texture_below_the_cap_is_left_at_the_size_it_was(tmp_path):
    _, world_layer, _ = written_library(tmp_path, maximum_texture_size=4096)

    copied = world_layer.parent / "Assets" / "wall_a" / "Materials" / "Textures"
    with Image.open(copied / "surface.png") as image:
        assert image.size == (64, 64)


# %% the collision the library authors


def test_every_asset_carries_a_collision_proxy_the_renderer_leaves_alone(tmp_path):
    # The scanned surface is far too dense to collide against, and a guide is what
    # keeps the proxy out of the picture while a physics engine still reads it.
    _, _, library = written_library(tmp_path)

    proxies = [prim for prim in library.TraverseAll() if prim.IsA(UsdGeom.Cube)]
    assert len(proxies) == len(OBJECT_NAMES)
    assert all(
        UsdGeom.Imageable(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
        for prim in proxies
    )


def test_a_collision_proxy_encloses_the_geometry_it_stands_for(tmp_path):
    source, _, library = written_library(tmp_path)

    [proxy] = [
        prim
        for prim in library.TraverseAll()
        if prim.IsA(UsdGeom.Cube) and "wall_a" in prim.GetPath().pathString
    ]
    bounds = (
        UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide]
        )
        .ComputeWorldBound(proxy)
        .ComputeAlignedRange()
    )
    mesh_bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(mesh_under(source, "wall_a"))
        .ComputeAlignedRange()
    )

    np.testing.assert_allclose(bounds.GetMin(), mesh_bounds.GetMin(), atol=1e-9)
    np.testing.assert_allclose(bounds.GetMax(), mesh_bounds.GetMax(), atol=1e-9)


def test_collision_is_written_apart_from_the_geometry_it_stands_for(tmp_path):
    # Collision is the part of an asset that gets retuned; the geometry beside it is a
    # binary crate nobody wants to rewrite to change a box.
    _, world_layer, _ = written_library(tmp_path)

    assert physics_in(asset_layer(world_layer, "wall_a", "geo.usd")) == []
    assert physics_in(asset_layer(world_layer, "wall_a", "physics.usda")) == [
        "/wall_a/collision"
    ]


def test_a_surface_collided_against_is_said_so_apart_from_the_surface(tmp_path):
    _, world_layer, _ = written_library(
        tmp_path, collision_proxy=CollisionProxy.CONVEX_DECOMPOSITION
    )

    assert physics_in(asset_layer(world_layer, "wall_a", "geo.usd")) == []
    assert physics_in(asset_layer(world_layer, "wall_a", "physics.usda")) == [
        "/wall_a/mesh"
    ]


def test_collision_is_written_as_text_a_person_can_edit(tmp_path):
    _, world_layer, _ = written_library(tmp_path)

    physics = world_layer.parent / "Assets" / "wall_a" / "wall_a_physics.usda"
    assert physics.read_text().startswith("#usda")


def test_a_material_binding_points_inside_the_asset_it_was_written_with(tmp_path):
    # A copied relationship keeps the path it was authored with, and a path into the
    # stage the asset came from resolves to nothing once the asset stands on its own.
    _, _, library = written_library(tmp_path)

    material, _ = UsdShade.MaterialBindingAPI(
        mesh_under(library, "wall_a")
    ).ComputeBoundMaterial()

    assert material
    assert material.GetPath().pathString.endswith("/wall_a/material")


# %% reading the library back


def test_the_library_parses_into_the_scene_the_stage_described(tmp_path):
    source, world_layer, _ = written_library(tmp_path)

    from_source = USDSceneParser(stage=source, prefix="scene").parse()
    from_library = USDSceneParser.from_file(str(world_layer), prefix="scene").parse()

    assert placed_bodies(from_library) == placed_bodies(from_source)


# %% sharing the vertices loose triangles do not


@dataclass
class SplitSoup:
    """
    The triangle-soup fixture and the library it was split into.

    Both stages are held, because a prim of a stage nothing refers to any more stops
    answering the moment that stage is collected.
    """

    source: Usd.Stage
    library: Usd.Stage

    @property
    def original(self) -> UsdGeom.Mesh:
        return UsdGeom.Mesh(mesh_under(self.source, "wall_a"))

    @property
    def written(self) -> UsdGeom.Mesh:
        return UsdGeom.Mesh(mesh_under(self.library, "wall_a"))


def soup_library(tmp_path, **arguments) -> SplitSoup:
    source = build_scene_stage_with_a_triangle_soup()
    world_layer = USDAssetLibrary(stage=source, **arguments).write(tmp_path / "library")
    return SplitSoup(source=source, library=Usd.Stage.Open(str(world_layer)))


def corner_values(mesh, values, interpolation):
    """
    Spread a mesh's attribute values out to one per face corner, whichever way they were
    stored, so two meshes storing the same thing differently compare equal.
    """
    values = np.asarray(values)
    if interpolation == UsdGeom.Tokens.uniform:
        return np.repeat(values, np.array(mesh.GetFaceVertexCountsAttr().Get()), axis=0)
    if interpolation == UsdGeom.Tokens.faceVarying:
        return values
    return values[np.array(mesh.GetFaceVertexIndicesAttr().Get())]


def corner_points(mesh):
    points = np.array(mesh.GetPointsAttr().Get())
    return points[np.array(mesh.GetFaceVertexIndicesAttr().Get())]


def test_vertices_are_kept_as_authored_by_default(tmp_path):
    split = soup_library(tmp_path)

    assert len(split.written.GetPointsAttr().Get()) == len(
        split.original.GetPointsAttr().Get()
    )


def test_sharing_vertices_keeps_one_point_per_position(tmp_path):
    split = soup_library(tmp_path, vertex_sharing=VertexSharing.BY_POSITION)

    assert len(split.original.GetPointsAttr().Get()) == 6
    assert len(split.written.GetPointsAttr().Get()) == 4


def test_sharing_vertices_keeps_the_surface_it_was_given(tmp_path):
    split = soup_library(tmp_path, vertex_sharing=VertexSharing.BY_POSITION)

    np.testing.assert_array_equal(
        corner_points(split.written), corner_points(split.original)
    )


def test_sharing_vertices_keeps_a_texture_coordinate_per_corner(tmp_path):
    split = soup_library(tmp_path, vertex_sharing=VertexSharing.BY_POSITION)
    written, original = split.written, split.original

    written_st = UsdGeom.PrimvarsAPI(written).GetPrimvar("st")
    original_st = UsdGeom.PrimvarsAPI(original).GetPrimvar("st")

    assert written_st.GetInterpolation() == UsdGeom.Tokens.faceVarying
    np.testing.assert_array_equal(
        corner_values(written, written_st.Get(), written_st.GetInterpolation()),
        corner_values(original, original_st.Get(), original_st.GetInterpolation()),
    )


def test_sharing_vertices_keeps_a_flat_normal_once_per_face(tmp_path):
    # A scan gives every corner of a face the same normal, so one per face says it.
    split = soup_library(tmp_path, vertex_sharing=VertexSharing.BY_POSITION)
    written, original = split.written, split.original

    assert written.GetNormalsInterpolation() == UsdGeom.Tokens.uniform
    assert len(written.GetNormalsAttr().Get()) == 2
    np.testing.assert_array_equal(
        corner_values(
            written, written.GetNormalsAttr().Get(), written.GetNormalsInterpolation()
        ),
        corner_values(
            original,
            original.GetNormalsAttr().Get(),
            original.GetNormalsInterpolation(),
        ),
    )


def test_sharing_vertices_stops_a_renderer_subdividing_the_result(tmp_path):
    # A scan is a polygon mesh, and USD's unauthored default is catmullClark - harmless
    # while no face shares an edge, a smoothed building once they do.
    split = soup_library(tmp_path, vertex_sharing=VertexSharing.BY_POSITION)

    assert split.written.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none


# %% simulating the library


def composed_bounds(stage):
    bounds = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        .ComputeWorldBound(stage.GetDefaultPrim())
        .ComputeAlignedRange()
    )
    return np.array(bounds.GetMin()), np.array(bounds.GetMax())


def test_the_library_declares_a_scene_for_physics_to_run_in(tmp_path):
    # Without one nothing simulates until a caller authors it, which a library meant
    # to be opened and played should not require.
    _, _, library = written_library(tmp_path)

    scenes = [prim for prim in library.TraverseAll() if prim.IsA(UsdPhysics.Scene)]

    assert len(scenes) == 1


# %% where the library stands


def test_the_library_keeps_the_coordinates_it_was_authored_in_by_default(tmp_path):
    source, _, library = written_library(tmp_path)

    np.testing.assert_allclose(
        composed_bounds(library), composed_bounds(source), atol=1e-6
    )


def test_the_library_can_stand_its_scene_on_the_origin(tmp_path):
    # A scan is authored wherever it was captured, which for the innolab is 250 m from
    # its stage's origin - far enough that anything placed by hand misses it.
    _, _, library = written_library(tmp_path, root_placement=RootPlacement.SCENE_GROUND)

    low, high = composed_bounds(library)

    np.testing.assert_allclose((low + high)[:2] / 2, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(low[2], 0.0, atol=1e-6)


def test_standing_the_scene_on_the_origin_moves_every_object_together(tmp_path):
    source, _, standing = written_library(
        tmp_path, root_placement=RootPlacement.SCENE_GROUND
    )

    def separation(stage):
        return (
            world_transform(mesh_under(stage, "wall_a"))[3, :3]
            - world_transform(mesh_under(stage, "floor_a"))[3, :3]
        )

    np.testing.assert_allclose(separation(standing), separation(source), atol=1e-6)


# %% what the library is collided against


def test_the_surface_itself_can_be_collided_against_instead_of_a_box(tmp_path):
    _, _, library = written_library(
        tmp_path, collision_proxy=CollisionProxy.CONVEX_DECOMPOSITION
    )

    assert [prim for prim in library.TraverseAll() if prim.IsA(UsdGeom.Cube)] == []
    meshes = [prim for prim in library.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
    assert meshes
    assert all(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
    assert {
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        for prim in meshes
    } == {UsdPhysics.Tokens.convexDecomposition}


def test_a_box_is_what_the_library_is_collided_against_by_default(tmp_path):
    _, _, library = written_library(tmp_path)

    meshes = [prim for prim in library.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
    assert not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
    assert len(
        [prim for prim in library.TraverseAll() if prim.IsA(UsdGeom.Cube)]
    ) == len(OBJECT_NAMES)
