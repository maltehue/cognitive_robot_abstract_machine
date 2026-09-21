from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_digital_twin.adapters.usd.asset_library import USDAssetLibrary
from semantic_digital_twin.adapters.usd.scene_parser import USDSceneParser
from semantic_digital_twin.adapters.usd.stage_parser import Usd, UsdGeom, UsdShade

from .usd_stages import (
    PXR_AVAILABLE,
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
    ]


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
