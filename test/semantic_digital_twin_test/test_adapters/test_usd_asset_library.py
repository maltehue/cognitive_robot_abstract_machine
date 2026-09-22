from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from semantic_digital_twin.adapters.usd.asset_library import (
    ASSETS_DIRECTORY,
    BASE_LAYER_NAME,
    DEFAULT_PART_NAME,
    GEOMETRIES_LAYER_NAME,
    INSTANCES_LAYER_NAME,
    MATERIALS_LAYER_NAME,
    PAYLOADS_DIRECTORY,
    PHYSICS_DIRECTORY,
    PHYSICS_LAYER_NAME,
    TEXTURES_DIRECTORY,
    CollisionProxy,
    MeshPart,
    MeshSegmentation,
    USDAssetLibrary,
    VertexSharing,
    WrittenLibrary,
)
from semantic_digital_twin.adapters.usd.scene_parser import USDSceneParser
from semantic_digital_twin.adapters.usd.stage_parser import (
    Gf,
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
    build_scene_stage_with_an_object_of_several_faces,
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


@dataclass
class SplitScene:
    """
    The fixture stage, what writing it out produced, and the library composed back.

    Both stages are held, because a prim of a stage nothing refers to any more stops
    answering the moment that stage is collected.
    """

    source: Usd.Stage
    """
    The stage the library was written from.
    """

    written: WrittenLibrary
    """
    What the writer reported having written.
    """

    library: Usd.Stage
    """
    The library, composed from its world layer.
    """

    @property
    def world_layer(self) -> Path:
        """
        :return: The layer placing every asset of the library.
        """
        return self.written.world_layer

    def asset_directory(self, object_name: str) -> Path:
        """
        :param object_name: The object whose asset to look in.
        :return: That asset's own directory.
        """
        return self.world_layer.parent / ASSETS_DIRECTORY / object_name

    def payloads(self, object_name: str) -> Path:
        """
        :param object_name: The object whose asset to look in.
        :return: The directory holding that asset's contents.
        """
        return self.asset_directory(object_name) / PAYLOADS_DIRECTORY

    def layer(self, object_name: str, *names: str) -> Sdf.Layer:
        """
        :param object_name: The object whose asset to look in.
        :param names: The path of the layer beneath the asset's payloads.
        :return: That layer, read without composing it.
        """
        return Sdf.Layer.FindOrOpen(str(self.payloads(object_name).joinpath(*names)))


def written_library(tmp_path, **arguments) -> SplitScene:
    """
    Split the scanned-scene fixture into an asset library and open it again.

    :param tmp_path: The directory to write into.
    :param arguments: Further fields of the library.
    :return: The source stage, what was written, and the composed library.
    """
    source = build_scene_stage_with_textured_objects(texture_file(tmp_path))
    written = USDAssetLibrary(stage=source, **arguments).write(tmp_path / "library")
    return SplitScene(
        source=source,
        written=written,
        library=Usd.Stage.Open(str(written.world_layer)),
    )


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


def cubes_in(stage) -> list:
    return [prim for prim in stage.TraverseAll() if prim.IsA(UsdGeom.Cube)]


# %% the files a library is written as


def test_write_gives_every_object_an_asset_of_its_own(tmp_path):
    split = written_library(tmp_path)

    assets = split.world_layer.parent / ASSETS_DIRECTORY
    assert sorted(path.name for path in assets.iterdir()) == sorted(OBJECT_NAMES)


def test_an_asset_is_an_interface_over_the_payloads_beside_it(tmp_path):
    split = written_library(tmp_path)

    assert sorted(path.name for path in split.asset_directory("wall_a").iterdir()) == [
        PAYLOADS_DIRECTORY,
        "wall_a.usda",
    ]


def test_an_asset_keeps_every_concern_in_a_layer_of_its_own(tmp_path):
    split = written_library(tmp_path)

    assert sorted(path.name for path in split.payloads("wall_a").iterdir()) == sorted(
        [
            PHYSICS_DIRECTORY,
            TEXTURES_DIRECTORY,
            BASE_LAYER_NAME,
            GEOMETRIES_LAYER_NAME,
            INSTANCES_LAYER_NAME,
            MATERIALS_LAYER_NAME,
        ]
    )
    assert [
        path.name for path in (split.payloads("wall_a") / PHYSICS_DIRECTORY).iterdir()
    ] == [PHYSICS_LAYER_NAME]


def test_only_the_mesh_data_is_written_in_usd_binary_encoding(tmp_path):
    # Mesh arrays are large and never edited by hand; everything else is small and read
    # and retuned constantly, so it stays text.
    split = written_library(tmp_path)

    binary = [
        path.name
        for path in split.asset_directory("wall_a").rglob("*.usd*")
        if not path.read_bytes().startswith(b"#usda")
    ]
    assert binary == [GEOMETRIES_LAYER_NAME]


def test_the_hierarchy_layer_carries_no_mesh_data(tmp_path):
    split = written_library(tmp_path)

    assert "points" not in split.layer("wall_a", BASE_LAYER_NAME).ExportToString()


def test_nothing_the_library_writes_is_instanceable(tmp_path):
    # An instance hides the children a caller has to reach: the collision boxes beneath
    # a part, and the overrides the physics layer authors on them.
    split = written_library(tmp_path)

    written = list(split.asset_directory("wall_a").rglob("*.usda"))
    assert written
    assert not any("instanceable" in path.read_text() for path in written)


def test_the_library_keeps_the_name_the_stage_gave_its_root(tmp_path):
    # The library stands in for the stage it was written from, so anything addressing
    # the scene by the name of its root goes on working.
    split = written_library(tmp_path)

    assert (
        split.library.GetDefaultPrim().GetName()
        == split.source.GetDefaultPrim().GetName()
    )


def test_the_world_layer_carries_no_geometry_of_its_own(tmp_path):
    # The point of the split: the file a person edits stays small enough to read.
    split = written_library(tmp_path)

    text = split.world_layer.read_text()
    assert "points" not in text
    assert text.count("prepend references") == len(OBJECT_NAMES)


def test_the_written_library_says_which_files_and_parts_each_object_became(tmp_path):
    split = written_library(tmp_path)

    assert sorted(asset.files.name for asset in split.written.assets) == sorted(
        OBJECT_NAMES
    )
    [wall] = [asset for asset in split.written.assets if asset.files.name == "wall_a"]
    assert wall.source_path == Sdf.Path("/scene/Wall/wall_a")
    assert wall.category == "Wall"
    assert [part.name for part in wall.parts] == [DEFAULT_PART_NAME]


# %% what the library composes back to


def test_the_library_places_every_object_where_the_stage_had_it(tmp_path):
    split = written_library(tmp_path)

    for name in OBJECT_NAMES:
        np.testing.assert_allclose(
            world_transform(mesh_under(split.library, name)),
            world_transform(mesh_under(split.source, name)),
            atol=1e-9,
        )


def test_the_library_carries_the_geometry_the_stage_held(tmp_path):
    split = written_library(tmp_path)

    for name in OBJECT_NAMES:
        np.testing.assert_allclose(
            UsdGeom.Mesh(mesh_under(split.library, name)).GetPointsAttr().Get(),
            UsdGeom.Mesh(mesh_under(split.source, name)).GetPointsAttr().Get(),
        )


def test_every_asset_is_a_component_deferred_behind_a_payload(tmp_path):
    split = written_library(tmp_path)

    unloaded = Usd.Stage.Open(str(split.world_layer), Usd.Stage.LoadNone)
    assert [prim for prim in unloaded.TraverseAll() if prim.IsA(UsdGeom.Mesh)] == []
    assert [
        Usd.ModelAPI(prim).GetKind()
        for prim in unloaded.TraverseAll()
        if Usd.ModelAPI(prim).GetKind()
    ] == ["component"] * len(OBJECT_NAMES)


def test_opening_the_library_composes_its_physics_without_being_asked(tmp_path):
    # Physics sits in a variant so that a scene can be opened without it; a variant
    # with no selection authored composes to nothing, which is a world that never
    # collides and says nothing about why.
    split = written_library(tmp_path)

    assert len(cubes_in(split.library)) == len(OBJECT_NAMES)


# %% the textures the library holds


def test_a_texture_is_copied_beside_the_asset_that_reads_it(tmp_path):
    split = written_library(tmp_path)

    copied = split.payloads("wall_a") / TEXTURES_DIRECTORY
    assert [path.name for path in copied.iterdir()] == ["surface.png"]

    [reader] = [
        prim
        for prim in split.library.TraverseAll()
        if prim.IsA(UsdShade.Shader)
        and UsdShade.Shader(prim).GetShaderId() == "UsdUVTexture"
        and "wall_a" in prim.GetPath().pathString
    ]
    asset_path = UsdShade.Shader(reader).GetInput("file").Get()
    assert asset_path.path.startswith("./")
    assert Path(asset_path.resolvedPath).samefile(copied / "surface.png")


def test_a_texture_is_downscaled_to_the_given_cap(tmp_path):
    split = written_library(tmp_path, maximum_texture_size=16)

    with Image.open(
        split.payloads("wall_a") / TEXTURES_DIRECTORY / "surface.png"
    ) as image:
        assert max(image.size) == 16


def test_a_texture_below_the_cap_is_left_at_the_size_it_was(tmp_path):
    split = written_library(tmp_path, maximum_texture_size=4096)

    with Image.open(
        split.payloads("wall_a") / TEXTURES_DIRECTORY / "surface.png"
    ) as image:
        assert image.size == (64, 64)


# %% the collision the library authors


def test_every_asset_carries_a_collision_proxy_the_renderer_leaves_alone(tmp_path):
    # The scanned surface is far too dense to collide against, and a guide is what
    # keeps the proxy out of the picture while a physics engine still reads it.
    split = written_library(tmp_path)

    proxies = cubes_in(split.library)
    assert len(proxies) == len(OBJECT_NAMES)
    assert all(
        UsdGeom.Imageable(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
        for prim in proxies
    )


def test_a_collision_proxy_encloses_the_geometry_it_stands_for(tmp_path):
    split = written_library(tmp_path)

    [proxy] = [
        prim
        for prim in cubes_in(split.library)
        if "wall_a" in prim.GetPath().pathString
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
        .ComputeWorldBound(mesh_under(split.source, "wall_a"))
        .ComputeAlignedRange()
    )

    np.testing.assert_allclose(bounds.GetMin(), mesh_bounds.GetMin(), atol=1e-9)
    np.testing.assert_allclose(bounds.GetMax(), mesh_bounds.GetMax(), atol=1e-9)


def test_collision_is_written_apart_from_the_geometry_it_stands_for(tmp_path):
    # Collision is the part of an asset that gets retuned; the geometry beside it is a
    # binary crate nobody wants to rewrite to change a box.
    split = written_library(tmp_path)

    assert physics_in(split.layer("wall_a", GEOMETRIES_LAYER_NAME)) == []
    assert physics_in(split.layer("wall_a", PHYSICS_DIRECTORY, PHYSICS_LAYER_NAME)) == [
        "/wall_a/collision"
    ]


def test_a_surface_collided_against_is_said_so_apart_from_the_surface(tmp_path):
    split = written_library(
        tmp_path, collision_proxy=CollisionProxy.CONVEX_DECOMPOSITION
    )

    assert physics_in(split.layer("wall_a", GEOMETRIES_LAYER_NAME)) == []
    assert physics_in(split.layer("wall_a", PHYSICS_DIRECTORY, PHYSICS_LAYER_NAME)) == [
        f"/wall_a/Geometry/{DEFAULT_PART_NAME}/{DEFAULT_PART_NAME}"
    ]


def test_collision_is_written_as_text_a_person_can_edit(tmp_path):
    split = written_library(tmp_path)

    physics = split.payloads("wall_a") / PHYSICS_DIRECTORY / PHYSICS_LAYER_NAME
    assert physics.read_text().startswith("#usda")


def test_the_physics_layer_stands_for_the_asset_even_when_it_says_nothing(tmp_path):
    # A payload whose layer names no default prim composes to nothing, so a caller
    # authoring its own collision into the layer later would find it ignored.
    split = written_library(tmp_path, collision_proxy=CollisionProxy.NONE)

    layer = split.layer("wall_a", PHYSICS_DIRECTORY, PHYSICS_LAYER_NAME)
    assert layer.defaultPrim == "wall_a"
    assert cubes_in(split.library) == []
    assert not any(
        prim.HasAPI(UsdPhysics.CollisionAPI) for prim in split.library.TraverseAll()
    )


def test_a_material_binding_points_inside_the_asset_it_was_written_with(tmp_path):
    # A copied relationship keeps the path it was authored with, and a path into the
    # stage the asset came from resolves to nothing once the asset stands on its own.
    split = written_library(tmp_path)

    material, _ = UsdShade.MaterialBindingAPI(
        mesh_under(split.library, "wall_a")
    ).ComputeBoundMaterial()

    assert material
    assert material.GetPath().pathString.endswith(
        f"/wall_a/Geometry/{DEFAULT_PART_NAME}/Material"
    )


# %% reading the library back


def test_the_library_parses_into_the_scene_the_stage_described(tmp_path):
    # A part is a body of its own and is named after the part, so the names read back
    # differ; what has to hold is that the same objects stand in the same places.
    split = written_library(tmp_path)

    from_source = USDSceneParser(stage=split.source, prefix="scene").parse()
    from_library = USDSceneParser.from_file(
        str(split.world_layer), prefix="scene"
    ).parse()

    assert sorted(placed_bodies(from_library).values()) == sorted(
        placed_bodies(from_source).values()
    )


# %% cutting an object into parts


class SecondFaceOfTheMesh(MeshSegmentation):
    """
    A segmentation taking one named part out of every object it is shown.

    Which face it takes is fixed, so the part it cuts out is the same on every run.
    """

    name = "door_0"
    """
    What the part it cuts out is called.
    """

    pivot = Gf.Vec3d(1.0, 0.0, 0.0)
    """
    Where the part it cuts out is written about.
    """

    def parts_of(self, object_prim: Usd.Prim, mesh: UsdGeom.Mesh) -> list:
        """
        :param object_prim: The object being written as an asset.
        :param mesh: The object's surface.
        :return: The one part it takes.
        """
        faces = np.zeros(len(mesh.GetFaceVertexCountsAttr().Get()), dtype=bool)
        faces[1] = True
        return [MeshPart(name=self.name, faces=faces, pivot=self.pivot)]


def segmented_library(tmp_path, **arguments) -> SplitScene:
    """
    :param tmp_path: The directory to write into.
    :param arguments: Further fields of the library.
    :return: A library written from a two-faced object with one face cut out of it.
    """
    source = build_scene_stage_with_an_object_of_several_faces(texture_file(tmp_path))
    written = USDAssetLibrary(
        stage=source, segmentation=SecondFaceOfTheMesh(), **arguments
    ).write(tmp_path / "library")
    return SplitScene(
        source=source,
        written=written,
        library=Usd.Stage.Open(str(written.world_layer)),
    )


def part_mesh(stage, part_name: str) -> UsdGeom.Mesh:
    """
    :param stage: The composed library.
    :param part_name: The part to look for.
    :return: The mesh that part is made of.
    """
    [mesh] = [
        prim
        for prim in stage.TraverseAll()
        if prim.IsA(UsdGeom.Mesh) and prim.GetParent().GetName() == part_name
    ]
    return UsdGeom.Mesh(mesh)


def test_a_segmentation_gives_each_part_a_prim_of_its_own(tmp_path):
    split = segmented_library(tmp_path)

    [wall] = split.written.assets
    assert [part.name for part in wall.parts] == [
        SecondFaceOfTheMesh.name,
        DEFAULT_PART_NAME,
    ]
    assert wall.parts[0].pivot == SecondFaceOfTheMesh.pivot
    assert split.library.GetPrimAtPath(
        f"/scene/Wall/wall_a/Geometry/{SecondFaceOfTheMesh.name}"
    )


def test_the_faces_a_segmentation_did_not_take_stay_with_the_object(tmp_path):
    split = segmented_library(tmp_path)

    assert list(
        part_mesh(split.library, SecondFaceOfTheMesh.name)
        .GetFaceVertexCountsAttr()
        .Get()
    ) == [4]
    assert list(
        part_mesh(split.library, DEFAULT_PART_NAME).GetFaceVertexCountsAttr().Get()
    ) == [4]


def test_a_part_is_written_about_the_pivot_it_was_given(tmp_path):
    # A door leaf turns about its hinge, so the points of the leaf are written relative
    # to it and the prim holding them carries the hinge.
    split = segmented_library(tmp_path)

    leaf = split.library.GetPrimAtPath(
        f"/scene/Wall/wall_a/Geometry/{SecondFaceOfTheMesh.name}"
    )
    translation = UsdGeom.Xformable(leaf).GetLocalTransformation().ExtractTranslation()

    np.testing.assert_allclose(translation, SecondFaceOfTheMesh.pivot, atol=1e-9)
    np.testing.assert_allclose(
        corner_points(part_mesh(split.library, SecondFaceOfTheMesh.name)),
        np.array([(1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)])
        - np.array(SecondFaceOfTheMesh.pivot),
        atol=1e-6,
    )


def test_a_part_stands_where_the_faces_it_took_stood(tmp_path):
    split = segmented_library(tmp_path)

    leaf = part_mesh(split.library, SecondFaceOfTheMesh.name)
    to_world = world_transform(leaf.GetPrim())

    np.testing.assert_allclose(
        corner_points(leaf) @ to_world[:3, :3] + to_world[3, :3],
        [(1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)],
        atol=1e-6,
    )


def test_a_part_keeps_what_the_mesh_it_came_from_said_about_itself(tmp_path):
    # A scan is a polygon mesh, and a part that loses the subdivision scheme saying so
    # is smoothed by a renderer that reads USD's unauthored default instead.
    split = segmented_library(tmp_path)

    assert (
        part_mesh(split.library, SecondFaceOfTheMesh.name)
        .GetSubdivisionSchemeAttr()
        .Get()
        == UsdGeom.Tokens.none
    )


def test_a_part_keeps_the_texture_coordinates_of_the_faces_it_took(tmp_path):
    split = segmented_library(tmp_path)

    leaf = part_mesh(split.library, SecondFaceOfTheMesh.name)
    coordinates = UsdGeom.PrimvarsAPI(leaf).GetPrimvar("st")

    np.testing.assert_allclose(
        corner_values(
            leaf, coordinates.ComputeFlattened(), coordinates.GetInterpolation()
        ),
        [(0.5, 0), (1, 0), (1, 1), (0.5, 1)],
        atol=1e-6,
    )


def test_every_part_is_covered_in_the_material_the_object_carried(tmp_path):
    split = segmented_library(tmp_path)

    for part in (SecondFaceOfTheMesh.name, DEFAULT_PART_NAME):
        material, _ = UsdShade.MaterialBindingAPI(
            part_mesh(split.library, part).GetPrim()
        ).ComputeBoundMaterial()
        assert material
        assert material.GetPath().pathString.endswith(f"/{part}/Material")


def test_a_part_becomes_a_body_of_its_own_when_the_library_is_read_back(tmp_path):
    split = segmented_library(tmp_path)

    world = USDSceneParser.from_file(str(split.world_layer), prefix="scene").parse()

    assert sorted(
        body.name.name for body in world.bodies if body is not world.root
    ) == sorted([SecondFaceOfTheMesh.name, DEFAULT_PART_NAME])


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
    written = USDAssetLibrary(stage=source, **arguments).write(tmp_path / "library")
    return SplitSoup(source=source, library=Usd.Stage.Open(str(written.world_layer)))


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
    split = written_library(tmp_path)

    scenes = [
        prim for prim in split.library.TraverseAll() if prim.IsA(UsdPhysics.Scene)
    ]

    assert len(scenes) == 1


# %% where the library stands


def test_the_library_keeps_the_coordinates_it_was_authored_in_by_default(tmp_path):
    split = written_library(tmp_path)

    np.testing.assert_allclose(
        composed_bounds(split.library), composed_bounds(split.source), atol=1e-6
    )


def test_the_library_can_stand_its_scene_on_the_origin(tmp_path):
    # A scan is authored wherever it was captured, which for the innolab is 250 m from
    # its stage's origin - far enough that anything placed by hand misses it.
    split = written_library(tmp_path, root_placement=RootPlacement.SCENE_GROUND)

    low, high = composed_bounds(split.library)

    np.testing.assert_allclose((low + high)[:2] / 2, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(low[2], 0.0, atol=1e-6)


def test_standing_the_scene_on_the_origin_moves_every_object_together(tmp_path):
    split = written_library(tmp_path, root_placement=RootPlacement.SCENE_GROUND)

    def separation(stage):
        return (
            world_transform(mesh_under(stage, "wall_a"))[3, :3]
            - world_transform(mesh_under(stage, "floor_a"))[3, :3]
        )

    np.testing.assert_allclose(
        separation(split.library), separation(split.source), atol=1e-6
    )


# %% what the library is collided against


def test_the_surface_itself_can_be_collided_against_instead_of_a_box(tmp_path):
    split = written_library(
        tmp_path, collision_proxy=CollisionProxy.CONVEX_DECOMPOSITION
    )

    assert cubes_in(split.library) == []
    meshes = [prim for prim in split.library.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
    assert meshes
    assert all(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
    assert {
        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        for prim in meshes
    } == {UsdPhysics.Tokens.convexDecomposition}


def test_a_box_is_what_the_library_is_collided_against_by_default(tmp_path):
    split = written_library(tmp_path)

    meshes = [prim for prim in split.library.TraverseAll() if prim.IsA(UsdGeom.Mesh)]
    assert not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in meshes)
    assert len(cubes_in(split.library)) == len(OBJECT_NAMES)
