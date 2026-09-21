from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Dict, Optional, Self, Tuple

from semantic_digital_twin.adapters.usd.exceptions import (
    PrimDefinedOutsideRootLayerError,
)
from semantic_digital_twin.adapters.usd.stage_parser import (
    Gf,
    Kind,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    downscaled_texture_path,
    geometry_owning_prims,
    readable_texture_path,
)

# %% the layout a library is written in

WORLD_LAYER_NAME = "world.usda"
"""
The layer that references every asset and places it, and the only one a person edits to
rearrange a scene.
"""

ASSETS_DIRECTORY = "Assets"
"""
The directory holding one directory per asset, beside the world layer.
"""

TEXTURES_DIRECTORY = Path("Materials") / "Textures"
"""
Where an asset's own copies of the images its materials read are kept, relative to the
asset's directory.
"""

UNNAMED_LIBRARY_ROOT = "Root"
"""
What the world layer's default prim is called when the stage being split named no
default prim of its own for it to be called after.
"""

TEXTURE_SHADER_ID = "UsdUVTexture"
"""
The shader that reads an image file, and so the one whose asset path is rewritten to
the asset's own copy.
"""

TEXTURE_FILE_INPUT = "file"
"""
The input of a texture shader naming the image it reads.
"""


class VertexSharing(StrEnum):
    """
    Whether the faces of a written mesh share the vertices they meet at.
    """

    AS_AUTHORED = "as_authored"
    """
    Every face keeps the vertices it was written with, so a mesh exported as loose
    triangles stays loose triangles.
    """

    BY_POSITION = "by_position"
    """
    Faces meeting at a position share one vertex there, and whatever differs between
    them is written per face corner, or once per face where a face agrees with itself.

    A scanned surface arrives as loose triangles, storing a position, a normal and a
    texture coordinate once for every triangle touching it, and sharing costs it
    nothing: the surface, its shading and its texturing all come back unchanged.
    """


@dataclass(frozen=True)
class AssetFiles:
    """
    The files one asset is written as.

    Geometry and materials are kept apart so either can be read without the other, and
    both sit behind a payload so a scene can be opened without any of them.
    """

    directory: Path
    """
    The asset's own directory.
    """

    name: str
    """
    The asset's name, which every one of its files is named after.
    """

    @property
    def interface(self) -> Path:
        """
        The layer a scene references, naming the asset and deferring its contents.
        """
        return self.directory / f"{self.name}.usda"

    @property
    def payload(self) -> Path:
        """
        The layer the interface defers to, gathering the contents.
        """
        return self.directory / f"{self.name}_payload.usda"

    @property
    def geometry(self) -> Path:
        """
        The layer holding the asset's surfaces, written in USD's binary encoding.
        """
        return self.directory / f"{self.name}_geo.usd"

    @property
    def material(self) -> Path:
        """
        The layer holding the asset's materials.
        """
        return self.directory / f"{self.name}_look.usda"

    @property
    def textures(self) -> Path:
        """
        The directory holding the asset's own copies of the images it reads.
        """
        return self.directory / TEXTURES_DIRECTORY


# %% sharing the vertices a mesh's faces meet at

CORNER_INTERPOLATIONS = (UsdGeom.Tokens.vertex, UsdGeom.Tokens.varying)
"""
The interpolations holding one value per point, and so the ones sharing points changes.
"""


def _unique_rows(rows: NDArray) -> Tuple[NDArray, NDArray]:
    """
    :param rows: The rows to deduplicate.
    :return: The distinct rows, and for each original row the index of its distinct one.
    """
    rows = np.ascontiguousarray(rows)
    as_records = rows.view([("", rows.dtype)] * rows.shape[1])
    distinct, inverse = np.unique(as_records, return_inverse=True)
    return distinct.view(rows.dtype).reshape(-1, rows.shape[1]), inverse.ravel()


def _face_of_corner(counts: NDArray) -> NDArray:
    """
    :param counts: How many corners each face has.
    :return: For each corner, the index of the first corner of the face it belongs to.
    """
    first_corners = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return np.repeat(first_corners, counts)


def _values_per_corner(
    values: NDArray, interpolation: str, corner_indices: NDArray, counts: NDArray
) -> Optional[NDArray]:
    """
    :param values: The values as authored.
    :param interpolation: The interpolation they were authored with.
    :param corner_indices: The point each face corner uses.
    :param counts: How many corners each face has.
    :return: One value per face corner, or ``None`` if sharing points cannot change
        how the values are held.
    """
    if interpolation in CORNER_INTERPOLATIONS:
        return values[corner_indices]
    if interpolation == UsdGeom.Tokens.uniform:
        return np.repeat(values, counts, axis=0)
    if interpolation == UsdGeom.Tokens.faceVarying:
        return values
    return None


def _tightest_holding(per_corner: NDArray, counts: NDArray) -> Tuple[str, NDArray]:
    """
    :param per_corner: One value per face corner.
    :param counts: How many corners each face has.
    :return: The fewest values saying the same thing, and the interpolation holding
        them - one per face where a face agrees with itself, one per corner otherwise.
    """
    first_corners = _face_of_corner(counts)
    if np.array_equal(per_corner, per_corner[first_corners]):
        return UsdGeom.Tokens.uniform, per_corner[np.unique(first_corners)]
    return UsdGeom.Tokens.faceVarying, per_corner


def _share_mesh_vertices(mesh: UsdGeom.Mesh) -> None:
    """
    Give a mesh one vertex per position, holding whatever differed between the faces
    meeting there on the faces instead.

    The surface, its shading and its texturing are unchanged - only how few values it
    takes to say them. The subdivision scheme is pinned to ``none`` as well, because
    USD's unauthored default smooths a mesh whose faces share edges, which loose
    triangles never did.

    :param mesh: The mesh to rewrite in place.
    """
    points = np.asarray(mesh.GetPointsAttr().Get())
    corner_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    shared_points, point_of_corner = _unique_rows(points)
    if len(shared_points) == len(points):
        return

    for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        per_corner = _values_per_corner(
            np.asarray(primvar.ComputeFlattened()),
            primvar.GetInterpolation(),
            corner_indices,
            counts,
        )
        if per_corner is None:
            continue
        interpolation, values = _tightest_holding(per_corner, counts)
        primvar.BlockIndices()
        primvar.SetInterpolation(interpolation)
        primvar.Set(values)

    normals = mesh.GetNormalsAttr().Get()
    if normals is not None:
        per_corner = _values_per_corner(
            np.asarray(normals),
            mesh.GetNormalsInterpolation(),
            corner_indices,
            counts,
        )
        interpolation, values = _tightest_holding(per_corner, counts)
        mesh.SetNormalsInterpolation(interpolation)
        mesh.GetNormalsAttr().Set(values)

    mesh.GetPointsAttr().Set(shared_points)
    mesh.GetFaceVertexIndicesAttr().Set(
        point_of_corner[corner_indices].astype(np.int32)
    )
    mesh.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)


# %% asset library


@dataclass
class USDAssetLibrary:
    """
    The referenced asset library a monolithic stage is rewritten as.

    A scanned building arrives as a single file holding every surface, every material
    and every texture at once, which has to be read in full to be opened at all. Each
    prim owning geometry becomes an asset of its own here, split across a layer per
    concern, and a small world layer references them and places them - the shape a
    hand-built scene has, and one a person can read and rearrange.

    ..note:: This rewrites USD as USD and never builds a
        :class:`~semantic_digital_twin.world.World`, so the material networks a stage
        carries survive it untouched.
    """

    stage: Usd.Stage
    """
    The stage to split.
    """

    maximum_texture_size: Optional[int] = None
    """
    Longest side a written texture may have, in pixels, or ``None`` to copy every
    texture at the size it was authored.
    """

    vertex_sharing: VertexSharing = VertexSharing.AS_AUTHORED
    """
    Whether the faces of a written mesh share the vertices they meet at.
    """

    # %% construction

    @classmethod
    def from_file(cls, file_path: str, **arguments) -> Self:
        """
        :param file_path: Path of the stage to split.
        :param arguments: Further fields of the library.
        :return: The library the stage at that path becomes.
        """
        return cls(stage=Usd.Stage.Open(file_path), **arguments)

    # %% entry point

    def write(self, directory: Path) -> Path:
        """
        Write the library out, one directory per asset beneath a world layer.

        :param directory: The directory to write into, created if it does not exist.
        :return: The path of the written world layer.
        :raises PrimDefinedOutsideRootLayerError: If a prim owning geometry is not
            defined entirely in the stage's root layer.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        world_layer = Sdf.Layer.CreateNew(str(directory / WORLD_LAYER_NAME))
        world = Usd.Stage.Open(world_layer)
        UsdGeom.SetStageUpAxis(world, UsdGeom.GetStageUpAxis(self.stage))
        UsdGeom.SetStageMetersPerUnit(world, UsdGeom.GetStageMetersPerUnit(self.stage))
        root_path = Sdf.Path(f"/{self._root_name()}")
        world.SetDefaultPrim(UsdGeom.Xform.Define(world, root_path).GetPrim())

        names: Dict[str, int] = {}
        for object_prim in geometry_owning_prims(self.stage):
            self._require_root_layer_definition(object_prim)
            files = self._write_asset(object_prim, directory, names)
            self._place(world, root_path, object_prim, files)

        world_layer.Save()
        return directory / WORLD_LAYER_NAME

    # %% one asset

    def _write_asset(
        self, object_prim: Usd.Prim, directory: Path, names: Dict[str, int]
    ) -> AssetFiles:
        """
        Write every layer of the asset one geometry-owning prim becomes.

        :param object_prim: The prim to write as an asset.
        :param directory: The library's directory.
        :param names: The asset names used so far and how often, extended in place so
            two prims of the same name become two assets.
        :return: The files written.
        """
        name = self._unique_name(object_prim.GetName(), names)
        files = AssetFiles(directory=directory / ASSETS_DIRECTORY / name, name=name)
        files.directory.mkdir(parents=True, exist_ok=True)

        self._write_geometry(object_prim, files)
        self._write_material(object_prim, files)
        self._write_payload(files)
        self._write_interface(object_prim, files)
        return files

    @staticmethod
    def _unique_name(name: str, names: Dict[str, int]) -> str:
        """
        :param name: The name the prim carries.
        :param names: The names used so far and how often, extended in place.
        :return: The name, numbered if the library already holds one.
        """
        seen = names.get(name, 0)
        names[name] = seen + 1
        return name if seen == 0 else f"{name}_{seen}"

    def _write_geometry(self, object_prim: Usd.Prim, files: AssetFiles) -> None:
        """
        Write the asset's surfaces, and the box standing in for them in collision.

        :param object_prim: The prim whose geometry to write.
        :param files: The files the asset is written as.
        """
        layer = self._new_asset_layer(files.geometry, files.name)
        for child in object_prim.GetChildren():
            if child.IsA(UsdShade.Material):
                continue
            Sdf.CopySpec(
                self.stage.GetRootLayer(),
                child.GetPath(),
                layer,
                self._path_in_asset(object_prim, child, files.name),
            )
        self._retarget_relationships(layer, object_prim, files.name)
        if self.vertex_sharing is VertexSharing.BY_POSITION:
            self._share_vertices(layer)
        self._author_collision_proxy(layer, object_prim, files.name)
        layer.Save()

    def _write_material(self, object_prim: Usd.Prim, files: AssetFiles) -> None:
        """
        Write the asset's materials, with every texture they read copied beside them.

        :param object_prim: The prim whose materials to write.
        :param files: The files the asset is written as.
        """
        layer = self._new_asset_layer(files.material, files.name)
        for child in object_prim.GetChildren():
            if not child.IsA(UsdShade.Material):
                continue
            Sdf.CopySpec(
                self.stage.GetRootLayer(),
                child.GetPath(),
                layer,
                self._path_in_asset(object_prim, child, files.name),
            )
        self._retarget_relationships(layer, object_prim, files.name)
        self._copy_textures(object_prim, files, layer)
        layer.Save()

    @staticmethod
    def _write_payload(files: AssetFiles) -> None:
        """
        Write the layer gathering the asset's contents, which its interface defers to.

        :param files: The files the asset is written as.
        """
        layer = Sdf.Layer.CreateNew(str(files.payload))
        layer.subLayerPaths = [f"./{files.geometry.name}", f"./{files.material.name}"]
        stage = Usd.Stage.Open(layer)
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, f"/{files.name}").GetPrim())
        layer.Save()

    def _write_interface(self, object_prim: Usd.Prim, files: AssetFiles) -> None:
        """
        Write the layer a scene references, which names the asset a component and holds
        back its contents until they are asked for.

        :param object_prim: The prim the asset was written from.
        :param files: The files the asset is written as.
        """
        layer = Sdf.Layer.CreateNew(str(files.interface))
        stage = Usd.Stage.Open(layer)
        asset = UsdGeom.Xform.Define(stage, f"/{files.name}")
        stage.SetDefaultPrim(asset.GetPrim())
        Usd.ModelAPI(asset).SetKind(Kind.Tokens.component)
        asset.GetPrim().GetPayloads().AddPayload(f"./{files.payload.name}")
        UsdGeom.ModelAPI(asset).SetExtentsHint(
            UsdGeom.ModelAPI(asset).ComputeExtentsHint(
                UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            )
        )
        layer.Save()

    # %% textures

    def _copy_textures(
        self, object_prim: Usd.Prim, files: AssetFiles, layer: Sdf.Layer
    ) -> None:
        """
        Copy every image the asset's materials read into the asset's own directory, and
        point the written material at the copy.

        :param object_prim: The prim whose textures to copy.
        :param files: The files the asset is written as.
        :param layer: The written material layer, whose asset paths are rewritten.
        """
        for prim in Usd.PrimRange(object_prim):
            shader = UsdShade.Shader(prim)
            if not prim.IsA(UsdShade.Shader):
                continue
            if shader.GetShaderId() != TEXTURE_SHADER_ID:
                continue
            file_input = shader.GetInput(TEXTURE_FILE_INPUT)
            asset_path = file_input.Get() if file_input else None
            if asset_path is None or not asset_path.resolvedPath:
                continue

            readable = readable_texture_path(asset_path.resolvedPath)
            written = downscaled_texture_path(readable, self.maximum_texture_size)
            files.textures.mkdir(parents=True, exist_ok=True)
            copy = files.textures / Path(asset_path.path).name
            shutil.copyfile(written, copy)

            attribute_path = self._path_in_asset(
                object_prim, prim, files.name
            ).AppendProperty(file_input.GetFullName())
            layer.GetAttributeAtPath(attribute_path).default = Sdf.AssetPath(
                f"./{TEXTURES_DIRECTORY.as_posix()}/{copy.name}"
            )

    @staticmethod
    def _retarget_relationships(
        layer: Sdf.Layer, object_prim: Usd.Prim, name: str
    ) -> None:
        """
        Point every relationship copied into an asset at the asset's own copy of what
        it named.

        A copied relationship keeps the path it was authored with, so a material
        binding would still name the stage the asset was lifted out of and resolve to
        nothing once the asset stands on its own.

        :param layer: The written layer to retarget within.
        :param object_prim: The prim the asset was written from.
        :param name: The asset's name.
        """
        source_root = object_prim.GetPath()
        asset_root = Sdf.Path(f"/{name}")

        def retarget(path: Sdf.Path) -> None:
            relationship = layer.GetRelationshipAtPath(path)
            if relationship is None:
                return
            targets = relationship.targetPathList
            targets.explicitItems = [
                (
                    asset_root.AppendPath(target.MakeRelativePath(source_root))
                    if target.HasPrefix(source_root) and target != source_root
                    else target
                )
                for target in targets.explicitItems
            ]

        layer.Traverse(asset_root, retarget)

    # %% sharing vertices

    @staticmethod
    def _share_vertices(layer: Sdf.Layer) -> None:
        """
        Give every mesh in a layer one vertex per position, moving whatever differed
        between the faces meeting there onto the faces themselves.

        :param layer: The written geometry layer to rewrite the meshes of.
        """
        stage = Usd.Stage.Open(layer)
        for prim in stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                _share_mesh_vertices(UsdGeom.Mesh(prim))

    # %% collision

    @staticmethod
    def _author_collision_proxy(
        layer: Sdf.Layer, object_prim: Usd.Prim, name: str
    ) -> None:
        """
        Author the box a physics engine collides against in place of the asset's own
        surfaces, as a guide the renderer leaves out of the picture.

        :param layer: The geometry layer to author into.
        :param object_prim: The prim whose geometry the box encloses.
        :param name: The asset's name.
        """
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeUntransformedBound(object_prim)
            .ComputeAlignedRange()
        )
        if bounds.IsEmpty():
            return

        stage = Usd.Stage.Open(layer)
        proxy = UsdGeom.Cube.Define(stage, f"/{name}/collision")
        proxy.GetSizeAttr().Set(1.0)
        proxy.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        proxy.AddTranslateOp().Set(Gf.Vec3d(bounds.GetMidpoint()))
        proxy.AddScaleOp().Set(Gf.Vec3f(bounds.GetMax() - bounds.GetMin()))
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())

    # %% placement

    def _root_name(self) -> str:
        """
        :return: What the world layer's default prim is called, which is what the
            stage called its own so that the library stands in for it unchanged.
        """
        default_prim = self.stage.GetDefaultPrim()
        if not default_prim:
            return UNNAMED_LIBRARY_ROOT
        return default_prim.GetName()

    @staticmethod
    def _place(
        world: Usd.Stage,
        root_path: Sdf.Path,
        object_prim: Usd.Prim,
        files: AssetFiles,
    ) -> None:
        """
        Reference an asset into the world layer, where the prim it was written from
        stood.

        The placement carries the prim's whole world transform, because the asset was
        written in a space of its own, so the groups it sits under are left as plain
        scopes that only say what a thing is.

        :param world: The world stage to place the asset in.
        :param root_path: The path every placement sits beneath.
        :param object_prim: The prim the asset was written from.
        :param files: The files the asset was written as.
        """
        category = object_prim.GetParent().GetName()
        parent_path = root_path.AppendChild(category) if category else root_path
        if category:
            UsdGeom.Scope.Define(world, parent_path)

        placement = UsdGeom.Xform.Define(world, parent_path.AppendChild(files.name))
        placement.GetPrim().GetReferences().AddReference(
            f"./{ASSETS_DIRECTORY}/{files.name}/{files.interface.name}"
        )
        placement.MakeMatrixXform().Set(
            UsdGeom.Xformable(object_prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
        )

    # %% source layers

    @staticmethod
    def _new_asset_layer(path: Path, name: str) -> Sdf.Layer:
        """
        :param path: The file to create the layer at.
        :param name: The asset's name, which its root prim carries.
        :return: A layer holding an empty root prim for the asset.
        """
        layer = Sdf.Layer.CreateNew(str(path))
        root = Sdf.CreatePrimInLayer(layer, Sdf.Path(f"/{name}"))
        root.specifier = Sdf.SpecifierDef
        root.typeName = UsdGeom.Tokens.Xform
        layer.defaultPrim = name
        return layer

    @staticmethod
    def _path_in_asset(object_prim: Usd.Prim, prim: Usd.Prim, name: str) -> Sdf.Path:
        """
        :param object_prim: The prim the asset was written from.
        :param prim: A prim in that prim's subtree.
        :return: Where that prim sits inside the written asset.
        """
        relative = prim.GetPath().MakeRelativePath(object_prim.GetPath())
        return Sdf.Path(f"/{name}").AppendPath(relative)

    def _require_root_layer_definition(self, prim: Usd.Prim) -> None:
        """
        :param prim: The prim about to be copied out of the stage.
        :raises PrimDefinedOutsideRootLayerError: If any opinion about the prim lives
            outside the stage's root layer, which copying would leave behind.
        """
        root_layer = self.stage.GetRootLayer()
        layers = [
            spec.layer.identifier
            for spec in prim.GetPrimStack()
            if spec.layer != root_layer
        ]
        if not layers:
            return
        raise PrimDefinedOutsideRootLayerError(
            file_path=root_layer.identifier,
            prim_path=prim.GetPath().pathString,
            layers=layers,
        )
