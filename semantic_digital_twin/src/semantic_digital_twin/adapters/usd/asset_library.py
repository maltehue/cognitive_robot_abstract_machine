from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Dict, List, Optional, Self, Tuple

from semantic_digital_twin.adapters.usd.exceptions import (
    PrimDefinedOutsideRootLayerError,
)
from semantic_digital_twin.adapters.usd.stage_parser import (
    Gf,
    Kind,
    RootPlacement,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    UsdShade,
    Vt,
    downscaled_texture_path,
    geometry_owning_prims,
    readable_texture_path,
    scene_ground,
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

PAYLOADS_DIRECTORY = "payloads"
"""
The directory of an asset holding everything its interface defers to.
"""

TEXTURES_DIRECTORY = "Textures"
"""
The directory holding an asset's own copies of the images its materials read.
"""

PHYSICS_DIRECTORY = "Physics"
"""
The directory holding what a physics engine reads and nothing else does.
"""

BASE_LAYER_NAME = "base.usda"
"""
The layer holding an asset's hierarchy and the transform of every part of it.
"""

GEOMETRIES_LAYER_NAME = "geometries.usdc"
"""
The layer holding the mesh arrays, and the only one written in USD's binary encoding.
"""

INSTANCES_LAYER_NAME = "instances.usda"
"""
The layer putting each mesh together with the material covering it.
"""

MATERIALS_LAYER_NAME = "materials.usda"
"""
The layer holding an asset's materials.
"""

PHYSICS_LAYER_NAME = "physics.usda"
"""
The layer holding what a physics engine collides and moves an asset as.
"""

GEOMETRY_SCOPE_NAME = "Geometry"
"""
What an asset calls the scope holding its parts.
"""

GEOMETRIES_SCOPE_NAME = "Geometries"
"""
What the binary crate calls the scope holding its meshes.
"""

INSTANCES_SCOPE_NAME = "Instances"
"""
What the assembly layer calls the scope holding its parts.
"""

MATERIALS_SCOPE_NAME = "Materials"
"""
What the material layer calls the scope holding its materials.
"""

BOUND_MATERIAL_NAME = "Material"
"""
What a part calls the material covering it.
"""

PHYSICS_VARIANT_SET_NAME = "Physics"
"""
The variant set deciding whether an asset is loaded with physics at all.
"""

PHYSICS_VARIANT_NAME = "physics"
"""
The variant loading what a physics engine reads, and the one selected when written.
"""

NO_PHYSICS_VARIANT_NAME = "none"
"""
The variant loading nothing but the asset's surfaces.
"""

DEFAULT_PART_NAME = "surface"
"""
What the part holding every face no segmentation took is called.
"""

COLLISION_PROXY_NAME = "collision"
"""
What the box standing in for an asset's surfaces is called.
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

PHYSICS_SCENE_NAME = "physicsScene"
"""
The scene a physics engine runs the library in, without which nothing simulates until
a caller authors one.
"""

TEXTURE_FILE_INPUT = "file"
"""
The input of a texture shader naming the image it reads.
"""

MESH_ARRAY_ATTRIBUTES = frozenset(
    {
        UsdGeom.Tokens.points,
        UsdGeom.Tokens.faceVertexCounts,
        UsdGeom.Tokens.faceVertexIndices,
        UsdGeom.Tokens.normals,
        UsdGeom.Tokens.extent,
    }
)
"""
The attributes of a mesh that say which faces it is made of, and so the ones a part
holding some of those faces writes for itself rather than copying.
"""


class CollisionProxy(StrEnum):
    """
    What a physics engine is given to collide an asset against.
    """

    NONE = "none"
    """
    Nothing. The physics layer is written empty but composed, so a caller can author
    its own collision into it without rewriting the asset around it.
    """

    BOUNDING_BOX = "bounding_box"
    """
    One box enclosing the asset, authored as a guide so a renderer leaves it out of
    the picture. The cheapest thing to collide against, and a close fit only for what
    is already box shaped - a floor, a flat wall.
    """

    CONVEX_DECOMPOSITION = "convex_decomposition"
    """
    The asset's own surface, which a physics engine approximates by convex pieces when
    it loads the library. Follows the surface far more closely than a box, at the cost
    of that engine having to work the pieces out from every triangle.
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

    Geometry, materials and physics are kept apart so any of them can be read without
    the others, and all of them sit behind the interface's payload so a scene can be
    opened without any. Only the mesh arrays are written in USD's binary encoding; what
    gets retuned - the hierarchy, the materials, the collision - stays text.
    """

    directory: Path
    """
    The asset's own directory.
    """

    name: str
    """
    The asset's name, which its root prim and its interface are called after.
    """

    @property
    def interface(self) -> Path:
        """
        The layer a scene references, naming the asset and deferring its contents.
        """
        return self.directory / f"{self.name}.usda"

    @property
    def payloads(self) -> Path:
        """
        The directory holding everything the interface defers to.
        """
        return self.directory / PAYLOADS_DIRECTORY

    @property
    def base(self) -> Path:
        """
        The layer holding the asset's hierarchy and where each of its parts sits.
        """
        return self.payloads / BASE_LAYER_NAME

    @property
    def geometries(self) -> Path:
        """
        The layer holding the asset's mesh arrays.
        """
        return self.payloads / GEOMETRIES_LAYER_NAME

    @property
    def instances(self) -> Path:
        """
        The layer putting each of the asset's meshes together with its material.
        """
        return self.payloads / INSTANCES_LAYER_NAME

    @property
    def materials(self) -> Path:
        """
        The layer holding the asset's materials.
        """
        return self.payloads / MATERIALS_LAYER_NAME

    @property
    def textures(self) -> Path:
        """
        The directory holding the asset's own copies of the images it reads.
        """
        return self.payloads / TEXTURES_DIRECTORY

    @property
    def physics(self) -> Path:
        """
        The layer holding what a physics engine collides and moves the asset as.
        """
        return self.payloads / PHYSICS_DIRECTORY / PHYSICS_LAYER_NAME


# %% cutting a mesh into parts


@dataclass(frozen=True)
class MeshPart:
    """
    A piece of a mesh that is written as a part of its own.

    A part is what a physics engine can move by itself: a door leaf cut out of the wall
    it was scanned as part of turns about its hinge while the wall stays where it is.
    """

    name: str
    """
    What the part is called inside the asset.
    """

    faces: NDArray
    """
    Which faces of the mesh belong to the part, one entry per face.
    """

    pivot: Gf.Vec3d
    """
    The point in the object's own space the part is written about, which is where a
    joint holding the part sits.
    """


class MeshSegmentation(ABC):
    """
    Decides which pieces of an object's surface are written as parts of their own.
    """

    @abstractmethod
    def parts_of(self, object_prim: Usd.Prim, mesh: UsdGeom.Mesh) -> List[MeshPart]:
        """
        :param object_prim: The object being written as an asset.
        :param mesh: The object's surface.
        :return: The parts to cut out of it, in the order they are written. Every face
            no part takes stays with the object.
        """


@dataclass(frozen=True)
class FaceSelection:
    """
    Which of a mesh's points, face corners and faces a part made of some of its faces
    keeps, and where each of them moved to in the part.
    """

    points: NDArray
    """
    The index in the source mesh of each point the part keeps.
    """

    corners: NDArray
    """
    The index in the source mesh of each face corner the part keeps.
    """

    faces: NDArray
    """
    The index in the source mesh of each face the part keeps.
    """

    counts: NDArray
    """
    How many corners each kept face has.
    """

    corner_indices: NDArray
    """
    The point each kept face corner uses, numbered within the part.
    """

    @classmethod
    def of(cls, counts: NDArray, corner_indices: NDArray, taken: NDArray) -> Self:
        """
        :param counts: How many corners each face of the source mesh has.
        :param corner_indices: The point each of its face corners uses.
        :param taken: Which of its faces the part keeps, one entry per face.
        :return: What the part is made of.
        """
        corners = np.flatnonzero(np.repeat(taken, counts))
        kept_corner_indices = corner_indices[corners]
        points = np.unique(kept_corner_indices)
        return cls(
            points=points,
            corners=corners,
            faces=np.flatnonzero(taken),
            counts=counts[taken],
            corner_indices=np.searchsorted(points, kept_corner_indices),
        )

    def values(self, values, interpolation: str):
        """
        :param values: One mesh attribute of the source mesh, flattened.
        :param interpolation: The interpolation it is held at.
        :return: The part's share of it, held the same way.
        """
        if interpolation == UsdGeom.Tokens.constant:
            return values
        if interpolation == UsdGeom.Tokens.uniform:
            return np.asarray(values)[self.faces]
        if interpolation == UsdGeom.Tokens.faceVarying:
            return np.asarray(values)[self.corners]
        return np.asarray(values)[self.points]


# %% one part, as the writer works on it


@dataclass(frozen=True)
class PartGeometry:
    """
    A part of an object as the writer holds it: which mesh it comes from, which of that
    mesh's faces it keeps, and where it is written about.
    """

    name: str
    """
    What the part is called inside the asset.
    """

    mesh: UsdGeom.Mesh
    """
    The mesh of the stage the part's faces are taken from.
    """

    faces: Optional[NDArray] = None
    """
    Which faces of that mesh the part keeps, or ``None`` for all of them.
    """

    pivot: Gf.Vec3d = field(default_factory=lambda: Gf.Vec3d(0.0, 0.0, 0.0))
    """
    The point in the object's own space the part is written about.
    """


NUMERIC_ARRAY_KINDS = frozenset("biuf")
"""
The kinds of numpy array USD holds as an array of numbers, and so the ones that can be
handed over whole rather than one value at a time.
"""


def _held_as_usd_holds_it(values, type_name: Sdf.ValueTypeName):
    """
    Put a part's share of an attribute into the array type USD stores it in.

    A million points handed over as Python objects one at a time is most of what writing
    a scanned building costs; handed over as one buffer it costs nothing.

    :param values: The values the part keeps.
    :param type_name: The value type the attribute is authored with.
    :return: Them in USD's own array type, or unchanged if they are not numbers.
    """
    if (
        not isinstance(values, np.ndarray)
        or values.dtype.kind not in NUMERIC_ARRAY_KINDS
    ):
        return values
    return type_name.type.pythonClass.FromNumpy(np.ascontiguousarray(values))


# %% what writing a library produced


@dataclass(frozen=True)
class WrittenPart:
    """
    One part of a written asset, as a caller reaching into the asset finds it.
    """

    name: str
    """
    What the part is called inside the asset.
    """

    pivot: Gf.Vec3d
    """
    Where inside the asset the part stands, which is the point its own geometry is
    written about.
    """


@dataclass(frozen=True)
class WrittenAsset:
    """
    One object of a stage, and the asset it was written as.
    """

    source_path: Sdf.Path
    """
    Where the object stood in the stage the library was written from.
    """

    category: str
    """
    What the stage grouped the object under, which the world layer keeps as a scope.
    """

    files: AssetFiles
    """
    The files the asset was written as.
    """

    parts: List[WrittenPart]
    """
    Every part of the asset, in the order they were written.
    """


@dataclass(frozen=True)
class WrittenLibrary:
    """
    What one run of the writer produced, so a caller can find its way back from an
    object of the stage to the files it became.
    """

    world_layer: Path
    """
    The layer that references every asset and places it.
    """

    assets: List[WrittenAsset]
    """
    Every asset written, in the order the stage held the objects.
    """


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
    concern behind an interface, and a small world layer references them and places
    them - the shape a hand-built scene has, and one a person can read and rearrange.

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

    collision_proxy: CollisionProxy = CollisionProxy.BOUNDING_BOX
    """
    What a physics engine is given to collide the library's assets against.
    """

    root_placement: RootPlacement = RootPlacement.STAGE_ORIGIN
    """
    Where the library's own root sits, which decides whether the scene keeps the
    coordinates it was captured in.
    """

    segmentation: Optional[MeshSegmentation] = None
    """
    What decides which pieces of an object's surface become parts of their own, or
    ``None`` to write every object as one piece.
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

    def write(self, directory: Path) -> WrittenLibrary:
        """
        Write the library out, one directory per asset beneath a world layer.

        :param directory: The directory to write into, created if it does not exist.
        :return: What was written, object by object.
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
        root = UsdGeom.Xform.Define(world, root_path)
        world.SetDefaultPrim(root.GetPrim())
        self._stand_scene(root)
        UsdPhysics.Scene.Define(world, root_path.AppendChild(PHYSICS_SCENE_NAME))

        names: Dict[str, int] = {}
        written = []
        for object_prim in geometry_owning_prims(self.stage):
            self._require_root_layer_definition(object_prim)
            asset = self._write_asset(object_prim, directory, names)
            self._place(world, root_path, object_prim, asset.files)
            written.append(asset)

        world_layer.Save()
        return WrittenLibrary(world_layer=directory / WORLD_LAYER_NAME, assets=written)

    # %% one asset

    def _write_asset(
        self, object_prim: Usd.Prim, directory: Path, names: Dict[str, int]
    ) -> WrittenAsset:
        """
        Write every layer of the asset one geometry-owning prim becomes.

        :param object_prim: The prim to write as an asset.
        :param directory: The library's directory.
        :param names: The asset names used so far and how often, extended in place so
            two prims of the same name become two assets.
        :return: The asset written.
        """
        name = self._unique_name(object_prim.GetName(), names)
        files = AssetFiles(directory=directory / ASSETS_DIRECTORY / name, name=name)
        files.payloads.mkdir(parents=True, exist_ok=True)
        files.physics.parent.mkdir(parents=True, exist_ok=True)

        parts = self._parts_of(object_prim)
        self._write_geometries(object_prim, files, parts)
        self._write_materials(object_prim, files)
        self._write_instances(object_prim, files, parts)
        self._write_base(files, parts)
        self._write_physics(object_prim, files, parts)
        self._write_interface(files)
        return WrittenAsset(
            source_path=object_prim.GetPath(),
            category=object_prim.GetParent().GetName(),
            files=files,
            parts=[WrittenPart(name=part.name, pivot=part.pivot) for part in parts],
        )

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

    # %% the parts an object is made of

    def _parts_of(self, object_prim: Usd.Prim) -> List[PartGeometry]:
        """
        :param object_prim: The prim being written as an asset.
        :return: Every part of it, each naming the mesh it is cut from and the faces it
            keeps.
        """
        meshes = [
            UsdGeom.Mesh(child)
            for child in object_prim.GetChildren()
            if child.IsA(UsdGeom.Mesh)
        ]
        if len(meshes) != 1 or self.segmentation is None:
            return [
                PartGeometry(name=self._sole_part_name(meshes, mesh), mesh=mesh)
                for mesh in meshes
            ]

        [mesh] = meshes
        taken = np.zeros(len(mesh.GetFaceVertexCountsAttr().Get()), dtype=bool)
        parts = []
        for part in self.segmentation.parts_of(object_prim, mesh):
            faces = np.asarray(part.faces, dtype=bool) & ~taken
            taken |= faces
            parts.append(
                PartGeometry(name=part.name, mesh=mesh, faces=faces, pivot=part.pivot)
            )
        if taken.all():
            return parts
        return parts + [PartGeometry(name=DEFAULT_PART_NAME, mesh=mesh, faces=~taken)]

    @staticmethod
    def _sole_part_name(meshes: List[UsdGeom.Mesh], mesh: UsdGeom.Mesh) -> str:
        """
        :param meshes: Every mesh the object holds.
        :param mesh: The one being named.
        :return: What to call the part that mesh becomes - an object made of one mesh
            names its part the same way whatever the stage called the mesh, and one
            made of several keeps the names telling them apart.
        """
        return DEFAULT_PART_NAME if len(meshes) == 1 else mesh.GetPrim().GetName()

    def _write_geometries(
        self, object_prim: Usd.Prim, files: AssetFiles, parts: List[PartGeometry]
    ) -> None:
        """
        Write the mesh arrays of every part, in the object's own space.

        :param object_prim: The prim whose geometry to write.
        :param files: The files the asset is written as.
        :param parts: The parts the object is made of.
        """
        layer = Sdf.Layer.CreateNew(str(files.geometries))
        stage = Usd.Stage.Open(layer)
        scope = Sdf.Path(f"/{GEOMETRIES_SCOPE_NAME}")
        UsdGeom.Scope.Define(stage, scope)
        for part in parts:
            UsdGeom.Xform.Define(stage, scope.AppendChild(part.name))
            self._write_part_mesh(
                stage,
                scope.AppendChild(part.name).AppendChild(part.name),
                object_prim,
                part,
            )
        if self.vertex_sharing is VertexSharing.BY_POSITION:
            for prim in stage.TraverseAll():
                if prim.IsA(UsdGeom.Mesh):
                    _share_mesh_vertices(UsdGeom.Mesh(prim))
        layer.Save()

    @staticmethod
    def _write_part_mesh(
        stage: Usd.Stage, path: Sdf.Path, object_prim: Usd.Prim, part: PartGeometry
    ) -> None:
        """
        Write the faces one part keeps as a mesh of its own, about the part's pivot.

        :param stage: The binary crate, open.
        :param path: Where to define the mesh.
        :param object_prim: The prim the asset is written from, whose space the mesh is
            written in.
        :param part: The part to write.
        """
        source = part.mesh
        counts = np.asarray(source.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        corner_indices = np.asarray(
            source.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
        )
        taken = (
            part.faces if part.faces is not None else np.ones(len(counts), dtype=bool)
        )
        selection = FaceSelection.of(counts, corner_indices, taken)

        written = UsdGeom.Mesh.Define(stage, path)
        to_object = np.array(
            UsdGeom.Xformable(source.GetPrim()).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            * UsdGeom.Xformable(object_prim)
            .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            .GetInverse(),
            dtype=float,
        )
        points = np.asarray(source.GetPointsAttr().Get(), dtype=float)[selection.points]
        points = points @ to_object[:3, :3] + to_object[3, :3] - np.asarray(part.pivot)
        written_points = Vt.Vec3fArray.FromNumpy(
            np.ascontiguousarray(points, dtype=np.float32)
        )
        written.CreatePointsAttr(written_points)
        written.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(selection.counts.astype(np.int32))
        )
        written.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(selection.corner_indices.astype(np.int32))
        )
        written.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(written_points))

        normals = source.GetNormalsAttr().Get()
        if normals is not None:
            interpolation = source.GetNormalsInterpolation()
            turned = np.asarray(normals, dtype=float) @ to_object[:3, :3]
            written.CreateNormalsAttr(
                Vt.Vec3fArray.FromNumpy(
                    np.ascontiguousarray(
                        selection.values(turned, interpolation), dtype=np.float32
                    )
                )
            )
            written.SetNormalsInterpolation(interpolation)

        for primvar in UsdGeom.PrimvarsAPI(source).GetPrimvars():
            interpolation = primvar.GetInterpolation()
            values = primvar.ComputeFlattened()
            if values is None:
                continue
            UsdGeom.PrimvarsAPI(written).CreatePrimvar(
                primvar.GetPrimvarName(), primvar.GetTypeName(), interpolation
            ).Set(
                _held_as_usd_holds_it(
                    selection.values(values, interpolation), primvar.GetTypeName()
                )
            )

        for attribute in source.GetPrim().GetAuthoredAttributes():
            name = attribute.GetName()
            value = attribute.Get()
            if name in MESH_ARRAY_ATTRIBUTES or value is None:
                continue
            if name.startswith("primvars:") or name.startswith("xformOp"):
                continue
            written.GetPrim().CreateAttribute(
                name,
                attribute.GetTypeName(),
                variability=attribute.GetVariability(),
            ).Set(value)

    # %% the materials an asset carries

    def _write_materials(self, object_prim: Usd.Prim, files: AssetFiles) -> None:
        """
        Write the asset's materials, with every texture they read copied beside them.

        :param object_prim: The prim whose materials to write.
        :param files: The files the asset is written as.
        """
        layer = Sdf.Layer.CreateNew(str(files.materials))
        scope = Sdf.CreatePrimInLayer(layer, Sdf.Path(f"/{MATERIALS_SCOPE_NAME}"))
        scope.specifier = Sdf.SpecifierDef
        scope.typeName = UsdGeom.Tokens.Scope
        for material in self._materials_of(object_prim):
            written = Sdf.Path(f"/{MATERIALS_SCOPE_NAME}/{material.GetName()}")
            Sdf.CopySpec(self.stage.GetRootLayer(), material.GetPath(), layer, written)
            _retarget(layer, written, material.GetPath(), written)
            self._copy_textures(material, files, layer, written)
        layer.Save()

    @staticmethod
    def _materials_of(object_prim: Usd.Prim) -> List[Usd.Prim]:
        """
        :param object_prim: The prim the asset is written from.
        :return: Every material defined beneath it.
        """
        return [
            prim for prim in Usd.PrimRange(object_prim) if prim.IsA(UsdShade.Material)
        ]

    def _write_instances(
        self, object_prim: Usd.Prim, files: AssetFiles, parts: List[PartGeometry]
    ) -> None:
        """
        Put each part's mesh together with the material covering it.

        :param object_prim: The prim the asset is written from.
        :param files: The files the asset is written as.
        :param parts: The parts the object is made of.
        """
        layer = Sdf.Layer.CreateNew(str(files.instances))
        stage = Usd.Stage.Open(layer)
        scope = Sdf.Path(f"/{INSTANCES_SCOPE_NAME}")
        UsdGeom.Scope.Define(stage, scope)
        for part in parts:
            path = scope.AppendChild(part.name)
            assembled = UsdGeom.Xform.Define(stage, path)
            assembled.GetPrim().GetReferences().AddReference(
                f"./{GEOMETRIES_LAYER_NAME}",
                Sdf.Path(f"/{GEOMETRIES_SCOPE_NAME}/{part.name}"),
            )
            material = self._material_of(part)
            if material is None:
                continue
            bound = UsdShade.Material.Define(
                stage, path.AppendChild(BOUND_MATERIAL_NAME)
            )
            bound.GetPrim().GetReferences().AddReference(
                f"./{MATERIALS_LAYER_NAME}",
                Sdf.Path(f"/{MATERIALS_SCOPE_NAME}/{material.GetName()}"),
            )
            surface = stage.OverridePrim(path.AppendChild(part.name))
            UsdShade.MaterialBindingAPI.Apply(surface).Bind(bound)
        layer.Save()

    @staticmethod
    def _material_of(part: PartGeometry) -> Optional[Usd.Prim]:
        """
        :param part: The part to look up.
        :return: The material the stage covered the part's mesh in, or ``None``.
        """
        material, _ = UsdShade.MaterialBindingAPI(
            part.mesh.GetPrim()
        ).ComputeBoundMaterial()
        return material.GetPrim() if material else None

    @staticmethod
    def _write_base(files: AssetFiles, parts: List[PartGeometry]) -> None:
        """
        Write the asset's hierarchy: one prim per part, standing at its pivot.

        :param files: The files the asset is written as.
        :param parts: The parts the object is made of.
        """
        layer = Sdf.Layer.CreateNew(str(files.base))
        stage = Usd.Stage.Open(layer)
        asset = UsdGeom.Xform.Define(stage, f"/{files.name}")
        stage.SetDefaultPrim(asset.GetPrim())
        geometry = UsdGeom.Scope.Define(
            stage, asset.GetPath().AppendChild(GEOMETRY_SCOPE_NAME)
        )
        for part in parts:
            placed = UsdGeom.Xform.Define(
                stage, geometry.GetPath().AppendChild(part.name)
            )
            placed.GetPrim().GetReferences().AddReference(
                f"./{INSTANCES_LAYER_NAME}",
                Sdf.Path(f"/{INSTANCES_SCOPE_NAME}/{part.name}"),
            )
            placed.AddTranslateOp().Set(Gf.Vec3d(part.pivot))
        layer.Save()

    def _write_physics(
        self, object_prim: Usd.Prim, files: AssetFiles, parts: List[PartGeometry]
    ) -> None:
        """
        Write what a physics engine collides the asset against, on top of the surfaces
        it stands for so that editing it never means rewriting them.

        :param object_prim: The prim whose geometry is collided against.
        :param files: The files the asset is written as.
        :param parts: The parts the object is made of.
        """
        layer = Sdf.Layer.CreateNew(str(files.physics))
        stage = Usd.Stage.Open(layer)
        asset = stage.OverridePrim(f"/{files.name}")
        stage.SetDefaultPrim(asset)
        self._author_collision(stage, object_prim, files.name, parts)
        layer.Save()

    @staticmethod
    def _write_interface(files: AssetFiles) -> None:
        """
        Write the layer a scene references, which names the asset a component, holds
        back its contents until they are asked for, and says whether physics is among
        them.

        :param files: The files the asset is written as.
        """
        layer = Sdf.Layer.CreateNew(str(files.interface))
        stage = Usd.Stage.Open(layer)
        asset = UsdGeom.Xform.Define(stage, f"/{files.name}")
        stage.SetDefaultPrim(asset.GetPrim())
        Usd.ModelAPI(asset).SetKind(Kind.Tokens.component)
        asset.GetPrim().GetPayloads().AddPayload(
            f"./{PAYLOADS_DIRECTORY}/{BASE_LAYER_NAME}"
        )

        variants = (
            asset.GetPrim().GetVariantSets().AddVariantSet(PHYSICS_VARIANT_SET_NAME)
        )
        variants.AddVariant(NO_PHYSICS_VARIANT_NAME)
        variants.AddVariant(PHYSICS_VARIANT_NAME)
        variants.SetVariantSelection(PHYSICS_VARIANT_NAME)
        with variants.GetVariantEditContext():
            asset.GetPrim().GetPayloads().AddPayload(
                f"./{PAYLOADS_DIRECTORY}/{PHYSICS_DIRECTORY}/{PHYSICS_LAYER_NAME}"
            )

        UsdGeom.ModelAPI(asset).SetExtentsHint(
            UsdGeom.ModelAPI(asset).ComputeExtentsHint(
                UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            )
        )
        layer.Save()

    # %% textures

    def _copy_textures(
        self,
        material: Usd.Prim,
        files: AssetFiles,
        layer: Sdf.Layer,
        written: Sdf.Path,
    ) -> None:
        """
        Copy every image a material reads into the asset's own directory, and point the
        written material at the copy.

        :param material: The material whose textures to copy.
        :param files: The files the asset is written as.
        :param layer: The written material layer, whose asset paths are rewritten.
        :param written: Where the material sits in that layer.
        """
        for prim in Usd.PrimRange(material):
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
            downscaled = downscaled_texture_path(readable, self.maximum_texture_size)
            files.textures.mkdir(parents=True, exist_ok=True)
            copy = files.textures / Path(asset_path.path).name
            shutil.copyfile(downscaled, copy)

            attribute_path = written.AppendPath(
                prim.GetPath().MakeRelativePath(material.GetPath())
            ).AppendProperty(file_input.GetFullName())
            layer.GetAttributeAtPath(attribute_path).default = Sdf.AssetPath(
                f"./{TEXTURES_DIRECTORY}/{copy.name}"
            )

    # %% collision

    def _author_collision(
        self,
        stage: Usd.Stage,
        object_prim: Usd.Prim,
        name: str,
        parts: List[PartGeometry],
    ) -> None:
        """
        Give a physics engine something to collide the asset against.

        :param stage: The asset's physics layer, open.
        :param object_prim: The prim whose geometry is collided against.
        :param name: The asset's name.
        :param parts: The parts the object is made of.
        """
        if self.collision_proxy is CollisionProxy.NONE:
            return
        if self.collision_proxy is CollisionProxy.CONVEX_DECOMPOSITION:
            self._collide_against_the_surface(stage, name, parts)
            return
        self._author_collision_proxy(stage, object_prim, name)

    @staticmethod
    def _collide_against_the_surface(
        stage: Usd.Stage, name: str, parts: List[PartGeometry]
    ) -> None:
        """
        Mark the asset's own surfaces as what it is collided against, for a physics
        engine to approximate by convex pieces when it loads them.

        :param stage: The asset's physics layer, open.
        :param name: The asset's name.
        :param parts: The parts the object is made of.
        """
        for part in parts:
            surface = stage.OverridePrim(
                f"/{name}/{GEOMETRY_SCOPE_NAME}/{part.name}/{part.name}"
            )
            UsdPhysics.CollisionAPI.Apply(surface)
            UsdPhysics.MeshCollisionAPI.Apply(surface).CreateApproximationAttr().Set(
                UsdPhysics.Tokens.convexDecomposition
            )

    @staticmethod
    def _author_collision_proxy(
        stage: Usd.Stage, object_prim: Usd.Prim, name: str
    ) -> None:
        """
        Author the box a physics engine collides against in place of the asset's own
        surfaces, as a guide the renderer leaves out of the picture.

        :param stage: The asset's physics layer, open.
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

        proxy = UsdGeom.Cube.Define(stage, f"/{name}/{COLLISION_PROXY_NAME}")
        proxy.GetSizeAttr().Set(1.0)
        proxy.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        proxy.AddTranslateOp().Set(Gf.Vec3d(bounds.GetMidpoint()))
        proxy.AddScaleOp().Set(Gf.Vec3f(bounds.GetMax() - bounds.GetMin()))
        UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())

    # %% placement

    def _stand_scene(self, root: UsdGeom.Xform) -> None:
        """
        Move the library's root so the scene stands where it was asked to.

        :param root: The library's root, which every placement sits beneath.
        """
        if self.root_placement is RootPlacement.STAGE_ORIGIN:
            return
        ground = scene_ground(self.stage, geometry_owning_prims(self.stage))
        root.AddTranslateOp().Set(-ground)

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


# %% copied opinions


def _retarget(
    layer: Sdf.Layer, root: Sdf.Path, source: Sdf.Path, destination: Sdf.Path
) -> None:
    """
    Point every relationship and connection copied into a layer at the copy of what it
    named.

    A copied path keeps the one it was authored with, so a shader connection would
    still name the stage the asset was lifted out of and resolve to nothing once the
    asset stands on its own.

    :param layer: The written layer to retarget within.
    :param root: The path to traverse from.
    :param source: The prefix the copied paths carry.
    :param destination: The prefix to give them instead.
    """

    def moved(path: Sdf.Path) -> Sdf.Path:
        if not path.HasPrefix(source):
            return path
        return path.ReplacePrefix(source, destination)

    def retarget(path: Sdf.Path) -> None:
        relationship = layer.GetRelationshipAtPath(path)
        if relationship is not None:
            targets = relationship.targetPathList
            targets.explicitItems = [moved(target) for target in targets.explicitItems]
            return
        attribute = layer.GetAttributeAtPath(path)
        if attribute is None:
            return
        connections = attribute.connectionPathList
        connections.explicitItems = [
            moved(connection) for connection in connections.explicitItems
        ]

    layer.Traverse(root, retarget)
