from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import NoneType

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from numpy.typing import NDArray
from typing_extensions import Dict, List, Optional, Self, Sequence, Tuple, Type

from krrood.class_diagrams.mocking import MockedClass, MockedModule

from semantic_digital_twin.adapters.package_resolver import (
    CompositePathResolver,
    PathResolver,
)
from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdGeometryTypeError,
)
from semantic_digital_twin.adapters.world_model_parser import WorldModelParser
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    MeshFileType,
    Scale,
    Shape,
    Sphere,
)
from semantic_digital_twin.world_description.mesh_file_storage import (
    MeshFileStorage,
)
from semantic_digital_twin.world_description.inertial_properties import (
    Inertial,
    InertiaTensor,
    PrincipalAxes,
    PrincipalMoments,
)
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


# %% pxr fallbacks
#
# Usd.Prim, Usd.Stage, Gf.Matrix4d, and Sdf.Path are used as dataclass field and
# parameter types here and in the modules building on this one. Leaving them unbound
# when usd-core is missing crashes any code that resolves those type hints (e.g.
# ORM/class-diagram generation), not just code that actually parses a USD stage.
# Binding them to these mocks keeps every annotation resolvable; actually parsing a
# stage still fails loudly, via krrood.class_diagrams.mocking.MockedClass.


@dataclass
class _MockedUsdPrim(MockedClass):
    """
    Mocked class for Usd.Prim in pxr.
    """


@dataclass
class _MockedUsdStage(MockedClass):
    """
    Mocked class for Usd.Stage in pxr.
    """


@dataclass
class _MockedUsdModule(MockedModule):
    """
    Mocked module for pxr.Usd.
    """

    Prim: Type[_MockedUsdPrim] = _MockedUsdPrim
    Stage: Type[_MockedUsdStage] = _MockedUsdStage


@dataclass
class _MockedGfMatrix4d(MockedClass):
    """
    Mocked class for Gf.Matrix4d in pxr.
    """


@dataclass
class _MockedGfModule(MockedModule):
    """
    Mocked module for pxr.Gf.
    """

    Matrix4d: Type[_MockedGfMatrix4d] = _MockedGfMatrix4d


@dataclass
class _MockedSdfPath(MockedClass):
    """
    Mocked class for Sdf.Path in pxr.
    """


@dataclass
class _MockedSdfModule(MockedModule):
    """
    Mocked module for pxr.Sdf.
    """

    Path: Type[_MockedSdfPath] = _MockedSdfPath


try:
    from pxr import Ar, Gf, Kind, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    logger.warning(
        "usd-core is required for USD parsing. Please install it using "
        "'pip install usd-core'"
    )
    Usd = _MockedUsdModule()
    Gf = _MockedGfModule()
    Sdf = _MockedSdfModule()
    # No member of these is used as a dataclass field type, so a bare mock is enough to
    # keep this module - and the ones importing these names from it - importable.
    Ar = MockedModule()
    Kind = MockedModule()
    UsdGeom = MockedModule()
    UsdPhysics = MockedModule()
    UsdShade = MockedModule()

try:
    from pxr import UsdSemantics
except ImportError:
    # UsdSemantics (UsdSemantics.LabelsAPI) is only available from usd-core 24.11
    # onward; an older install simply never yields UsdSemanticLabels annotations.
    UsdSemantics = NoneType


def _usd_pose_to_transform(
    position: Gf.Vec3d, rotation: Gf.Quatf, **kwargs
) -> HomogeneousTransformationMatrix:
    """
    Build a transform from a USD position and quaternion rotation.

    :param position: The translation.
    :param rotation: The rotation.
    :param kwargs: Forwarded to ``HomogeneousTransformationMatrix.from_xyz_quaternion``
        (typically ``reference_frame``/``child_frame``).
    :return: The built transform.
    """
    imaginary = rotation.GetImaginary()
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=position[0],
        pos_y=position[1],
        pos_z=position[2],
        quat_x=imaginary[0],
        quat_y=imaginary[1],
        quat_z=imaginary[2],
        quat_w=rotation.GetReal(),
        **kwargs,
    )


def _decompose_local_transform(
    prim: Usd.Prim, link_to_world: Gf.Matrix4d
) -> Tuple[HomogeneousTransformationMatrix, Gf.Vec3d]:
    """
    Decompose a prim's pose relative to its enclosing link into a rigid transform and a
    scale, regardless of how the prim's ``xformOpOrder`` was authored.

    :param prim: The prim whose pose to decompose.
    :param link_to_world: The enclosing link's local-to-world transform.
    :return: The prim's rigid pose relative to the link (with no ``reference_frame`` set
        - the caller sets it), and its scale relative to the link.
    """
    prim_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    prim_to_link = prim_to_world * link_to_world.GetInverse()
    transform = Gf.Transform(prim_to_link)
    translation = transform.GetTranslation()
    rotation_quat = transform.GetRotation().GetQuat()
    origin = _usd_pose_to_transform(translation, rotation_quat)
    return origin, transform.GetScale()


class Shading(StrEnum):
    """
    How a renderer is asked to light a surface.
    """

    LIT = "lit"
    """
    Lit by the scene's lights, the usual way a modelled surface is drawn.
    """

    UNLIT = "unlit"
    """
    Drawn at the brightness its texture holds, whatever the lights. A photogrammetric
    texture is a photograph, so the lighting is already in its pixels and lighting it
    again darkens it.
    """


FULLY_EMISSIVE = (1.0, 1.0, 1.0)
"""
The emissive colour that has a renderer draw a surface at the brightness its texture
holds, rather than lighting it.
"""


class UsdGeomPrimType(StrEnum):
    """
    The USD prim type names (``UsdGeom.Xxx.Define``'s ``Xxx``) this parser builds a
    Shape for.
    """

    MESH = "Mesh"
    CUBE = "Cube"
    SPHERE = "Sphere"
    CYLINDER = "Cylinder"

    def create_shape(
        self,
        prim: Usd.Prim,
        link_to_world: Gf.Matrix4d,
        shading: Shading,
        maximum_texture_size: Optional[int],
    ) -> Shape:
        """
        Creates the Shape a prim of this type describes.

        :param prim: The prim to create a shape for.
        :param link_to_world: The enclosing link's local-to-world transform.
        :param shading: How a renderer is asked to light the shape.
        :param maximum_texture_size: Longest side the shape's texture may have,
            in pixels, or ``None`` to keep it at the size it was authored.
        :return: The created shape.
        """
        builders: Dict[UsdGeomPrimType, Type[UsdShapeBuilder]] = {
            UsdGeomPrimType.MESH: UsdMeshShapeBuilder,
            UsdGeomPrimType.CUBE: UsdCubeShapeBuilder,
            UsdGeomPrimType.SPHERE: UsdSphereShapeBuilder,
            UsdGeomPrimType.CYLINDER: UsdCylinderShapeBuilder,
        }
        return builders[self](
            prim, link_to_world, shading, maximum_texture_size
        ).build()


class UsdAxis(StrEnum):
    """
    A USD local-frame axis token, as authored on a joint's ``axis`` attribute or a
    ``UsdGeom.Cylinder``'s.
    """

    X = "X"
    Y = "Y"
    Z = "Z"

    @property
    def index(self) -> int:
        """
        :return: The position this axis takes in a three-component vector.
        """
        return {UsdAxis.X: 0, UsdAxis.Y: 1, UsdAxis.Z: 2}[self]

    def vector(self, reference_frame: Body) -> Vector3:
        """
        :param reference_frame: The frame the returned vector is expressed in.
        :return: The unit vector this axis token denotes.
        """
        unit_vectors = {
            UsdAxis.X: (1.0, 0.0, 0.0),
            UsdAxis.Y: (0.0, 1.0, 0.0),
            UsdAxis.Z: (0.0, 0.0, 1.0),
        }
        return Vector3(*unit_vectors[self], reference_frame=reference_frame)


@dataclass
class UsdShapeBuilder(ABC):
    """
    Builds the :class:`~semantic_digital_twin.world_description.geometry.Shape` one USD
    geometry prim describes, relative to its enclosing link.
    """

    prim: Usd.Prim
    """
    The prim to build a shape for.
    """

    link_to_world: Gf.Matrix4d
    """
    The enclosing link's local-to-world transform.
    """

    shading: Shading = Shading.LIT
    """
    How a renderer is asked to light the shape.
    """

    maximum_texture_size: Optional[int] = None
    """
    Longest side the shape's texture may have, in pixels, or ``None`` to keep it at
    the size it was authored.
    """

    @abstractmethod
    def build(self) -> Shape:
        """
        :return: The shape :attr:`prim` describes.
        """


@dataclass
class UsdCubeShapeBuilder(UsdShapeBuilder):
    """
    Builds the Box shape a ``UsdGeom.Cube`` prim describes.
    """

    def build(self) -> Box:
        origin, scale = _decompose_local_transform(self.prim, self.link_to_world)
        side = UsdGeom.Cube(self.prim).GetSizeAttr().Get()
        return Box(
            origin=origin,
            scale=Scale(side * scale[0], side * scale[1], side * scale[2]),
        )


@dataclass
class UsdSphereShapeBuilder(UsdShapeBuilder):
    """
    Builds the Sphere shape a ``UsdGeom.Sphere`` prim describes.

    A sphere has no per-axis size, so a non-uniform scale is approximated by its average
    factor across the three axes.
    """

    def build(self) -> Sphere:
        origin, scale = _decompose_local_transform(self.prim, self.link_to_world)
        radius = UsdGeom.Sphere(self.prim).GetRadiusAttr().Get()
        average_scale = (scale[0] + scale[1] + scale[2]) / 3.0
        return Sphere(origin=origin, radius=radius * average_scale)


@dataclass
class UsdCylinderShapeBuilder(UsdShapeBuilder):
    """
    Builds the Cylinder shape a ``UsdGeom.Cylinder`` prim describes.

    :class:`~semantic_digital_twin.world_description.geometry.Cylinder` is always
    aligned with its local Z axis, so a cylinder authored along X or Y gets an extra
    rotation folded into its origin to align it.
    """

    def build(self) -> Cylinder:
        origin, scale = _decompose_local_transform(self.prim, self.link_to_world)
        usd_cylinder = UsdGeom.Cylinder(self.prim)
        axis = usd_cylinder.GetAxisAttr().Get()
        # scale is in the prim's own local axes, unaffected by the alignment rotation
        # below (which only reorients origin), so the axis also picks out which scale
        # components are the cylinder's height vs. its two radial directions.
        if axis == UsdAxis.X:
            alignment = HomogeneousTransformationMatrix.from_xyz_rpy(pitch=math.pi / 2)
            height_scale, radial_scale = scale[0], (scale[1] + scale[2]) / 2.0
        elif axis == UsdAxis.Y:
            alignment = HomogeneousTransformationMatrix.from_xyz_rpy(roll=-math.pi / 2)
            height_scale, radial_scale = scale[1], (scale[0] + scale[2]) / 2.0
        else:
            alignment = HomogeneousTransformationMatrix()
            height_scale, radial_scale = scale[2], (scale[0] + scale[1]) / 2.0
        origin = origin @ alignment
        return Cylinder(
            origin=origin,
            width=usd_cylinder.GetRadiusAttr().Get() * 2 * radial_scale,
            height=usd_cylinder.GetHeightAttr().Get() * height_scale,
        )


@dataclass
class UsdMeshShapeBuilder(UsdShapeBuilder):
    """
    Builds the Mesh shape one USD mesh prim describes, positioned relative to its link.

    Applied directly to the raw vertex positions rather than split into a rotation,
    translation, and :class:`~semantic_digital_twin.world_description.geometry.Scale`
    for the shape's origin, since a mesh's local-to-link transform can carry a non-
    uniform scale or shear: decomposing a general affine transform into
    translation/rotation/scale is ill-posed in the presence of shear, while applying the
    matrix to the points themselves is exact regardless.
    """

    def build(self) -> Mesh:
        mesh_to_world = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        mesh_to_link = mesh_to_world * self.link_to_world.GetInverse()

        mesh_geometry = UsdGeom.Mesh(self.prim)
        local_vertices = np.array(mesh_geometry.GetPointsAttr().Get())
        vertices = self._transform_points(local_vertices, mesh_to_link)
        faces = self._triangulate(
            mesh_geometry.GetFaceVertexCountsAttr().Get(),
            mesh_geometry.GetFaceVertexIndicesAttr().Get(),
        )
        if mesh_geometry.GetDoubleSidedAttr().Get():
            faces = self._both_windings(faces)
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        texture_file_path = self._diffuse_texture_path(self.prim)
        uv_per_point = self._uv_coordinates(self.prim)
        if texture_file_path is None or uv_per_point is None:
            return Mesh.from_trimesh(
                mesh=trimesh_mesh,
                origin=HomogeneousTransformationMatrix(),
                file_type=MeshFileType.GLB,
            )

        texture_file_path = downscaled_texture_path(
            texture_file_path, self.maximum_texture_size
        )

        # The st primvar is per point, so it indexes the vertices as they already are;
        # handing the uv to Mesh.from_trimesh instead would have it split every vertex
        # per face corner again, which for a surface holding both windings duplicates
        # the whole mesh.
        trimesh_mesh.visual = trimesh.visual.TextureVisuals(uv=uv_per_point)
        if self.shading is Shading.LIT:
            return Mesh.from_trimesh(
                mesh=trimesh_mesh,
                origin=HomogeneousTransformationMatrix(),
                texture_file_path=texture_file_path,
                file_type=MeshFileType.GLB,
            )

        texture = Image.open(texture_file_path)
        trimesh_mesh.visual.material = PBRMaterial(
            name=Path(texture_file_path).stem,
            baseColorTexture=texture,
            metallicFactor=0.0,
            roughnessFactor=1.0,
            emissiveFactor=FULLY_EMISSIVE,
            emissiveTexture=texture,
        )
        return Mesh.from_trimesh(
            mesh=trimesh_mesh,
            origin=HomogeneousTransformationMatrix(),
            file_type=MeshFileType.GLB,
        )

    @staticmethod
    def _both_windings(faces: NDArray[np.int64]) -> NDArray[np.int64]:
        """
        Adds the reverse of every face.

        A face is drawn from the side its winding faces, so a renderer that discards
        back faces leaves a surface authored ``doubleSided`` - a scanned sheet, which
        has no inside - invisible from behind. Holding both windings makes it show from
        either side in any renderer, since it no longer depends on one honouring the
        flag.

        :param faces: The triangles of the surface.
        :return: Those triangles followed by their reverses.
        """
        return np.vstack([faces, faces[:, ::-1]])

    @staticmethod
    def _transform_points(
        points: NDArray[np.float64], matrix: Gf.Matrix4d
    ) -> NDArray[np.float64]:
        """
        Applies a USD transform to an array of points.

        :param points: An ``(n, 3)`` array of points in the transform's source frame.
        :param matrix: The transform to apply.
        :return: An ``(n, 3)`` array of the transformed points.
        """
        points_homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        return (points_homogeneous @ np.array(matrix))[:, :3]

    @staticmethod
    def _diffuse_texture_path(mesh_prim: Usd.Prim) -> Optional[str]:
        """
        Resolves the file path of the diffuse texture bound to a mesh prim's material.

        :param mesh_prim: The mesh prim to look up.
        :return: The resolved path to the diffuse texture image, or ``None`` if the prim
            has no bound material, its surface shader has no ``diffuseColor`` input, or
            that input is not connected to a texture (e.g. a flat colour).
        """
        material, _ = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()
        if not material:
            return None

        surface_source = material.GetSurfaceOutput().GetConnectedSource()
        if surface_source is None:
            return None
        surface_shader = UsdShade.Shader(surface_source[0])

        diffuse_input = surface_shader.GetInput("diffuseColor")
        diffuse_source = diffuse_input.GetConnectedSource() if diffuse_input else None
        if diffuse_source is None:
            return None
        texture_shader = UsdShade.Shader(diffuse_source[0])

        file_input = texture_shader.GetInput("file")
        asset_path = file_input.Get() if file_input else None
        if asset_path is None or not asset_path.resolvedPath:
            return None
        return readable_texture_path(asset_path.resolvedPath)

    @staticmethod
    def _uv_coordinates(mesh_prim: Usd.Prim) -> Optional[NDArray[np.float64]]:
        """
        Reads a mesh prim's per-point UV coordinates from its ``st`` primvar.

        :param mesh_prim: The mesh prim to look up.
        :return: An ``(n_points, 2)`` array of UV coordinates, or ``None`` if the prim
            has no ``st`` primvar, or its interpolation is not per-point
            (``vertex``/``varying``).
        """
        primvar = UsdGeom.PrimvarsAPI(mesh_prim).GetPrimvar("st")
        if not primvar.IsDefined():
            return None
        if primvar.GetInterpolation() not in (
            UsdGeom.Tokens.vertex,
            UsdGeom.Tokens.varying,
        ):
            return None
        values = primvar.Get()
        if not values:
            return None
        return np.array(values, dtype=np.float64)

    @staticmethod
    def _triangulate(
        face_vertex_counts: Sequence[int], face_vertex_indices: Sequence[int]
    ) -> NDArray[np.int64]:
        """
        Fan-triangulates a USD mesh's polygonal faces.

        :param face_vertex_counts: The number of vertices of each face.
        :param face_vertex_indices: The faces' vertex indices, flattened in
            ``face_vertex_counts`` order.
        :return: An ``(n, 3)`` array of triangle vertex indices.
        """
        triangles = []
        cursor = 0
        for count in face_vertex_counts:
            face = face_vertex_indices[cursor : cursor + count]
            for i in range(1, count - 1):
                triangles.append((face[0], face[i], face[i + 1]))
            cursor += count
        return np.array(triangles, dtype=np.int64)


# %% the prims a stage holds


def geometry_owning_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    :param stage: The stage to search.
    :return: Every prim of the stage that directly holds renderable geometry, in stage
        order - each geometry prim belongs to exactly one of them, so no geometry of
        the stage is left out.
    """
    owning_prims = []
    seen_paths = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        parent = prim.GetParent()
        if parent.GetPath() in seen_paths:
            continue
        seen_paths.add(parent.GetPath())
        owning_prims.append(parent)
    return owning_prims


# %% texture files


def readable_texture_path(resolved_path: str) -> str:
    """
    Makes a resolved texture path one an image reader can open.

    A texture inside a USD package (a ``.usdz``) resolves to a path of the form
    ``package[path/inside]``, which only USD's asset resolver can read, so its bytes
    are written out to a file of their own.

    :param resolved_path: The texture's resolved asset path.
    :return: A filesystem path holding the texture's bytes.
    """
    if not Ar.IsPackageRelativePath(resolved_path):
        return resolved_path

    _, path_inside_package = Ar.SplitPackageRelativePathInner(resolved_path)
    extracted_path = (
        MeshFileStorage().allocate_directory() / Path(path_inside_package).name
    )
    asset = Ar.GetResolver().OpenAsset(Ar.ResolvedPath(resolved_path))
    extracted_path.write_bytes(asset.GetBuffer())
    return str(extracted_path)


def downscaled_texture_path(
    texture_file_path: str, maximum_texture_size: Optional[int]
) -> str:
    """
    Shrinks a texture whose longest side is past a maximum, keeping its shape.

    A scanned surface can carry a texture of hundreds of megapixels, which a viewer
    holds decoded in memory whatever the size of the file it came from.

    :param texture_file_path: Path of the texture image.
    :param maximum_texture_size: Longest side the texture may have, in pixels, or
        ``None`` to keep it at the size it was authored.
    :return: A filesystem path holding a texture within the maximum.
    """
    if maximum_texture_size is None:
        return texture_file_path

    texture = Image.open(texture_file_path)
    if max(texture.size) <= maximum_texture_size:
        return texture_file_path

    texture.thumbnail((maximum_texture_size, maximum_texture_size))
    downscaled_path = (
        MeshFileStorage().allocate_directory() / Path(texture_file_path).name
    )
    texture.save(downscaled_path)
    return str(downscaled_path)


# %% stage parser


@dataclass
class USDStageParser(WorldModelParser, ABC):
    """
    Base for every parser that turns a USD stage into a world.

    Holds what reading any stage takes - opening it, identifying the prim to treat as
    its root, and building the shapes, inertial properties, and semantic labels of a
    body - and leaves what a stage's *structure* is to each subclass.

    .. note::
        Requires the ``usd-core`` package (``pxr``) to read the stage.

    .. note::
        USD is right-handed. Unlike the axis convention, the up axis and the unit scale
        are not fixed: they are stage metadata (``UsdGeom.GetStageUpAxis``, typically Y
        or Z, and ``UsdGeom.GetStageMetersPerUnit``). Neither is ever assumed here -
        every transform built is relative to a prim's own parent, computed straight from
        the authored ``xformOpOrder``, so it comes out correct in whatever up axis and
        unit scale the stage itself declares.
    """

    stage: Usd.Stage
    """
    The USD stage to parse.

    Unlike URDF/SDF, a USD stage is not a flat block of source text one file happens to
    hold: composition (references, sublayers, ...) makes the opened stage itself, not
    any single file's text, the thing that fully describes what to parse. So this - not
    a file path - is the parser's payload field; :meth:`from_file` opens the file first.
    """

    prefix: Optional[str] = None
    """
    The prefix for every name used in this world.
    """

    path_resolver: PathResolver = field(default_factory=CompositePathResolver)
    """
    The path resolver used for the asset references of this stage.
    """

    shading: Shading = Shading.LIT
    """
    How a renderer is asked to light the shapes of this stage.
    """

    maximum_texture_size: Optional[int] = None
    """
    Longest side a texture of this stage may have, in pixels, or ``None`` to keep each
    at the size it was authored.
    """

    def __post_init__(self):
        if self.prefix is None:
            self.prefix = Path(self.stage.GetRootLayer().identifier).stem

    # %% construction

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: Optional[str] = None,
        path_resolver: Optional[PathResolver] = None,
    ) -> Self:
        """
        Creates a parser for a USD stage file.

        :param file_path: The path of the stage file to parse.
        :param prefix: The prefix for every name used in this world.
        :param path_resolver: The resolver for the asset references of the stage.
        :return: A parser for the described world.
        """
        path_resolver = path_resolver or CompositePathResolver()
        resolved_path = path_resolver.resolve(file_path)
        parser = cls(stage=Usd.Stage.Open(resolved_path), prefix=prefix)
        parser.path_resolver = path_resolver
        return parser

    # %% diagnostics

    @property
    def source_description(self) -> str:
        """
        :return: A human-readable identifier of :attr:`stage`, used in error messages -
            its file path if it was opened from one, its in-memory layer identifier
            otherwise.
        """
        return self.stage.GetRootLayer().identifier

    # %% stage structure

    def _root_prim(self) -> Optional[Usd.Prim]:
        """
        :return: The stage's default prim, or its single top-level prim if it has no
            default prim and exactly one top-level prim, or ``None`` if neither
            identifies a single prim unambiguously.
        """
        default_prim = self.stage.GetDefaultPrim()
        if default_prim.IsValid():
            return default_prim

        top_level_prims = self._top_level_prims()
        if len(top_level_prims) != 1:
            return None
        return top_level_prims[0]

    def _top_level_prims(self) -> List[Usd.Prim]:
        """
        :return: Every top-level prim of the stage (the pseudo-root's direct children).
        """
        return list(self.stage.GetPseudoRoot().GetChildren())

    # %% shapes

    def _create_shape(
        self, prim: Usd.Prim, link_to_world: Gf.Matrix4d
    ) -> Optional[Shape]:
        """
        Creates the Shape a mesh/primitive prim describes.

        A prim that is not a renderable geometric primitive at all (a plain ``Xform``
        grouping node, a camera, a shader, ...) is not shape geometry and is silently
        skipped; one that is (:class:`~pxr.UsdGeom.Gprim`) but of a type this parser
        does not build a Shape for (e.g. ``Cone``, ``Capsule``) raises instead, the same
        way an unrecognised joint type does, rather than silently vanishing from the
        built world.

        A prim marked as a guide is skipped too: a guide is what a renderer draws
        nothing for, which is the shape a collision proxy authored beside the surface
        it stands for takes.

        :param prim: The prim to create a shape for.
        :param link_to_world: The enclosing link's local-to-world transform.
        :return: The created shape, or ``None`` if ``prim`` is not shape geometry.
        :raises UnsupportedUsdGeometryTypeError: If ``prim`` is a renderable geometric
            primitive of a type this parser does not build a Shape for.
        """
        if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
            return None

        type_name = prim.GetTypeName()
        try:
            geom_type = UsdGeomPrimType(type_name)
        except ValueError as error:
            if prim.IsA(UsdGeom.Gprim):
                raise UnsupportedUsdGeometryTypeError(
                    file_path=self.source_description,
                    prim_path=str(prim.GetPath()),
                    geometry_type=type_name,
                    supported_types=list(UsdGeomPrimType),
                ) from error
            return None
        return geom_type.create_shape(
            prim, link_to_world, self.shading, self.maximum_texture_size
        )

    # %% inertials

    @staticmethod
    def _parse_inertial(link_prim: Usd.Prim, body: Body) -> Optional[Inertial]:
        """
        Parses a link prim's :class:`~pxr.UsdPhysics.MassAPI` inertial properties.

        :param link_prim: The link's root USD prim.
        :param body: The body the properties belong to, used as their reference frame.
        :return: The inertial properties, or ``None`` if ``UsdPhysics.MassAPI`` is not
            applied to the prim.
        """
        if not link_prim.HasAPI(UsdPhysics.MassAPI):
            return None

        mass_api = UsdPhysics.MassAPI(link_prim)
        mass = mass_api.GetMassAttr().Get()
        if mass is None or mass <= 0.0:
            return None

        center_of_mass = mass_api.GetCenterOfMassAttr().Get() or (0.0, 0.0, 0.0)
        diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get() or (0.0, 0.0, 0.0)
        principal_axes = mass_api.GetPrincipalAxesAttr().Get()

        principal_moments = PrincipalMoments.from_values(
            i1=diagonal_inertia[0], i2=diagonal_inertia[1], i3=diagonal_inertia[2]
        )
        axes_rotation = _usd_pose_to_transform(
            Gf.Vec3d(0, 0, 0),
            principal_axes if principal_axes is not None else Gf.Quatf(1, 0, 0, 0),
        )
        inertia_tensor = InertiaTensor.from_principal_moments_and_axes(
            moments=principal_moments,
            axes=PrincipalAxes.from_rotation_matrix(
                RotationMatrix(data=axes_rotation.to_np()[:3, :3])
            ),
        )

        return Inertial(
            mass=mass,
            center_of_mass=Point3(*center_of_mass, reference_frame=body),
            inertia=inertia_tensor,
        )

    # %% semantics

    @classmethod
    def _attach_semantic_labels(cls, world: World, prim: Usd.Prim, body: Body) -> None:
        """
        Attaches a :class:`UsdSemanticLabels` annotation to ``body`` for every
        ``UsdSemantics.LabelsAPI`` taxonomy directly authored on ``prim``, if any.

        Must be called inside ``world``'s modification context, alongside adding
        ``body`` itself.

        :param world: The world to add the annotation to.
        :param prim: The USD prim ``body`` was built from.
        :param body: The already-added body the annotation's root is.
        """
        for taxonomy, labels in cls._read_semantic_labels(prim).items():
            world.add_semantic_annotation(
                UsdSemanticLabels(root=body, taxonomy=taxonomy, labels=labels)
            )

    @staticmethod
    def _read_semantic_labels(prim: Usd.Prim) -> Dict[str, List[str]]:
        """
        Reads every ``UsdSemantics.LabelsAPI`` taxonomy directly authored on a prim.

        Only labels authored directly on ``prim`` are read, the same way
        :meth:`_parse_inertial` only reads a link's own ``UsdPhysics.MassAPI`` - not
        the taxonomies USD's inheritance semantics would additionally consider
        accumulated from an ancestor prim.

        :param prim: The prim to read semantic labels from.
        :return: A mapping of taxonomy to the labels authored under it, empty if
            ``usd-core`` predates ``UsdSemantics`` or the prim has none.
        """
        if UsdSemantics is NoneType:
            return {}
        return {
            taxonomy: list(
                UsdSemantics.LabelsAPI.Get(prim, taxonomy).GetLabelsAttr().Get() or ()
            )
            for taxonomy in UsdSemantics.LabelsAPI.GetDirectTaxonomies(prim)
        }
