from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Dict, List, Optional, Self

from semantic_digital_twin.adapters.usd.stage_parser import (
    Gf,
    UsdAxis,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
    RootPlacement,
    USDStageParser,
    _usd_pose_to_transform,
    geometry_owning_prims,
    scene_ground,
    unique_prim_names,
)
from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdStageOrigin
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% rigid poses


def _rigid_world_pose(prim: Usd.Prim) -> Gf.Matrix4d:
    """
    Build the rigid part of a prim's local-to-world transform, dropping any scale or
    shear an enclosing prim contributes.

    A connection places a body rigidly, so a body's frame can only be the rigid part;
    what is dropped here is what the shape builders bake into the geometry instead,
    which they are given this same pose to do.

    :param prim: The prim whose world pose to build.
    :return: The prim's rigid local-to-world transform.
    """
    transform = Gf.Transform(
        UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    )
    pose = Gf.Matrix4d(1.0)
    pose.SetRotate(transform.GetRotation().GetQuat())
    pose.SetTranslateOnly(transform.GetTranslation())
    return pose


def _translation(offset: Gf.Vec3d) -> Gf.Matrix4d:
    """
    :param offset: The translation the pose consists of.
    :return: A pose translated by ``offset``, with no rotation.
    """
    pose = Gf.Matrix4d(1.0)
    pose.SetTranslateOnly(offset)
    return pose


def _relative_transform(
    parent_pose: Gf.Matrix4d, child_pose: Gf.Matrix4d, reference_frame: Body
) -> HomogeneousTransformationMatrix:
    """
    Build the transform of one rigid world pose relative to another.

    :param parent_pose: The pose the result is expressed relative to.
    :param child_pose: The pose to express.
    :param reference_frame: The frame the result is expressed in.
    :return: The child's pose relative to the parent.
    """
    transform = Gf.Transform(child_pose * parent_pose.GetInverse())
    return _usd_pose_to_transform(
        transform.GetTranslation(),
        transform.GetRotation().GetQuat(),
        reference_frame=reference_frame,
    )


def _stage_origin_in(root_pose: Gf.Matrix4d, root_body: Body) -> Point3:
    """
    :param root_pose: The rigid world pose the root body sits at.
    :param root_body: The body the result is expressed in.
    :return: The stage's origin, in the root body's frame.
    """
    origin = root_pose.GetInverse().Transform(Gf.Vec3d(0, 0, 0))
    return Point3(origin[0], origin[1], origin[2], reference_frame=root_body)


# %% placed objects


@dataclass
class PlacedObject:
    """
    One object of a scene stage: the body built for a geometry-owning prim, and the
    rigid world pose it is placed at.
    """

    body: Body
    """
    The body built for the object.
    """

    world_pose: Gf.Matrix4d
    """
    The rigid part of the object's local-to-world transform.
    """

    prim: Optional[Usd.Prim] = None
    """
    The prim the object was built from, ``None`` for a synthetic root that no prim of
    the stage corresponds to.
    """


# %% scene parser


@dataclass
class USDSceneParser(USDStageParser):
    """
    Parses a USD stage describing separately placed static objects into a world.

    Where an articulated asset's structure is its physics joints, a scene's is its
    transform hierarchy: a laser scan or a dressed set carries no physics at all, only
    prims that own geometry and the ``Xform`` groups that place them. Every prim that
    directly holds geometry becomes a body, fixed where the stage places it - rigidly,
    since a measured object is not freely posable.

    A grouping prim that owns no geometry itself is not a body; its transform still
    reaches the objects it holds, through their own local-to-world transforms.

    The surfaces the stage holds are reported as visual geometry. What a body is
    collided against is what the stage says it is: every geometry prim below the object
    that carries :class:`~pxr.UsdPhysics.CollisionAPI`, guides included, since a guide is
    the shape a collision box authored beside a surface takes. A stage authoring no
    collision - a scan straight from the vendor - is collided against by nothing, because
    handing a scanned surface to a collision detector reads all of it back into memory;
    apply a :class:`~semantic_digital_twin.pipeline.mesh_decomposition.base.MeshDecomposer`
    step to decide what such a scene collides as.

    .. note::
        A stage describing one physically articulated asset is read by
        :class:`~semantic_digital_twin.adapters.usd.parser.USDParser` instead.
    """

    root_placement: RootPlacement = RootPlacement.STAGE_ORIGIN
    """
    Where the world root is placed.
    """

    # %% construction

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: Optional[str] = None,
        path_resolver: Optional[PathResolver] = None,
        root_placement: RootPlacement = RootPlacement.STAGE_ORIGIN,
    ) -> Self:
        """
        Creates a parser for a USD scene stage file.

        :param file_path: The path of the stage file to parse.
        :param prefix: The prefix for every name used in this world.
        :param path_resolver: The resolver for the asset references of the stage.
        :param root_placement: Where the world root is placed.
        :return: A parser for the described world.
        """
        parser = super().from_file(file_path, prefix, path_resolver)
        parser.root_placement = root_placement
        return parser

    # %% entry point

    def parse(self) -> World:
        """
        Parses the stage into a world.

        :return: The parsed world.
        :raises UnsupportedUsdGeometryTypeError: If the stage contains a renderable
            geometric primitive of a type this parser does not build a Shape for.
        """
        object_prims = self._object_prims()
        root_prim = self._root_prim()
        root_path = root_prim.GetPath() if root_prim is not None else None
        root_pose = self._root_pose(root_prim, object_prims)

        # Every shape is built before the world exists: a World left partway through a
        # failed modification is unusable, and unsupported geometry raises.
        root_shapes = (
            self._object_shapes(root_prim, root_pose) if root_prim is not None else []
        )
        body_names = unique_prim_names(
            [prim for prim in object_prims if prim.GetPath() != root_path]
        )
        objects = [
            self._create_object(prim, body_names[prim.GetPath().pathString])
            for prim in object_prims
            if prim.GetPath() != root_path
        ]

        world = World.create_with_root_body(
            root_prim.GetName() if root_prim is not None else self.prefix, self.prefix
        )
        root = PlacedObject(body=world.root, world_pose=root_pose, prim=root_prim)
        visual = ShapeCollection(root_shapes, reference_frame=world.root)
        visual.transform_all_shapes_to_own_frame()
        world.root.visual = visual

        objects_by_path: Dict[Sdf.Path, PlacedObject] = {
            placed_object.prim.GetPath(): placed_object for placed_object in objects
        }
        if root_path is not None:
            objects_by_path[root_path] = root

        with world.modify_world():
            world.add_semantic_annotation(
                UsdStageOrigin(
                    root=world.root, position=_stage_origin_in(root_pose, world.root)
                )
            )
            if root_prim is not None:
                self._attach_semantic_labels(world, root_prim, world.root)
            for placed_object in objects:
                world.add_body(placed_object.body)
                self._attach_semantic_labels(
                    world, placed_object.prim, placed_object.body
                )
            for placed_object in objects:
                self._connect(world, placed_object, objects_by_path, root)
        return world

    # %% root placement

    def _root_pose(
        self, root_prim: Optional[Usd.Prim], object_prims: List[Usd.Prim]
    ) -> Gf.Matrix4d:
        """
        :param root_prim: The prim the world is rooted at, ``None`` for a synthetic
            root no prim of the stage corresponds to.
        :param object_prims: Every geometry-owning prim of the stage.
        :return: The rigid world pose the root body sits at.
        """
        if self.root_placement is RootPlacement.SCENE_GROUND:
            return _translation(scene_ground(self.stage, object_prims))
        if root_prim is None:
            return Gf.Matrix4d(1.0)
        return _rigid_world_pose(root_prim)

    # %% objects

    def _object_prims(self) -> List[Usd.Prim]:
        """
        :return: Every prim of the stage that directly holds renderable geometry, in
            stage order - each geometry prim belongs to exactly one of them, so no
            geometry of the stage is left without a body to hold it.
        """
        return geometry_owning_prims(self.stage)

    def _create_object(self, object_prim: Usd.Prim, name: str) -> PlacedObject:
        """
        Creates the body for one geometry-owning prim, with a Shape for every geometry
        prim it holds and its :class:`~pxr.UsdPhysics.MassAPI` inertial properties, if
        applied.

        :param object_prim: The prim to build an object for.
        :param name: What to call the body, unique across the scene.
        :return: The created object, its body not yet added to a world.
        """
        world_pose = _rigid_world_pose(object_prim)
        body = Body(
            name=PrefixedName(name, self.prefix),
            visual=ShapeCollection(self._object_shapes(object_prim, world_pose)),
            collision=ShapeCollection(self._collision_shapes(object_prim, world_pose)),
        )
        inertial = self._parse_inertial(object_prim, body)
        if inertial is not None:
            body.inertial = inertial
        return PlacedObject(body=body, world_pose=world_pose, prim=object_prim)

    def _object_shapes(
        self, object_prim: Usd.Prim, world_pose: Gf.Matrix4d
    ) -> List[Shape]:
        """
        Creates the Shape for every geometry prim one object holds.

        :param object_prim: The prim whose geometry to build shapes for.
        :param world_pose: The object's rigid local-to-world transform, which its
            shapes are positioned relative to.
        :return: The created shapes.
        """
        shapes = [
            self._create_shape(child, world_pose) for child in object_prim.GetChildren()
        ]
        return [shape for shape in shapes if shape is not None]

    def _collision_shapes(
        self, object_prim: Usd.Prim, world_pose: Gf.Matrix4d
    ) -> List[Shape]:
        """
        Creates the Shape for every geometry prim a physics engine collides one object
        against: the prims below it carrying :class:`~pxr.UsdPhysics.CollisionAPI`,
        however deeply an asset library nested them, but not those of an object of its
        own held beneath it, which collides as itself.

        :param object_prim: The prim whose collision to build shapes for.
        :param world_pose: The object's rigid local-to-world transform, which its
            shapes are positioned relative to.
        :return: The created shapes.
        """
        objects_of_their_own = {prim.GetPath() for prim in self._object_prims()}
        shapes = []
        below = iter(Usd.PrimRange(object_prim))
        for prim in below:
            if prim != object_prim and prim.GetPath() in objects_of_their_own:
                below.PruneChildren()
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            shape = self._build_shape(prim, world_pose)
            if shape is not None:
                shapes.append(shape)
        return shapes

    # %% placement

    def _connect(
        self,
        world: World,
        placed_object: PlacedObject,
        objects_by_path: Dict[Sdf.Path, PlacedObject],
        root: PlacedObject,
    ) -> None:
        """
        Fixes one object to the object whose subtree holds it.

        :param world: The world to add the connection to.
        :param placed_object: The object to fix in place.
        :param objects_by_path: Every object of the stage, by the path of its prim.
        :param root: The object to fall back to when no prim above this one holds
            geometry.
        """
        parent = self._enclosing_object(placed_object.prim, objects_by_path, root)
        world.add_connection(
            FixedConnection.create_with_dofs(
                world=world,
                parent=parent.body,
                child=placed_object.body,
                parent_T_connection_expression=_relative_transform(
                    parent.world_pose, placed_object.world_pose, parent.body
                ),
            )
        )

    @staticmethod
    def _enclosing_object(
        object_prim: Usd.Prim,
        objects_by_path: Dict[Sdf.Path, PlacedObject],
        root: PlacedObject,
    ) -> PlacedObject:
        """
        :param object_prim: The prim whose enclosing object to find.
        :param objects_by_path: Every object of the stage, by the path of its prim.
        :param root: The object to fall back to when no prim above ``object_prim``
            holds geometry.
        :return: The nearest object above ``object_prim`` in the stage.
        """
        ancestor = object_prim.GetParent()
        while ancestor.IsValid() and not ancestor.IsPseudoRoot():
            enclosing_object = objects_by_path.get(ancestor.GetPath())
            if enclosing_object is not None:
                return enclosing_object
            ancestor = ancestor.GetParent()
        return root
