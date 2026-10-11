from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import rustworkx

from semantic_digital_twin.adapters.usd import parser as usd
from semantic_digital_twin.adapters.usd.exceptions import (
    UsdSceneTopologyError,
    UsdScaledJointFrameError,
)
from semantic_digital_twin.adapters.usd.parser import USDParser
from semantic_digital_twin.adapters.usd.semantics import UsdSceneSemantics
from semantic_digital_twin.adapters.world_model_parser import JointDescription
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from typing_extensions import Iterator, Self
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from semantic_digital_twin.adapters.package_resolver import PathResolver
from semantic_digital_twin.world_description.world_entity import Body

# %% scene object ownership


class UsdObjectKind(StrEnum):
    """
    USD model kinds that own their internal geometry.
    """

    COMPONENT = "component"
    SUBCOMPONENT = "subcomponent"


@dataclass
class UsdSceneGeometry:
    """Visible shapes and authored collision shapes owned by one scene body."""

    visual: list[Shape] = field(default_factory=list)
    """Shapes visible in the stage."""

    collision: list[Shape] = field(default_factory=list)
    """Shapes with enabled authored collision."""

    has_authored_collision: bool = False
    """Whether collision was explicitly authored, including disabled colliders."""


@dataclass
class USDSceneParser(USDParser):
    """
    Import separate scene objects and tree-shaped articulations in stage coordinates.
    """

    semantics: UsdSceneSemantics = field(
        default_factory=UsdSceneSemantics, kw_only=True
    )
    """Interpretation of authored scene support labels."""

    world_T_scene: HomogeneousTransformationMatrix = field(
        default_factory=HomogeneousTransformationMatrix, kw_only=True
    )
    """Rigid placement of authored stage coordinates in the world frame."""

    _body_paths: set[str] = field(default_factory=set, init=False)
    """
    Object boundaries excluded from an enclosing body's geometry traversal.
    """

    _joint_paths: set[str] = field(default_factory=set, init=False)
    """Explicit joint endpoints that remain separate within model components."""

    # %% scene construction

    @classmethod
    def from_file(
        cls,
        file_path: str,
        prefix: str | None = None,
        path_resolver: PathResolver | None = None,
        *,
        world_T_scene: HomogeneousTransformationMatrix | None = None,
        semantics: UsdSceneSemantics | None = None,
    ) -> Self:
        """Open a composed scene with an optional world placement and label vocabulary."""
        parser = super().from_file(
            file_path, prefix=prefix, path_resolver=path_resolver
        )
        if world_T_scene is not None:
            parser.world_T_scene = world_T_scene
        if semantics is not None:
            parser.semantics = semantics
        return parser

    def parse(self) -> World:
        """
        Build the scene without flattening objects or discarding unrelated geometry.
        """
        joint_prims = [
            prim for prim in self.stage.Traverse() if prim.IsA(usd.UsdPhysics.Joint)
        ]
        object_prims = self._collect_object_prims(joint_prims)
        self._body_paths = {str(path) for path in object_prims}
        bodies: dict[str, Body] = {}
        for path, prim in object_prims.items():
            body = self._create_link_body(
                prim, self._resolve_rigid_world_transform(prim)
            )
            body.name = PrefixedName(str(path).lstrip("/"), self.prefix)
            bodies[str(path)] = body

        # A stage frame is distinct from authored prims, including the default prim.
        world = World.create_with_root_body(
            self.stage.GetPseudoRoot().GetName(), self.prefix
        )
        descriptions = [
            self._describe_joint(prim, world.root, bodies) for prim in joint_prims
        ]
        self._complete_connections(world.root, object_prims, bodies, descriptions)
        self._validate_tree(world.root, descriptions)
        for description in descriptions:
            if description.parent is world.root:
                description.parent_T_connection = (
                    self.world_T_scene @ description.parent_T_connection
                )
        with world.modify_world():
            for path, body in bodies.items():
                world.add_body(body)
                self._attach_semantic_labels(
                    world, self.stage.GetPrimAtPath(path), body
                )
            for description in descriptions:
                world.add_connection(self._create_connection(world, description))
        return world

    def _resolve_object_prim(self, prim_path: usd.Sdf.Path) -> usd.Usd.Prim:
        """
        Resolve geometry to its nearest rigid body or model component.
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or prim.IsPseudoRoot() or prim.IsA(usd.UsdPhysics.Joint):
            raise UsdSceneTopologyError(
                file_path=self.source_description, prim_path=str(prim_path)
            )
        owner = (
            prim.GetParent()
            if prim.IsA(usd.UsdGeom.Gprim) and not prim.GetParent().IsPseudoRoot()
            else prim
        )
        ancestor = owner
        while not ancestor.IsPseudoRoot():
            model = usd.Usd.ModelAPI(ancestor)
            if (
                str(ancestor.GetPath()) in self._joint_paths
                or self._read_semantic_labels(ancestor)
                or ancestor.HasAPI(usd.UsdPhysics.RigidBodyAPI)
                or (
                    any(
                        model.IsKind(kind, usd.Usd.ModelAPI.KindValidationNone)
                        for kind in UsdObjectKind
                    )
                )
            ):
                return ancestor
            ancestor = ancestor.GetParent()
        return owner

    def _collect_object_prims(
        self, joint_prims: list[usd.Usd.Prim]
    ) -> dict[usd.Sdf.Path, usd.Usd.Prim]:
        """Find geometry owners and explicit joint endpoints throughout the stage."""
        objects: dict[usd.Sdf.Path, usd.Usd.Prim] = {}
        self._joint_paths = set()
        for prim in joint_prims:
            joint = usd.UsdPhysics.Joint(prim)
            for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                targets = relationship.GetTargets()
                if len(targets) > 1:
                    raise UsdSceneTopologyError(
                        file_path=self.source_description, prim_path=str(prim.GetPath())
                    )
                for target in targets:
                    endpoint = self.stage.GetPrimAtPath(target)
                    if (
                        not endpoint.IsValid()
                        or endpoint.IsPseudoRoot()
                        or endpoint.IsA(usd.UsdPhysics.Joint)
                    ):
                        raise UsdSceneTopologyError(
                            file_path=self.source_description, prim_path=str(target)
                        )
                    if (
                        endpoint.IsA(usd.UsdGeom.Gprim)
                        and not endpoint.GetParent().IsPseudoRoot()
                    ):
                        endpoint = endpoint.GetParent()
                    self._joint_paths.add(str(endpoint.GetPath()))
        for prim in joint_prims:
            joint = usd.UsdPhysics.Joint(prim)
            for relationship in (joint.GetBody0Rel(), joint.GetBody1Rel()):
                for target in relationship.GetTargets():
                    owner = self._resolve_object_prim(target)
                    authored = usd.UsdGeom.Xformable(
                        owner
                    ).ComputeLocalToWorldTransform(usd.Usd.TimeCode.Default())
                    if not np.allclose(
                        np.array(authored),
                        np.array(self._resolve_rigid_world_transform(owner)),
                    ):
                        raise UsdScaledJointFrameError(
                            file_path=self.source_description, prim_path=str(target)
                        )
                    objects[owner.GetPath()] = owner
        for prim in self.stage.Traverse():
            if prim.IsA(usd.UsdGeom.Gprim) and (
                usd.UsdGeom.Imageable(prim).ComputePurpose() != usd.UsdGeom.Tokens.guide
                or prim.HasAPI(usd.UsdPhysics.CollisionAPI)
            ):
                owner = self._resolve_object_prim(prim.GetPath())
                objects[owner.GetPath()] = owner
        return objects

    # %% visual and collision geometry

    def _geometry_prims(self, link_prim: usd.Usd.Prim) -> Iterator[usd.Usd.Prim]:
        """Yield geometry belonging to this object, excluding nested body boundaries."""
        iterator = iter(usd.Usd.PrimRange(link_prim))
        for prim in iterator:
            if prim != link_prim and str(prim.GetPath()) in self._body_paths:
                iterator.PruneChildren()
                continue
            if prim.IsA(usd.UsdGeom.Gprim):
                yield prim

    def _collect_geometry(
        self, link_prim: usd.Usd.Prim, link_to_world: usd.Gf.Matrix4d
    ) -> UsdSceneGeometry:
        """Separate visible shapes from authored colliders within one object."""
        geometry = UsdSceneGeometry()
        for prim in self._geometry_prims(link_prim):
            imageable = usd.UsdGeom.Imageable(prim)
            visible = imageable.ComputePurpose() != usd.UsdGeom.Tokens.guide
            authored_collision = prim.HasAPI(usd.UsdPhysics.CollisionAPI)
            geometry.has_authored_collision |= authored_collision
            collision_enabled = (
                authored_collision
                and usd.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            )
            if not visible and not collision_enabled:
                continue
            shape = self._create_shape(prim, link_to_world)
            if visible:
                geometry.visual.append(shape)
            if collision_enabled:
                geometry.collision.append(shape)
        if not geometry.has_authored_collision:
            geometry.collision = list(geometry.visual)
        return geometry

    def _create_link_body(
        self, link_prim: usd.Usd.Prim, link_to_world: usd.Gf.Matrix4d | None = None
    ) -> Body:
        """Build a body with distinct visual and collision shape collections."""
        if link_to_world is None:
            link_to_world = self._resolve_rigid_world_transform(link_prim)
        geometry = self._collect_geometry(link_prim, link_to_world)
        body = Body(
            name=PrefixedName(link_prim.GetName(), self.prefix),
            visual=ShapeCollection(geometry.visual),
            collision=ShapeCollection(geometry.collision),
        )
        inertial = self._parse_inertial(link_prim, body)
        if inertial is not None:
            body.inertial = inertial
        return body

    # %% scene connection tree

    def _resolve_link_body(
        self, link_bodies: dict[str, Body], prim_path: usd.Sdf.Path
    ) -> Body:
        """
        Return the already-created body for a validated joint endpoint.
        """
        return link_bodies[str(self._resolve_object_prim(prim_path).GetPath())]

    def _complete_connections(
        self,
        root_body: Body,
        object_prims: dict[usd.Sdf.Path, usd.Usd.Prim],
        bodies: dict[str, Body],
        descriptions: list[JointDescription],
    ) -> None:
        """
        Attach objects not controlled by physics joints to their enclosing body.
        """
        connected_children = set()
        for description in descriptions:
            if description.child in connected_children:
                raise UsdSceneTopologyError(
                    file_path=self.source_description,
                    prim_path=str(description.child.name),
                )
            connected_children.add(description.child)
        for path, prim in object_prims.items():
            child = bodies[str(path)]
            if child in connected_children:
                continue
            parent_prim = prim.GetParent()
            while (
                not parent_prim.IsPseudoRoot()
                and parent_prim.GetPath() not in object_prims
            ):
                parent_prim = parent_prim.GetParent()
            parent = (
                root_body
                if parent_prim.IsPseudoRoot()
                else bodies[str(parent_prim.GetPath())]
            )
            world_T_parent = (
                usd.Gf.Matrix4d(1.0)
                if parent is root_body
                else self._resolve_rigid_world_transform(parent_prim)
            )
            parent_T_child = (
                self._resolve_rigid_world_transform(prim) * world_T_parent.GetInverse()
            )
            descriptions.append(
                JointDescription(
                    connection_type=FixedConnection,
                    parent=parent,
                    child=child,
                    parent_T_connection=HomogeneousTransformationMatrix(
                        data=np.array(parent_T_child).T,
                        reference_frame=parent,
                        child_frame=child,
                    ),
                    connection_T_child=HomogeneousTransformationMatrix(),
                )
            )

    def _validate_tree(
        self, root_body: Body, descriptions: list[JointDescription]
    ) -> None:
        """
        Reject cycles before the scene is added to the world.
        """
        graph = rustworkx.PyDiGraph()
        bodies = {root_body}
        bodies.update(description.child for description in descriptions)
        indices = {body: graph.add_node(body) for body in bodies}
        for description in descriptions:
            graph.add_edge(
                indices[description.parent], indices[description.child], description
            )
        cycle = rustworkx.digraph_find_cycle(graph)
        if not cycle:
            return
        _, child_index = cycle[0]
        raise UsdSceneTopologyError(
            file_path=self.source_description, prim_path=str(graph[child_index].name)
        )

    # %% inherited scene labels

    def _attach_semantic_labels(
        self, world: World, prim: usd.Usd.Prim, body: Body
    ) -> None:
        """Attach labels from the object and its ancestors, nearest labels first."""
        labels_by_taxonomy: dict[str, list[str]] = {}
        support_type = None
        while not prim.IsPseudoRoot():
            direct_labels = self._read_semantic_labels(prim)
            if support_type is None:
                support_type = self.semantics.find_support_type(direct_labels)
            for taxonomy, labels in direct_labels.items():
                inherited = labels_by_taxonomy.setdefault(taxonomy, [])
                inherited.extend(label for label in labels if label not in inherited)
            prim = prim.GetParent()
        self.semantics.apply(world, body, labels_by_taxonomy, support_type)
