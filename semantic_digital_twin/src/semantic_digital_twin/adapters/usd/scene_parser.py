from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from semantic_digital_twin.adapters.usd import parser as usd
from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdPhysicsJointTypeError,
)
from semantic_digital_twin.adapters.usd.parser import USDParser
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Shape
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from typing_extensions import Iterator
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
    Import separate static scene objects in authored stage coordinates.
    """

    _body_paths: set[str] = field(default_factory=set, init=False)
    """
    Object boundaries excluded from an enclosing body's geometry traversal.
    """

    def parse(self) -> World:
        """
        Build independently named bodies without duplicating nested geometry.
        """
        object_prims = self._collect_object_prims()
        self._body_paths = {str(path) for path in object_prims}
        bodies: dict[str, Body] = {}
        world = World.create_with_root_body(
            self.stage.GetPseudoRoot().GetName(), self.prefix
        )
        with world.modify_world():
            for path, prim in object_prims.items():
                body = self._create_link_body(
                    prim, self._resolve_rigid_world_transform(prim)
                )
                body.name = PrefixedName(str(path).lstrip("/"), self.prefix)
                bodies[str(path)] = body
                world.add_body(body)
                self._attach_semantic_labels(world, prim, body)
            for path, prim in object_prims.items():
                parent_prim = prim.GetParent()
                while (
                    not parent_prim.IsPseudoRoot()
                    and parent_prim.GetPath() not in object_prims
                ):
                    parent_prim = parent_prim.GetParent()
                parent = (
                    world.root
                    if parent_prim.IsPseudoRoot()
                    else bodies[str(parent_prim.GetPath())]
                )
                world_T_parent = (
                    usd.Gf.Matrix4d(1.0)
                    if parent is world.root
                    else self._resolve_rigid_world_transform(parent_prim)
                )
                parent_T_child = (
                    self._resolve_rigid_world_transform(prim)
                    * world_T_parent.GetInverse()
                )
                child = bodies[str(path)]
                world.add_connection(
                    FixedConnection(
                        parent=parent,
                        child=child,
                        parent_T_connection_expression=HomogeneousTransformationMatrix(
                            data=np.array(parent_T_child).T,
                            reference_frame=parent,
                            child_frame=child,
                        ),
                    )
                )
        return world

    def _resolve_object_prim(self, prim_path: usd.Sdf.Path) -> usd.Usd.Prim:
        """
        Resolve geometry to its nearest rigid body or model component.
        """
        prim = self.stage.GetPrimAtPath(prim_path)
        owner = (
            prim.GetParent()
            if prim.IsA(usd.UsdGeom.Gprim) and not prim.GetParent().IsPseudoRoot()
            else prim
        )
        ancestor = owner
        while not ancestor.IsPseudoRoot():
            model = usd.Usd.ModelAPI(ancestor)
            if ancestor.HasAPI(usd.UsdPhysics.RigidBodyAPI) or (
                any(
                    model.IsKind(kind, usd.Usd.ModelAPI.KindValidationNone)
                    for kind in UsdObjectKind
                )
            ):
                return ancestor
            ancestor = ancestor.GetParent()
        return owner

    def _collect_object_prims(self) -> dict[usd.Sdf.Path, usd.Usd.Prim]:
        """
        Find geometry owners while rejecting unsupported scene articulations.
        """
        objects: dict[usd.Sdf.Path, usd.Usd.Prim] = {}
        for prim in self.stage.Traverse():
            if prim.IsA(usd.UsdPhysics.Joint):
                raise UnsupportedUsdPhysicsJointTypeError(
                    file_path=self.source_description,
                    joint_path=str(prim.GetPath()),
                    joint_type=prim.GetTypeName(),
                )
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
